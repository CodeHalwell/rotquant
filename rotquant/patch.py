"""Model patching: walk a HF model, replace ``nn.Linear`` with ``QuantLinear``,
and enforce the rotation-consistency rules.

The consistency invariant: every rotated weight must have its matching activation
rotation, and the inverse transform is fused into dequant -- no mixed bases. Three
patch modes are exposed for E7:

* ``consistent``    -- weight and activation share one rotation per layer (correct).
* ``fused_inverse`` -- same as consistent, recording that the inverse is folded
  into dequant (the production path); behaviourally identical to ``consistent`` for
  a single linear, kept distinct for bookkeeping/plots.
* ``mismatched``    -- the weight is rotated but the activation is rotated by a
  *different* (or absent) basis, deliberately breaking consistency to surface the
  cross-layer drift the trap predicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .linear import QuantLinear
from .quantize import QuantConfig
from .rotate import Identity, LearnedRotation, Rotation, build_rotation
from .utils import get_logger

logger = get_logger()

PATCH_MODES = ("consistent", "fused_inverse", "mismatched")


@dataclass
class PatchConfig:
    quant: QuantConfig
    rotation: str = "fwht"            # none | fwht | dense | learned
    block: int = 128
    mode: str = "consistent"          # see PATCH_MODES
    # kwargs for rotquant.train_rotation.RotationTrainConfig (e.g.
    # {"steps": 200, "lr": 1e-3}). Only meaningful with rotation="learned":
    # theta is optimised per layer (data-free, alternating minimisation) before
    # quantisation. None leaves theta at its ~identity init (and warns).
    train_rotation: Optional[Dict] = None
    include: Optional[Sequence[str]] = None
    # Substrings of layer names to leave in fp16. The default keeps the output
    # head (and its tied embedding) unquantised -- the convention every baseline
    # (GPTQ/AWQ/QuIP#/AQLM) follows, so results stay comparable. Pass () to
    # quantise everything.
    exclude: Sequence[str] = ("lm_head", "embed_out")
    fallback: bool = False
    seed: int = 0


def _get_parent(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _make_rotations(in_features: int, cfg: PatchConfig, layer_seed: int,
                    device=None):
    """Return (weight_rotation, act_rotation) honouring the consistency mode.

    ``device`` should be the target weight's device: the rotation is applied to
    that weight *before* the module tree is ever moved, so its buffers must be
    built where the weight lives.
    """
    weight_rot = build_rotation(cfg.rotation, in_features, block=cfg.block,
                                seed=layer_seed, device=device)
    if cfg.mode in ("consistent", "fused_inverse"):
        act_rot: Rotation = weight_rot           # matched basis -- the invariant
    elif cfg.mode == "mismatched":
        # Deliberately break it: rotate the weight but leave activations un-rotated.
        act_rot = Identity(in_features)
    else:
        raise ValueError(f"unknown patch mode: {cfg.mode}; pick from {PATCH_MODES}")
    return weight_rot, act_rot


def patch_model(model: nn.Module, cfg: PatchConfig,
                hessians: Optional[Dict[str, torch.Tensor]] = None,
                stats_out: Optional[Dict] = None) -> nn.Module:
    """Replace targeted ``nn.Linear`` layers with ``QuantLinear`` in-place.

    ``stats_out``: optional dict filled with patching side-info (currently
    per-run rotation-training aggregates under ``"rotation_train"``).
    """
    if cfg.mode not in PATCH_MODES:
        raise ValueError(f"unknown patch mode: {cfg.mode}")
    if cfg.mode == "mismatched":
        logger.warning("patch mode 'mismatched' active -- consistency invariant "
                       "intentionally violated (E7 only)")
    learned_kind = cfg.rotation in ("learned", "cayley", "stiefel")
    if learned_kind and cfg.train_rotation is None:
        logger.warning(
            "rotation='learned' starts at ~identity (theta init 1e-3): without "
            "patch.train_rotation (e.g. {steps: 200}) this arm measures a "
            "no-rotation control, not a learned rotation.")
    hessians = hessians or {}

    include_terms = tuple(cfg.include) if cfg.include is not None else None
    exclude_terms = tuple(cfg.exclude or ())
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and (include_terms is None or any(k in n for k in include_terms))
               and not any(k in n for k in exclude_terms)]
    if not targets:
        logger.warning(
            "patch_model found NO nn.Linear layers to quantise (include=%s, "
            "exclude=%s). GPT-2-style models use transformers Conv1D, which is "
            "not supported -- the model is still full-precision!",
            include_terms, exclude_terms)
        return model

    train_stats: list = []
    for i, (name, linear) in enumerate(targets):
        weight_rot, act_rot = _make_rotations(linear.in_features, cfg,
                                               layer_seed=cfg.seed + i,
                                               device=linear.weight.device)
        if isinstance(weight_rot, LearnedRotation) and cfg.train_rotation is not None:
            from .train_rotation import RotationTrainConfig, train_layer_rotation
            stats = train_layer_rotation(weight_rot, linear.weight, cfg.quant,
                                         RotationTrainConfig(**cfg.train_rotation))
            train_stats.append(stats)
            logger.debug("trained rotation for %s: mse %.5f -> %.5f",
                         name, stats["initial_mse"], stats["final_mse"])
        H = hessians.get(name)
        if H is not None:
            # Hessians may have been offloaded to CPU by collect_hessians;
            # bring them back next to the weight for the rotation + GPTQ solve.
            H = H.to(linear.weight.device)
        if H is not None and cfg.rotation not in ("none", "identity"):
            # Rotate the Hessian into the same basis as the rotated weight:
            # H' = R H R^T so GPTQ sees the consistent input statistics.
            R = weight_rot.as_matrix(device=H.device, dtype=torch.float64)
            H = (R @ H.to(torch.float64) @ R.transpose(-1, -2)).to(torch.float32)
        qlin = QuantLinear.from_linear(linear, cfg.quant,
                                       weight_rotation=weight_rot,
                                       act_rotation=act_rot, H=H,
                                       fallback=cfg.fallback)
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, qlin)
        if (i + 1) % 32 == 0:
            logger.info("patched %d/%d layers (last: %s)", i + 1, len(targets), name)

    if train_stats:
        agg = {
            "layers": len(train_stats),
            "steps": train_stats[0]["steps"],
            "mean_initial_mse": sum(s["initial_mse"] for s in train_stats) / len(train_stats),
            "mean_final_mse": sum(s["final_mse"] for s in train_stats) / len(train_stats),
        }
        logger.info("rotation training: mean quant-MSE %.5f -> %.5f over %d layers",
                    agg["mean_initial_mse"], agg["mean_final_mse"], agg["layers"])
        if stats_out is not None:
            stats_out["rotation_train"] = agg

    logger.info("Patched %d linear layers (rotation=%s, mode=%s)",
                len(targets), cfg.rotation, cfg.mode)
    return model
