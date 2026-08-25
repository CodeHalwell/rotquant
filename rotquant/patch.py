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
from .rotate import Identity, Rotation, build_rotation
from .utils import get_logger

logger = get_logger()

PATCH_MODES = ("consistent", "fused_inverse", "mismatched")


@dataclass
class PatchConfig:
    quant: QuantConfig
    rotation: str = "fwht"            # none | fwht | dense | learned
    block: int = 128
    mode: str = "consistent"          # see PATCH_MODES
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


def _make_rotations(in_features: int, cfg: PatchConfig, layer_seed: int):
    """Return (weight_rotation, act_rotation) honouring the consistency mode."""
    weight_rot = build_rotation(cfg.rotation, in_features, block=cfg.block,
                                seed=layer_seed)
    if cfg.mode in ("consistent", "fused_inverse"):
        act_rot: Rotation = weight_rot           # matched basis -- the invariant
    elif cfg.mode == "mismatched":
        # Deliberately break it: rotate the weight but leave activations un-rotated.
        act_rot = Identity(in_features)
    else:
        raise ValueError(f"unknown patch mode: {cfg.mode}; pick from {PATCH_MODES}")
    return weight_rot, act_rot


def patch_model(model: nn.Module, cfg: PatchConfig,
                hessians: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
    """Replace targeted ``nn.Linear`` layers with ``QuantLinear`` in-place."""
    if cfg.mode not in PATCH_MODES:
        raise ValueError(f"unknown patch mode: {cfg.mode}")
    if cfg.mode == "mismatched":
        logger.warning("patch mode 'mismatched' active -- consistency invariant "
                       "intentionally violated (E7 only)")
    if cfg.rotation in ("learned", "cayley", "stiefel"):
        logger.warning(
            "rotation='learned' starts at ~identity (theta init 1e-3): without a "
            "training step on theta this arm measures a no-rotation control, not "
            "a learned rotation.")
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

    for i, (name, linear) in enumerate(targets):
        weight_rot, act_rot = _make_rotations(linear.in_features, cfg,
                                               layer_seed=cfg.seed + i)
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

    logger.info("Patched %d linear layers (rotation=%s, mode=%s)",
                len(targets), cfg.rotation, cfg.mode)
    return model
