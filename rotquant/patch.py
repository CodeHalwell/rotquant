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

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

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
    include: Optional[Iterable[str]] = None
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
        logger.warning("patch mode 'mismatched' active -- consistency invariant "
                       "intentionally violated (E7 only)")
    else:
        raise ValueError(f"unknown patch mode: {cfg.mode}; pick from {PATCH_MODES}")
    return weight_rot, act_rot


def patch_model(model: nn.Module, cfg: PatchConfig,
                hessians: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
    """Replace targeted ``nn.Linear`` layers with ``QuantLinear`` in-place."""
    if cfg.mode not in PATCH_MODES:
        raise ValueError(f"unknown patch mode: {cfg.mode}")
    hessians = hessians or {}

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and (cfg.include is None or any(k in n for k in cfg.include))]

    for i, (name, linear) in enumerate(targets):
        weight_rot, act_rot = _make_rotations(linear.in_features, cfg,
                                               layer_seed=cfg.seed + i)
        H = hessians.get(name)
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

    logger.info("Patched %d linear layers (rotation=%s, mode=%s)",
                len(targets), cfg.rotation, cfg.mode)
    return model
