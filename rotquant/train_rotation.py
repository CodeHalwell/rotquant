"""Data-free training for :class:`~rotquant.rotate.LearnedRotation` (E1).

The learned arm is only meaningful if theta is actually optimised: the Cayley
parametrisation initialises at ~identity, i.e. a no-rotation control.

Objective (per layer, no calibration data): minimise the quantisation error of
the rotated weight,

    L(theta) = || Q(W R(theta)^T) - W R(theta)^T ||^2

Because R is exactly orthogonal, ``||W R^T||_F`` is invariant and the effective
un-rotated weight error ``||Q(W R^T) R - W||_F`` equals the rotated-domain error
-- so this is exactly the layer's output MSE under isotropic inputs, and the
rotation can only improve the loss by reshaping the weight distribution to fit
the codebook grid, never by shrinking it.

Optimisation is alternating minimisation in the Lloyd style: each step the
quantisation assignments (indices *and* group scales) are recomputed for the
current rotation and then held fixed (detached), and one Adam step is taken on
theta through the differentiable Cayley map. A plain straight-through gradient
of ``||Q(v) - v||^2`` would cancel to zero; fixing the assignments gives the
exact gradient of the current-assignment objective.

``error_comp`` (GPTQ / residual passes) is deliberately ignored here: those
compensate *after* rounding, so the rotation is trained on the raw grid fit it
actually determines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .quantize import QuantConfig, Quantizer, _quantize_groups
from .rotate import LearnedRotation
from .utils import get_logger

logger = get_logger()


@dataclass
class RotationTrainConfig:
    steps: int = 100
    lr: float = 1e-3


def train_layer_rotation(rotation: LearnedRotation, weight: torch.Tensor,
                         quant_cfg: QuantConfig,
                         config: RotationTrainConfig = None) -> Dict[str, float]:
    """Optimise ``rotation.theta`` in place for one layer's weight.

    Returns ``{"initial_mse", "final_mse", "steps"}`` (rotated-domain
    quantisation MSE before the first and after the last step). The rotation is
    left in eval mode with its matrix cache warm.
    """
    cfg = config or RotationTrainConfig()
    quantizer = Quantizer(quant_cfg)
    w = weight.detach().to(torch.float32)

    def quantize_current(v: torch.Tensor) -> torch.Tensor:
        scales = quantizer._select_scales(v)
        q, _ = _quantize_groups(v, scales, quantizer.codebook,
                                quant_cfg.group_size)
        return q

    @torch.no_grad()
    def current_mse() -> float:
        v = rotation.rotate_weight(w)
        return (v - quantize_current(v)).pow(2).mean().item()

    initial = current_mse()
    rotation.train()
    opt = torch.optim.Adam(rotation.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):
        v = rotation.rotate_weight(w)          # differentiable via Cayley map
        with torch.no_grad():                  # freeze this step's assignments
            q = quantize_current(v)
        loss = (v - q).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    rotation.eval()
    rotation.matrix()  # warm the eval-mode cache with the trained theta
    return {"initial_mse": initial, "final_mse": current_mse(),
            "steps": int(cfg.steps)}
