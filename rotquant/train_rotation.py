"""Layerwise training for dense and structured orthogonal rotations.

The learned arm is only meaningful if theta is actually optimised: the Cayley
parametrisation initialises at ~identity, i.e. a no-rotation control.

The original data-free objective minimises quantisation error of the rotated
weight,

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

For structured butterfly rotations an activation-aware objective is also
available. Given source-model layer inputs ``X``, it directly minimises relative
output reconstruction error while the base weight remains frozen:

    ||X W^T - (X R^T) Q(W R^T)^T||^2 / ||X W^T||^2.

Only a bounded token sample is needed. Quantisation assignments and scales are
still alternated/frozen per optimiser step, making this rotation-aware PTQ rather
than full-model fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from ._internal import quantize_groups, rotate_hessian
from .quantize import QuantConfig, Quantizer
from .rotate import ButterflyRotation, LearnedRotation, Rotation
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class RotationTrainConfig:
    steps: int = 100
    lr: float = 1e-3
    objective: str = "weight"          # weight | activation | hessian
    max_tokens: int = 64
    selection_tokens: int = 0
    # MSE scale search evaluates dozens of candidates per optimiser step. Using
    # RMS assignments for training is much faster; final packing still uses the
    # experiment's actual scale strategy.
    assignment_scale: str | None = None
    max_grad_norm: float = 1.0
    restore_best: bool = True
    # After fast proxy-objective training, require this relative improvement
    # under the experiment's actual final quantizer or restore exact FWHT.
    selection_min_improvement: float = 0.0
    learn_signs: bool = False
    sign_temperature: float = 1.0
    # Initial |logit| of the trainable signs; must be reachable by lr * steps
    # or no sign can ever flip (see ButterflyRotation.enable_sign_training).
    sign_init_magnitude: float = 0.1

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("rotation training steps must be >= 1")
        if self.lr <= 0:
            raise ValueError("rotation training lr must be > 0")
        if self.objective not in {"weight", "activation", "hessian"}:
            raise ValueError(
                "rotation training objective must be 'weight', 'activation', "
                "or 'hessian'"
            )
        if self.max_tokens < 1:
            raise ValueError("rotation training max_tokens must be >= 1")
        if self.selection_tokens < 0:
            raise ValueError("rotation training selection_tokens must be >= 0")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0")
        if not 0 <= self.selection_min_improvement < 1:
            raise ValueError("selection_min_improvement must be in [0, 1)")
        if not isinstance(self.learn_signs, bool):
            raise TypeError("learn_signs must be boolean")
        if self.sign_temperature <= 0:
            raise ValueError("sign_temperature must be positive")
        if self.sign_init_magnitude <= 0:
            raise ValueError("sign_init_magnitude must be positive")


@torch.no_grad()
def hessian_reconstruction_error(rotation: Rotation, weight: torch.Tensor,
                                 quant_cfg: QuantConfig,
                                 hessian: torch.Tensor) -> float:
    """Hessian-weighted layer-output error of the exact deployed quantizer.

    Returns ``tr(E H E^T) / tr(W H W^T)`` for the un-rotated weight error
    ``E = (W R^T - Q(W R^T)) R`` of the packed weight the experiment would
    actually deploy, including GPTQ with the rotated Hessian when configured.
    """
    w = weight.detach().to(torch.float32)
    H = hessian.detach().to(device=w.device, dtype=torch.float32)
    if H.shape != (w.shape[1], w.shape[1]):
        raise ValueError("rotation Hessian shape does not match weight input")
    rotated = rotation.rotate_weight(w)
    rotated_hessian = (
        rotate_hessian(rotation, H) if quant_cfg.error_comp == "gptq" else None
    )
    q = Quantizer(quant_cfg).quantize_weight(
        rotated, H=rotated_hessian).dequantize().to(w.device)
    error = rotation.inverse_activation(rotated - q)
    return ((error @ H * error).sum()
            / (w @ H * w).sum().clamp_min(1e-12)).item()


@torch.no_grad()
def select_butterfly_checkpoint_hessian(rotation: ButterflyRotation,
                                        reference: ButterflyRotation,
                                        weight: torch.Tensor,
                                        quant_cfg: QuantConfig,
                                        hessian: torch.Tensor,
                                        min_improvement: float = 0.0,
                                        ) -> dict[str, Any]:
    """Deployed-quantizer gate for Hessian-objective butterflies.

    The training proxy (RMS assignment, no error compensation) is not the
    deployed quantizer.  Keep the trained angles/signs only when the exact
    packed weight beats the seeded FWHT reference on the Hessian-weighted
    output error by the required margin; otherwise restore FWHT.
    """
    reference_error = hessian_reconstruction_error(
        reference, weight, quant_cfg, hessian)
    candidate_error = hessian_reconstruction_error(
        rotation, weight, quant_cfg, hessian)
    accepted = candidate_error <= reference_error * (1.0 - min_improvement)
    if not accepted:
        rotation.theta.copy_(reference.theta)
        rotation.signs.copy_(reference.signs)
    return {
        "selection_reference_mse": reference_error,
        "selection_candidate_mse": candidate_error,
        "selection_accepted": bool(accepted),
        "selection_objective": "hessian",
    }


@torch.no_grad()
def activation_reconstruction_error(rotation: Rotation, weight: torch.Tensor,
                                    quant_cfg: QuantConfig,
                                    activations: torch.Tensor,
                                    max_tokens: int = 64) -> float:
    """Relative layer-output error using the exact final quantized weight."""
    if quant_cfg.error_comp == "gptq":
        raise ValueError(
            "exact rotation checkpoint selection with GPTQ requires a Hessian")
    w = weight.detach().to(torch.float32)
    x = activations.detach().reshape(-1, w.shape[1])[:max_tokens]
    x = x.to(device=w.device, dtype=torch.float32)
    rotated = rotation.rotate_weight(w)
    q = Quantizer(quant_cfg).quantize_weight(rotated).dequantize().to(w.device)
    target = x @ w.T
    pred = rotation.rotate_activation(x) @ q.T
    return ((pred - target).pow(2).mean()
            / target.pow(2).mean().clamp_min(1e-12)).item()


@torch.no_grad()
def select_butterfly_checkpoint(rotation: ButterflyRotation,
                                reference: ButterflyRotation,
                                weight: torch.Tensor,
                                quant_cfg: QuantConfig,
                                activations: torch.Tensor,
                                max_tokens: int = 64,
                                min_improvement: float = 0.0) -> dict[str, Any]:
    """Keep a trained butterfly only when the deployed quantizer prefers it.

    ``reference`` is the exact seeded FWHT initialization. If the trained
    candidate misses the required margin, its angles are restored to reference;
    this prevents a cheap training proxy from silently degrading final packing.
    """
    reference_error = activation_reconstruction_error(
        reference, weight, quant_cfg, activations, max_tokens)
    candidate_error = activation_reconstruction_error(
        rotation, weight, quant_cfg, activations, max_tokens)
    accepted = candidate_error <= reference_error * (1.0 - min_improvement)
    if not accepted:
        rotation.theta.copy_(reference.theta)
        rotation.signs.copy_(reference.signs)
    return {
        "selection_reference_mse": reference_error,
        "selection_candidate_mse": candidate_error,
        "selection_accepted": bool(accepted),
    }


def train_layer_rotation(rotation: Rotation, weight: torch.Tensor,
                         quant_cfg: QuantConfig,
                         config: RotationTrainConfig | None = None,
                         activations: torch.Tensor | None = None,
                         hessian: torch.Tensor | None = None) -> dict[str, Any]:
    """Optimise a trainable rotation in place for one layer's frozen weight.

    Returns ``{"initial_mse", "final_mse", "steps"}`` (rotated-domain
    weight MSE, or relative output MSE for the activation objective). The
    rotation is left in eval mode with its inference cache warm.
    """
    cfg = config or RotationTrainConfig()
    if not isinstance(rotation, (LearnedRotation, ButterflyRotation)):
        raise TypeError("rotation training requires a trainable rotation")
    if cfg.objective == "activation" and activations is None:
        raise ValueError("activation-aware rotation training requires activations")
    if cfg.objective == "hessian" and hessian is None:
        raise ValueError("Hessian-aware rotation training requires a Hessian")

    train_quant_cfg = replace(
        quant_cfg,
        error_comp="none",
        scale=cfg.assignment_scale or quant_cfg.scale,
    )
    quantizer = Quantizer(train_quant_cfg)
    w = weight.detach().to(torch.float32)
    initial_signs = None
    if cfg.learn_signs:
        if not isinstance(rotation, ButterflyRotation):
            raise TypeError("learned signs require a ButterflyRotation")
        initial_signs = rotation.signs.detach().clone()
        rotation.enable_sign_training(
            cfg.sign_temperature, cfg.sign_init_magnitude)

    x = target = target_power = None
    if cfg.objective == "activation":
        x = activations.detach().reshape(-1, w.shape[1])[:cfg.max_tokens]
        x = x.to(device=w.device, dtype=torch.float32)
        target = x @ w.T
        target_power = target.pow(2).mean().clamp_min(1e-12)
    H = None
    if cfg.objective == "hessian":
        H = hessian.detach().to(device=w.device, dtype=torch.float32)
        if H.shape != (w.shape[1], w.shape[1]):
            raise ValueError("rotation Hessian shape does not match weight input")
        target_power = (w @ H * w).sum().clamp_min(1e-12)

    def quantize_current(v: torch.Tensor) -> torch.Tensor:
        scales = quantizer.select_scales(v)
        q, _ = quantize_groups(v, scales, quantizer.codebook,
                                train_quant_cfg.group_size)
        return q

    @torch.no_grad()
    def current_mse() -> float:
        v = rotation.rotate_weight(w)
        q = quantize_current(v)
        if cfg.objective == "activation":
            pred = rotation.rotate_activation(x) @ q.T
            return ((pred - target).pow(2).mean() / target_power).item()
        if cfg.objective == "hessian":
            effective_error = rotation.inverse_activation(v - q)
            return ((effective_error @ H * effective_error).sum()
                    / target_power).item()
        return (v - q).pow(2).mean().item()

    initial = current_mse()
    best_loss = initial
    best_step = 0
    best_params = [p.detach().clone() for p in rotation.parameters()]
    rotation.train()
    rotation.requires_grad_(True)
    opt = torch.optim.Adam(rotation.parameters(), lr=cfg.lr)
    for step in range(cfg.steps):
        v = rotation.rotate_weight(w)          # differentiable orthogonal map
        with torch.no_grad():                  # freeze this step's assignments
            q = quantize_current(v)
        if cfg.objective == "activation":
            pred = rotation.rotate_activation(x) @ q.T
            loss = (pred - target).pow(2).mean() / target_power
        elif cfg.objective == "hessian":
            effective_error = rotation.inverse_activation(v - q)
            loss = ((effective_error @ H * effective_error).sum()
                    / target_power)
        else:
            loss = (v - q).pow(2).mean()
        loss_value = loss.detach().item()
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_params = [p.detach().clone() for p in rotation.parameters()]
        opt.zero_grad()
        loss.backward()
        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(rotation.parameters(), cfg.max_grad_norm)
        opt.step()

    candidate = current_mse()
    if candidate < best_loss:
        best_loss = candidate
        best_step = cfg.steps
        best_params = [p.detach().clone() for p in rotation.parameters()]
    elif cfg.restore_best:
        with torch.no_grad():
            for parameter, best in zip(rotation.parameters(), best_params):
                parameter.copy_(best)

    sign_flip_rate = 0.0
    if isinstance(rotation, ButterflyRotation) and rotation.sign_logits is not None:
        final_signs = torch.where(
            rotation.sign_logits.detach() >= 0,
            torch.ones_like(rotation.signs),
            -torch.ones_like(rotation.signs),
        )
        sign_flip_rate = float((final_signs != initial_signs).float().mean().item())
        rotation.commit_signs()
    rotation.eval()
    rotation.requires_grad_(False)
    if isinstance(rotation, LearnedRotation):
        rotation.matrix()
    else:
        rotation._trig()
    final = best_loss if cfg.restore_best else candidate
    return {"initial_mse": initial, "final_mse": final,
            "steps": int(cfg.steps), "objective": cfg.objective,
            "tokens": int(x.shape[0]) if x is not None else 0,
            "best_step": int(best_step),
            "sign_flip_rate": sign_flip_rate}
