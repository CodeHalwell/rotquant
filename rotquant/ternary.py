"""Trainable ternary weights with fixed FWHT and exact FP16 scale storage.

The three legal symbols use physical two-bit slots. Code 3 is reserved and
never represents a fourth quantisation level.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ._internal import expand_scales
from .codebooks import ScalarCodebook
from .linear import QuantLinear
from .pack import PackedTensor, pack_indices, unpack_indices
from .quantize import QuantizedWeight
from .rotate import RandomizedHadamard


@dataclass(frozen=True)
class TernaryConfig:
    group_size: int = 128
    rotation_block: int = 128
    seed: int = 0
    learn_scales: bool = True

    def __post_init__(self):
        for name in ("group_size", "rotation_block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rotation_block & (self.rotation_block - 1):
            raise ValueError("rotation_block must be a power of two")
        if not isinstance(self.learn_scales, bool):
            raise TypeError("learn_scales must be boolean")


def _initial_scales(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    padding = (-weight.shape[1]) % group_size
    groups = F.pad(weight, (0, padding)).reshape(weight.shape[0], -1, group_size)
    valid = F.pad(torch.ones_like(weight), (0, padding)).reshape_as(groups)
    count = valid.sum(-1).clamp_min(1)
    scales = (groups.square().sum(-1) / count).sqrt() * 1.224
    for _ in range(8):
        selected = (groups.abs() > scales.unsqueeze(-1) / 2) * valid
        scales = (groups.abs() * selected).sum(-1) / selected.sum(-1).clamp_min(1)
    return scales.clamp(2 ** -24, 65504)


class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized):
        ctx.save_for_backward(normalized)
        return torch.where(normalized.abs() > 0.5, normalized.sign(), 0)

    @staticmethod
    def backward(ctx, gradient):
        (normalized,) = ctx.saved_tensors
        return gradient * (normalized.abs() <= 1)


class TernaryLinear(nn.Module):
    """Hard ternary forward, clipped identity STE for trainable latent weights."""

    def __init__(self, linear: nn.Linear, config: TernaryConfig):
        super().__init__()
        if linear.in_features % config.rotation_block:
            raise ValueError("rotation_block must divide the ternary input width")
        if not torch.isfinite(linear.weight).all():
            raise ValueError("ternary source weights must be finite")
        self.config = config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_rotation = RandomizedHadamard(
            self.in_features, block=config.rotation_block, seed=config.seed,
            device=linear.weight.device)
        with torch.no_grad():
            rotated = self.act_rotation.rotate_weight(linear.weight.detach().float())
            scales = _initial_scales(rotated, config.group_size)
        self.latent_weight = nn.Parameter(rotated)
        self.log_scales = nn.Parameter(scales.log(), requires_grad=config.learn_scales)
        self.register_buffer("fixed_codes", None, persistent=False)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: TernaryConfig | None = None):
        return cls(linear, config or TernaryConfig())

    @classmethod
    def from_quant_linear(cls, module: QuantLinear, latent_state: dict | None = None):
        from .ternary_latents import validate_ternary_module
        validate_ternary_module(module)
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.config = TernaryConfig(module.qweight.group_size, module.act_rotation.block)
        result.in_features, result.out_features = module.in_features, module.out_features
        result.act_rotation = copy.deepcopy(module.act_rotation)
        latent = module.qweight.dequantize().float()
        scales = module.qweight.scales.float().log()
        if latent_state is not None:
            latent = latent_state["latent_weight"].to(latent)
            scales = latent_state["log_scales"].to(scales)
        if latent.shape != (result.out_features, result.in_features) or scales.shape != module.qweight.scales.shape:
            raise ValueError("retained ternary latent shapes differ from the packed checkpoint")
        if not torch.isfinite(latent).all() or not torch.isfinite(scales).all():
            raise ValueError("retained ternary latents must be finite")
        result.latent_weight = nn.Parameter(latent.detach().clone())
        result.log_scales = nn.Parameter(scales.detach().clone())
        result.register_buffer("fixed_codes", None, persistent=False)
        result.register_buffer("bias", None if module.bias is None else module.bias.detach().clone())
        expected = packed_ternary_state(module)
        actual = packed_ternary_state(result.to_quant_linear())
        if any(not torch.equal(actual[key], expected[key]) for key in ("packed", "scales", "signs")):
            raise ValueError("retained ternary latents do not reproduce the packed checkpoint")
        return result

    def stored_scales(self) -> torch.Tensor:
        scales = self.log_scales.clamp(-16.635532, 11.089866).exp()
        stored = scales.to(torch.float16).float()
        return scales + (stored - scales).detach()

    def hard_codes(self) -> torch.Tensor:
        if self.fixed_codes is not None:
            return self.fixed_codes
        expanded = expand_scales(self.stored_scales(), self.config.group_size, self.in_features)
        normalized = self.latent_weight / expanded
        return torch.where(normalized.abs() > 0.5, normalized.sign(), 0)

    def quantized_weight(self) -> torch.Tensor:
        scales = self.stored_scales()
        expanded = expand_scales(scales, self.config.group_size, self.in_features)
        if self.fixed_codes is not None:
            return self.fixed_codes * expanded
        # Detaching scale in the latent surrogate leaves its gradient equal to
        # the exact fixed-assignment derivative: sum(dL/dq * ternary_symbol).
        normalized = self.latent_weight / expanded.detach()
        codes = _TernarySTE.apply(normalized)
        return codes * expanded

    @torch.no_grad()
    def freeze_assignments(self):
        self.fixed_codes = self.hard_codes().detach().to(torch.int8)
        self.latent_weight = nn.Parameter(torch.empty(0, device=self.latent_weight.device),
                                          requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rotated = self.act_rotation.rotate_activation(value)
        bias = None if self.bias is None else self.bias.to(rotated.dtype)
        return F.linear(rotated, self.quantized_weight().to(rotated.dtype), bias)

    @torch.no_grad()
    def to_quant_linear(self) -> QuantLinear:
        indices = (self.hard_codes() + 1).to(torch.int64)
        weight = QuantizedWeight(
            packed=pack_indices(indices, 2),
            scales=self.stored_scales().to(torch.float16),
            codebook=ScalarCodebook([-1, 0, 1], name="ternary"),
            group_size=self.config.group_size,
            out_features=self.out_features, in_features=self.in_features,
            scale_bits_main=16.0)
        return QuantLinear(weight, copy.deepcopy(self.act_rotation), self.bias)


def packed_ternary_state(module: QuantLinear) -> dict:
    """Compact completed-block state, containing no latent or optimiser arrays."""
    qw = module.qweight
    return {
        "packed": qw.packed.data.detach().cpu().clone(),
        "scales": qw.scales.detach().cpu().clone(),
        "signs": module.act_rotation.signs.detach().cpu().clone(),
        "rotation_block": module.act_rotation.block,
        "group_size": qw.group_size,
        "in_features": qw.in_features, "out_features": qw.out_features,
    }


def restore_packed_ternary(state: dict, source: nn.Linear) -> QuantLinear:
    """Validate the reserved code and metadata before restoring packed weights."""
    shape = (source.out_features, source.in_features)
    if (state["out_features"], state["in_features"]) != shape:
        raise ValueError("ternary checkpoint shape differs from source")
    config = TernaryConfig(state["group_size"], state["rotation_block"])
    packed = PackedTensor(state["packed"], shape, 2, shape[0] * shape[1])
    if (unpack_indices(packed) > 2).any():
        raise ValueError("ternary checkpoint contains reserved code 3")
    scales = state["scales"]
    expected = (shape[0], (shape[1] + config.group_size - 1) // config.group_size)
    if scales.dtype != torch.float16 or tuple(scales.shape) != expected:
        raise ValueError("ternary checkpoint requires correctly shaped FP16 scales")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("ternary checkpoint scales must be finite and positive")
    signs = state["signs"]
    if signs.dtype != torch.int8 or tuple(signs.shape) != (shape[1],):
        raise ValueError("invalid ternary rotation signs")
    if not ((signs == 1) | (signs == -1)).all():
        raise ValueError("ternary rotation signs must be -1 or +1")
    rotation = RandomizedHadamard(shape[1], block=config.rotation_block)
    if rotation.block != config.rotation_block:
        raise ValueError("ternary rotation block does not divide input width")
    rotation.signs.copy_(signs)
    qw = QuantizedWeight(packed, scales, ScalarCodebook([-1, 0, 1], "ternary"),
                         config.group_size, *shape, scale_bits_main=16.0)
    return QuantLinear(qw, rotation, source.bias).to(source.weight.device)
