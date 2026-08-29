"""Rotation-aware quantization primitives for attention KV caches.

Keys are stored after an orthogonal rotation and queries receive the matching
rotation, preserving their dot product before quantization. Values are stored in
a separately rotated basis; attention accumulates in that basis and applies one
inverse rotation to the weighted result. Both persistent K and V traffic is
therefore packed, while all numerically sensitive reductions remain fp32.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .codebooks import build_scalar_codebook
from .pack import packed_bytes
from .quantize import (
    QuantConfig,
    QuantizedWeight,
    Quantizer,
    _group_scales_rms,
    _quantize_groups,
)
from .rotate import ButterflyRotation, RandomizedHadamard, Rotation


@dataclass(frozen=True)
class KVQuantConfig:
    bits: int = 4
    key_bits: int | None = None
    value_bits: int | None = None
    group_size: int = 64
    scale_bits: float = 16.0
    codebook: str = "gaussian"
    rotation_block: int = 128
    seed: int = 0

    def bits_for(self, *, value: bool = False) -> int:
        selected = self.value_bits if value else self.key_bits
        return self.bits if selected is None else selected

    def quant_config(self, *, value: bool = False) -> QuantConfig:
        return QuantConfig(
            bits=self.bits_for(value=value),
            group_size=self.group_size,
            scale_bits=self.scale_bits,
            codebook=self.codebook,
            scale="rms",
            error_comp="none",
            seed=self.seed,
        )


@dataclass
class QuantizedKV:
    qweight: QuantizedWeight
    shape: tuple[int, ...]
    rotation: Rotation

    def dequantize(self, *, original_basis: bool = False) -> torch.Tensor:
        tensor = self.qweight.dequantize().reshape(self.shape)
        return self.rotation.inverse_activation(tensor) if original_basis else tensor

    def packed_state_bytes(self) -> int:
        size = packed_bytes(self.qweight.packed)
        if self.qweight.scales is not None:
            size += self.qweight.scales.numel() * self.qweight.scales.element_size()
        return size


def build_kv_rotation(dim: int, config: KVQuantConfig, *, value: bool = False,
                      learned: bool = False, device=None) -> Rotation:
    seed = config.seed + (1 if value else 0)
    cls = ButterflyRotation if learned else RandomizedHadamard
    return cls(
        dim, block=config.rotation_block, seed=seed, device=device)


@torch.no_grad()
def quantize_kv(tensor: torch.Tensor, rotation: Rotation,
                config: KVQuantConfig, *, value: bool = False) -> QuantizedKV:
    if tensor.ndim < 2:
        raise ValueError("KV tensors require at least two dimensions")
    if tensor.shape[-1] != rotation.dim:
        raise ValueError("KV head dimension and rotation dimension differ")
    rotated = rotation.rotate_activation(tensor.float())
    rows = rotated.reshape(-1, rotated.shape[-1])
    qweight = Quantizer(config.quant_config(value=value)).quantize_weight(rows)
    return QuantizedKV(qweight=qweight, shape=tuple(tensor.shape), rotation=rotation)


def _attention_weights(queries: torch.Tensor, keys: torch.Tensor,
                       *, causal: bool) -> torch.Tensor:
    scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2))
    scores = scores / math.sqrt(queries.shape[-1])
    if causal:
        n_query, n_key = scores.shape[-2:]
        diagonal = 1 + n_key - n_query
        mask = torch.ones(
            n_query, n_key, device=scores.device, dtype=torch.bool).triu(diagonal)
        scores = scores.masked_fill(mask, -torch.inf)
    return F.softmax(scores, dim=-1)


def reference_attention(queries: torch.Tensor, keys: torch.Tensor,
                        values: torch.Tensor, *, causal: bool = True,
                        ) -> torch.Tensor:
    return torch.matmul(_attention_weights(queries, keys, causal=causal), values.float())


@torch.no_grad()
def rotquant_attention(queries: torch.Tensor, keys: torch.Tensor,
                       values: torch.Tensor, key_rotation: Rotation,
                       value_rotation: Rotation, config: KVQuantConfig,
                       *, causal: bool = True,
                       ) -> tuple[torch.Tensor, QuantizedKV, QuantizedKV]:
    packed_keys = quantize_kv(keys, key_rotation, config)
    packed_values = quantize_kv(values, value_rotation, config, value=True)
    rotated_queries = key_rotation.rotate_activation(queries.float())
    weights = _attention_weights(
        rotated_queries, packed_keys.dequantize(), causal=causal)
    rotated_output = torch.matmul(weights, packed_values.dequantize())
    output = value_rotation.inverse_activation(rotated_output)
    return output, packed_keys, packed_values


def _fake_quant(tensor: torch.Tensor, config: KVQuantConfig,
                *, value: bool = False) -> torch.Tensor:
    rows = tensor.reshape(-1, tensor.shape[-1])
    scales = _group_scales_rms(rows, config.group_size)
    codebook = build_scalar_codebook(
        config.codebook, 1 << config.bits_for(value=value))
    quantized, _ = _quantize_groups(rows, scales, codebook, config.group_size)
    quantized = quantized.reshape_as(tensor)
    return tensor + (quantized - tensor).detach()


def _fake_rotquant_attention(queries, keys, values, key_rotation,
                             value_rotation, config, causal):
    rotated_queries = key_rotation.rotate_activation(queries.float())
    rotated_keys = _fake_quant(
        key_rotation.rotate_activation(keys.float()), config)
    weights = _attention_weights(rotated_queries, rotated_keys, causal=causal)
    rotated_values = _fake_quant(
        value_rotation.rotate_activation(values.float()), config, value=True)
    return value_rotation.inverse_activation(
        torch.matmul(weights, rotated_values))


def train_kv_rotations(
    train_triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: KVQuantConfig,
    *,
    steps: int = 32,
    lr: float = 1e-3,
    causal: bool = True,
) -> tuple[ButterflyRotation, ButterflyRotation, dict[str, float | int]]:
    """Train exactly orthogonal K/V butterflies against attention-output MSE."""
    if steps < 1 or lr <= 0:
        raise ValueError("KV rotation training requires steps >= 1 and lr > 0")
    queries, keys, values = train_triplet
    v_queries, v_keys, v_values = validation_triplet
    dim = queries.shape[-1]
    device = queries.device
    key_rotation = build_kv_rotation(
        dim, config, learned=True, device=device)
    value_rotation = build_kv_rotation(
        dim, config, value=True, learned=True, device=device)
    reference = reference_attention(queries, keys, values, causal=causal)
    validation_reference = reference_attention(
        v_queries, v_keys, v_values, causal=causal)
    optimizer = torch.optim.Adam(
        [key_rotation.theta, value_rotation.theta], lr=lr)

    def validation_loss() -> torch.Tensor:
        candidate = _fake_rotquant_attention(
            v_queries, v_keys, v_values, key_rotation, value_rotation,
            config, causal)
        return F.mse_loss(candidate, validation_reference)

    with torch.no_grad():
        initial = float(validation_loss().item())
    best = initial
    best_step = 0
    best_key = copy.deepcopy(key_rotation.state_dict())
    best_value = copy.deepcopy(value_rotation.state_dict())
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        candidate = _fake_rotquant_attention(
            queries, keys, values, key_rotation, value_rotation,
            config, causal)
        loss = F.mse_loss(candidate, reference)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current = float(validation_loss().item())
        if current < best:
            best = current
            best_step = step
            best_key = copy.deepcopy(key_rotation.state_dict())
            best_value = copy.deepcopy(value_rotation.state_dict())
    key_rotation.load_state_dict(best_key)
    value_rotation.load_state_dict(best_value)
    key_rotation.requires_grad_(False).eval()
    value_rotation.requires_grad_(False).eval()
    return key_rotation, value_rotation, {
        "steps": steps,
        "best_step": best_step,
        "initial_validation_mse": initial,
        "best_validation_mse": best,
    }


@torch.no_grad()
def kv_fidelity_metrics(queries: torch.Tensor, keys: torch.Tensor,
                        values: torch.Tensor, config: KVQuantConfig,
                        key_rotation: Rotation | None = None,
                        value_rotation: Rotation | None = None,
                        *, causal: bool = True) -> dict[str, float | int]:
    key_rotation = key_rotation or build_kv_rotation(
        keys.shape[-1], config, device=keys.device)
    value_rotation = value_rotation or build_kv_rotation(
        values.shape[-1], config, value=True, device=values.device)
    reference = reference_attention(queries, keys, values, causal=causal)
    candidate, packed_keys, packed_values = rotquant_attention(
        queries, keys, values, key_rotation, value_rotation, config,
        causal=causal)
    denominator = reference.pow(2).mean().clamp_min(1e-12)
    fp16_bytes = (keys.numel() + values.numel()) * 2
    packed_bytes_total = (
        packed_keys.packed_state_bytes() + packed_values.packed_state_bytes())
    return {
        "bits": config.bits,
        "key_bits": config.bits_for(),
        "value_bits": config.bits_for(value=True),
        "group_size": config.group_size,
        "relative_attention_mse": float(
            ((candidate - reference).pow(2).mean() / denominator).item()),
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1, candidate.shape[-1]),
            reference.reshape(-1, reference.shape[-1]), dim=-1).mean().item()),
        "fp16_bytes": fp16_bytes,
        "packed_bytes": packed_bytes_total,
        "compression_ratio": fp16_bytes / packed_bytes_total,
        "effective_bpv": packed_bytes_total * 8 / (
            keys.numel() + values.numel()),
    }
