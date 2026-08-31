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

from ._internal import group_scales_rms, quantize_groups
from .codebooks import build_scalar_codebook
from .pack import packed_bytes
from .quantize import QuantConfig, QuantizedWeight, Quantizer
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
    codebook_dim: int | None = None
    bias_correction: str = "none"

    def bits_for(self, *, value: bool = False) -> int:
        selected = self.value_bits if value else self.key_bits
        return self.bits if selected is None else selected

    def quant_config(self, *, value: bool = False) -> QuantConfig:
        spherical = self.codebook.lower() in {
            "sphere", "spherical", "beta", "finite_beta"}
        return QuantConfig(
            bits=self.bits_for(value=value),
            group_size=self.group_size,
            scale_bits=self.scale_bits,
            codebook=self.codebook,
            codebook_dim=(
                self.codebook_dim or self.group_size
            ) if spherical else None,
            scale="rms",
            error_comp="none",
            bias_correction=self.bias_correction,
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


@torch.no_grad()
def retrieval_rotquant_decode(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_rotation: Rotation,
    value_rotation: Rotation,
    config: KVQuantConfig,
    *,
    retrieval_k: int,
    recent_window: int = 0,
    sink_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, QuantizedKV, QuantizedKV]:
    """Decode attention by scanning packed keys and gathering selected values.

    This is the reference semantics for a TurboVec-like cache path.  A production
    kernel would score code indices directly and dequantize only the selected V
    rows; this Python oracle materializes reconstructed keys solely to keep the
    quality experiment independent of a future SIMD/CUDA implementation.

    ``retrieval_k`` is the total candidate budget.  The newest
    ``recent_window`` positions and first ``sink_tokens`` positions are forced
    into that budget, then approximate key scores fill the remaining slots.
    """
    if queries.shape[-2] != 1:
        raise ValueError("retrieval attention currently supports decode queries only")
    if keys.shape[:-2] != queries.shape[:-2] or values.shape[:-2] != keys.shape[:-2]:
        raise ValueError("queries, keys, and values must share batch/head dimensions")
    if keys.shape[-2] != values.shape[-2] or keys.shape[-1] != values.shape[-1]:
        raise ValueError("keys and values must have matching sequence/head dimensions")
    sequence_length = keys.shape[-2]
    if not 1 <= retrieval_k <= sequence_length:
        raise ValueError("retrieval_k must be in [1, cache sequence length]")
    if recent_window < 0 or sink_tokens < 0:
        raise ValueError("recent_window and sink_tokens must be nonnegative")

    packed_keys = quantize_kv(keys, key_rotation, config)
    packed_values = quantize_kv(values, value_rotation, config, value=True)
    rotated_queries = key_rotation.rotate_activation(queries.float())
    reconstructed_keys = packed_keys.dequantize()
    scores = torch.matmul(
        rotated_queries, reconstructed_keys.transpose(-1, -2)
    ) / math.sqrt(queries.shape[-1])

    forced = torch.zeros(sequence_length, dtype=torch.bool, device=scores.device)
    forced[:min(sink_tokens, sequence_length)] = True
    if recent_window:
        forced[max(0, sequence_length - recent_window):] = True
    forced_count = int(forced.sum().item())
    if forced_count > retrieval_k:
        raise ValueError(
            "retrieval_k is smaller than the mandatory sink/recent candidate set")
    forced_shape = (1,) * (scores.ndim - 1) + (sequence_length,)
    ranking_scores = scores.masked_fill(forced.reshape(forced_shape), torch.inf)
    selected = torch.topk(
        ranking_scores, k=retrieval_k, dim=-1, sorted=False
    ).indices
    selected_scores = torch.gather(scores, -1, selected)
    weights = F.softmax(selected_scores, dim=-1)

    reconstructed_values = packed_values.dequantize()
    expanded_values = reconstructed_values.unsqueeze(-3).expand(
        *reconstructed_values.shape[:-2],
        queries.shape[-2],
        sequence_length,
        reconstructed_values.shape[-1],
    )
    selected_values = torch.gather(
        expanded_values,
        -2,
        selected.unsqueeze(-1).expand(
            *selected.shape, reconstructed_values.shape[-1]
        ),
    )
    rotated_output = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
    output = value_rotation.inverse_activation(rotated_output)
    return output, selected, packed_keys, packed_values


@torch.no_grad()
def kv_retrieval_metrics(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    config: KVQuantConfig,
    *,
    retrieval_k: int,
    recent_window: int = 0,
    sink_tokens: int = 0,
) -> dict[str, float | int]:
    """Evaluate selective KV retrieval against full-precision dense attention."""
    key_rotation = build_kv_rotation(keys.shape[-1], config, device=keys.device)
    value_rotation = build_kv_rotation(
        values.shape[-1], config, value=True, device=values.device)
    reference = reference_attention(queries, keys, values, causal=True)
    candidate, selected, packed_keys, packed_values = retrieval_rotquant_decode(
        queries,
        keys,
        values,
        key_rotation,
        value_rotation,
        config,
        retrieval_k=retrieval_k,
        recent_window=recent_window,
        sink_tokens=sink_tokens,
    )
    dense_weights = _attention_weights(queries, keys, causal=True)
    selected_mass = torch.gather(dense_weights, -1, selected).sum(dim=-1)
    denominator = reference.square().mean().clamp_min(1e-12)
    return {
        "retrieval_k": retrieval_k,
        "sequence_length": keys.shape[-2],
        "selected_fraction": retrieval_k / keys.shape[-2],
        "recent_window": recent_window,
        "sink_tokens": sink_tokens,
        "reference_attention_mass_coverage": float(selected_mass.mean().item()),
        "relative_attention_mse": float(
            ((candidate - reference).square().mean() / denominator).item()),
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1, candidate.shape[-1]),
            reference.reshape(-1, reference.shape[-1]),
            dim=-1,
        ).mean().item()),
        "packed_kv_bytes": (
            packed_keys.packed_state_bytes() + packed_values.packed_state_bytes()),
        # K is scanned; only this fraction of V vectors needs to cross the
        # dequantization/value-accumulation path in a fused implementation.
        "value_vector_read_fraction": retrieval_k / keys.shape[-2],
    }


@torch.no_grad()
def oracle_value_retrieval_curve(
    attention_weights: torch.Tensor,
    values: torch.Tensor,
    retrieval_counts: tuple[int, ...] | list[int],
    *,
    reference_values: torch.Tensor | None = None,
    recent_window: int = 0,
    sink_tokens: int = 0,
    mass_threshold: float | None = None,
) -> list[dict[str, float | int]]:
    """Measure the upper bound from gathering only high-attention V rows.

    This consumes real dense attention probabilities and cache values, making it
    useful before an architecture-specific query hook exists. Selection by the
    dense probabilities is an oracle: it measures whether value sparsity is
    present, not whether packed-key scoring can recover the same candidates.
    The latter remains a separate required experiment.

    ``attention_weights`` has shape ``[batch, query_heads, queries, sequence]``
    and ``values`` has shape ``[batch, kv_heads, sequence, head_dim]``. Pass the
    source cache as ``reference_values`` when ``values`` is quantized so the
    reported error includes both V quantization and selective retrieval. Grouped
    query attention is expanded using the normal consecutive-head mapping.
    """

    if attention_weights.ndim != 4 or values.ndim != 4:
        raise ValueError(
            "attention weights and values must have four dimensions"
        )
    batch, query_heads, queries, sequence = attention_weights.shape
    if values.shape[0] != batch or values.shape[2] != sequence:
        raise ValueError("attention weights and values have incompatible shapes")
    if reference_values is not None and reference_values.shape != values.shape:
        raise ValueError("reference values must have the same shape as values")
    kv_heads = values.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if recent_window < 0 or sink_tokens < 0:
        raise ValueError("recent_window and sink_tokens must be nonnegative")
    if mass_threshold is not None and not 0.0 < mass_threshold <= 1.0:
        raise ValueError("mass_threshold must be in (0, 1]")
    counts = tuple(sorted({int(value) for value in retrieval_counts}))
    if not counts or counts[0] < 1 or counts[-1] > sequence:
        raise ValueError("retrieval counts must be within the sequence length")
    weights = attention_weights.detach().float()
    if (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError("attention weights must be finite and nonnegative")
    row_mass = weights.sum(dim=-1, keepdim=True)
    if (row_mass <= 0).any():
        raise ValueError("attention rows must have positive mass")
    weights = weights / row_mass

    repeat = query_heads // kv_heads
    candidate_values = values.detach().float().repeat_interleave(repeat, dim=1)
    expanded_values = candidate_values.unsqueeze(2).expand(
        batch, query_heads, queries, sequence, values.shape[-1]
    )
    reference_values = values if reference_values is None else reference_values
    source_values = reference_values.detach().float().repeat_interleave(
        repeat, dim=1
    )
    reference = torch.matmul(weights, source_values)
    dense_candidate = torch.matmul(weights, candidate_values)
    reference_energy = reference.square().mean().clamp_min(1e-12)
    dense_value_relative_mse = float(
        ((dense_candidate - reference).square().mean() / reference_energy).item()
    )
    dense_value_cosine = float(F.cosine_similarity(
        dense_candidate.reshape(-1, dense_candidate.shape[-1]),
        reference.reshape(-1, reference.shape[-1]),
        dim=-1,
    ).mean().item())

    forced = torch.zeros(sequence, dtype=torch.bool, device=weights.device)
    forced[:min(sink_tokens, sequence)] = True
    if recent_window:
        forced[max(0, sequence - recent_window):] = True
    forced_count = int(forced.sum().item())

    results = []
    for retrieval_k in counts:
        if forced_count > retrieval_k:
            raise ValueError(
                "retrieval count is smaller than mandatory sink/recent tokens"
            )
        ranking = weights.masked_fill(
            forced.reshape(1, 1, 1, sequence), torch.inf
        )
        selected = torch.topk(
            ranking, k=retrieval_k, dim=-1, sorted=False
        ).indices
        selected_weights = torch.gather(weights, -1, selected)
        selected_mass = selected_weights.sum(dim=-1)
        normalized = selected_weights / selected_mass.unsqueeze(-1).clamp_min(1e-12)
        selected_values = torch.gather(
            expanded_values,
            -2,
            selected.unsqueeze(-1).expand(*selected.shape, values.shape[-1]),
        )
        candidate = (normalized.unsqueeze(-1) * selected_values).sum(dim=-2)
        relative_mse = (
            (candidate - reference).square().mean() / reference_energy
        )
        cosine = F.cosine_similarity(
            candidate.reshape(-1, candidate.shape[-1]),
            reference.reshape(-1, reference.shape[-1]),
            dim=-1,
        ).mean()
        fallback_fraction = 0.0
        effective_read_fraction = retrieval_k / sequence
        gated_relative_mse = float(relative_mse.item())
        if mass_threshold is not None:
            fallback = selected_mass < mass_threshold
            fallback_fraction = float(fallback.float().mean().item())
            # A confidence fallback reads every stored V row. When ``values``
            # is quantized this is the dense quantized result, not an impossible
            # fallback to source values that are no longer resident.
            gated = torch.where(
                fallback.unsqueeze(-1), dense_candidate, candidate
            )
            gated_relative_mse = float(
                ((gated - reference).square().mean() / reference_energy).item()
            )
            effective_read_fraction = (
                (1.0 - fallback_fraction) * retrieval_k / sequence
                + fallback_fraction
            )
        results.append({
            "retrieval_k": retrieval_k,
            "sequence_length": sequence,
            "selected_fraction": retrieval_k / sequence,
            "mean_attention_mass": float(selected_mass.mean().item()),
            "p05_attention_mass": float(
                torch.quantile(selected_mass.flatten(), 0.05).item()
            ),
            "dense_value_relative_attention_mse": dense_value_relative_mse,
            "dense_value_cosine": dense_value_cosine,
            "relative_attention_mse": float(relative_mse.item()),
            "cosine": float(cosine.item()),
            "mass_threshold": mass_threshold or 0.0,
            "dense_fallback_fraction": fallback_fraction,
            "gated_relative_attention_mse": gated_relative_mse,
            "effective_value_read_fraction": effective_read_fraction,
        })
    return results


def _fake_quant(tensor: torch.Tensor, config: KVQuantConfig,
                *, value: bool = False) -> torch.Tensor:
    rows = tensor.reshape(-1, tensor.shape[-1])
    scales = group_scales_rms(rows, config.group_size)
    spherical = config.codebook.lower() in {
        "sphere", "spherical", "beta", "finite_beta"}
    dimension = (config.codebook_dim or config.group_size) if spherical else None
    codebook = build_scalar_codebook(
        config.codebook, 1 << config.bits_for(value=value), dimension)
    quantized, _ = quantize_groups(rows, scales, codebook, config.group_size)
    if config.bias_correction == "length":
        energy = rows.square().sum(dim=1)
        alignment = (rows * quantized).sum(dim=1)
        usable = (energy > 0) & (
            alignment > torch.finfo(rows.dtype).eps * energy)
        correction = torch.ones_like(energy)
        correction[usable] = energy[usable] / alignment[usable]
        quantized = quantized * correction.unsqueeze(1)
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
    reconstructed_keys = packed_keys.dequantize(original_basis=True).float()
    reconstructed_values = packed_values.dequantize(original_basis=True).float()
    scale = math.sqrt(queries.shape[-1])
    reference_logits = torch.matmul(
        queries.float(), keys.float().transpose(-1, -2)) / scale
    candidate_logits = torch.matmul(
        queries.float(), reconstructed_keys.transpose(-1, -2)) / scale
    if causal:
        n_query, n_key = reference_logits.shape[-2:]
        diagonal = 1 + n_key - n_query
        valid = ~torch.ones(
            n_query, n_key, device=queries.device, dtype=torch.bool
        ).triu(diagonal)
        reference_logit_values = reference_logits[..., valid]
        candidate_logit_values = candidate_logits[..., valid]
    else:
        reference_logit_values = reference_logits.reshape(-1)
        candidate_logit_values = candidate_logits.reshape(-1)
    logit_error = candidate_logit_values - reference_logit_values
    logit_energy = reference_logit_values.square().sum().clamp_min(1e-12)
    key_energy = keys.float().square().sum().clamp_min(1e-12)
    value_energy = values.float().square().sum().clamp_min(1e-12)
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
        "relative_attention_logit_mse": float(
            logit_error.square().sum().div(logit_energy).item()),
        "attention_logit_bias": float(logit_error.mean().item()),
        "attention_logit_mae": float(logit_error.abs().mean().item()),
        "key_nmse": float(
            (reconstructed_keys - keys.float()).square().sum().div(
                key_energy).item()),
        "value_nmse": float(
            (reconstructed_values - values.float()).square().sum().div(
                value_energy).item()),
        "key_self_dot_ratio": float(
            (keys.float() * reconstructed_keys).sum().div(key_energy).item()),
        "value_self_dot_ratio": float(
            (values.float() * reconstructed_values).sum().div(
                value_energy).item()),
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1, candidate.shape[-1]),
            reference.reshape(-1, reference.shape[-1]), dim=-1).mean().item()),
        "fp16_bytes": fp16_bytes,
        "packed_bytes": packed_bytes_total,
        "compression_ratio": fp16_bytes / packed_bytes_total,
        "effective_bpv": packed_bytes_total * 8 / (
            keys.numel() + values.numel()),
    }
