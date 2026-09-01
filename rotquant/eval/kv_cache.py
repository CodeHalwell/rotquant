"""End-to-end fidelity evaluation at the Transformers KV-cache boundary.

The simulator packs the actual post-RoPE keys and values produced during
prefill. It reconstructs them for the unmodified attention implementation and
intercepts subsequent cache writes so every new K/V pair is rotated, quantized,
and reconstructed exactly once. This isolates cache quantization quality while
the native fused runtime is developed.
"""
from __future__ import annotations

import copy
import inspect
import types
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from rotquant.kv_cache import (
    KVQuantConfig,
    QuantizedKV,
    build_kv_rotation,
    quantize_kv,
)
from rotquant.utils import get_logger

from .statistics import bootstrap_report

logger = get_logger(__name__)


@dataclass(frozen=True)
class KVCacheEvalConfig:
    bits: int = 4
    key_bits: int | None = None
    value_bits: int | None = None
    group_size: int = 64
    scale_bits: float = 16.0
    scale_quant_group_size: int = 256
    codebook: str = "gaussian"
    rotation_block: int = 128
    batches: int = 2
    eval_offset_batches: int = 0
    prompt_len: int = 128
    continuation_len: int = 8
    skip: int = 384
    temperature: float = 1.0
    seed: int = 0
    dynamic: dict[str, Any] | None = None
    frozen_recipe: list[dict[str, int]] | None = None
    codebook_dim: int | None = None
    bias_correction: str = "none"
    sink_tokens: int = 0
    recent_window: int = 0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 17

    def __post_init__(self) -> None:
        if self.batches < 1:
            raise ValueError("KV-cache evaluation requires batches >= 1")
        if self.prompt_len < 2 or self.continuation_len < 1:
            raise ValueError("KV-cache prompt/continuation lengths are too small")
        if self.eval_offset_batches < 0:
            raise ValueError("KV-cache eval_offset_batches must be nonnegative")
        if self.skip < 0 or self.temperature <= 0:
            raise ValueError("KV-cache skip must be nonnegative and temperature positive")
        if self.bootstrap_draws < 1 or self.bootstrap_seed < 0:
            raise ValueError("bootstrap draws must be positive and seed nonnegative")
        if self.dynamic is not None and self.frozen_recipe is not None:
            raise ValueError("dynamic and frozen_recipe are mutually exclusive")
        if self.frozen_recipe is not None and not self.frozen_recipe:
            raise ValueError("frozen_recipe must contain at least one layer")

    def quant_config(self, *, seed: int | None = None) -> KVQuantConfig:
        return KVQuantConfig(
            bits=self.bits,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
            scale_quant_group_size=self.scale_quant_group_size,
            codebook=self.codebook,
            rotation_block=self.rotation_block,
            seed=self.seed if seed is None else seed,
            codebook_dim=self.codebook_dim,
            bias_correction=self.bias_correction,
            sink_tokens=self.sink_tokens,
            recent_window=self.recent_window,
        )


@dataclass(frozen=True)
class KVDynamicConfig:
    """Held-out, exact-byte mixed-precision allocation for K/V states."""

    candidate_bits: tuple[int, ...] = (3, 4, 8)
    key_candidate_bits: tuple[int, ...] | None = None
    value_candidate_bits: tuple[int, ...] | None = None
    target_bpv: float = 4.25
    selection_batches: int = 2
    nll_weight: float = 0.0

    def __post_init__(self) -> None:
        candidates = tuple(sorted({int(value) for value in self.candidate_bits}))
        if not candidates or any(value < 1 or value > 16 for value in candidates):
            raise ValueError("candidate_bits must contain integers in [1, 16]")
        object.__setattr__(self, "candidate_bits", candidates)
        for field in ("key_candidate_bits", "value_candidate_bits"):
            values = getattr(self, field)
            if values is None:
                continue
            values = tuple(sorted({int(value) for value in values}))
            if not values or any(value not in candidates for value in values):
                raise ValueError(
                    f"{field} must be a nonempty subset of candidate_bits")
            object.__setattr__(self, field, values)
        if self.target_bpv <= 0:
            raise ValueError("target_bpv must be > 0")
        if self.selection_batches < 1:
            raise ValueError("selection_batches must be >= 1")
        if self.nll_weight < 0:
            raise ValueError("nll_weight must be nonnegative")

    def bits_for(self, side: str) -> tuple[int, ...]:
        selected = (self.key_candidate_bits if side == "key"
                    else self.value_candidate_bits)
        return self.candidate_bits if selected is None else selected


def _clone_tree(value):
    """Clone every tensor reachable through plain containers.

    Transformers cache layers have stored linear-attention conv/recurrent
    state both as direct tensor attributes (5.9) and as ``dict[int, Tensor]``
    attributes (5.16).  Those states are updated *in place* during decode, so a
    shallow container copy would silently share them between the source and
    packed caches and the second decode pass would see state already advanced
    by the first.  Every tensor must therefore be cloned wherever it lives.
    """
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.copy(value)


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _storage_keys(cache) -> set[tuple[str, int]]:
    keys = set()
    for layer in cache.layers:
        for tensor in _iter_tensors(vars(layer)):
            if tensor.numel():
                keys.add((str(tensor.device), tensor.untyped_storage().data_ptr()))
    for name, value in vars(cache).items():
        if name == "layers":
            continue
        for tensor in _iter_tensors(value):
            if tensor.numel():
                keys.add((str(tensor.device), tensor.untyped_storage().data_ptr()))
    return keys


def _clone_cache(cache):
    """Clone all cache state so source and packed decode passes never interact."""
    cloned = copy.copy(cache)
    cloned.layers = []
    for layer in cache.layers:
        layer_copy = copy.copy(layer)
        for name, value in vars(layer).items():
            setattr(layer_copy, name, _clone_tree(value))
        cloned.layers.append(layer_copy)
    for name, value in vars(cache).items():
        if name != "layers" and isinstance(value, (torch.Tensor, dict, list, tuple)):
            setattr(cloned, name, _clone_tree(value))
    shared = _storage_keys(cache) & _storage_keys(cloned)
    if shared:
        raise RuntimeError(
            "cloned cache still shares tensor storage with the source cache; "
            "the source/packed decode passes would corrupt each other"
        )
    return cloned


def _cache_tensor_bytes(cache) -> int:
    storages: dict[tuple[str, int], int] = {}
    for layer in cache.layers:
        for value in _iter_tensors(vars(layer)):
            if value.numel() == 0:
                continue
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            storages[key] = storage.nbytes()
    return sum(storages.values())


def _reconstruct(packed: QuantizedKV, reference: torch.Tensor) -> torch.Tensor:
    return packed.dequantize(original_basis=True).to(
        device=reference.device, dtype=reference.dtype)


def simulate_packed_kv_cache(
    cache,
    config: KVQuantConfig,
    layer_configs: dict[int, KVQuantConfig] | None = None,
) -> tuple[Any, dict[str, float | int]]:
    """Return a cache with simulated rotate-before-store K/V quantization.

    Full-attention layers expose nonempty ``keys`` and ``values`` tensors.
    Recurrent/linear-attention state is cloned unchanged and included in total
    cache accounting, but is not mislabeled as K/V compression.
    """
    if not hasattr(cache, "layers") or not hasattr(cache, "update"):
        raise TypeError("expected a Transformers-style Cache with layers/update")
    simulated = _clone_cache(cache)
    source_total_bytes = _cache_tensor_bytes(cache)
    source_kv_bytes = 0
    source_kv_elements = 0
    key_elements = 0
    value_elements = 0
    key_code_bits = 0
    value_code_bits = 0
    key_squared_error = 0.0
    value_squared_error = 0.0
    key_signal_energy = 0.0
    value_signal_energy = 0.0
    packed_kv_bytes = 0
    full_precision_kv_elements = 0
    rotations: dict[int, tuple[Any, Any]] = {}
    n_kv_layers = 0

    for layer_idx, layer in enumerate(simulated.layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            continue
        if keys.numel() == 0 or values.numel() == 0:
            continue
        if keys.shape[-1] != values.shape[-1]:
            raise ValueError(f"K/V head dimensions differ at cache layer {layer_idx}")
        selected = ((layer_configs or {}).get(layer_idx, config))
        layer_config = replace(selected, seed=config.seed + 2 * layer_idx)
        key_rotation = build_kv_rotation(
            keys.shape[-1], layer_config, device=keys.device)
        value_rotation = build_kv_rotation(
            values.shape[-1], layer_config, value=True, device=values.device)
        packed_keys = quantize_kv(keys, key_rotation, layer_config)
        packed_values = quantize_kv(
            values, value_rotation, layer_config, value=True)
        reconstructed_keys = _reconstruct(packed_keys, keys)
        reconstructed_values = _reconstruct(packed_values, values)
        key_squared_error += float((
            reconstructed_keys.float() - keys.float()).square().sum().item())
        value_squared_error += float((
            reconstructed_values.float() - values.float()).square().sum().item())
        key_signal_energy += float(keys.float().square().sum().item())
        value_signal_energy += float(values.float().square().sum().item())
        layer.keys = reconstructed_keys
        layer.values = reconstructed_values
        rotations[layer_idx] = (key_rotation, value_rotation)
        source_kv_bytes += keys.numel() * keys.element_size()
        source_kv_bytes += values.numel() * values.element_size()
        source_kv_elements += keys.numel() + values.numel()
        key_elements += keys.numel()
        value_elements += values.numel()
        key_code_bits += keys.numel() * layer_config.bits_for()
        value_code_bits += (
            values.numel() * layer_config.bits_for(value=True))
        packed_kv_bytes += packed_keys.packed_state_bytes()
        packed_kv_bytes += packed_values.packed_state_bytes()
        if packed_keys.full_precision_rows is not None:
            full_precision_kv_elements += packed_keys.full_precision_rows.numel()
        if packed_values.full_precision_rows is not None:
            full_precision_kv_elements += packed_values.full_precision_rows.numel()
        n_kv_layers += 1

    if not rotations:
        raise ValueError("cache contains no initialized full-attention K/V layers")

    original_update = simulated.update

    def quantized_update(_self, key_states, value_states, layer_idx, *args, **kwargs):
        pair = rotations.get(layer_idx)
        if pair is not None:
            selected = ((layer_configs or {}).get(layer_idx, config))
            layer_config = replace(selected, seed=config.seed + 2 * layer_idx)
            packed_keys = quantize_kv(key_states, pair[0], layer_config)
            packed_values = quantize_kv(
                value_states, pair[1], layer_config, value=True)
            key_states = _reconstruct(packed_keys, key_states)
            value_states = _reconstruct(packed_values, value_states)
        return original_update(
            key_states, value_states, layer_idx, *args, **kwargs)

    simulated.update = types.MethodType(quantized_update, simulated)
    non_kv_state_bytes = max(source_total_bytes - source_kv_bytes, 0)
    deployed_total_bytes = non_kv_state_bytes + packed_kv_bytes
    return simulated, {
        "kv_layers": n_kv_layers,
        "key_bits": key_code_bits / max(key_elements, 1),
        "value_bits": value_code_bits / max(value_elements, 1),
        "source_kv_bytes": source_kv_bytes,
        "source_kv_elements": source_kv_elements,
        "packed_kv_bytes": packed_kv_bytes,
        "kv_compression_ratio": source_kv_bytes / max(packed_kv_bytes, 1),
        "effective_kv_bpv": packed_kv_bytes * 8 / max(source_kv_elements, 1),
        "full_precision_kv_fraction": (
            full_precision_kv_elements / max(source_kv_elements, 1)),
        "sink_tokens": config.sink_tokens,
        "recent_window": config.recent_window,
        "prefill_key_nmse": key_squared_error / max(key_signal_energy, 1e-12),
        "prefill_value_nmse": (
            value_squared_error / max(value_signal_energy, 1e-12)),
        "prefill_kv_nmse": (
            (key_squared_error + value_squared_error)
            / max(key_signal_energy + value_signal_energy, 1e-12)),
        "non_kv_state_bytes": non_kv_state_bytes,
        "source_total_cache_bytes": source_total_bytes,
        "deployed_total_cache_bytes": deployed_total_bytes,
        "total_cache_compression_ratio": source_total_bytes / max(
            deployed_total_bytes, 1),
    }


def _mean(items: list[float]) -> float:
    return sum(items) / max(len(items), 1)


def _accepts_position_ids(model: torch.nn.Module) -> bool:
    """Whether the public model forward accepts explicit absolute positions."""
    try:
        return "position_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


@torch.inference_mode()
def _evaluate_kv_cache(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    config: KVCacheEvalConfig,
    device,
    layer_configs: dict[int, KVQuantConfig] | None = None,
) -> dict[str, Any]:
    """Compare source and simulated packed-cache next-token distributions."""
    if len(batches) < config.batches:
        raise ValueError("not enough KV-cache evaluation batches")
    kls: list[float] = []
    cosine: list[float] = []
    top1: list[float] = []
    source_nll: list[float] = []
    quant_nll: list[float] = []
    size_metrics: list[dict[str, float | int]] = []
    evaluated_tokens = 0
    explicit_decode_positions = _accepts_position_ids(model)

    for batch_idx, batch in enumerate(batches[:config.batches]):
        ids = batch["input_ids"].to(device)
        required = config.prompt_len + config.continuation_len + 1
        if ids.shape[1] < required:
            raise ValueError(
                f"KV batch requires {required} tokens, got {ids.shape[1]}")
        prefill_ids = ids[:, :config.prompt_len]
        prefill_mask = torch.ones_like(prefill_ids)
        source_prefill = model(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            use_cache=True,
        )
        source_cache = source_prefill.past_key_values
        if source_cache is None:
            raise RuntimeError("model did not return a cache with use_cache=True")
        packed_cache, cache_metrics = simulate_packed_kv_cache(
            source_cache, config.quant_config(
                seed=config.seed + batch_idx * 1009),
            layer_configs=layer_configs)
        size_metrics.append(cache_metrics)

        for step in range(config.continuation_len):
            position = config.prompt_len + step
            current = ids[:, position:position + 1]
            target = ids[:, position + 1]
            attention_mask = torch.ones(
                ids.shape[0], position + 1, dtype=torch.long, device=device)
            decode_kwargs: dict[str, torch.Tensor] = {}
            if explicit_decode_positions:
                # Unified multimodal wrappers such as Qwen3.5 retain M-RoPE
                # state after ``generate``.  If they infer positions from the
                # full cache attention mask, a one-token decode can incorrectly
                # receive ``position_ids`` with ``position + 1`` entries.  The
                # evaluator is text-only, so provide its unambiguous absolute
                # one-token position and avoid stateful wrapper inference.
                decode_kwargs["position_ids"] = torch.full(
                    (ids.shape[0], 1), position,
                    dtype=torch.long, device=device)
            source = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=source_cache,
                use_cache=True,
                **decode_kwargs,
            )
            candidate = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=packed_cache,
                use_cache=True,
                **decode_kwargs,
            )
            source_cache = source.past_key_values
            packed_cache = candidate.past_key_values
            source_logits = source.logits[:, -1].float()
            candidate_logits = candidate.logits[:, -1].float()
            temperature = config.temperature
            source_log_probs = F.log_softmax(source_logits / temperature, dim=-1)
            candidate_log_probs = F.log_softmax(
                candidate_logits / temperature, dim=-1)
            source_probs = source_log_probs.exp()
            kl = (source_probs * (
                source_log_probs - candidate_log_probs)).sum(dim=-1)
            kls.extend((kl * temperature**2).detach().cpu().tolist())
            cosine.extend(F.cosine_similarity(
                source_logits, candidate_logits, dim=-1).detach().cpu().tolist())
            top1.extend((
                source_logits.argmax(dim=-1)
                == candidate_logits.argmax(dim=-1)
            ).float().detach().cpu().tolist())
            source_nll.extend(F.cross_entropy(
                source_logits, target, reduction="none"
            ).detach().cpu().tolist())
            quant_nll.extend(F.cross_entropy(
                candidate_logits, target, reduction="none"
            ).detach().cpu().tolist())
            evaluated_tokens += target.numel()

    averaged_sizes = {
        key: _mean([float(metrics[key]) for metrics in size_metrics])
        for key in size_metrics[0]
        if key != "kv_layers"
    }
    nll_deltas = [candidate - source
                  for source, candidate in zip(source_nll, quant_nll)]
    result: dict[str, Any] = {
        "batches": config.batches,
        "prompt_len": config.prompt_len,
        "continuation_len": config.continuation_len,
        "evaluated_tokens": evaluated_tokens,
        "kv_layers": int(size_metrics[0]["kv_layers"]),
        "mean_teacher_kl": _mean(kls),
        "mean_logit_cosine": _mean(cosine),
        "top1_agreement": _mean(top1),
        "source_nll": _mean(source_nll),
        "quantized_cache_nll": _mean(quant_nll),
        "nll_delta": _mean(quant_nll) - _mean(source_nll),
        **averaged_sizes,
    }
    result["paired_token_bootstrap"] = bootstrap_report(
        {
            "mean_teacher_kl": kls,
            "mean_logit_cosine": cosine,
            "top1_agreement": top1,
            "nll_delta": nll_deltas,
        },
        draws=config.bootstrap_draws,
        seed=config.bootstrap_seed,
    )
    return result


def _score(metrics: dict[str, float | int], config: KVDynamicConfig) -> float:
    return (float(metrics["mean_teacher_kl"])
            + config.nll_weight * max(float(metrics["nll_delta"]), 0.0))


def _discover_kv_layers(model, batch, config: KVCacheEvalConfig, device) -> list[int]:
    ids = batch["input_ids"].to(device)[:, :config.prompt_len]
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
    )
    cache = output.past_key_values
    if cache is None or not hasattr(cache, "layers"):
        raise RuntimeError("model did not return a Transformers-style cache")
    layers = []
    for index, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if (isinstance(keys, torch.Tensor) and keys.numel()
                and isinstance(values, torch.Tensor) and values.numel()):
            layers.append(index)
    if not layers:
        raise ValueError("cache contains no initialized full-attention K/V layers")
    return layers


def _uniform_recipe(
    layers: list[int],
    base: KVQuantConfig,
    bits: int,
) -> dict[int, KVQuantConfig]:
    return {
        layer: replace(base, key_bits=bits, value_bits=bits)
        for layer in layers
    }


def _materialize_frozen_recipe(
    records: list[dict[str, int]],
    layers: list[int],
    base: KVQuantConfig,
) -> dict[int, KVQuantConfig]:
    """Validate and materialize a serialized per-layer K/V bit recipe."""
    recipe: dict[int, KVQuantConfig] = {}
    for record in records:
        missing = {"layer", "key_bits", "value_bits"} - set(record)
        if missing:
            raise ValueError(
                "frozen_recipe entry is missing: " + ", ".join(sorted(missing)))
        layer = int(record["layer"])
        if layer in recipe:
            raise ValueError(f"frozen_recipe contains duplicate layer {layer}")
        key_bits = int(record["key_bits"])
        value_bits = int(record["value_bits"])
        if not 1 <= key_bits <= 16 or not 1 <= value_bits <= 16:
            raise ValueError("frozen_recipe bits must be in [1, 16]")
        recipe[layer] = replace(
            base, key_bits=key_bits, value_bits=value_bits)

    expected = set(layers)
    actual = set(recipe)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing layers {missing}")
        if extra:
            details.append(f"unknown layers {extra}")
        raise ValueError("frozen_recipe does not match cache: " + "; ".join(details))
    return recipe


@torch.inference_mode()
def select_dynamic_kv_quantization(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    config: KVCacheEvalConfig,
    device,
) -> tuple[dict[int, KVQuantConfig], dict[str, Any]]:
    """Allocate cache bits by held-out end-to-end KL per exact saved byte.

    Candidate effects are screened one K/V state at a time from the highest
    precision recipe. The resulting multiple-choice allocation is then measured
    jointly and retained only when it beats the best uniform recipe that fits the
    same byte budget.
    """
    dynamic = KVDynamicConfig(**(config.dynamic or {}))
    if len(batches) < dynamic.selection_batches:
        raise ValueError("not enough KV-cache dynamic-selection batches")
    selection = batches[:dynamic.selection_batches]
    selection_config = replace(
        config, batches=dynamic.selection_batches, dynamic=None)
    layers = _discover_kv_layers(model, selection[0], selection_config, device)
    base = selection_config.quant_config()
    highest = max(dynamic.candidate_bits)
    high_recipe = _uniform_recipe(layers, base, highest)
    high_metrics = _evaluate_kv_cache(
        model, selection, selection_config, device, high_recipe)
    high_bytes = round(float(high_metrics["packed_kv_bytes"]))
    elements = round(float(high_metrics["source_kv_elements"]))
    target_bytes = int(dynamic.target_bpv * elements // 8)

    candidates: dict[tuple[int, str], list[dict[str, float | int]]] = {}
    total = sum(len(dynamic.bits_for(side))
                for _ in layers for side in ("key", "value"))
    progress = 0
    for layer in layers:
        for side in ("key", "value"):
            options = []
            for bits in dynamic.bits_for(side):
                recipe = dict(high_recipe)
                current = recipe[layer]
                recipe[layer] = replace(
                    current,
                    key_bits=bits if side == "key" else current.bits_for(),
                    value_bits=(bits if side == "value"
                                else current.bits_for(value=True)),
                )
                metrics = (high_metrics if bits == highest else _evaluate_kv_cache(
                    model, selection, selection_config, device, recipe))
                candidate_bytes = round(float(metrics["packed_kv_bytes"]))
                options.append({
                    "bits": bits,
                    "bytes_delta": candidate_bytes - high_bytes,
                    "teacher_kl": float(metrics["mean_teacher_kl"]),
                    "nll_delta": float(metrics["nll_delta"]),
                    "score": _score(metrics, dynamic),
                })
                progress += 1
                if bits != highest:
                    logger.info(
                        "KV dynamic scored %d/%d layer %d %s %db: KL %.4g",
                        progress, total, layer, side, bits,
                        metrics["mean_teacher_kl"])
            candidates[(layer, side)] = sorted(
                options, key=lambda item: int(item["bits"]))

    selected = {component: len(options) - 1
                for component, options in candidates.items()}

    def stored_bytes() -> int:
        return high_bytes + sum(
            int(candidates[component][index]["bytes_delta"])
            for component, index in selected.items())

    while stored_bytes() > target_bytes:
        best = None
        for order, (component, index) in enumerate(selected.items()):
            if index == 0:
                continue
            current = candidates[component][index]
            lower = candidates[component][index - 1]
            savings = (int(current["bytes_delta"])
                       - int(lower["bytes_delta"]))
            if savings <= 0:
                continue
            penalty = max(
                float(lower["score"]) - float(current["score"]), 0.0)
            key = (penalty / savings, -savings, order)
            if best is None or key < best[0]:
                best = (key, component)
        if best is None:
            break
        selected[best[1]] -= 1

    recipe = dict(high_recipe)
    for (layer, side), index in selected.items():
        bits = int(candidates[(layer, side)][index]["bits"])
        current = recipe[layer]
        recipe[layer] = replace(
            current,
            key_bits=bits if side == "key" else current.bits_for(),
            value_bits=(bits if side == "value"
                        else current.bits_for(value=True)),
        )
    selected_metrics = _evaluate_kv_cache(
        model, selection, selection_config, device, recipe)

    uniform_trials = []
    best_uniform = None
    common_bits = sorted(
        set(dynamic.bits_for("key")) & set(dynamic.bits_for("value")))
    for bits in common_bits:
        uniform = _uniform_recipe(layers, base, bits)
        metrics = (high_metrics if bits == highest else _evaluate_kv_cache(
            model, selection, selection_config, device, uniform))
        trial = {
            "bits": bits,
            "packed_kv_bytes": round(float(metrics["packed_kv_bytes"])),
            "teacher_kl": float(metrics["mean_teacher_kl"]),
            "nll_delta": float(metrics["nll_delta"]),
            "score": _score(metrics, dynamic),
        }
        uniform_trials.append(trial)
        if (trial["packed_kv_bytes"] <= target_bytes
                and (best_uniform is None
                     or trial["score"] < best_uniform[0]["score"])):
            best_uniform = (trial, uniform, metrics)

    selected_score = _score(selected_metrics, dynamic)
    restored_uniform = bool(
        best_uniform is not None and best_uniform[0]["score"] < selected_score)
    if restored_uniform:
        recipe = best_uniform[1]
        deployed_metrics = best_uniform[2]
    else:
        deployed_metrics = selected_metrics

    key_counts: dict[str, int] = {}
    value_counts: dict[str, int] = {}
    details = []
    for layer in layers:
        key_bits = recipe[layer].bits_for()
        value_bits = recipe[layer].bits_for(value=True)
        key_counts[str(key_bits)] = key_counts.get(str(key_bits), 0) + 1
        value_counts[str(value_bits)] = value_counts.get(str(value_bits), 0) + 1
        details.append({
            "layer": layer,
            "key_bits": key_bits,
            "value_bits": value_bits,
        })
    stats = {
        "config": asdict(dynamic),
        "kv_layers": len(layers),
        "target_bpv": dynamic.target_bpv,
        "target_bytes": target_bytes,
        "selected_estimated_bytes": stored_bytes(),
        "deployed_bytes": round(float(deployed_metrics["packed_kv_bytes"])),
        "target_reached": (
            float(deployed_metrics["packed_kv_bytes"]) <= target_bytes),
        "selected_teacher_kl": float(selected_metrics["mean_teacher_kl"]),
        "deployed_teacher_kl": float(deployed_metrics["mean_teacher_kl"]),
        "restored_uniform": restored_uniform,
        "key_counts_by_bits": key_counts,
        "value_counts_by_bits": value_counts,
        "uniform_trials": uniform_trials,
        "recipe": details,
        "candidate_scores": [
            {"layer": layer, "side": side, "options": options}
            for (layer, side), options in candidates.items()
        ],
    }
    return recipe, stats


@torch.inference_mode()
def evaluate_kv_cache(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    config: KVCacheEvalConfig,
    device,
) -> dict[str, Any]:
    """Compare source and packed-cache distributions, optionally allocating bits."""
    if not config.dynamic:
        required = config.eval_offset_batches + config.batches
        if len(batches) < required:
            raise ValueError(
                f"KV evaluation requires {required} batches, got {len(batches)}")
        evaluation = batches[config.eval_offset_batches:required]
        evaluation_config = replace(config, eval_offset_batches=0)
        layer_configs = None
        if config.frozen_recipe is not None:
            layers = _discover_kv_layers(
                model, evaluation[0], evaluation_config, device)
            layer_configs = _materialize_frozen_recipe(
                config.frozen_recipe, layers, evaluation_config.quant_config())
        metrics: dict[str, Any] = _evaluate_kv_cache(
            model, evaluation, evaluation_config, device, layer_configs)
        if config.frozen_recipe is not None:
            metrics["frozen_recipe"] = {
                "recipe": [dict(record) for record in config.frozen_recipe],
                "validated_layers": len(layer_configs or {}),
            }
        return metrics
    dynamic = KVDynamicConfig(**config.dynamic)
    evaluation_offset = max(
        dynamic.selection_batches, config.eval_offset_batches)
    required = evaluation_offset + config.batches
    if len(batches) < required:
        raise ValueError(
            f"dynamic KV evaluation requires {required} batches, got {len(batches)}")
    selection = batches[:dynamic.selection_batches]
    evaluation = batches[evaluation_offset:required]
    recipe, stats = select_dynamic_kv_quantization(
        model, selection, config, device)
    final: dict[str, Any] = _evaluate_kv_cache(
        model, evaluation,
        replace(config, dynamic=None, eval_offset_batches=0), device, recipe)
    final["dynamic"] = stats
    return final
