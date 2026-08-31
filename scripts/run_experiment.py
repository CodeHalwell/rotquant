#!/usr/bin/env python
"""config -> quantise -> eval -> write results/<run_id>.json

Reads a single experiment YAML (see ``configs/``), loads the HF model, optionally
collects real-activation Hessians, patches it with ``QuantLinear`` and runs the
fixed eval protocol. Every run writes a JSON with config, git SHA, library
versions, GPU, all metrics, and wall-clock.

Config resolution: if a ``_base.yaml`` sits next to the experiment YAML it is
deep-merged underneath it (experiment keys win, nested dicts merge, lists are
replaced wholesale). ``--model``, ``--device``, ``--seed`` and dotted
``--set key=value`` override the merged config from the CLI so the sweeps the
configs describe never require editing YAML:

    python scripts/run_experiment.py configs/e1_rotation.yaml --seed 1
    python scripts/run_experiment.py configs/e2_codebook.yaml --model facebook/opt-125m
    python scripts/run_experiment.py configs/e1_rotation.yaml --set patch.rotation=dense
    python scripts/run_experiment.py configs/e1_rotation.yaml --set patch.enabled=false
    python scripts/run_experiment.py configs/e8_footprint.yaml --set patch.fallback=true

Run ids of CLI-modified runs get the overridden values appended, so sweep
results never overwrite each other.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.utils import (
    Timer,
    enable_default_logging,
    environment_record,
    get_logger,
    peak_vram_bytes,
    reset_peak_vram,
    set_seed,
    write_result,
)

logger = get_logger(__name__)

BASE_CONFIG_NAME = "_base.yaml"
MAX_RUN_ID_LENGTH = 220  # leaves room for the .json suffix under NAME_MAX=255
MODEL_LOADERS = ("auto", "causal_lm", "multimodal_lm")
class TokenBatch(dict):
    """Model-input mapping carrying a non-forwarded source-row identity."""

    def __init__(self, input_ids: torch.Tensor, source_row: int):
        super().__init__(input_ids=input_ids)
        self.source_row = source_row


_CALIB_CACHE: OrderedDict[
    str, tuple[tuple[torch.Tensor, int], ...]
] = OrderedDict()
_CALIB_CACHE_LIMIT = 32
_TRAJECTORY_REFERENCE_CACHE: dict[str, Any] = {}
_LOGIT_REFERENCE_CACHE: dict[str, Any] = {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``. Lists are replaced."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict[str, Any]:
    """Load an experiment YAML, deep-merging ``_base.yaml`` from the same dir."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = os.path.join(os.path.dirname(os.path.abspath(path)), BASE_CONFIG_NAME)
    if os.path.basename(path) != BASE_CONFIG_NAME and os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)
    return cfg


def apply_set_overrides(cfg: dict[str, Any], sets) -> None:
    """Apply ``[(dotted.key, value), ...]`` overrides in place, creating
    intermediate dicts as needed."""
    for key, value in sets:
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=+-]+", "-", str(text)).strip("-")


def override_slug(model: str | None, sets, device: str | None = None) -> str:
    """Filename fragment describing the CLI overrides that change results."""
    parts = []
    if model:
        parts.append(_slug(model.rstrip("/").split("/")[-1]))
    if device:
        parts.append(_slug(f"device={device}"))
    for key, value in sets:
        parts.append(_slug(f"{key}={value}"))
    return "_".join(parts)


def derive_run_id(cfg: dict[str, Any], config_path: str,
                  slug: str = "", seed_overridden: bool = False) -> str:
    """An explicit ``run_id`` with no CLI overrides is used verbatim. Otherwise
    the base name (``run_id`` > ``label`` > config file stem) gets the override
    slug and a ``_s<seed>`` suffix appended, so neither seed sweeps nor
    ``--model``/``--set`` sweeps ever overwrite each other's results."""
    explicit = cfg.get("run_id")
    if explicit and not slug and not seed_overridden:
        return _bounded_run_id(str(explicit))
    stem = os.path.splitext(os.path.basename(config_path))[0]
    parts = [str(explicit) if explicit else (cfg.get("label") or stem)]
    if slug:
        parts.append(slug)
    if not explicit or seed_overridden:
        parts.append(f"s{int(cfg.get('seed', 0))}")
    return _bounded_run_id("_".join(parts))


def _bounded_run_id(run_id: str, max_length: int = MAX_RUN_ID_LENGTH) -> str:
    """Bound long IDs deterministically while preserving ``_sN`` grouping.

    macOS and most Linux filesystems cap one filename component at 255 bytes.
    CLI sweep descriptions can exceed that, especially with nested training
    overrides. The digest prevents prefix collisions; retaining the seed suffix
    lets ``aggregate.py`` continue merging seeds of the same experiment cell.
    """
    if len(run_id.encode("utf-8")) <= max_length:
        return run_id
    match = re.search(r"(_s\d+)$", run_id)
    seed_suffix = match.group(1) if match else ""
    body = run_id[:-len(seed_suffix)] if seed_suffix else run_id
    # Hash the seedless body so seed variants retain an identical aggregate key.
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    reserved = len(digest) + 1 + len(seed_suffix)
    prefix = body[:max_length - reserved].rstrip("_-")
    return f"{prefix}_{digest}{seed_suffix}"


def resolve_device_dtype(cfg: dict[str, Any]) -> tuple:
    """Honour the config but never crash on a CUDA-less box: fall back to CPU
    (and to fp32, since fp16 matmuls on CPU are unsupported/slow)."""
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = cfg.get("dtype", "float16")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("config requests device=%s but CUDA is unavailable; "
                       "falling back to cpu", device)
        device = "cpu"
    if device == "cpu" and dtype_name in ("float16", "half", "bfloat16"):
        logger.warning("dtype=%s on cpu is slow/unsupported; using float32",
                       dtype_name)
        dtype_name = "float32"
    return device, getattr(torch, dtype_name)


def apply_device_defaults(cfg: dict[str, Any], device) -> bool:
    """Apply safe execution defaults required by a backend.

    MPS has no fused packed matmul in this project. Re-unpacking int32 weights on
    every forward is dramatically slower than the matmul, so quality experiments
    cache the dequantized source-dtype weight once. Packed storage accounting is
    still reported, but MPS runs must not be used for packed throughput numbers.
    Returns whether the config was changed.
    """
    if str(device).startswith("mps"):
        patch_cfg = cfg.setdefault("patch", {})
        if not patch_cfg.get("enabled", True):
            return False
        if not patch_cfg.get("fallback", False):
            patch_cfg["fallback"] = True
            logger.warning(
                "MPS has no fused packed QuantLinear kernel; enabling "
                "patch.fallback=true for quality evaluation. Do not use this run "
                "for packed throughput or peak-memory measurements.")
            return True
    return False


def resolve_model_loader(config, requested: str = "auto") -> str:
    """Choose the appropriate Transformers auto-model family.

    Most experiments use decoder-only ``AutoModelForCausalLM`` checkpoints.
    Newer unified text/vision checkpoints, including Qwen3.5, expose a nested
    ``vision_config`` and must instead be constructed through
    ``AutoModelForMultimodalLM`` even when an evaluation batch contains text
    only.  Keep an explicit config override for unusual/custom architectures.
    """
    if requested not in MODEL_LOADERS:
        raise ValueError(
            f"unknown model_loader={requested!r}; pick from {MODEL_LOADERS}")
    if requested != "auto":
        return requested
    return ("multimodal_lm"
            if getattr(config, "vision_config", None) is not None
            else "causal_lm")


def load_hf_model(model_name: str, dtype: torch.dtype, device,
                  model_loader: str = "auto", model_revision: str | None = None):
    """Load a text or unified multimodal Hugging Face model plus tokenizer.

    RotQuant's language-quality harness intentionally uses the tokenizer and
    text-only forward path. Vision inputs require the model's AutoProcessor at
    serving/evaluation time, but are not needed for WikiText/C4 perplexity.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, revision=model_revision)
    selected = resolve_model_loader(config, model_loader)
    if selected == "causal_lm":
        model_cls = AutoModelForCausalLM
    else:
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "model_loader='multimodal_lm' requires a Transformers release "
                "that provides AutoModelForMultimodalLM") from exc
        model_cls = AutoModelForMultimodalLM

    logger.info("loading %s with %s", model_name, model_cls.__name__)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, revision=model_revision)
    model = model_cls.from_pretrained(
        model_name, config=config, dtype=dtype, revision=model_revision).to(device)
    return model, tokenizer, selected


def build_calib_loader(tokenizer, n_seq: int, seq_len: int, device,
                       skip: int = 0, revision: str | None = None):
    """Tokenised C4/WikiText-train calibration sequences (128-512 typical)."""
    tokenizer_identity = {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens": getattr(tokenizer, "special_tokens_map", None),
        "revision": (getattr(tokenizer, "init_kwargs", {}) or {}).get(
            "revision"
        ),
        "commit_hash": (getattr(tokenizer, "init_kwargs", {}) or {}).get(
            "_commit_hash"
        ),
    }
    key = hashlib.sha256(json.dumps({
        "tokenizer": tokenizer_identity,
        "revision": revision,
        "n_seq": n_seq,
        "seq_len": seq_len,
        "skip": skip,
    }, sort_keys=True, default=str).encode()).hexdigest()
    cached = _CALIB_CACHE.get(key)
    if cached is not None:
        _CALIB_CACHE.move_to_end(key)
        return [TokenBatch(ids.to(device), source_row) for ids, source_row in cached]

    from datasets import load_dataset
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        revision=revision,
    )
    batches, count, eligible = [], 0, 0
    for source_row, row in enumerate(ds):
        ids = tokenizer(row["text"], return_tensors="pt").input_ids
        if ids.shape[1] < seq_len:
            continue
        if eligible < skip:
            eligible += 1
            continue
        batches.append(TokenBatch(ids[:, :seq_len].cpu(), source_row))
        count += 1
        if count >= n_seq:
            break
    cached_batches = tuple(
        (batch["input_ids"], batch.source_row) for batch in batches
    )
    _CALIB_CACHE[key] = cached_batches
    _CALIB_CACHE.move_to_end(key)
    while len(_CALIB_CACHE) > _CALIB_CACHE_LIMIT:
        _CALIB_CACHE.popitem(last=False)
    return [
        TokenBatch(ids.to(device), source_row)
        for ids, source_row in cached_batches
    ]


def token_batch_manifest(batches, *, dataset: str, split: str,
                         revision: str | None, skip: int,
                         seq_len: int) -> dict[str, Any]:
    """Return immutable identities for exact token batches used by a stage."""

    hashes = [hashlib.sha256(
        batch["input_ids"].detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest() for batch in batches]
    return {
        "dataset": dataset,
        "split": split,
        "revision": revision,
        "skip": skip,
        "seq_len": seq_len,
        "batches": len(hashes),
        "source_rows": [getattr(batch, "source_row", None) for batch in batches],
        "token_hashes": hashes,
        "digest": hashlib.sha256("".join(hashes).encode()).hexdigest(),
    }


def _reference_cache_key(kind: str, model_name: str,
                         model_revision: str | None, config,
                         batches) -> str:
    payload = {
        "kind": kind,
        "model": model_name,
        "model_revision": model_revision,
        "config": vars(config),
        "token_hashes": token_batch_manifest(
            batches, dataset="allenai/c4", split="train",
            revision=None, skip=int(getattr(config, "skip", 0)),
            seq_len=int(getattr(
                config, "prompt_len", getattr(config, "seq_len", 0)
            )),
        )["token_hashes"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def footprint_metrics(model: torch.nn.Module, cfg_model: dict[str, Any]) -> dict[str, Any]:
    """True bits/weight + packed-vs-fp16 storage across all QuantLinear layers.

    Enforces the equal-bits discipline: if the config declares ``claimed_bpw``,
    every layer's BitBudget must match it (``BitBudget.assert_matches``).
    """
    from rotquant.linear import QuantLinear
    metrics: dict[str, Any] = {}
    bpws, packed_bytes, fp16_bytes = [], 0, 0
    rotation_parameter_bytes, adapter_parameter_bytes = 0, 0
    codebook_bytes = fallback_cache_bytes = 0
    total_bits, total_weights = 0.0, 0
    claimed = cfg_model.get("claimed_bpw")
    tol = float(cfg_model.get("claimed_bpw_tol", 1e-6))
    for _, mod in model.named_modules():
        if not isinstance(mod, QuantLinear):
            continue
        budget = mod.qweight.bit_budget()
        if claimed is not None:
            budget.assert_matches(float(claimed), tol=tol)
        n_weights = mod.out_features * mod.in_features
        bpws.append(budget.bits_per_weight)
        total_bits += budget.bits_per_weight * n_weights
        total_weights += n_weights
        packed_bytes += mod.packed_state_bytes()
        codebook = getattr(mod.qweight, "codebook", None)
        centroids = getattr(codebook, "centroids", None)
        if centroids is not None:
            codebook_bytes += centroids.numel() * centroids.element_size()
        if mod._fp_cache is not None:
            fallback_cache_bytes += (
                mod._fp_cache.numel() * mod._fp_cache.element_size()
            )
        fp16_bytes += n_weights * 2
        rotation_parameter_bytes += sum(
            p.numel() * p.element_size() for p in mod.act_rotation.parameters())
        adapter_parameter_bytes += mod.adapter_state_bytes()
    if bpws:
        metrics["n_quant_layers"] = len(bpws)
        # Size-weighted: total stored bits over total quantised weights, so a
        # small layer with high per-in_features overhead (e.g. TurboQuant
        # per-row scales) cannot skew the model-wide rate used for equal-bit
        # comparisons. min/max stay per-layer extremes.
        metrics["bits_per_weight_mean"] = total_bits / total_weights
        metrics["bits_per_weight_min"] = min(bpws)
        metrics["bits_per_weight_max"] = max(bpws)
        metrics["packed_weight_bytes"] = packed_bytes
        metrics["fp16_weight_bytes"] = fp16_bytes
        metrics["compression_ratio"] = fp16_bytes / max(packed_bytes, 1)
        if adapter_parameter_bytes:
            metrics["adapter_parameter_bytes"] = adapter_parameter_bytes
        if rotation_parameter_bytes or adapter_parameter_bytes:
            effective_bytes = (packed_bytes + rotation_parameter_bytes
                               + adapter_parameter_bytes)
            metrics["rotation_parameter_bytes"] = rotation_parameter_bytes
            metrics["packed_plus_rotation_bytes"] = (
                packed_bytes + rotation_parameter_bytes)
            metrics["packed_plus_auxiliary_bytes"] = effective_bytes
            metrics["effective_bits_per_weight"] = (
                effective_bytes * 8 / total_weights)
            metrics["effective_compression_ratio"] = (
                fp16_bytes / effective_bytes)
    registered_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(model.parameters()) + list(model.buffers())
    )
    metrics["registered_model_bytes"] = registered_bytes
    metrics["codebook_bytes"] = codebook_bytes
    metrics["complete_persistent_model_bytes"] = (
        registered_bytes + packed_bytes + codebook_bytes
    )
    metrics["fallback_cache_bytes"] = fallback_cache_bytes
    metrics["quality_runtime_model_bytes"] = (
        metrics["complete_persistent_model_bytes"] + fallback_cache_bytes
    )
    return metrics


@dataclass
class _CalibrationArtifacts:
    """Calibration inputs produced by :func:`_prepare_calibration`."""

    hessians: dict[str, torch.Tensor] | None = None
    activations: dict[str, torch.Tensor] | None = None
    block_calls: Any = None
    distill_calls: Any = None
    dynamic_calls: Any = None
    needs_dynamic: bool = False
    needs_block_calls: bool = False


@dataclass
class _ReferenceCaptures:
    """Source-model captures taken before patching, replayed afterwards."""

    fp_capture: Any = None
    drift_batch: Any = None
    trajectory_config: Any = None
    trajectory_references: Any = None
    logit_fidelity_config: Any = None
    logit_references: Any = None


def _prepare_calibration(cfg: dict[str, Any], model, tokenizer, device: str,
                         qcfg: QuantConfig, pcfg: PatchConfig,
                         metrics: dict[str, Any]) -> _CalibrationArtifacts:
    """Collect Hessian / activation / teacher / block calibration artifacts.

    One C4 loader is built lazily and reused by later stages when its batch
    count suffices. The data manifest records the sequence length the shared
    loader was *actually built with*, which can differ from a stage's own
    default when an earlier stage built the loader first (e.g. Hessian
    calibration at 2048 tokens feeding block capture whose default is 256).
    """
    calibration_revision = cfg.get("calib_dataset_revision")
    art = _CalibrationArtifacts()
    rotation_train_cfg = pcfg.train_rotation or {}
    dynamic_cfg = pcfg.dynamic or {}
    art.needs_dynamic = pcfg.enabled and bool(pcfg.dynamic)
    art.needs_block_calls = (pcfg.enabled
                             and rotation_train_cfg.get("objective") == "block")
    needs_hessians = pcfg.enabled and (
        qcfg.error_comp == "gptq" or cfg.get("calibrate", False))
    needs_activations = (pcfg.enabled and (
        rotation_train_cfg.get("objective") == "activation"
        or art.needs_dynamic))

    calib_loader = None
    calib_loader_seq_len: int | None = None

    def build_shared_loader(n_batches: int, seq_len: int):
        nonlocal calib_loader, calib_loader_seq_len
        calib_loader = build_calib_loader(
            tokenizer, n_batches, seq_len, device,
            revision=calibration_revision)
        calib_loader_seq_len = seq_len
        return calib_loader

    if needs_hessians:
        from rotquant.calibrate import collect_hessians
        build_shared_loader(int(cfg.get("n_calib", 128)),
                            int(cfg.get("calib_seq_len", 2048)))
        metrics["data_manifest"]["hessian_calibration"] = token_batch_manifest(
            calib_loader, dataset="allenai/c4", split="train",
            revision=calibration_revision, skip=0,
            seq_len=calib_loader_seq_len,
        )
        with Timer() as t:
            # damp_frac=0: the GPTQ solver applies (auto-increasing) percdamp
            # itself; damping here as well would double it.
            calib = collect_hessians(model, calib_loader, device,
                                     include=pcfg.include,
                                     exclude=pcfg.exclude,
                                     damp_frac=0.0)
        art.hessians = calib.hessians
        metrics["calib_seconds"] = t.elapsed

    if needs_activations:
        from rotquant.calibrate import collect_activations
        rotation_tokens = 0
        if rotation_train_cfg.get("objective") == "activation":
            train_tokens = int(rotation_train_cfg.get("max_tokens", 64))
            selection_tokens = int(
                rotation_train_cfg.get("selection_tokens", 0))
            rotation_tokens = train_tokens + selection_tokens
        dynamic_tokens = int(dynamic_cfg.get("max_tokens", 32)) \
            if art.needs_dynamic else 0
        max_tokens = max(rotation_tokens, dynamic_tokens)
        if calib_loader is None:
            calib_seq_len = int(cfg.get("calib_seq_len", 2048))
            minimum_batches = max(1, (max_tokens + calib_seq_len - 1) // calib_seq_len)
            n_batches = int(cfg.get("rotation_n_calib", minimum_batches))
            build_shared_loader(n_batches, calib_seq_len)
        metrics["data_manifest"]["activation_calibration"] = token_batch_manifest(
            calib_loader, dataset="allenai/c4", split="train",
            revision=calibration_revision, skip=0,
            seq_len=calib_loader_seq_len,
        )
        with Timer() as t:
            activation_calib = collect_activations(
                model, calib_loader, device,
                include=pcfg.include, exclude=pcfg.exclude,
                max_tokens=max_tokens)
        art.activations = activation_calib.activations
        metrics["activation_calib_seconds"] = t.elapsed

    dynamic_global_batches = int(dynamic_cfg.get("global_kl_batches", 0)) \
        if art.needs_dynamic else 0
    if dynamic_global_batches:
        from rotquant.block_train import collect_teacher_calls
        dynamic_teacher_skip = int(cfg.get("dynamic_teacher_skip", 0))
        dynamic_seq_len = int(cfg.get("calib_seq_len", 256))
        dynamic_loader = build_calib_loader(
            tokenizer, dynamic_global_batches,
            dynamic_seq_len, device,
            skip=dynamic_teacher_skip,
            revision=calibration_revision)
        metrics["data_manifest"]["allocation_teacher"] = token_batch_manifest(
            dynamic_loader, dataset="allenai/c4", split="train",
            revision=calibration_revision, skip=dynamic_teacher_skip,
            seq_len=dynamic_seq_len,
        )
        with Timer() as t:
            art.dynamic_calls = collect_teacher_calls(
                model, dynamic_loader, device,
                max_batches=dynamic_global_batches)
        metrics["dynamic_teacher_calib_seconds"] = t.elapsed

    if art.needs_block_calls:
        from rotquant.block_train import (
            collect_block_calls,
            collect_teacher_calls,
            find_transformer_blocks,
        )
        train_batches = int(rotation_train_cfg.get("train_batches", 1))
        validation_batches = int(rotation_train_cfg.get("validation_batches", 1))
        selection_batches = int(rotation_train_cfg.get("selection_batches", 1))
        n_block_batches = train_batches + validation_batches + selection_batches
        distill_steps = int(rotation_train_cfg.get("distill_steps", 0))
        n_distill_batches = 0
        if distill_steps:
            n_distill_batches = (
                int(rotation_train_cfg.get("distill_train_batches", 1))
                + int(rotation_train_cfg.get("distill_validation_batches", 1))
                + int(rotation_train_cfg.get("distill_selection_batches", 1)))
        total_block_batches = n_block_batches + n_distill_batches
        if calib_loader is None or len(calib_loader) < total_block_batches:
            build_shared_loader(total_block_batches,
                                int(cfg.get("calib_seq_len", 256)))
        metrics["data_manifest"]["block_calibration"] = token_batch_manifest(
            calib_loader[:total_block_batches], dataset="allenai/c4",
            split="train", revision=calibration_revision, skip=0,
            seq_len=calib_loader_seq_len,
        )
        with Timer() as t:
            art.block_calls = collect_block_calls(
                model, calib_loader[:n_block_batches], device,
                blocks=find_transformer_blocks(model),
                max_batches=n_block_batches)
            if n_distill_batches:
                art.distill_calls = collect_teacher_calls(
                    model, calib_loader[n_block_batches:total_block_batches],
                    device, max_batches=n_distill_batches)
        metrics["block_calib_seconds"] = t.elapsed

    return art


def _capture_references(cfg: dict[str, Any], eval_cfg: dict[str, Any],
                        model, tokenizer, device: str, model_name: str,
                        metrics: dict[str, Any]) -> _ReferenceCaptures:
    """Capture source-model outputs that patched-model evaluation replays.

    Layer-drift capture (E7), multi-token trajectories, and teacher logits all
    must be taken BEFORE patching so the same batches can be compared on the
    quantized model. Trajectory and logit references are cached per
    (model, revision, config, token-hash) so seed sweeps do not recompute them.
    """
    calibration_revision = cfg.get("calib_dataset_revision")
    refs = _ReferenceCaptures()

    if eval_cfg.get("layer_mse", False):
        from rotquant.eval.layer_mse import capture_outputs
        layer_mse_skip = int(eval_cfg.get("layer_mse_skip", 0))
        layer_mse_seq_len = int(eval_cfg.get("layer_mse_seq_len", 512))
        refs.drift_batch = build_calib_loader(
            tokenizer, 1, layer_mse_seq_len, device,
            skip=layer_mse_skip,
            revision=calibration_revision)[0]
        metrics["data_manifest"]["layer_drift"] = token_batch_manifest(
            [refs.drift_batch], dataset="allenai/c4", split="train",
            revision=calibration_revision, skip=layer_mse_skip,
            seq_len=layer_mse_seq_len,
        )
        refs.fp_capture = capture_outputs(model, refs.drift_batch, device)

    trajectory_requested = eval_cfg.get("trajectory", False)
    if trajectory_requested:
        from rotquant.eval.trajectory import TrajectoryConfig, capture_trajectories
        trajectory_kwargs = (trajectory_requested
                             if isinstance(trajectory_requested, dict) else {})
        refs.trajectory_config = TrajectoryConfig(**trajectory_kwargs)
        trajectory_batches = build_calib_loader(
            tokenizer, refs.trajectory_config.batches,
            refs.trajectory_config.prompt_len, device,
            skip=refs.trajectory_config.skip,
            revision=calibration_revision)
        metrics["data_manifest"]["trajectory"] = token_batch_manifest(
            trajectory_batches, dataset="allenai/c4", split="train",
            revision=calibration_revision, skip=refs.trajectory_config.skip,
            seq_len=refs.trajectory_config.prompt_len,
        )
        trajectory_cache_key = _reference_cache_key(
            "trajectory", model_name, cfg.get("model_revision"),
            refs.trajectory_config, trajectory_batches,
        )
        refs.trajectory_references = _TRAJECTORY_REFERENCE_CACHE.get(
            trajectory_cache_key
        )
        metrics["source_trajectory_cache_hit"] = (
            refs.trajectory_references is not None
        )
        if refs.trajectory_references is None:
            with Timer() as t:
                refs.trajectory_references = capture_trajectories(
                    model, tokenizer, trajectory_batches, device,
                    refs.trajectory_config)
            metrics["source_trajectory_seconds"] = t.elapsed
            _TRAJECTORY_REFERENCE_CACHE[trajectory_cache_key] = (
                refs.trajectory_references
            )
        else:
            metrics["source_trajectory_seconds"] = 0.0

    logit_fidelity_requested = eval_cfg.get("logit_fidelity", False)
    if logit_fidelity_requested:
        from rotquant.eval.logit_fidelity import (
            LogitFidelityConfig,
            capture_logit_references,
        )
        logit_kwargs = (
            logit_fidelity_requested
            if isinstance(logit_fidelity_requested, dict) else {}
        )
        refs.logit_fidelity_config = LogitFidelityConfig(**logit_kwargs)
        logit_batches = build_calib_loader(
            tokenizer, refs.logit_fidelity_config.batches,
            refs.logit_fidelity_config.prompt_len, device,
            skip=refs.logit_fidelity_config.skip,
            revision=calibration_revision,
        )
        metrics["data_manifest"]["logit_fidelity"] = token_batch_manifest(
            logit_batches, dataset="allenai/c4", split="train",
            revision=calibration_revision, skip=refs.logit_fidelity_config.skip,
            seq_len=refs.logit_fidelity_config.prompt_len,
        )
        logit_cache_key = _reference_cache_key(
            "logit_fidelity", model_name, cfg.get("model_revision"),
            refs.logit_fidelity_config, logit_batches,
        )
        refs.logit_references = _LOGIT_REFERENCE_CACHE.get(logit_cache_key)
        metrics["source_logit_fidelity_cache_hit"] = (
            refs.logit_references is not None
        )
        if refs.logit_references is None:
            with Timer() as t:
                refs.logit_references = capture_logit_references(
                    model, logit_batches, device, refs.logit_fidelity_config
                )
            metrics["source_logit_fidelity_seconds"] = t.elapsed
            _LOGIT_REFERENCE_CACHE[logit_cache_key] = refs.logit_references
        else:
            metrics["source_logit_fidelity_seconds"] = 0.0

    return refs


def _apply_quantization(cfg: dict[str, Any], model, pcfg: PatchConfig,
                        art: _CalibrationArtifacts,
                        metrics: dict[str, Any]) -> None:
    """Run optional dynamic allocation, then patch or block-train the model."""
    patch_stats: dict[str, Any] = {}
    if art.needs_dynamic:
        from rotquant.dynamic import select_dynamic_quantization
        with Timer() as t:
            pcfg.layer_quant, dynamic_stats = select_dynamic_quantization(
                model, pcfg, activations=art.activations,
                teacher_calls=art.dynamic_calls)
        dynamic_stats["seconds"] = t.elapsed
        patch_stats["dynamic_quantization"] = dynamic_stats

    reset_peak_vram()
    with Timer() as t:
        if art.needs_block_calls:
            from rotquant.block_train import train_and_patch_blocks
            train_and_patch_blocks(model, pcfg, art.block_calls,
                                   distill_calls=art.distill_calls,
                                   stats_out=patch_stats)
        else:
            patch_model(model, pcfg, hessians=art.hessians,
                        activations=art.activations,
                        stats_out=patch_stats)
    metrics["patch_seconds"] = t.elapsed
    metrics["peak_vram_bytes_patch"] = peak_vram_bytes()
    metrics.update(patch_stats)
    metrics.update(footprint_metrics(model, cfg))


def _run_evaluations(cfg: dict[str, Any], eval_cfg: dict[str, Any],
                     model, tokenizer, device: str, seed: int,
                     refs: _ReferenceCaptures,
                     metrics: dict[str, Any]) -> None:
    """Run the fixed eval protocol on the patched model."""
    calibration_revision = cfg.get("calib_dataset_revision")
    reset_peak_vram()
    if refs.fp_capture is not None:
        from rotquant.eval.layer_mse import capture_outputs, drift_between
        q_capture = capture_outputs(model, refs.drift_batch, device)
        drift = drift_between(refs.fp_capture, q_capture)
        metrics["layer_mse"] = {"mse": drift.mse, "cosine": drift.cosine,
                                "order": drift.order}

    if refs.trajectory_references is not None:
        from rotquant.eval.trajectory import evaluate_trajectories
        with Timer() as t:
            metrics["trajectory"] = evaluate_trajectories(
                model, tokenizer, refs.trajectory_references, device,
                refs.trajectory_config)
        metrics["trajectory"]["seconds"] = t.elapsed

    if refs.logit_references is not None:
        from rotquant.eval.logit_fidelity import evaluate_logit_fidelity
        with Timer() as t:
            metrics["logit_fidelity"] = evaluate_logit_fidelity(
                model, refs.logit_references, device, refs.logit_fidelity_config
            )
        metrics["logit_fidelity"]["seconds"] = t.elapsed

    kv_cache_requested = eval_cfg.get("kv_cache", False)
    if kv_cache_requested:
        from rotquant.eval.kv_cache import KVCacheEvalConfig, evaluate_kv_cache
        kv_cache_kwargs = (kv_cache_requested
                           if isinstance(kv_cache_requested, dict) else {})
        kv_cache_kwargs = dict(kv_cache_kwargs)
        kv_cache_kwargs.setdefault("seed", seed)
        kv_cache_config = KVCacheEvalConfig(**kv_cache_kwargs)
        kv_selection_batches = 0
        if kv_cache_config.dynamic:
            from rotquant.eval.kv_cache import KVDynamicConfig
            kv_selection_batches = KVDynamicConfig(
                **kv_cache_config.dynamic).selection_batches
        kv_eval_offset = max(
            kv_selection_batches, kv_cache_config.eval_offset_batches)
        kv_cache_batches = build_calib_loader(
            tokenizer,
            kv_cache_config.batches + kv_eval_offset,
            (kv_cache_config.prompt_len
             + kv_cache_config.continuation_len + 1),
            device,
            skip=kv_cache_config.skip,
            revision=calibration_revision,
        )
        with Timer() as t:
            metrics["kv_cache"] = evaluate_kv_cache(
                model, kv_cache_batches, kv_cache_config, device)
        metrics["kv_cache"]["seconds"] = t.elapsed

    if eval_cfg.get("perplexity", True):
        from rotquant.eval.perplexity import PPLConfig, perplexity_details
        ppl_cfg = PPLConfig(**(eval_cfg.get("ppl") or {}))
        for ds in eval_cfg.get("ppl_datasets", ["wikitext2", "c4"]):
            with Timer() as t:
                ppl_details = perplexity_details(
                    model, tokenizer, ds, ppl_cfg, device
                )
                metrics[f"ppl_{ds}"] = ppl_details["ppl"]
                metrics[f"ppl_{ds}_details"] = ppl_details
                metrics["data_manifest"][f"ppl_{ds}"] = {
                    "dataset": ds,
                    "split": "test" if ds == "wikitext2" else "validation",
                    "revision": (
                        ppl_cfg.wikitext_revision
                        if ds == "wikitext2" else ppl_cfg.c4_revision
                    ),
                    "digest": ppl_details["input_digest"],
                    "window_hashes": ppl_details["window_hashes"],
                }
            metrics[f"ppl_{ds}_seconds"] = t.elapsed

    tp_requested = eval_cfg.get("throughput", False)
    if tp_requested:
        from rotquant.eval.throughput import ThroughputConfig, measure_throughput
        tp_kwargs = tp_requested if isinstance(tp_requested, dict) else {}
        metrics["throughput"] = measure_throughput(
            model, tokenizer, device, ThroughputConfig(**tp_kwargs))

    if eval_cfg.get("zeroshot", False):
        from rotquant.eval.zeroshot import zeroshot
        metrics["zeroshot"] = zeroshot(model, tokenizer,
                                       tasks=eval_cfg.get("tasks"),
                                       batch_size=eval_cfg.get("zeroshot_batch_size", 8),
                                       device=device,
                                       limit=eval_cfg.get("limit"))

    metrics["peak_vram_bytes_eval"] = peak_vram_bytes()


def _export_checkpoint(cfg: dict[str, Any], model, tokenizer, model_name: str,
                       selected_loader: str, export_dir: str,
                       export_overwrite: bool, export_processor: bool,
                       export_deployment_metadata: str | None,
                       metrics: dict[str, Any]) -> None:
    """Write the packed pickle-free checkpoint for the (patched) model."""
    deployment_metadata = None
    if export_deployment_metadata:
        with open(export_deployment_metadata, encoding="utf-8") as handle:
            deployment_metadata = json.load(handle)
        if not isinstance(deployment_metadata, dict):
            raise TypeError(
                "--export-deployment-metadata must contain a JSON object")
    processor = None
    if export_processor:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_name, revision=cfg.get("model_revision"))
        except Exception as exc:
            logger.warning(
                "could not save AutoProcessor metadata for %s: %s",
                model_name, exc)
    from rotquant.checkpoint import save_packed_checkpoint
    with Timer() as t:
        export_metrics = save_packed_checkpoint(
            model,
            export_dir,
            base_model=model_name,
            base_model_revision=cfg.get("model_revision"),
            model_loader=selected_loader,
            tokenizer=tokenizer,
            processor=processor,
            deployment_metadata=deployment_metadata,
            overwrite=export_overwrite,
        )
    export_metrics["seconds"] = t.elapsed
    metrics["packed_checkpoint"] = export_metrics
    logger.info(
        "exported packed checkpoint to %s (%.3f GB)",
        export_dir,
        export_metrics["artifact_bytes"] / 1e9,
    )


def run(config_path: str, output_dir: str = "results",
        overrides: dict[str, Any] | None = None,
        sets: list | None = None,
        export_dir: str | None = None,
        export_overwrite: bool = False,
        export_processor: bool = False,
        export_deployment_metadata: str | None = None) -> dict[str, Any]:
    """Execute one experiment config end to end and write its result JSON.

    Stages, in order: resolve config and run id, load the model, collect
    calibration artifacts (:func:`_prepare_calibration`), capture source-model
    references (:func:`_capture_references`), quantize
    (:func:`_apply_quantization`), evaluate (:func:`_run_evaluations`), and
    optionally export the packed checkpoint (:func:`_export_checkpoint`).
    """
    cfg = load_config(config_path)
    overrides = overrides or {}
    sets = list(sets or [])
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    apply_set_overrides(cfg, sets)
    device, dtype = resolve_device_dtype(cfg)
    effective_sets = list(sets)
    if apply_device_defaults(cfg, device):
        effective_sets.append(("patch.fallback", True))
    run_id = derive_run_id(cfg, config_path,
                           slug=override_slug(overrides.get("model"), effective_sets,
                                              overrides.get("device")),
                           seed_overridden=overrides.get("seed") is not None)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    model_name = cfg["model"]
    logger.info("run %s: model=%s device=%s dtype=%s seed=%d",
                run_id, model_name, device, dtype, seed)

    with Timer() as model_load_timer:
        model, tokenizer, selected_loader = load_hf_model(
            model_name, dtype, device, cfg.get("model_loader", "auto"),
            cfg.get("model_revision"))
    model.eval()

    # Let a per-block ``seed:`` override the top-level one, but don't explode if
    # the same key appears in both the block and the explicit kwarg.
    # ``or {}`` handles an explicit ``quant: null`` in the YAML (which makes
    # cfg.get("quant", {}) return None and dict(None) raise TypeError).
    quant_kwargs = dict(cfg.get("quant") or {})
    patch_kwargs = dict(cfg.get("patch") or {})
    qcfg = QuantConfig(seed=quant_kwargs.pop("seed", seed), **quant_kwargs)
    pcfg = PatchConfig(quant=qcfg, seed=patch_kwargs.pop("seed", seed), **patch_kwargs)

    metrics: dict[str, Any] = {
        "model_loader": selected_loader,
        "model_load_seconds": model_load_timer.elapsed,
        "data_manifest": {},
    }
    eval_cfg = cfg.get("eval") or {}

    art = _prepare_calibration(cfg, model, tokenizer, device, qcfg, pcfg, metrics)
    refs = _capture_references(cfg, eval_cfg, model, tokenizer, device,
                               model_name, metrics)
    _apply_quantization(cfg, model, pcfg, art, metrics)
    _run_evaluations(cfg, eval_cfg, model, tokenizer, device, seed, refs, metrics)
    if export_dir:
        _export_checkpoint(cfg, model, tokenizer, model_name, selected_loader,
                           export_dir, export_overwrite, export_processor,
                           export_deployment_metadata, metrics)

    payload = {
        "run_id": run_id,
        "config": cfg,
        "metrics": metrics,
        "environment": environment_record(),
    }
    out_path = os.path.join(output_dir, f"{run_id}.json")
    write_result(out_path, payload)
    logger.info("wrote %s", out_path)
    return payload


def main(argv: list[str] | None = None) -> None:
    enable_default_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to experiment YAML")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--model", default=None,
                    help="override the config's model (HF id or local path)")
    ap.add_argument("--device", default=None, help="override the config's device")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config's seed (for >=3-seed sweeps)")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override any dotted config key, YAML-typed value "
                         "(repeatable), e.g. --set patch.rotation=dense "
                         "--set quant.bits=4")
    ap.add_argument(
        "--export-dir",
        default=None,
        help="write a self-contained packed safetensors checkpoint after the run",
    )
    ap.add_argument(
        "--export-overwrite",
        action="store_true",
        help="allow replacing files in a non-empty export directory",
    )
    ap.add_argument(
        "--export-processor",
        action="store_true",
        help="also save AutoProcessor metadata for multimodal serving",
    )
    ap.add_argument(
        "--export-deployment-metadata",
        default=None,
        metavar="JSON",
        help=("embed a JSON object in the packed manifest, for example a "
              "validated K/V cache deployment recipe"),
    )
    args = ap.parse_args(argv)
    sets = []
    for item in args.sets:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        sets.append((key.strip(), yaml.safe_load(raw) if raw.strip() else None))
    run(args.config, args.output_dir,
        overrides={"model": args.model, "device": args.device, "seed": args.seed},
        sets=sets,
        export_dir=args.export_dir,
        export_overwrite=args.export_overwrite,
        export_processor=args.export_processor,
        export_deployment_metadata=args.export_deployment_metadata)


if __name__ == "__main__":
    main()
