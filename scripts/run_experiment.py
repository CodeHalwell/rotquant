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
from typing import Any, Dict, List, Optional

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotquant.utils import (  # noqa: E402
    Timer, environment_record, get_logger, set_seed, write_result,
    peak_vram_bytes, reset_peak_vram,
)
from rotquant.quantize import QuantConfig  # noqa: E402
from rotquant.patch import PatchConfig, patch_model  # noqa: E402

logger = get_logger()

BASE_CONFIG_NAME = "_base.yaml"
MAX_RUN_ID_LENGTH = 220  # leaves room for the .json suffix under NAME_MAX=255
MODEL_LOADERS = ("auto", "causal_lm", "multimodal_lm")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``. Lists are replaced."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load an experiment YAML, deep-merging ``_base.yaml`` from the same dir."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_path = os.path.join(os.path.dirname(os.path.abspath(path)), BASE_CONFIG_NAME)
    if os.path.basename(path) != BASE_CONFIG_NAME and os.path.exists(base_path):
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)
    return cfg


def apply_set_overrides(cfg: Dict[str, Any], sets) -> None:
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


def override_slug(model: Optional[str], sets, device: Optional[str] = None) -> str:
    """Filename fragment describing the CLI overrides that change results."""
    parts = []
    if model:
        parts.append(_slug(model.rstrip("/").split("/")[-1]))
    if device:
        parts.append(_slug(f"device={device}"))
    for key, value in sets:
        parts.append(_slug(f"{key}={value}"))
    return "_".join(parts)


def derive_run_id(cfg: Dict[str, Any], config_path: str,
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


def resolve_device_dtype(cfg: Dict[str, Any]) -> tuple:
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


def apply_device_defaults(cfg: Dict[str, Any], device) -> bool:
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
    from datasets import load_dataset
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        revision=revision,
    )
    batches, count, eligible = [], 0, 0
    for row in ds:
        ids = tokenizer(row["text"], return_tensors="pt").input_ids
        if ids.shape[1] < seq_len:
            continue
        if eligible < skip:
            eligible += 1
            continue
        batches.append({"input_ids": ids[:, :seq_len].to(device)})
        count += 1
        if count >= n_seq:
            break
    return batches


def footprint_metrics(model: torch.nn.Module, cfg_model: Dict[str, Any]) -> Dict[str, Any]:
    """True bits/weight + packed-vs-fp16 storage across all QuantLinear layers.

    Enforces the equal-bits discipline: if the config declares ``claimed_bpw``,
    every layer's BitBudget must match it (``BitBudget.assert_matches``).
    """
    from rotquant.linear import QuantLinear
    metrics: Dict[str, Any] = {}
    bpws, packed_bytes, fp16_bytes = [], 0, 0
    rotation_parameter_bytes, adapter_parameter_bytes = 0, 0
    total_bits, total_weights = 0.0, 0
    claimed = cfg_model.get("claimed_bpw")
    tol = float(cfg_model.get("claimed_bpw_tol", 1e-6))
    for name, mod in model.named_modules():
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
    return metrics


def run(config_path: str, output_dir: str = "results",
        overrides: Optional[Dict[str, Any]] = None,
        sets: Optional[List] = None,
        export_dir: Optional[str] = None,
        export_overwrite: bool = False,
        export_processor: bool = False,
        export_deployment_metadata: Optional[str] = None) -> Dict[str, Any]:
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

    metrics: Dict[str, Any] = {"model_loader": selected_loader}
    eval_cfg = cfg.get("eval") or {}
    calibration_revision = cfg.get("calib_dataset_revision")

    hessians = None
    activations = None
    block_calls = None
    distill_calls = None
    dynamic_calls = None
    trajectory_references = None
    trajectory_config = None
    needs_hessians = pcfg.enabled and (
        qcfg.error_comp == "gptq" or cfg.get("calibrate", False))
    rotation_train_cfg = pcfg.train_rotation or {}
    dynamic_cfg = pcfg.dynamic or {}
    needs_dynamic = pcfg.enabled and bool(pcfg.dynamic)
    needs_activations = (pcfg.enabled and (
        rotation_train_cfg.get("objective") == "activation" or needs_dynamic))
    needs_block_calls = (pcfg.enabled
                         and rotation_train_cfg.get("objective") == "block")
    calib_loader = None
    if needs_hessians:
        calib_loader = build_calib_loader(
            tokenizer, cfg.get("n_calib", 128),
            cfg.get("calib_seq_len", 2048), device,
            revision=calibration_revision)

    if needs_hessians:
        from rotquant.calibrate import collect_hessians
        with Timer() as t:
            # damp_frac=0: the GPTQ solver applies (auto-increasing) percdamp
            # itself; damping here as well would double it.
            calib = collect_hessians(model, calib_loader, device,
                                     include=pcfg.include,
                                     exclude=pcfg.exclude,
                                     damp_frac=0.0)
        hessians = calib.hessians
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
            if needs_dynamic else 0
        max_tokens = max(rotation_tokens, dynamic_tokens)
        if calib_loader is None:
            calib_seq_len = int(cfg.get("calib_seq_len", 2048))
            minimum_batches = max(1, (max_tokens + calib_seq_len - 1) // calib_seq_len)
            n_batches = int(cfg.get("rotation_n_calib", minimum_batches))
            calib_loader = build_calib_loader(
                tokenizer, n_batches, calib_seq_len, device,
                revision=calibration_revision)
        with Timer() as t:
            activation_calib = collect_activations(
                model, calib_loader, device,
                include=pcfg.include, exclude=pcfg.exclude,
                max_tokens=max_tokens)
        activations = activation_calib.activations
        metrics["activation_calib_seconds"] = t.elapsed

    dynamic_global_batches = int(dynamic_cfg.get("global_kl_batches", 0)) \
        if needs_dynamic else 0
    if dynamic_global_batches:
        from rotquant.block_train import collect_teacher_calls
        dynamic_loader = build_calib_loader(
            tokenizer, dynamic_global_batches,
            int(cfg.get("calib_seq_len", 256)), device,
            revision=calibration_revision)
        with Timer() as t:
            dynamic_calls = collect_teacher_calls(
                model, dynamic_loader, device,
                max_batches=dynamic_global_batches)
        metrics["dynamic_teacher_calib_seconds"] = t.elapsed

    if needs_block_calls:
        from rotquant.block_train import (
            collect_block_calls, collect_teacher_calls,
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
            calib_loader = build_calib_loader(
                tokenizer, total_block_batches,
                int(cfg.get("calib_seq_len", 256)), device,
                revision=calibration_revision)
        with Timer() as t:
            block_calls = collect_block_calls(
                model, calib_loader[:n_block_batches], device,
                blocks=find_transformer_blocks(model),
                max_batches=n_block_batches)
            if n_distill_batches:
                distill_calls = collect_teacher_calls(
                    model, calib_loader[n_block_batches:total_block_batches],
                    device, max_batches=n_distill_batches)
        metrics["block_calib_seconds"] = t.elapsed

    # Layer-drift diagnostic (E7): capture fp-model outputs on a fixed batch
    # BEFORE patching, so the same batch can be replayed on the patched model.
    fp_capture = None
    drift_batch = None
    if eval_cfg.get("layer_mse", False):
        from eval.layer_mse import capture_outputs
        drift_batch = build_calib_loader(
            tokenizer, 1, eval_cfg.get("layer_mse_seq_len", 512), device,
            revision=calibration_revision)[0]
        fp_capture = capture_outputs(model, drift_batch, device)

    trajectory_requested = eval_cfg.get("trajectory", False)
    if trajectory_requested:
        from eval.trajectory import TrajectoryConfig, capture_trajectories
        trajectory_kwargs = (trajectory_requested
                             if isinstance(trajectory_requested, dict) else {})
        trajectory_config = TrajectoryConfig(**trajectory_kwargs)
        trajectory_batches = build_calib_loader(
            tokenizer, trajectory_config.batches,
            trajectory_config.prompt_len, device,
            skip=trajectory_config.skip,
            revision=calibration_revision)
        with Timer() as t:
            trajectory_references = capture_trajectories(
                model, tokenizer, trajectory_batches, device,
                trajectory_config)
        metrics["source_trajectory_seconds"] = t.elapsed

    patch_stats: Dict[str, Any] = {}
    if needs_dynamic:
        from rotquant.dynamic import select_dynamic_quantization
        with Timer() as t:
            pcfg.layer_quant, dynamic_stats = select_dynamic_quantization(
                model, pcfg, activations=activations,
                teacher_calls=dynamic_calls)
        dynamic_stats["seconds"] = t.elapsed
        patch_stats["dynamic_quantization"] = dynamic_stats

    reset_peak_vram()
    with Timer() as t:
        if needs_block_calls:
            from rotquant.block_train import train_and_patch_blocks
            train_and_patch_blocks(model, pcfg, block_calls,
                                   distill_calls=distill_calls,
                                   stats_out=patch_stats)
        else:
            patch_model(model, pcfg, hessians=hessians, activations=activations,
                        stats_out=patch_stats)
    metrics["patch_seconds"] = t.elapsed
    metrics["peak_vram_bytes_patch"] = peak_vram_bytes()
    metrics.update(patch_stats)
    metrics.update(footprint_metrics(model, cfg))

    # Evaluation -----------------------------------------------------------
    if fp_capture is not None:
        from eval.layer_mse import capture_outputs, drift_between
        q_capture = capture_outputs(model, drift_batch, device)
        drift = drift_between(fp_capture, q_capture)
        metrics["layer_mse"] = {"mse": drift.mse, "cosine": drift.cosine,
                                "order": drift.order}

    if trajectory_references is not None:
        from eval.trajectory import evaluate_trajectories
        with Timer() as t:
            metrics["trajectory"] = evaluate_trajectories(
                model, tokenizer, trajectory_references, device,
                trajectory_config)
        metrics["trajectory"]["seconds"] = t.elapsed

    kv_cache_requested = eval_cfg.get("kv_cache", False)
    if kv_cache_requested:
        from eval.kv_cache import KVCacheEvalConfig, evaluate_kv_cache
        kv_cache_kwargs = (kv_cache_requested
                           if isinstance(kv_cache_requested, dict) else {})
        kv_cache_kwargs = dict(kv_cache_kwargs)
        kv_cache_kwargs.setdefault("seed", seed)
        kv_cache_config = KVCacheEvalConfig(**kv_cache_kwargs)
        kv_selection_batches = 0
        if kv_cache_config.dynamic:
            from eval.kv_cache import KVDynamicConfig
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
        from eval.perplexity import perplexity, PPLConfig
        ppl_cfg = PPLConfig(**(eval_cfg.get("ppl") or {}))
        for ds in eval_cfg.get("ppl_datasets", ["wikitext2", "c4"]):
            with Timer() as t:
                metrics[f"ppl_{ds}"] = perplexity(model, tokenizer, ds, ppl_cfg, device)
            metrics[f"ppl_{ds}_seconds"] = t.elapsed

    tp_requested = eval_cfg.get("throughput", False)
    if tp_requested:
        from eval.throughput import ThroughputConfig, measure_throughput
        tp_kwargs = tp_requested if isinstance(tp_requested, dict) else {}
        metrics["throughput"] = measure_throughput(
            model, tokenizer, device, ThroughputConfig(**tp_kwargs))

    if eval_cfg.get("zeroshot", False):
        from eval.zeroshot import zeroshot
        metrics["zeroshot"] = zeroshot(model, tokenizer,
                                       tasks=eval_cfg.get("tasks"),
                                       batch_size=eval_cfg.get("zeroshot_batch_size", 8),
                                       device=device,
                                       limit=eval_cfg.get("limit"))

    if export_dir:
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


def main(argv: Optional[List[str]] = None) -> None:
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
