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
    python scripts/run_experiment.py configs/e8_footprint.yaml --set patch.fallback=true

Run ids of CLI-modified runs get the overridden values appended, so sweep
results never overwrite each other.
"""
from __future__ import annotations

import argparse
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


def override_slug(model: Optional[str], sets) -> str:
    """Filename fragment describing the CLI overrides that change results."""
    parts = []
    if model:
        parts.append(_slug(model.rstrip("/").split("/")[-1]))
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
        return str(explicit)
    stem = os.path.splitext(os.path.basename(config_path))[0]
    parts = [str(explicit) if explicit else (cfg.get("label") or stem)]
    if slug:
        parts.append(slug)
    if not explicit or seed_overridden:
        parts.append(f"s{int(cfg.get('seed', 0))}")
    return "_".join(parts)


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


def build_calib_loader(tokenizer, n_seq: int, seq_len: int, device):
    """Tokenised C4/WikiText-train calibration sequences (128-512 typical)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    batches, count = [], 0
    for row in ds:
        ids = tokenizer(row["text"], return_tensors="pt").input_ids
        if ids.shape[1] < seq_len:
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
    claimed = cfg_model.get("claimed_bpw")
    tol = float(cfg_model.get("claimed_bpw_tol", 1e-6))
    for name, mod in model.named_modules():
        if not isinstance(mod, QuantLinear):
            continue
        budget = mod.qweight.bit_budget()
        if claimed is not None:
            budget.assert_matches(float(claimed), tol=tol)
        bpws.append(budget.bits_per_weight)
        packed_bytes += mod.packed_state_bytes()
        fp16_bytes += mod.out_features * mod.in_features * 2
    if bpws:
        metrics["n_quant_layers"] = len(bpws)
        metrics["bits_per_weight_mean"] = sum(bpws) / len(bpws)
        metrics["bits_per_weight_min"] = min(bpws)
        metrics["bits_per_weight_max"] = max(bpws)
        metrics["packed_weight_bytes"] = packed_bytes
        metrics["fp16_weight_bytes"] = fp16_bytes
        metrics["compression_ratio"] = fp16_bytes / max(packed_bytes, 1)
    return metrics


def run(config_path: str, output_dir: str = "results",
        overrides: Optional[Dict[str, Any]] = None,
        sets: Optional[List] = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    overrides = overrides or {}
    sets = list(sets or [])
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    apply_set_overrides(cfg, sets)
    run_id = derive_run_id(cfg, config_path,
                           slug=override_slug(overrides.get("model"), sets),
                           seed_overridden=overrides.get("seed") is not None)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]
    device, dtype = resolve_device_dtype(cfg)
    logger.info("run %s: model=%s device=%s dtype=%s seed=%d",
                run_id, model_name, device, dtype, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    # Let a per-block ``seed:`` override the top-level one, but don't explode if
    # the same key appears in both the block and the explicit kwarg.
    # ``or {}`` handles an explicit ``quant: null`` in the YAML (which makes
    # cfg.get("quant", {}) return None and dict(None) raise TypeError).
    quant_kwargs = dict(cfg.get("quant") or {})
    patch_kwargs = dict(cfg.get("patch") or {})
    qcfg = QuantConfig(seed=quant_kwargs.pop("seed", seed), **quant_kwargs)
    pcfg = PatchConfig(quant=qcfg, seed=patch_kwargs.pop("seed", seed), **patch_kwargs)

    metrics: Dict[str, Any] = {}
    eval_cfg = cfg.get("eval") or {}

    hessians = None
    if qcfg.error_comp == "gptq" or cfg.get("calibrate", False):
        from rotquant.calibrate import collect_hessians
        loader = build_calib_loader(tokenizer, cfg.get("n_calib", 128),
                                    cfg.get("calib_seq_len", 2048), device)
        with Timer() as t:
            # damp_frac=0: the GPTQ solver applies (auto-increasing) percdamp
            # itself; damping here as well would double it.
            calib = collect_hessians(model, loader, device,
                                     include=pcfg.include,
                                     exclude=pcfg.exclude,
                                     damp_frac=0.0)
        hessians = calib.hessians
        metrics["calib_seconds"] = t.elapsed

    # Layer-drift diagnostic (E7): capture fp-model outputs on a fixed batch
    # BEFORE patching, so the same batch can be replayed on the patched model.
    fp_capture = None
    drift_batch = None
    if eval_cfg.get("layer_mse", False):
        from eval.layer_mse import capture_outputs
        drift_batch = build_calib_loader(
            tokenizer, 1, eval_cfg.get("layer_mse_seq_len", 512), device)[0]
        fp_capture = capture_outputs(model, drift_batch, device)

    reset_peak_vram()
    with Timer() as t:
        patch_model(model, pcfg, hessians=hessians)
    metrics["patch_seconds"] = t.elapsed
    metrics["peak_vram_bytes_patch"] = peak_vram_bytes()
    metrics.update(footprint_metrics(model, cfg))

    # Evaluation -----------------------------------------------------------
    if fp_capture is not None:
        from eval.layer_mse import capture_outputs, drift_between
        q_capture = capture_outputs(model, drift_batch, device)
        drift = drift_between(fp_capture, q_capture)
        metrics["layer_mse"] = {"mse": drift.mse, "cosine": drift.cosine,
                                "order": drift.order}

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
    args = ap.parse_args(argv)
    sets = []
    for item in args.sets:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        sets.append((key.strip(), yaml.safe_load(raw) if raw.strip() else None))
    run(args.config, args.output_dir,
        overrides={"model": args.model, "device": args.device, "seed": args.seed},
        sets=sets)


if __name__ == "__main__":
    main()
