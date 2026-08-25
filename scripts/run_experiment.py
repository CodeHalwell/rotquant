#!/usr/bin/env python
"""config -> quantise -> eval -> write results/<run_id>.json

Reads a single experiment YAML (see ``configs/``), loads the HF model, optionally
collects real-activation Hessians, patches it with ``QuantLinear`` and runs the
fixed eval protocol. Every run writes a JSON with config, git SHA, library
versions, GPU, all metrics, and wall-clock.

Config resolution: if a ``_base.yaml`` sits next to the experiment YAML it is
deep-merged underneath it (experiment keys win, nested dicts merge, lists are
replaced wholesale). ``--model``, ``--device`` and ``--seed`` override the merged
config from the CLI so seed sweeps and model sweeps never require editing YAML:

    python scripts/run_experiment.py configs/e1_rotation.yaml --seed 1
    python scripts/run_experiment.py configs/e2_codebook.yaml --model facebook/opt-125m
"""
from __future__ import annotations

import argparse
import os
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


def derive_run_id(cfg: Dict[str, Any], config_path: str) -> str:
    """An explicit ``run_id`` is used verbatim; otherwise ``label`` (or the config
    file stem) gets a ``_s<seed>`` suffix so seed sweeps never overwrite each other."""
    explicit = cfg.get("run_id")
    if explicit:
        return str(explicit)
    stem = os.path.splitext(os.path.basename(config_path))[0]
    name = cfg.get("label") or stem
    return f"{name}_s{int(cfg.get('seed', 0))}"


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
        overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    run_id = derive_run_id(cfg, config_path)
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
            calib = collect_hessians(model, loader, device,
                                     include=pcfg.include,
                                     exclude=pcfg.exclude,
                                     damp_frac=qcfg.percdamp)
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
    args = ap.parse_args(argv)
    run(args.config, args.output_dir,
        overrides={"model": args.model, "device": args.device, "seed": args.seed})


if __name__ == "__main__":
    main()
