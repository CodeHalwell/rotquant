#!/usr/bin/env python
"""config -> quantise -> eval -> write results/<run_id>.json

Reads a single experiment YAML (see ``configs/``), loads the HF model, optionally
collects real-activation Hessians, patches it with ``QuantLinear`` and runs the
fixed eval protocol. Every run writes a JSON with config, git SHA, library
versions, GPU, all metrics, and wall-clock.
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


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


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


def run(config_path: str, output_dir: str = "results") -> Dict[str, Any]:
    cfg = load_config(config_path)
    run_id = cfg.get("run_id", os.path.splitext(os.path.basename(config_path))[0])
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.get("dtype", "float16"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    qcfg = QuantConfig(**cfg.get("quant", {}), seed=seed)
    pcfg = PatchConfig(quant=qcfg, seed=seed, **cfg.get("patch", {}))

    metrics: Dict[str, Any] = {}
    hessians = None
    if qcfg.error_comp == "gptq" or cfg.get("calibrate", False):
        from rotquant.calibrate import collect_hessians
        loader = build_calib_loader(tokenizer, cfg.get("n_calib", 128),
                                    cfg.get("calib_seq_len", 2048), device)
        with Timer() as t:
            calib = collect_hessians(model, loader, device,
                                     include=pcfg.include,
                                     damp_frac=qcfg.percdamp)
        hessians = calib.hessians
        metrics["calib_seconds"] = t.elapsed

    reset_peak_vram()
    with Timer() as t:
        patch_model(model, pcfg, hessians=hessians)
    metrics["patch_seconds"] = t.elapsed
    metrics["peak_vram_bytes_patch"] = peak_vram_bytes()

    # Evaluation -----------------------------------------------------------
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("perplexity", True):
        from eval.perplexity import perplexity, PPLConfig
        ppl_cfg = PPLConfig(**eval_cfg.get("ppl", {}))
        for ds in eval_cfg.get("ppl_datasets", ["wikitext2", "c4"]):
            metrics[f"ppl_{ds}"] = perplexity(model, tokenizer, ds, ppl_cfg, device)

    if eval_cfg.get("zeroshot", False):
        from eval.zeroshot import zeroshot
        metrics["zeroshot"] = zeroshot(model, tokenizer,
                                       tasks=eval_cfg.get("tasks"),
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
    args = ap.parse_args(argv)
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
