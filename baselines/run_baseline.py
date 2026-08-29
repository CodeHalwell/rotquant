#!/usr/bin/env python
"""Thin wrappers to quantise the same model with external baselines and push the
result through the *identical* perplexity/zero-shot harness.

Baselines are non-negotiable: a finding only counts placed next to GPTQ/AWQ at
3-4 bit and QuIP#/AQLM/QTIP at 2 bit, on the same model and eval protocol.

Each backend is imported lazily and raises an informative error (with the install
/clone command) if it is not present, so the harness never silently skips one.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotquant.utils import environment_record, get_logger, write_result  # noqa: E402

logger = get_logger()

INSTALL_HINTS = {
    "gptq": "pip install gptqmodel",
    "awq": "pip install autoawq",
    "aqlm": "pip install aqlm[gpu]",
    "higgs": "pip install flute-kernel  # HIGGS runtime",
    "quip": "git clone https://github.com/Cornell-RelaxML/quip-sharp",
    "qtip": "git clone https://github.com/Cornell-RelaxML/qtip",
}

IMPLEMENTED_BACKENDS = ("gptq", "awq", "aqlm")


def baseline_run_id(backend: str, model: str, bits: int, group_size: int,
                    prequantized: bool, device: str) -> str:
    """Build an artifact id from every CLI option that can change the result."""
    model_slug = model.rstrip("/").split("/")[-1]
    source = "prequantized" if prequantized else "quantized"
    device_slug = device.replace(":", "-").replace("/", "-")
    return (f"baseline_{backend}_{model_slug}_{bits}bit_"
            f"g{group_size}_{source}_{device_slug}")


def _require(module: str, backend: str):
    try:
        return __import__(module)
    except Exception as exc:
        raise ImportError(
            f"baseline '{backend}' needs '{module}'. Install: {INSTALL_HINTS.get(backend, module)}"
        ) from exc


def _calib_texts(n: int = 256, min_chars: int = 2048) -> list:
    """Raw C4 texts for backends that quantise here (gptq/awq)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for row in ds:
        if len(row["text"]) < min_chars:
            continue
        texts.append(row["text"])
        if len(texts) >= n:
            break
    return texts


def load_baseline(backend: str, model_name: str, bits: int, device: str,
                  prequantized: bool = False, **kwargs):
    """Return (model, tokenizer) quantised by the requested external method.

    With ``prequantized`` the model id must point at an already-quantised
    checkpoint (e.g. a ``*-AWQ`` / ``*-GPTQ`` / ISTA-DASLab AQLM repo) and is
    loaded as-is; otherwise gptq/awq quantise ``model_name`` on the fly with a
    C4 calibration set. AQLM quantisation takes GPU-days, so that backend
    *requires* ``prequantized``.
    """
    backend = backend.lower()
    if backend == "gptq":
        _require("gptqmodel", backend)
        from gptqmodel import GPTQModel, QuantizeConfig
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if prequantized:
            model = GPTQModel.load(model_name, device=device)
        else:
            qcfg = QuantizeConfig(bits=bits, group_size=kwargs.get("group_size", 128))
            model = GPTQModel.load(model_name, qcfg)
            logger.info("gptq: quantising %s at %d bits on C4 calibration data",
                        model_name, bits)
            model.quantize(_calib_texts())
        return getattr(model, "model", model), tok
    if backend == "awq":
        _require("awq", backend)
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if prequantized:
            model = AutoAWQForCausalLM.from_quantized(model_name)
        else:
            model = AutoAWQForCausalLM.from_pretrained(model_name)
            logger.info("awq: quantising %s at %d bits", model_name, bits)
            model.quantize(tok, quant_config={
                "w_bit": bits, "q_group_size": kwargs.get("group_size", 128),
                "zero_point": True, "version": "GEMM"})
        return getattr(model, "model", model), tok
    if backend == "aqlm":
        _require("aqlm", backend)
        if not prequantized:
            raise ValueError(
                "AQLM quantisation takes GPU-days; pass --prequantized with an "
                "AQLM checkpoint id (e.g. ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf).")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        return model, tok
    if backend in ("quip", "qtip", "higgs"):
        raise NotImplementedError(
            f"baseline '{backend}' is not integrated yet; installing its package "
            "alone is insufficient because this harness has no checkpoint loader for it")
    raise ValueError(f"unknown baseline backend: {backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=IMPLEMENTED_BACKENDS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--prequantized", action="store_true",
                    help="--model is an already-quantised checkpoint; load as-is "
                         "instead of quantising here (required for aqlm)")
    args = ap.parse_args()

    model, tok = load_baseline(args.backend, args.model, args.bits, args.device,
                               prequantized=args.prequantized,
                               group_size=args.group_size)
    model.eval()

    from eval.perplexity import perplexity, PPLConfig
    metrics: Dict[str, Any] = {}
    for ds in ("wikitext2", "c4"):
        metrics[f"ppl_{ds}"] = perplexity(model, tok, ds, PPLConfig(), args.device)
    if args.zeroshot:
        from eval.zeroshot import zeroshot
        metrics["zeroshot"] = zeroshot(model, tok, device=args.device)

    run_id = baseline_run_id(args.backend, args.model, args.bits, args.group_size,
                             args.prequantized, args.device)
    write_result(os.path.join(args.output_dir, f"{run_id}.json"), {
        "run_id": run_id,
        "config": {"experiment": "baseline", "backend": args.backend,
                   "model": args.model, "bits": args.bits,
                   "group_size": args.group_size,
                   "prequantized": args.prequantized,
                   "device": args.device, "label": run_id},
        "metrics": metrics,
        "environment": environment_record(),
    })


if __name__ == "__main__":
    main()
