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


def _require(module: str, backend: str):
    try:
        return __import__(module)
    except Exception as exc:
        raise ImportError(
            f"baseline '{backend}' needs '{module}'. Install: {INSTALL_HINTS.get(backend, module)}"
        ) from exc


def load_baseline(backend: str, model_name: str, bits: int, device: str,
                  **kwargs):
    """Return (model, tokenizer) quantised by the requested external method."""
    backend = backend.lower()
    if backend == "gptq":
        _require("gptqmodel", backend)
        from gptqmodel import GPTQModel, QuantizeConfig
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        qcfg = QuantizeConfig(bits=bits, group_size=kwargs.get("group_size", 128))
        model = GPTQModel.load(model_name, qcfg)
        return model, tok
    if backend == "awq":
        _require("awq", backend)
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoAWQForCausalLM.from_quantized(model_name)
        return model, tok
    if backend == "aqlm":
        _require("aqlm", backend)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        return model, tok
    if backend in ("quip", "qtip", "higgs"):
        _require({"quip": "quip", "qtip": "qtip", "higgs": "flute"}[backend], backend)
        raise NotImplementedError(
            f"{backend} requires its repo's loader; wire it here once cloned.")
    raise ValueError(f"unknown baseline backend: {backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=list(INSTALL_HINTS.keys()))
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--zeroshot", action="store_true")
    args = ap.parse_args()

    model, tok = load_baseline(args.backend, args.model, args.bits, args.device,
                               group_size=args.group_size)

    from eval.perplexity import perplexity, PPLConfig
    metrics: Dict[str, Any] = {}
    for ds in ("wikitext2", "c4"):
        metrics[f"ppl_{ds}"] = perplexity(model, tok, ds, PPLConfig(), args.device)
    if args.zeroshot:
        from eval.zeroshot import zeroshot
        metrics["zeroshot"] = zeroshot(model, tok, device=args.device)

    run_id = f"baseline_{args.backend}_{args.bits}bit"
    write_result(os.path.join(args.output_dir, f"{run_id}.json"), {
        "run_id": run_id,
        "config": {"experiment": "baseline", "backend": args.backend,
                   "model": args.model, "bits": args.bits, "label": run_id},
        "metrics": metrics,
        "environment": environment_record(),
    })


if __name__ == "__main__":
    main()
