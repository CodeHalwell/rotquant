#!/usr/bin/env python
"""Load a RotQuant packed checkpoint and run text generation with Transformers."""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotquant.checkpoint import load_packed_model


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="packed checkpoint directory")
    parser.add_argument("--prompt", default="Explain rotation-aware quantization.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default=None
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="cache dequantized weights (faster, but forfeits compressed memory)",
    )
    parser.add_argument(
        "--audit-no-fallback-cache",
        action="store_true",
        help="fail unless every restored QuantLinear starts without an fp cache",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = args.device or _default_device()
    dtype = args.dtype
    if dtype is None and device == "cpu":
        dtype = "float32"
    model = load_packed_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        fallback=args.fallback,
        trust_remote_code=args.trust_remote_code,
    )
    if args.audit_no_fallback_cache:
        from rotquant.linear import QuantLinear

        quantized = [
            module for module in model.modules() if isinstance(module, QuantLinear)
        ]
        if not quantized:
            raise RuntimeError("packed checkpoint restored no QuantLinear modules")
        cached = [module for module in quantized if module._fp_cache is not None]
        if cached:
            raise RuntimeError(
                f"{len(cached)} of {len(quantized)} QuantLinear modules restored "
                "with fallback caches"
            )
        print(f"Fallback-cache audit: PASS ({len(quantized)} packed modules)")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=args.trust_remote_code
    )
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
