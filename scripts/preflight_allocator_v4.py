#!/usr/bin/env python3
"""Check all six allocator formats against deployment before an expensive run.

No model or dataset download is needed. This is a runtime/correctness smoke
test, not evidence of Qwen quality or packed-kernel throughput.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch import nn

from rotquant import RotQuantConfig, optimize_model
from rotquant.dynamic import (
    DynamicQuantConfig,
    _candidate_context,
    _candidate_linear,
    _candidate_specs,
)
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.utils import write_result


def preflight(device: str = "cpu") -> dict:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    dtype = torch.float16 if device == "cuda" else torch.float32
    with (ROOT / "configs/qwen35_4b_allocator_v4_cuda.yaml").open() as handle:
        config = yaml.safe_load(handle)
    base_quant = QuantConfig(**config["quant"])
    dynamic = DynamicQuantConfig(**config["patch"]["dynamic"])
    source = nn.Linear(256, 16, bias=False).to(device=device, dtype=dtype).eval()
    activations = torch.randn(128, 256, device=device, dtype=dtype)
    hessian = activations.float().T @ activations.float() / len(activations)
    rows = []
    for name, quant in _candidate_specs("probe", dynamic, base_quant):
        started = time.perf_counter()
        patch = PatchConfig(quant=quant, block=128, seed=17, fallback=True)
        context = _candidate_context(source, patch, patch.seed, hessian)
        scored = _candidate_linear(source, quant, context, None)
        deployed = nn.Sequential(copy.deepcopy(source))
        patch_model(deployed, patch, hessians={"0": hessian})
        assert torch.equal(scored.qweight.packed.data, deployed[0].qweight.packed.data), name
        torch.testing.assert_close(
            scored.qweight.dequantize(), deployed[0].qweight.dequantize(), rtol=0, atol=0,
        )
        with torch.inference_mode():
            actual = deployed(activations)
            torch.testing.assert_close(scored(activations), actual, rtol=0, atol=0)
            assert torch.isfinite(actual).all(), name
        row = {"format": name, "packed_and_forward_match": True,
               "seconds": time.perf_counter() - started}
        rows.append(row)
        print(json.dumps(row), flush=True)

    tiny = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=63,
    )).to(device=device, dtype=dtype).eval()
    optimize_model(tiny, RotQuantConfig(group_size=8, rotation_block=8))
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3, 4]], device=device)
        assert torch.isfinite(tiny(ids).logits).all()
        assert tiny.generate(ids, max_new_tokens=2, do_sample=False).shape[1] >= 5
    print("Shared-rotation inference and generation passed.", flush=True)
    return {"protocol": "allocator-v4-runtime-preflight-v1", "device": device,
            "torch": torch.__version__, "passed": True, "formats": rows,
            "shared_inference_and_generation": True,
            "boundary": "Synthetic smoke only; does not measure Qwen quality or speed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = preflight(args.device)
    if args.output:
        write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
