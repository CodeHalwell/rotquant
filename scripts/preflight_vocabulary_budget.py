#!/usr/bin/env python3
"""Offline CPU/CUDA conformance smoke before the vocabulary-budget screen."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from rotquant.linear import QuantLinear
from rotquant.quantize import QuantConfig
from rotquant.rotate import RandomizedHadamard, fwht
from rotquant.utils import write_result
from rotquant.vocabulary import (
    VocabularyConfig,
    install_packed_vocabulary,
    quantize_vocabulary,
    vocabulary_prototype,
)


def qwen_hybrid_smoke(device="cpu", dtype=torch.float32):
    """Exercise real Qwen3.5 recurrent and attention blocks without downloads."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=129, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True, eos_token_id=128, pad_token_id=0,
    )).to(device=device, dtype=dtype).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.inference_mode():
        source = model(ids).logits.clone()
    owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                VocabularyConfig(bits=6, chunk_rows=64))
    with vocabulary_prototype(model, owner), torch.inference_mode():
        assert torch.isfinite(model(ids).logits).all()
        assert model.generate(ids, max_new_tokens=2, do_sample=False).shape[-1] >= 5
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, source, rtol=0, atol=0)
    install_packed_vocabulary(model, owner)
    with torch.inference_mode():
        assert torch.isfinite(model(ids).logits).all()
        assert model.generate(ids, max_new_tokens=2, do_sample=False).shape[-1] >= 5
    return {"architecture": "tiny_qwen3_5_hybrid_text", "vocabulary_bits": 6,
            "source_restored": True, "prototype_and_packed_generation": True}


def preflight(device="cpu"):
    from transformers import LlamaConfig, LlamaForCausalLM

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested without a CUDA runtime")
    torch.manual_seed(20260906)
    dtype = torch.float16 if device == "cuda" else torch.float32
    large = torch.zeros(1, 128, dtype=torch.float16, device=device)
    large[0, :2] = 40000
    torch.testing.assert_close(fwht(large), fwht(large.float()).half(), rtol=2e-3, atol=2)
    assert torch.isfinite(fwht(large)).all()
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=129, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=128,
    )).to(device=device, dtype=dtype).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.inference_mode():
        reference = model(ids).logits.clone()
    rows = []
    for bits in (8, 6):
        started = time.monotonic()
        owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                    VocabularyConfig(bits=bits, chunk_rows=64))
        with vocabulary_prototype(model, owner), torch.inference_mode():
            assert torch.isfinite(model(ids).logits).all()
            assert model.generate(ids, max_new_tokens=2, do_sample=False).shape[-1] >= 5
        with torch.inference_mode():
            torch.testing.assert_close(model(ids).logits, reference, rtol=0, atol=0)
        rows.append({"vocabulary_bits": bits, "source_restored": True,
                     "packed_payload_bytes": owner.packed_bytes(),
                     "seconds": time.monotonic() - started})
        print(json.dumps(rows[-1]), flush=True)
    projection = nn.Linear(128, 32, bias=False).to(device=device, dtype=dtype)
    rotation = RandomizedHadamard(128, block=128, seed=0, device=device)
    for bits in (4, 5, 6, 8):
        started = time.monotonic()
        quant = QuantConfig(bits=bits, scale="mse_search", error_comp="gptq",
                            scale_bits=8 if bits <= 5 else 16)
        layer = QuantLinear.from_linear(projection, quant, weight_rotation=rotation,
                                        H=torch.eye(128, device=device), fallback=True)
        before = layer.qweight.dequantize().clone()
        layer.bfloat16()
        torch.testing.assert_close(before, layer.qweight.dequantize(), rtol=0, atol=0)
        assert torch.isfinite(layer(torch.ones(2, 128, device=device, dtype=torch.bfloat16))).all()
        rows.append({"backbone_bits": bits, "metadata_preserved": True,
                     "seconds": time.monotonic() - started})
        print(json.dumps(rows[-1]), flush=True)
    rows.append(qwen_hybrid_smoke(device, dtype))
    print(json.dumps(rows[-1]), flush=True)
    return {"protocol": "vocabulary-budget-preflight-v1", "device": device,
            "passed": True, "checks": rows,
            "boundary": "Synthetic correctness checks; not full-Qwen quality or a throughput benchmark."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = preflight(args.device)
    if args.output:
        write_result(str(args.output), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
