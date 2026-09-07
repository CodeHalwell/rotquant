#!/usr/bin/env python3
"""Offline tiny multimodal-Qwen packed export and fresh-process reload smoke."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rotquant.checkpoint import load_packed_model, save_packed_checkpoint
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.utils import write_result
from rotquant.validation import (
    audit_artifact,
    capture_probes,
    compare_probes,
    packed_residency,
    probe_inputs,
    save_probes,
)
from rotquant.vocabulary import VocabularyConfig, quantize_vocabulary, vocabulary_prototype
from scripts.run_qwen35_packed_validation import packed_context


def tiny_model(device="cpu", dtype=torch.float32):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={"vocab_size": 129, "hidden_size": 128, "intermediate_size": 256,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                     "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                     "layer_types": ["linear_attention", "full_attention"],
                     "tie_word_embeddings": True, "eos_token_id": 128, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 128, "num_position_embeddings": 64,
                       "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2},
        image_token_id=126, video_token_id=127, tie_word_embeddings=True)
    return Qwen3_5ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def reload_probe(checkpoint, expected_path, device, dtype):
    from safetensors.torch import load_file

    expected = load_file(str(expected_path))
    model = load_packed_model(checkpoint, device=device, dtype=dtype, fallback=False)
    before = packed_residency(model)
    actual = capture_probes(model, probe_inputs(expected), device,
                            prompt_tokens=4, logit_positions=4, new_tokens=2)
    parity = compare_probes(actual, expected, reload=True)
    return {"passed": parity["passed"], "parity": parity,
            "residency_before": before, "residency_after": packed_residency(model),
            "artifact": audit_artifact(checkpoint)}


def preflight(device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = torch.float16 if device == "cuda" else torch.float32
    checks = []
    with tempfile.TemporaryDirectory(prefix="rotquant-packed-preflight-") as temporary:
        root = Path(temporary)
        for bits in (6, 8):
            torch.manual_seed(20260907)
            model = tiny_model(device, dtype)
            owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                        VocabularyConfig(bits=bits, chunk_rows=64))
            owner.projection_mode = "dense_equivalent"
            patch_model(model, PatchConfig(quant=QuantConfig(
                bits=5, group_size=128, scale_bits=8), block=128, fallback=True,
                include=["model.language_model.layers."],
                exclude=["linear_attn.in_proj_a", "linear_attn.in_proj_b"]))
            prompts = [{"input_ids": torch.tensor([[1, 2, 3, 4]])}]
            with vocabulary_prototype(model, owner):
                dense = capture_probes(model, prompts, device, prompt_tokens=4,
                                       logit_positions=4, new_tokens=2)
            with packed_context(model, owner, device, dtype):
                packed_residency(model)
                packed = capture_probes(model, prompts, device, prompt_tokens=4,
                                        logit_positions=4, new_tokens=2)
                parity = compare_probes(packed, dense)
                if not parity["passed"]:
                    raise ValueError(f"tiny Qwen W5/V{bits} prototype parity failed: {parity}")
                checkpoint = root / f"v{bits}"
                save_packed_checkpoint(model, checkpoint, model_loader="multimodal_lm")
            expected = root / f"v{bits}-probes.safetensors"
            save_probes(expected, packed)
            output = root / f"v{bits}-reload.json"
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--device", device, "--reload-dir", str(checkpoint),
                            "--expected", str(expected), "--output", str(output)], check=True)
            reloaded = json.loads(output.read_text())
            if not reloaded["passed"]:
                raise ValueError(f"tiny Qwen W5/V{bits} fresh reload failed")
            checks.append({"vocabulary_bits": bits, "prototype_parity": parity,
                           "fresh_process_reload": reloaded})
            print(f"passed tiny multimodal Qwen W5/V{bits} export/reload", flush=True)
    return {"protocol": "qwen35-packed-validation-preflight-v1", "passed": True,
            "device": device, "checks": checks,
            "boundary": "Tiny random model, text-path conformance only; no pretrained quality or vision accuracy result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reload-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.reload_dir:
        if args.expected is None:
            parser.error("reload requires expected probes")
        result = reload_probe(args.reload_dir, args.expected, args.device,
                              torch.float16 if args.device == "cuda" else torch.float32)
    else:
        result = preflight(args.device)
    if args.output:
        write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
