"""Offline random-HF-model -> saved checkpoint -> GGUF -> CPU/GPU conformance.

The tokenizer is a deliberately fake 256-token fixture; only frozen integer
IDs are evaluated. This does not validate the production tokenizer or 4B model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rotquant.checkpoint import save_packed_checkpoint
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.vocabulary import VocabularyConfig, install_packed_vocabulary, quantize_vocabulary
from scripts.build_rq3_runtime import digest
from scripts.check_rq3_model import execution_settings, numerical_metrics, runtime_identity
from scripts.export_rotquant_gguf_v2 import export_checkpoint
from scripts.rq3_test_runtime import NativeTests


def fake_vocab(converter):
    writer = converter.gguf_writer
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list([f"t{i}" for i in range(256)])
    writer.add_token_types([1] * 256)
    writer.add_token_merges(["t0 t1"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)


def check(library, llama_dir, output, backend):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if backend == "CPU":
        raise ValueError("choose a GPU; CPU and HF are always included as references")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(llama_dir.resolve()))
    sys.path.insert(0, str(llama_dir.resolve() / "gguf-py"))
    from conversion.qwen import Qwen3_5TextModel

    torch.set_num_threads(4)
    torch.manual_seed(331)
    config = Qwen3_5TextConfig(vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=128,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128,
        linear_value_head_dim=128, layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True, max_position_embeddings=512, architectures=["Qwen3_5ForCausalLM"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.,
                         "partial_rotary_factor": .5, "mrope_section": [11, 11, 10]})
    model = Qwen3_5ForCausalLM(config).half().eval()
    owner = quantize_vocabulary(model.get_input_embeddings().weight.detach(), VocabularyConfig(bits=6, chunk_rows=128))
    owner.execution_dtype, owner.projection_mode = torch.float16, "dense_equivalent"
    patch_model(model, PatchConfig(quant=QuantConfig(bits=5, group_size=128, scale_bits=8, scale="rms"),
                                  exclude=["lm_head", "in_proj_a", "in_proj_b"]))
    install_packed_vocabulary(model, owner)
    save_packed_checkpoint(model, output / "checkpoint")
    original_vocab = Qwen3_5TextModel.set_vocab
    try:
        Qwen3_5TextModel.set_vocab = fake_vocab
        export_checkpoint(output / "checkpoint", output / "export", llama_dir.resolve())
    finally:
        Qwen3_5TextModel.set_vocab = original_vocab
    runtime = NativeTests(library)
    rows = []
    original_gate = os.environ.get("ROTQUANT_REQUIRE_GPU")
    try:
        for device in ("CPU", backend):
            os.environ["ROTQUANT_REQUIRE_GPU"] = "0" if device == "CPU" else "1"
            with runtime.model(output / "export/model.gguf", device, 256) as native:
                for length in (4, 17, 64):
                    ids = torch.arange(3, length + 3)[None]
                    with torch.no_grad():
                        expected = model(ids, use_cache=False).logits[0, -4:].float().numpy()
                    actual = native.evaluate(ids[0].numpy(), reset=True, selected=4)
                    metrics = numerical_metrics(actual, expected)
                    row = {"backend": device, "tokens": length, "metrics": metrics,
                           "passed": metrics["max_abs_error"] <= .02 and metrics["mean_abs_error"] <= .002
                           and metrics["mean_kl"] <= 1e-5 and metrics["top1_agreement"] == 1.}
                    rows.append(row)
                    print("conversion check", json.dumps(row), flush=True)
    finally:
        if original_gate is None:
            os.environ.pop("ROTQUANT_REQUIRE_GPU", None)
        else:
            os.environ["ROTQUANT_REQUIRE_GPU"] = original_gate
    return {"protocol": "rq3-offline-hf-conversion-v1", "seed": 331, "backend": backend,
            "settings": execution_settings(), "runtime_files": runtime_identity(library),
            "export_sha256": digest(output / "export/export.json"), "cases": rows,
            "thresholds": {"max_abs_error": .02, "mean_abs_error": .002, "mean_kl": 1e-5, "top1_agreement": 1.},
            "passed": all(r["passed"] for r in rows),
            "scope": "random two-layer HF checkpoint, fake tokenizer, frozen integer IDs; not production 4B/tokenizer validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = check(args.library, args.llama_dir, args.output_dir, args.backend)
    except BaseException as error:
        args.report.write_text(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("HF/CPU/GPU conversion conformance failed")
