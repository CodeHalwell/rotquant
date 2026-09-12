"""Fail early on dependencies, hardware or invalid retained evidence; no model load."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rotquant.checkpoint import MANIFEST_NAME
from scripts.build_rq3_runtime import digest
from scripts.export_rotquant_gguf_v2 import validate_recipe
from scripts.run_rq3_retained_gpu import verified_source_evidence

PACKAGES = {"transformers": "5.9.0", "safetensors": "0.8.0", "sentencepiece": "0.2.1",
            "scipy": "1.15.3", "pyyaml": "6.0.3", "ninja": "1.13.0", "cmake": "3.31.10"}


def environment(backend, expected_torch, enforce_pins=True):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: F401

    if expected_torch and torch.__version__ != expected_torch:
        raise ValueError("PyTorch changed during dependency setup; stop before rebuilding CUDA")
    versions = {name: metadata.version(name) for name in PACKAGES}
    if enforce_pins and versions != PACKAGES:
        raise ValueError(f"unexpected dependency versions: {versions}")
    if backend == "CUDA":
        if not torch.cuda.is_available() or not shutil.which("nvcc"):
            raise ValueError("CUDA GPU and nvcc required; no CPU fallback")
        device = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
    else:
        if platform.system() != "Darwin" or not torch.backends.mps.is_available():
            raise ValueError("Metal validation requires a supported Mac")
        device, vram = "Metal", None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "torch": torch.__version__, "torch_file": torch.__file__, "cuda": torch.version.cuda,
            "backend": backend, "device": device, "total_vram_bytes": vram, "passed": True}


def source(arm, expected_bits):
    print(f"Checking every saved checkpoint hash: {arm}", flush=True)
    probes, manifest = verified_source_evidence(arm)
    validate_recipe(manifest)
    if manifest["tied_vocabulary"]["config"]["bits"] != expected_bits:
        raise ValueError("arm name and vocabulary bit width disagree")
    checkpoint = arm / "checkpoint"
    # The real exporter requires a saved tokenizer; never download a replacement.
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    expected = load_file(str(probes))
    vocab = manifest["tied_vocabulary"]["shape"][0]
    prefixes = sorted({k.split(".")[0] for k in expected})
    if not 1 <= len(prefixes) <= 8:
        raise ValueError("expected 1..8 bounded saved probes")
    allowed = set()
    for prefix in prefixes:
        ids = expected[prefix + ".input.input_ids"]
        logits = expected[prefix + ".logits"]
        generated = expected[prefix + ".generated"]
        if (ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1
                or not 1 <= ids.shape[1] <= 128 or (ids < 0).any() or (ids >= vocab).any()
                or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != vocab
                or not 1 <= logits.shape[1] <= min(16, ids.shape[1])
                or not logits.is_floating_point() or not torch.isfinite(logits).all()
                or generated.dtype != torch.int64 or generated.ndim != 2 or generated.shape[0] != 1
                or not 1 <= generated.shape[1] - ids.shape[1] <= 8
                or not torch.equal(generated[:, :ids.shape[1]], ids)
                or (generated < 0).any() or (generated >= vocab).any()):
            raise ValueError(f"unsupported saved probe: {prefix}")
        mask = expected.get(prefix + ".input.attention_mask")
        if mask is not None and (mask.shape != ids.shape or not torch.all(mask == 1)):
            raise ValueError("irregular/padded masks are unsupported")
        allowed.update(prefix + suffix for suffix in (".input.input_ids", ".logits", ".generated"))
        if mask is not None:
            allowed.add(prefix + ".input.attention_mask")
    if set(expected) != allowed:
        raise ValueError("unexpected/multimodal saved probe inputs")
    return {"passed": True, "checkpoint_sha256": digest(checkpoint / MANIFEST_NAME),
            "prepared_sha256": digest(arm / "prepared.json"), "probe_sha256": digest(probes),
            "vocabulary_bits": expected_bits, "probes": len(prefixes),
            "checkpoint_bytes": sum((checkpoint / p).stat().st_size for p in manifest["files_sha256"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("environment", "source"))
    parser.add_argument("--backend", choices=("CUDA", "Metal"), default="CUDA")
    parser.add_argument("--expected-torch")
    parser.add_argument("--allow-unpinned-local-test", action="store_true")
    parser.add_argument("--source-arm", type=Path)
    parser.add_argument("--bits", type=int, choices=(6, 8))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = environment(args.backend, args.expected_torch, not args.allow_unpinned_local_test) if args.mode == "environment" else source(args.source_arm, args.bits)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
