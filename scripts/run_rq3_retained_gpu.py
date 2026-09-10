"""Bounded native-GPU parity against saved retained-model probes, then timing.

No model downloads, quantization, training, or task sweeps. Source evidence is
read-only; all reports go to a new directory. Timing is skipped on any parity
failure. This does not establish a quality win or a speedup over another engine.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rotquant.checkpoint import MANIFEST_NAME, verify_checkpoint
from rotquant.validation import compare_probes
from scripts.build_rq3_runtime import digest
from scripts.check_rq3_model import execution_settings, runtime_identity
from scripts.rq3_test_runtime import NativeTests


def verified_source_evidence(arm):
    """Verify immutable preparation/checkpoint/probe bindings before any build."""
    prepared = json.loads((arm / "prepared.json").read_text())
    intent_path = arm / "preparation.json"
    intent = json.loads(intent_path.read_text())
    if prepared.get("status") != "prepared" or not prepared.get("prototype_parity", {}).get("passed"):
        raise ValueError("source preparation did not pass its canonical guards")
    core = {k: v for k, v in prepared.items() if k not in {"status", "export"}}
    if core != {k: v for k, v in intent.items() if k != "status"}:
        raise ValueError("prepared result is not bound to preparation evidence")
    checkpoint = arm / "checkpoint"
    manifest = verify_checkpoint(checkpoint, expected_manifest_sha256=prepared["export"]["manifest_sha256"])
    if manifest.get("deployment", {}).get("preparation_sha256") != digest(intent_path):
        raise ValueError("checkpoint preparation binding changed")
    probes = arm / "packed_probes.safetensors"
    if prepared["probe_files"].get(probes.name) != digest(probes):
        raise ValueError("canonical probe hash mismatch")
    return probes, manifest


def verified_evidence(arm, exported):
    probes, _ = verified_source_evidence(arm)
    checkpoint = arm / "checkpoint"
    export = json.loads((exported / "export.json").read_text())
    if (export.get("protocol") != "rotquant-gguf-v2-export-v1"
            or export["checkpoint_manifest_sha256"] != digest(checkpoint / MANIFEST_NAME)):
        raise ValueError("GGUF was not exported from this retained checkpoint")
    for name, identity in export["artifact_files"].items():
        if Path(name).name != name or not name or (exported / name).is_symlink():
            raise ValueError("unsafe export artifact path")
        path = exported / name
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise ValueError(f"export artifact changed: {name}")
    if "model.gguf" not in export["artifact_files"]:
        raise ValueError("export lacks model.gguf")
    if sum(v["bytes"] for v in export["artifact_files"].values()) != export["complete_model_payload_bytes"]:
        raise ValueError("export byte ledger does not reconcile")
    return probes, export


def capture_native(model, expected):
    prefixes = sorted({key.split(".")[0] for key in expected})
    if not 1 <= len(prefixes) <= 8:
        raise ValueError("preflight permits 1..8 saved prompts")
    actual = {}
    for prefix in prefixes:
        inputs = {k: v for k, v in expected.items() if k.startswith(prefix + ".input.")}
        if {k.split(".input.", 1)[1] for k in inputs} - {"input_ids", "attention_mask"}:
            raise ValueError("text-only native preflight cannot ignore extra model inputs")
        ids = expected[prefix + ".input.input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 128:
            raise ValueError("requires one saved prompt of at most 128 tokens")
        for key, value in inputs.items():
            if key.endswith("attention_mask") and (value.shape != ids.shape or not torch.all(value == 1)):
                raise ValueError("padded/irregular attention masks unsupported")
        logits = expected[prefix + ".logits"]
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != model.vocab or not 1 <= logits.shape[1] <= 16:
            raise ValueError("unexpected saved logit shape")
        selected = model.evaluate(ids[0].numpy(), reset=True, selected=logits.shape[1])
        actual.update(inputs)
        actual[prefix + ".logits"] = torch.from_numpy(selected.reshape(logits.shape).copy())
        generated = expected[prefix + ".generated"]
        if (generated.ndim != 2 or generated.shape[0] != 1
                or not torch.equal(generated[:, :ids.shape[1]], ids)
                or not 1 <= generated.shape[1] - ids.shape[1] <= 8):
            raise ValueError("requires the original bounded greedy trace")
        output = model.evaluate(ids[0].numpy(), reset=True)
        tokens = ids[0].tolist()
        for _ in range(generated.shape[1] - ids.shape[1]):
            token = int(output.argmax())
            tokens.append(token)
            if model.is_eog(token):
                break
            output = model.evaluate([token])
        actual[prefix + ".generated"] = torch.tensor([tokens], dtype=torch.int64)
        print(f"native parity {prefix}: {ids.shape[1]} input tokens, {logits.shape[1]} logit positions, {len(tokens)-ids.shape[1]} generated tokens", flush=True)
    if set(actual) != set(expected):
        raise ValueError("unexpected or missing canonical probe keys")
    return actual


class ProcessVRAM:
    """Sample this PID through nvidia-smi; Torch's allocator excludes GGML."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True, timeout=3)
                values = [float(row.split(",")[1]) for row in text.splitlines() if row.split(",")[0].strip() == str(os.getpid())]
                if values:
                    self.samples.append(sum(values))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)

    def report(self):
        return {"sampled_peak_process_vram_mib": max(self.samples) if self.samples else None,
                "samples": len(self.samples), "scope": "nvidia-smi process allocation, 0.5s sampling; not an exact transient peak; unavailable on Metal"}


def timing(model, ids, contexts=(128, 512, 2048), decode=16):
    rows = []
    for context in contexts:
        prompt = np.resize(ids, context)
        # Shape warmup is deliberately measured separately and excluded.
        for repetition in range(3):
            start = time.perf_counter()
            logits = model.evaluate(prompt, reset=True)
            prefill = time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(decode):
                logits = model.evaluate([int(logits.argmax())])
            elapsed = time.perf_counter() - start
            row = {"context": context, "repetition": repetition, "warmup": repetition == 0,
                   "prefill_seconds": prefill, "prefill_tokens_per_second": context / prefill,
                   "decode_seconds": elapsed, "decode_tokens_per_second": decode / elapsed,
                   "decode_steps": decode}
            rows.append(row)
            print("timing", json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--source-arm", type=Path, required=True)
    parser.add_argument("--export", dest="exported", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing", action="store_true", help="only after saved-probe parity passes")
    args = parser.parse_args()
    if args.backend == "CPU":
        raise ValueError("this stage requires a GPU; CPU fallback is forbidden")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "rq3-retained-native-gpu-v1", "passed": False, "backend": args.backend,
              "cpu_fallback_forbidden": True, "runtime_files": runtime_identity(args.library), "settings": execution_settings(),
              "boundary": "Native text-model probe parity and bounded timing, not task accuracy, multimodal execution or a matched-engine speedup."}
    path = args.output_dir / "report.json"
    def persist():
        path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        print("Verifying original checkpoint, export, and saved probe hashes", flush=True)
        probes, exported = verified_evidence(args.source_arm, args.exported)
        report.update(export_sha256=digest(args.exported / "export.json"), probe_sha256=digest(probes),
                      complete_model_payload_bytes=exported["complete_model_payload_bytes"],
                      export_files=exported["artifact_files"], checkpoint_manifest_sha256=exported["checkpoint_manifest_sha256"])
        os.environ["ROTQUANT_REQUIRE_GPU"] = "1"
        expected = load_file(str(probes))
        runtime = NativeTests(args.library)
        with ProcessVRAM() as memory:
            loading = time.perf_counter()
            with runtime.model(args.exported / "model.gguf", args.backend, 256) as model:
                report["load_seconds"] = time.perf_counter() - loading
                actual = capture_native(model, expected)
                report["gpu_custom_ops"] = model.custom_ops
            save_file(actual, str(args.output_dir / "native_probes.safetensors"))
            parity = compare_probes(actual, expected)
            parity["comparison"] = "cross-engine-native-v3"
            parity["guards"]["exact_generation"] = parity["generation_probes"] == parity["exact_generation_probes"]
            parity["passed"] = all(parity["guards"].values())
            report["parity"] = parity
            report["passed"] = parity["passed"] and report["gpu_custom_ops"] > 0
            persist()
            if not report["passed"]:
                raise ValueError("native parity failed; timings and task benchmarks are forbidden")
            if args.timing:
                with runtime.model(args.exported / "model.gguf", args.backend, 2304) as model:
                    report["timing"] = timing(model, expected["p0.input.input_ids"][0].numpy())
                    report["timing_context_capacity"] = 2304
            else:
                report["timing"] = {"skipped": "not requested"}
        report["memory"] = memory.report()
        persist()
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise


if __name__ == "__main__":
    main()
