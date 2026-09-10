"""Bounded whole-Qwen CPU/GPU conformance on a random, non-quality fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rq3_test_runtime import NativeTests


def numerical_metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid conformance logits")
    def log_softmax(value):
        shifted = value - value.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    la, lb = log_softmax(a), log_softmax(b)
    return {"max_abs_error": float(np.abs(a - b).max()), "mean_abs_error": float(np.abs(a - b).mean()),
            "mean_kl": max(0., float((np.exp(lb) * (lb - la)).sum(axis=-1).mean())),
            "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).mean())}


def runtime_identity(library):
    directory = Path(library).resolve().parent
    files = [p for p in directory.iterdir() if p.is_file() and (
        ".so" in p.name or ".dylib" in p.name or p.suffix == ".dll" or p.name == "default.metallib")]
    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    return {p.name: digest(p) for p in sorted(files)}


def execution_settings():
    return {"kv_dtype": "float16", "flash_attention": "auto", "batch": 1,
            "GGML_METAL_TENSOR_DISABLE": os.environ.get("GGML_METAL_TENSOR_DISABLE"),
            "GGML_METAL_TENSOR_ENABLE": os.environ.get("GGML_METAL_TENSOR_ENABLE")}


def check_model(library, model_path, backend):
    if backend == "CPU":
        raise ValueError("choose a GPU backend; the CPU is always the reference")
    runtime = NativeTests(library)
    inputs = [list(range(3, 3 + count)) for count in (1, 4, 17, 64)]
    captured = {}
    original_gate = os.environ.get("ROTQUANT_REQUIRE_GPU")
    try:
        for device in ("CPU", backend):
            os.environ["ROTQUANT_REQUIRE_GPU"] = "0" if device == "CPU" else "1"
            results, traces = [], []
            with runtime.model(model_path, device, 256) as model:
                for prompt in inputs:
                    logits = model.evaluate(prompt, reset=True)
                    results.append(logits.copy())
                    generated = []
                    for _ in range(8):
                        token = int(logits.argmax())
                        generated.append(token)
                        logits = model.evaluate([token])
                        results.append(logits.copy())
                    traces.append(generated)
                    print(f"{device}: {len(prompt)} prompt tokens + 8 cached steps completed", flush=True)
                if model.custom_ops == 0:
                    raise ValueError("fixture did not execute native-v3 operators")
                captured[device] = {"logits": np.stack(results), "traces": traces, "custom_ops": model.custom_ops}
    finally:
        if original_gate is None:
            os.environ.pop("ROTQUANT_REQUIRE_GPU", None)
        else:
            os.environ["ROTQUANT_REQUIRE_GPU"] = original_gate
    metrics = numerical_metrics(captured[backend]["logits"], captured["CPU"]["logits"])
    by_prompt = {str(len(prompt)): numerical_metrics(captured[backend]["logits"][i*9:(i+1)*9],
                 captured["CPU"]["logits"][i*9:(i+1)*9]) for i, prompt in enumerate(inputs)}
    thresholds = {"max_abs_error": .02, "mean_abs_error": .002, "mean_kl": 1e-5, "top1_agreement": 1.}
    guards = {k: metrics[k] >= v if k == "top1_agreement" else metrics[k] <= v for k, v in thresholds.items()}
    guards["exact_generation"] = captured[backend]["traces"] == captured["CPU"]["traces"]
    with Path(model_path).open("rb") as handle:
        model_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"protocol": "rq3-synthetic-model-check-v1", "backend": backend,
            "runtime_files": runtime_identity(library), "settings": execution_settings(), "model_sha256": model_digest,
            "metrics": metrics, "by_prompt_length": by_prompt, "thresholds": thresholds, "guards": guards, "passed": all(guards.values()),
            "gpu_custom_ops": captured[backend]["custom_ops"], "cpu_fallback_forbidden": True,
            "scope": "random two-layer Qwen graph; not retained-4B parity, accuracy or speed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = check_model(args.library, args.model, args.backend)
    except BaseException as error:
        args.output.write_text(json.dumps({"passed": False, "backend": args.backend,
            "settings": execution_settings(), "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("whole-model conformance failed; inspect report")
