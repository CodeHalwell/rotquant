"""Isolate ordinary-GGUF bridge errors from CPU/CUDA quantized-backend drift.

The independent caller uses public llama APIs and explicit token positions, but
shares the compiled llama/ggml libraries. This is a bridge-equivalence gate,
not an independent audit of upstream kernels, model quality or performance.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_conventional_control import SOURCE
from scripts.build_rq3_runtime import REVISION
from scripts.check_rq3_model import execution_settings, numerical_metrics, runtime_identity
from scripts.native_gpu_workflow import digest, write_json
from scripts.rq3_test_runtime import NativeTests

COUNTS = (1, 4, 17, 64)
# Unchanged historical CPU/GPU limits; still required for ordinary BF16.
CROSS_BACKEND_LIMITS = {"max_abs_error": .02, "mean_abs_error": .002,
                        "mean_kl": 1e-5, "top1_agreement": 1.}
# The *same* native backend must agree when called through a separate API path.
BRIDGE_LIMITS = {"max_abs_error": 1e-6, "mean_abs_error": 1e-7,
                 "mean_kl": 1e-10, "top1_agreement": 1.}


def validated_capture(logits, traces):
    logits, traces = np.asarray(logits), np.asarray(traces)
    if logits.shape != (36, 256) or not np.isfinite(logits).all():
        raise ValueError("invalid control logit shape/values")
    if traces.shape != (4, 8) or traces.dtype.kind not in "iu" or (traces < 0).any() or (traces >= 256).any():
        raise ValueError("invalid control greedy traces")
    # Every emitted token must actually be argmax at the preceding position.
    if not np.array_equal(logits.reshape(4, 9, 256)[:, :8].argmax(-1), traces):
        raise ValueError("trace does not follow the captured logits")
    return {"logits": logits, "traces": traces}


def read_capture(path):
    raw = Path(path).read_bytes()
    if len(raw) != 24 + 36 * 256 * 4 + 32 * 4 or raw[:8] != b"RQCTRL01":
        raise ValueError("invalid public-API probe record")
    if struct.unpack_from("<4I", raw, 8) != (4, 9, 256, 8):
        raise ValueError("unexpected public-API probe dimensions")
    return validated_capture(np.frombuffer(raw, dtype="<f4", count=36*256, offset=24).reshape(36, 256),
                             np.frombuffer(raw, dtype="<i4", count=32, offset=24+36*256*4).reshape(4, 8))


def capture_bridge(library, model_path, device):
    runtime = NativeTests(library)
    logits, traces = [], []
    previous = os.environ.get("ROTQUANT_REQUIRE_GPU")
    try:
        os.environ["ROTQUANT_REQUIRE_GPU"] = "0" if device == "CPU" else "1"
        with runtime.model(model_path, device, 256) as model:
            if model.vocab != 256:
                raise ValueError("requires tiny fixture vocabulary")
            for count in COUNTS:
                output = model.evaluate(list(range(3, 3 + count)), reset=True)
                logits.append(output.copy())
                generated = []
                for _ in range(8):
                    token = int(output.argmax())
                    generated.append(token)
                    output = model.evaluate([token])
                    logits.append(output.copy())
                traces.append(generated)
                print(f"private bridge {device}: {count} prompt tokens + 8 cached steps completed", flush=True)
            if model.custom_ops != 0:
                raise ValueError("ordinary fixture executed RotQuant operators")
    finally:
        if previous is None:
            os.environ.pop("ROTQUANT_REQUIRE_GPU", None)
        else:
            os.environ["ROTQUANT_REQUIRE_GPU"] = previous
    return validated_capture(np.stack(logits), np.asarray(traces, dtype=np.int32))


def compare(actual, expected, limits):
    actual = validated_capture(**actual)
    expected = validated_capture(**expected)
    metrics = numerical_metrics(actual["logits"], expected["logits"])
    guards = {k: metrics[k] >= v if k == "top1_agreement" else metrics[k] <= v for k, v in limits.items()}
    guards["exact_generation"] = bool(np.array_equal(actual["traces"], expected["traces"]))
    return {"metrics": metrics, "thresholds": limits, "guards": guards, "passed": all(guards.values()),
            "by_prompt_length": {str(n): numerical_metrics(actual["logits"][i*9:(i+1)*9],
                                  expected["logits"][i*9:(i+1)*9]) for i, n in enumerate(COUNTS)}}


def assess(private, public, backend, kind):
    if kind not in {"bf16", "q4_0"} or set(private) != set(public) or set(private) != {"CPU", backend}:
        raise ValueError("missing or unexpected conventional control")
    same_backend = {device: compare(private[device], public[device], BRIDGE_LIMITS) for device in private}
    guards = {f"public_api_matches_{device}": r["passed"] for device, r in same_backend.items()}
    if backend == "CPU":
        return {"same_backend": same_backend, "cross_backend": {}, "guards": guards,
                "passed": all(guards.values()), "cross_backend_numerics_role": "not executed; CPU-only local test"}
    drift = {"private_bridge": compare(private[backend], private["CPU"], CROSS_BACKEND_LIMITS),
             "public_api": compare(public[backend], public["CPU"], CROSS_BACKEND_LIMITS)}
    for caller, r in drift.items():
        guards[f"{caller}_cross_backend_top1"] = r["guards"]["top1_agreement"]
        guards[f"{caller}_cross_backend_generation"] = r["guards"]["exact_generation"]
        if kind == "bf16":
            guards[f"{caller}_bf16_cross_backend_numerics"] = r["passed"]
    return {"same_backend": same_backend, "cross_backend": drift, "guards": guards,
            "passed": all(guards.values()),
            "cross_backend_numerics_role": "required" if kind == "bf16" else "diagnostic; not a Q4_0 bridge-equivalence gate"}


def verify_fixture(model, llama_dir, kind):
    if model.stat().st_size > 16 * 1024**2:
        raise ValueError("tiny conventional fixtures only")
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(str(model))
    wanted = gguf.GGMLQuantizationType.BF16 if kind == "bf16" else gguf.GGMLQuantizationType.Q4_0
    types = {t.name: t.tensor_type for t in reader.tensors}
    if (any(k.startswith("rotquant.") for k in reader.fields)
            or types.get("token_embd.weight") != wanted
            or sum(t == wanted for t in types.values()) != 14
            or set(types.values()) - {wanted, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}):
        raise ValueError("fixture format does not match the declared conventional control")


def verify_control(receipt_path, library):
    record = json.loads(receipt_path.read_text())
    executable = Path(record["executable"])
    if (record.get("protocol") != "rq3-public-api-control-build-v1" or record.get("passed") is not True
            or record.get("base_revision") != REVISION or record.get("source_sha256") != digest(SOURCE)
            or record.get("builder_sha256") != digest(ROOT / "scripts/build_conventional_control.py")
            or record.get("runtime_files") != runtime_identity(library)
            or executable.is_symlink() or digest(executable) != record["executable_sha256"]):
        raise ValueError("public-API control identity mismatch")
    return record, executable


def check(library, model, backend, kind, control_receipt, llama_dir, output, *, cpu_only_local_test=False):
    if kind not in {"bf16", "q4_0"}:
        raise ValueError("choose bf16/q4_0")
    if backend not in {"CUDA0", "MTL0"} and not (backend == "CPU" and cpu_only_local_test):
        raise ValueError("choose CUDA0/MTL0; CPU-only execution requires explicit local-test opt-in")
    if output.exists():
        raise FileExistsError(output)
    directory = output.parent / "conventional-probes"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "rq3-conventional-bridge-check-v2", "passed": False, "backend": backend,
              "format": kind, "cpu_only_local_test": cpu_only_local_test, "gpu_executed": False,
              "settings": execution_settings(), "runtime_files": runtime_identity(library),
              "cpu_fallback_forbidden": backend != "CPU", "gpu_custom_ops": 0,
              "boundary": "Independent public-API caller sharing llama/ggml libraries. Bridge equivalence on tiny fixtures, not upstream kernel correctness, 4B quality, speed, or decode4 validation."}
    try:
        control, executable = verify_control(control_receipt, library)
        verify_fixture(model, llama_dir, kind)
        report.update(model_sha256=digest(model), control_receipt_sha256=digest(control_receipt),
                      public_api_control=control, prompt_lengths=list(COUNTS), decode_steps=8)
        public, private = {}, {}
        for device in dict.fromkeys(("CPU", backend)):
            env = dict(os.environ, ROTQUANT_REQUIRE_GPU="0" if device == "CPU" else "1",
                       LD_LIBRARY_PATH=str(library.resolve().parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
            path = directory / f"public-{device}.bin"
            subprocess.run([str(executable), str(model.resolve()), device, str(path.resolve())], check=True, env=env, timeout=90)
            public[device] = read_capture(path)
            private[device] = capture_bridge(library, model, device)
            if device != "CPU":
                report["gpu_executed"] = True
        probes = directory / "probes.json"
        write_json(probes, {"protocol": "rq3-conventional-captures-v1", "model_sha256": report["model_sha256"],
                   "captures": {caller: {device: {k: v.tolist() for k, v in data.items()}
                               for device, data in captures.items()} for caller, captures in (("public_api", public), ("private_bridge", private))}})
        report["probes_sha256"] = digest(probes)
        report["probes_file"] = str(probes)
        report.update(assess(private, public, backend, kind))
        if (runtime_identity(library) != report["runtime_files"] or digest(model) != report["model_sha256"]
                or digest(control_receipt) != report["control_receipt_sha256"]):
            raise ValueError("model/runtime/control receipt changed during conventional conformance")
        verify_control(control_receipt, library)
    except BaseException as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        write_json(output, report)
        raise
    write_json(output, report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA0", "MTL0"), required=True)
    parser.add_argument("--format", choices=("bf16", "q4_0"), required=True)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only-local-test", action="store_true")
    args = parser.parse_args()
    result = check(args.library, args.model, args.backend, args.format, args.control_receipt,
                   args.llama_dir, args.output, cpu_only_local_test=args.cpu_only_local_test)
    if not result["passed"]:
        raise SystemExit("conventional bridge equivalence failed; inspect saved probes/report")
