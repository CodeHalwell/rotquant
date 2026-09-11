"""One bounded native GPU context: synchronous prefill/decode and process VRAM.

Requires a matching saved-model parity pass. Repeated frozen token IDs are a
shape/performance fixture, not meaningful long-context text or an accuracy eval.
The private bridge includes synchronization, finite-logit checks, full-vocabulary
host copies and Python argmax. These are user-visible bridge timings, not pure
kernel timings and not a matched-engine speedup claim. Logging and diagnostic
checkpoint writes are excluded; rates do not describe total job wall time.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_rq3_model import execution_settings, runtime_identity
from scripts.native_gpu_workflow import digest, write_json
from scripts.native_pilot_controls import validate_controls
from scripts.rq3_test_runtime import NativeTests
from scripts.run_rq3_retained_gpu import ProcessVRAM


def summarize(rows):
    measured = [row for row in rows if not row["warmup"]]
    if not measured:
        return None
    seconds = sum(row["decode_seconds"] for row in measured)
    return {"measured_repetitions": len(measured),
        "prefill_tokens_per_second": sum(row["input_tokens"] for row in measured) / sum(row["prefill_seconds"] for row in measured),
        "decode_tokens_per_second": sum(row["decode_steps"] for row in measured) / seconds,
        "median_prefill_seconds": statistics.median(row["prefill_seconds"] for row in measured),
        "median_decode_step_ms": 1000 * statistics.median(step for row in measured for step in row["decode_step_seconds"]),
        "decode_tps_min": min(row["decode_tokens_per_second"] for row in measured),
        "decode_tps_max": max(row["decode_tokens_per_second"] for row in measured),
        "boundary": "Warmup excluded. Rates are total tokens / total time; min/max are observed repetitions, not confidence intervals."}


def measure(model, ids, *, context, decode, repetitions, minimum_tps, maximum_vram,
            memory, persist, clock=time.perf_counter):
    prompt = np.resize(ids, context).astype(np.int32)
    rows = []
    def memory_guard():
        peak = memory.report()["sampled_peak_process_vram_mib"]
        if peak is not None and peak > maximum_vram:
            raise RuntimeError(f"Cost guard: process VRAM {peak:.0f} MiB exceeds {maximum_vram:.0f} MiB")
    for repetition in range(repetitions + 1):
        memory_guard()
        print(f"PILOT context={context} repetition={repetition}/{repetitions} "
              f"{'warmup' if repetition == 0 else 'measured'}: prefill", flush=True)
        start = clock()
        logits = model.evaluate(prompt, reset=True)
        prefill = clock() - start
        row = {"input_tokens": context, "repetition": repetition, "warmup": repetition == 0,
               "prefill_seconds": prefill, "decode_step_seconds": [], "decode_steps": 0,
               "decode_seconds": 0., "completed": False}
        if prefill <= 0 or not np.isfinite(logits).all():
            raise ValueError("Invalid prefill measurement")
        persist(rows, row)
        for step in range(decode):
            memory_guard()
            start = clock()
            # Fixed number of cached steps, even at EOS: not task generation.
            token = int(logits.argmax())
            logits = model.evaluate([token])
            elapsed = clock() - start
            if elapsed <= 0 or not np.isfinite(logits).all():
                raise ValueError("Invalid cached decode measurement")
            row["decode_step_seconds"].append(elapsed)
            row["decode_steps"] += 1
            row["decode_seconds"] += elapsed
            # Persist outside the timed region, including partial repetitions.
            persist(rows, row)
            if (step + 1) % 8 == 0:
                print(f"PILOT context={context} rep={repetition}: decode {step+1}/{decode} "
                      f"({row['decode_steps']/row['decode_seconds']:.2f} tok/s)", flush=True)
        row.update(completed=True, prefill_tokens_per_second=context / prefill,
                   decode_tokens_per_second=decode / row["decode_seconds"])
        rows.append(row)
        persist(rows, None)
        memory_guard()
        if repetition and row["decode_tokens_per_second"] < minimum_tps:
            raise RuntimeError(f"Cost guard: decode {row['decode_tokens_per_second']:.2f} tok/s "
                               f"is below the {minimum_tps:.2f} tok/s pilot floor; later contexts skipped")
    return rows


def verify_inputs(library, exported, parity_path, probe_path):
    gate = json.loads(parity_path.read_text())
    identity = runtime_identity(library)
    if (gate.get("passed") is not True or gate.get("parity", {}).get("passed") is not True
            or gate.get("runtime_files") != identity or gate.get("cpu_fallback_forbidden") is not True
            or gate.get("gpu_custom_ops", 0) <= 0 or gate.get("backend") not in {"CUDA0", "MTL0"}
            or gate.get("export_sha256") != digest(exported / "export.json")
            or gate.get("probe_sha256") != digest(probe_path)):
        raise ValueError("A matching retained parity pass is required before timing")
    receipt = json.loads((exported / "export.json").read_text())
    if (receipt.get("checkpoint_manifest_sha256") != gate.get("checkpoint_manifest_sha256")
            or receipt.get("artifact_files") != gate.get("export_files")
            or "model.gguf" not in receipt.get("artifact_files", {})):
        raise ValueError("Pilot export disagrees with parity receipt")
    for name, value in receipt["artifact_files"].items():
        path = exported / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size != value["bytes"] or digest(path) != value["sha256"]):
            raise ValueError("Pilot export artifact changed after parity")
    return gate, identity


def run(args):
    validate_controls(args.context, args.decode_steps, args.repetitions, args.min_decode_tps, args.max_vram_mib)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "report.json"
    report = {"protocol": "rq3-native-performance-pilot-v1", "passed": False, "status": "running",
        "context": args.context, "context_capacity": ((args.context + args.decode_steps + 255) // 256) * 256,
        "controls": {"decode_steps": args.decode_steps, "repetitions": args.repetitions,
                     "warmup_repetitions": 1, "min_decode_tps": args.min_decode_tps,
                     "max_process_vram_mib": args.max_vram_mib},
        "settings": execution_settings(), "rows": [], "in_progress": None, "summary": None,
        "boundary": __doc__}
    memory = ProcessVRAM()
    def persist(rows=None, current=None):
        if rows is not None:
            report.update(rows=rows, in_progress=current, summary=summarize(rows))
        report["memory"] = memory.report()
        write_json(path, report)
    persist()
    try:
        gate, identity = verify_inputs(args.library, args.exported, args.parity_report, args.probes)
        report.update(runtime_files=identity, backend=gate["backend"], cpu_fallback_forbidden=True,
                      parity_report_sha256=digest(args.parity_report), export_sha256=gate["export_sha256"],
                      probe_sha256=gate["probe_sha256"], complete_model_payload_bytes=gate["complete_model_payload_bytes"])
        expected = load_file(str(args.probes))
        ids = expected["p0.input.input_ids"][0].numpy()
        os.environ["ROTQUANT_REQUIRE_GPU"] = "1"
        runtime = NativeTests(args.library)
        with memory:
            start = time.perf_counter()
            with runtime.model(args.exported / "model.gguf", gate["backend"], report["context_capacity"]) as model:
                report["load_seconds"] = time.perf_counter() - start
                measure(model, ids, context=args.context, decode=args.decode_steps, repetitions=args.repetitions,
                        minimum_tps=args.min_decode_tps, maximum_vram=args.max_vram_mib,
                        memory=memory, persist=persist)
                report["gpu_custom_ops"] = model.custom_ops
                if model.custom_ops <= 0:
                    raise ValueError("No packed native operations observed")
        if gate["backend"] == "CUDA0" and not memory.samples:
            raise ValueError("Process VRAM unavailable; pilot cannot claim a memory pass")
        peak = memory.report()["sampled_peak_process_vram_mib"]
        if peak is not None and peak > args.max_vram_mib:
            raise RuntimeError("Cost guard: final sampled VRAM exceeds the configured limit")
        report.update(passed=True, status="passed")
        persist()
    except BaseException as error:
        report.update(passed=False, status="stopped", error=f"{type(error).__name__}: {error}")
        persist()
        raise
    print(json.dumps({"context": args.context, "summary": report["summary"], "memory": report["memory"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--export", dest="exported", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, choices=(128, 512, 2048), required=True)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--min-decode-tps", type=float, default=2.)
    parser.add_argument("--max-vram-mib", type=float, default=16384.)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt("Pilot phase interrupted; partial measurements preserved")
    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
