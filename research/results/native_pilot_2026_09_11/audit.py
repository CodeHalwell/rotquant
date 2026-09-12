"""Audit the original pilot reports, not missing GPU binaries/model tensors.

Usage: python audit.py /path/to/extracted/original-reports
Prints a reproducible JSON audit; performs no downloads or GPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
REVISION = "1bc31d43acf62f669fe7e5d8af8a766e94698232"


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def audit(root):
    def read(name):
        return json.loads((root / name).read_text())
    def report(stage, name="report.json"):
        return read(f"stages/{stage}/attempt-001/{name}")
    def producer(name):
        return subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=REPO)
    def near(a, b):
        assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)
    files = {}
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink(), path
        if path.is_file():
            files[path.relative_to(root).as_posix()] = identity(path)
    workflow, summary = read("workflow.json"), read("summary.json")
    controls = workflow["controls"]
    assert controls["revision"] == REVISION and controls["dirty_local_test"] is False
    assert controls["backend"] == "CUDA" and controls["arms"] == ["b5_v6_s0"]
    assert controls["performance_pilot"] is True
    assert len(workflow["attempts"]) == len(summary["stages"]) == 14
    assert workflow["status"] == summary["status"] == "failed"
    assert "TimeoutError" in summary["error"]
    for i, (attempt, row) in enumerate(zip(workflow["attempts"], summary["stages"], strict=True)):
        assert attempt["name"] == row["name"]
        assert attempt["status"] == row["status"] == ("passed" if i < 13 else "failed")
        near(attempt["elapsed_seconds"], row["elapsed_seconds"])
    seconds = sum(row["elapsed_seconds"] for row in summary["stages"])
    near(seconds, workflow["active_seconds"])
    near(seconds / 60, summary["active_minutes"])
    original_root = PurePosixPath(workflow["attempts"][0]["directory"]).parents[2]
    checked, omitted = set(), set()
    for attempt in workflow["attempts"]:
        for name, expected in attempt.get("artifacts", {}).items():
            remote = PurePosixPath(name)
            if remote.is_relative_to(original_root):
                local = remote.relative_to(original_root).as_posix()
                assert files[local] == expected, local
                checked.add(local)
            else:
                omitted.add(name)
    build = report("native-build-and-load", "build-receipt.json")
    assert build["root_revision"] == REVISION and build["root_dirty"] is False
    assert build["load_validated"] and build["backend"] == "CUDA"
    code = {**controls["workflow_sources"], **build["external_sources"]}
    for name, expected in code.items():
        assert hashlib.sha256(producer(name)).hexdigest() == expected, name
    contract = json.loads(producer("integrations/llama.cpp/rotquant-native-v2-files.json"))
    assert build["patch_sha256"] == contract["patch_sha256"]
    assert build["base_revision"] == contract["base_revision"]
    runtime = report("binding-load", "load.json")["runtime_files"]
    assert runtime["librotquant_ggml_test.so"] == build["library_sha256"]
    for stage in ("operators-CPU", "operators-CUDA0", "whole-model-w6", "whole-model-w8", "hf-conversion"):
        data = report(stage)
        assert data["passed"] and data["runtime_files"] == runtime
        if "cases" in data:
            assert all(case["passed"] for case in data["cases"])
        if "guards" in data:
            assert all(data["guards"].values())
    retained = report("retained-b5_v6_s0", "retained/report.json")
    exported = report("export-b5_v6_s0", "export.json")
    source = report("source-b5_v6_s0", "source.json")
    assert retained["passed"] and retained["runtime_files"] == runtime
    assert retained["cpu_fallback_forbidden"] and retained["gpu_custom_ops"] > 0
    assert retained["export_files"] == exported["artifact_files"]
    assert retained["checkpoint_manifest_sha256"] == exported["checkpoint_manifest_sha256"] == source["checkpoint_sha256"]
    assert retained["probe_sha256"] == source["probe_sha256"]
    assert retained["export_sha256"] == files["stages/export-b5_v6_s0/attempt-001/export.json"]["sha256"]
    parity = retained["parity"]
    assert parity["passed"] and all(parity["guards"].values())
    for key in ("max_abs_error", "mean_abs_error", "mean_kl"):
        assert 0 <= parity[key] <= parity["thresholds"][key]
    assert parity["positions"] == 16 and parity["top1_agreement"] == 1.
    assert parity["generation_probes"] == parity["exact_generation_probes"] == 4
    sizes = {name: value["bytes"] for name, value in exported["artifact_files"].items()}
    assert sum(sizes.values()) == retained["complete_model_payload_bytes"]
    results = []
    for context, repetitions in ((128, 3), (512, 3), (2048, 1)):
        data = report(f"pilot-b5_v6_s0-ctx{context}", "pilot/report.json")
        assert data["runtime_files"] == runtime and data["backend"] == "CUDA0"
        assert data["export_sha256"] == retained["export_sha256"]
        assert data["probe_sha256"] == retained["probe_sha256"]
        assert data["parity_report_sha256"] == files["stages/retained-b5_v6_s0/attempt-001/retained/report.json"]["sha256"]
        assert data["passed"] == (context != 2048)
        rows = data["rows"]
        assert len(rows) == repetitions + 1
        for i, row in enumerate(rows):
            assert row["warmup"] == (i == 0) and row["input_tokens"] == context
            assert row["decode_steps"] == len(row["decode_step_seconds"]) == 32
            near(row["decode_seconds"], sum(row["decode_step_seconds"]))
            near(row["prefill_tokens_per_second"], context / row["prefill_seconds"])
            near(row["decode_tokens_per_second"], 32 / row["decode_seconds"])
        measured, aggregate = rows[1:], data["summary"]
        assert aggregate["measured_repetitions"] == repetitions
        near(aggregate["prefill_tokens_per_second"], repetitions * context / sum(r["prefill_seconds"] for r in measured))
        near(aggregate["decode_tokens_per_second"], repetitions * 32 / sum(r["decode_seconds"] for r in measured))
        near(aggregate["median_decode_step_ms"], 1000 * statistics.median(s for r in measured for s in r["decode_step_seconds"]))
        results.append({"context": context, "status": data["status"], **aggregate,
                        "sampled_peak_process_vram_mib": data["memory"]["sampled_peak_process_vram_mib"]})
    return {"protocol": "native-pilot-report-audit-v1", "reviewed_revision": REVISION,
        "audit_passed": True, "workflow_passed": False, "passed_stages": 13, "stopped_stages": 1,
        "active_minutes": seconds / 60, "build_minutes": summary["stages"][2]["elapsed_seconds"] / 60,
        "included_artifacts_verified": len(checked), "external_artifacts_unavailable": len(omitted),
        "producer_sources_verified": len(code), "bundle_index": files,
        "retained_parity": parity, "payload_bytes": sizes, "timing": results,
        "limits": ["2048 timing is one measured repetition only; it is not a completed context benchmark.",
            "Missing binaries, weights and raw logits prevent independent GPU/numerical replay.",
            "Rates include the synchronous private bridge, logit copies/checks and argmax; not pure kernels.",
            "Sampled process VRAM may miss transients. Text inference does not load the auxiliary sidecar.",
            "These are unchanged quantized-model parity probes, not new FP16/task quality evidence."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    print(json.dumps(audit(parser.parse_args().bundle), indent=2))
