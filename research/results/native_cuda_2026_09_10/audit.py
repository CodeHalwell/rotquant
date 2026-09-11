"""Recheck the supplied 06a4379 report bundle; no model execution or downloads.

Validates report integrity and aggregate guards, not KL recomputation from
missing probe tensors. Run with the extracted original-reports.tar.gz folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
REVISION = "06a4379c71074e4606116c835941388d1876599b"


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def producer_file(name):
    # Reproduce the historical audit after workflow code has evolved. Compare
    # the recorded producer revision, not today's edited working tree.
    return subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=REPO)


def audit(root):
    def read(name):
        return json.loads((root / name).read_text())
    def report(stage, name="report.json"):
        return read(f"stages/{stage}/attempt-001/{name}")
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Unexpected symlink: {p.name}")
        if p.is_file():
            files[p.relative_to(root).as_posix()] = identity(p)
    workflow, summary = read("workflow.json"), read("summary.json")
    controls = workflow["controls"]
    assert controls["revision"] == REVISION and controls["dirty_local_test"] is False
    assert controls["backend"] == "CUDA" and controls["arms"] == ["b5_v6_s0"]
    assert controls["synthetic_only"] is False and controls["timing"] is False
    assert len(workflow["attempts"]) == len(summary["stages"]) == 11
    assert workflow["status"] == summary["status"] == "passed" and summary["error"] is None
    for attempt, row in zip(workflow["attempts"], summary["stages"], strict=True):
        assert attempt["name"] == row["name"] and attempt["status"] == row["status"] == "passed"
        assert attempt["elapsed_seconds"] == row["elapsed_seconds"]
    total = sum(s["elapsed_seconds"] for s in summary["stages"])
    assert math.isclose(total, workflow["active_seconds"], abs_tol=1e-8)
    assert math.isclose(total / 60, summary["active_minutes"], abs_tol=1e-8)
    producer_root = PurePosixPath(workflow["attempts"][0]["directory"]).parents[2]
    checked, omitted = [], set()
    for attempt in workflow["attempts"]:
        for name, expected in attempt["artifacts"].items():
            path = PurePosixPath(name)
            if path.is_relative_to(producer_root):
                relative = path.relative_to(producer_root).as_posix()
                assert files[relative] == expected, relative
                checked.append(relative)
            else:
                omitted.add(name)

    build = report("native-build-and-load", "build-receipt.json")
    assert build["root_revision"] == REVISION and build["root_dirty"] is False
    assert build["backend"] == "CUDA" and build["load_validated"] is True
    code = {**controls["workflow_sources"], **build["external_sources"]}
    for name, expected in code.items():
        assert hashlib.sha256(producer_file(name)).hexdigest() == expected, name
    contract = json.loads(producer_file("integrations/llama.cpp/rotquant-native-v2-files.json"))
    assert build["patch_sha256"] == contract["patch_sha256"]
    assert build["base_revision"] == contract["base_revision"]
    build_artifacts = workflow["attempts"][2]["artifacts"]
    for name, expected in contract["files_sha256"].items():
        assert build_artifacts[controls["work_dir"] + "/llama-source/" + name]["sha256"] == expected
    runtime = report("binding-load", "load.json")["runtime_files"]
    assert runtime["librotquant_ggml_test.so"] == build["library_sha256"]
    for name, expected in runtime.items():
        # Workflow resolves symlinks; numerical receipts also name .so aliases.
        family = name.split(".so", 1)[0] + ".so"
        matches = [v for k, v in build_artifacts.items()
                   if PurePosixPath(k).name == family or PurePosixPath(k).name.startswith(family + ".")]
        assert matches and all(v["sha256"] == expected for v in matches), name
    env = report("environment", "environment.json")
    for name in ("bootstrap-before.json", "bootstrap-after.json"):
        bootstrap = report("environment", name)
        assert bootstrap["passed"] and bootstrap["torch"] == env["torch"]
        assert bootstrap["torch_file"] == env["torch_file"]
    operators = {}
    for backend in ("CPU", "CUDA0"):
        data = report("operators-" + backend)
        assert data["backend"] == backend and data["runtime_files"] == runtime
        expected_cases = {(bits, scale, width, tokens) for bits, scale in ((5, 8), (6, 16), (8, 16))
                          for width in (256, 512) for tokens in (1, 7, 33)}
        assert {(c["bits"], c["scale_bits"], c["width"], c["tokens"]) for c in data["cases"]} == expected_cases
        assert len(data["cases"]) == 18 and data["passed"] is True
        assert all(c["passed"] and 0 <= c["max_abs"] <= .001 for c in data["cases"])
        operators[backend] = {"cases": 18, "max_abs": max(c["max_abs"] for c in data["cases"])}
    def guards(data, metrics):
        for key in ("max_abs_error", "mean_abs_error", "mean_kl"):
            assert 0 <= metrics[key] <= data["thresholds"][key]
        minimum = data["thresholds"].get("top1_agreement_min", data["thresholds"].get("top1_agreement"))
        assert minimum <= metrics["top1_agreement"] <= 1
    synthetic = {}
    for bits in (6, 8):
        data = report("whole-model-w" + str(bits))
        assert data["runtime_files"] == runtime and data["backend"] == "CUDA0"
        assert data["passed"] and data["cpu_fallback_forbidden"] and data["gpu_custom_ops"] > 0
        assert all(data["guards"].values()) and set(data["by_prompt_length"]) == {"1", "4", "17", "64"}
        guards(data, data["metrics"])
        for metrics in data["by_prompt_length"].values():
            guards(data, metrics)
        synthetic[f"w{bits}"] = data["metrics"]
    conversion = report("hf-conversion")
    assert conversion["passed"] and conversion["runtime_files"] == runtime
    assert {(c["backend"], c["tokens"]) for c in conversion["cases"]} == {
        (backend, tokens) for backend in ("CPU", "CUDA0") for tokens in (4, 17, 64)}
    for case in conversion["cases"]:
        assert case["passed"]
        guards(conversion, case["metrics"])
    retained = report("retained-b5_v6_s0", "retained/report.json")
    source = report("source-b5_v6_s0", "source.json")
    exported = report("export-b5_v6_s0", "export.json")
    assert retained["passed"] and retained["runtime_files"] == runtime
    assert retained["backend"] == "CUDA0" and retained["cpu_fallback_forbidden"] and retained["gpu_custom_ops"] > 0
    assert retained["checkpoint_manifest_sha256"] == source["checkpoint_sha256"] == exported["checkpoint_manifest_sha256"]
    assert retained["probe_sha256"] == source["probe_sha256"]
    assert retained["export_sha256"] == files["stages/export-b5_v6_s0/attempt-001/export.json"]["sha256"]
    assert retained["export_files"] == exported["artifact_files"]
    size = sum(f["bytes"] for f in exported["artifact_files"].values())
    assert size == retained["complete_model_payload_bytes"] == exported["complete_model_payload_bytes"]
    assert len(exported["matrices"]) == 201
    assert sum(m["vocabulary"] for m in exported["matrices"]) == 1
    for matrix in exported["matrices"]:
        assert (matrix["bits"], matrix["scale_bits"]) == ((6, 16) if matrix["vocabulary"] else (5, 8))
    parity = retained["parity"]
    assert parity["thresholds"] == {"max_abs_error": .125, "mean_abs_error": .005,
                                    "mean_kl": .0001, "top1_agreement_min": .99}
    guards(parity, parity)
    assert all(parity["guards"].values()) and parity["positions"] == 16
    assert parity["generation_probes"] == parity["exact_generation_probes"] == source["probes"] == 4
    log = (root / "stages/retained-b5_v6_s0/attempt-001/logs/run_rq3_retained_gpu.log").read_text()
    probes = re.findall(r"native parity (p\d+): (\d+) input tokens, (\d+) logit positions, (\d+) generated tokens", log)
    assert probes == [(f"p{i}", "64", "4", "8") for i in range(4)]
    assert "offloaded 33/33 layers to GPU" in log and re.search(r"n_ctx\s+= 256\b", log)
    assert retained["timing"] == {"skipped": "not requested"}
    return {"protocol": "native-cuda-report-audit-v1", "reviewed_revision": REVISION,
            "passed_with_caveats": True, "stage_count": 11, "bundle_file_count": len(files),
            "bundle_index": files, "available_receipt_hashes_verified": len(checked),
            "source_code_hashes_verified": len(code), "recorded_patched_source_hashes_matched": len(contract["files_sha256"]),
            "recorded_external_artifacts_not_in_report_bundle": len(omitted),
            "active_minutes": total / 60, "build_minutes": summary["stages"][2]["elapsed_seconds"] / 60,
            "non_build_minutes": (total - summary["stages"][2]["elapsed_seconds"]) / 60,
            "environment": env, "operators": operators, "synthetic_models": synthetic,
            "retained_parity": parity, "model_payload_bytes": size,
            "model_payload_gb": size / 1e9, "model_payload_gib": size / 2**30,
            "text_gguf_bytes": exported["artifact_files"]["model.gguf"]["bytes"],
            "auxiliary_bytes": exported["artifact_files"]["auxiliary.safetensors"]["bytes"],
            "sampled_process_vram_mib": retained["memory"]["sampled_peak_process_vram_mib"],
            "sampled_process_vram_gib": retained["memory"]["sampled_peak_process_vram_mib"] / 1024,
            "vram_samples": retained["memory"]["samples"], "reserved_context": 256,
            "probe_shapes": probes, "timing_requested": False,
            "limits": ["Reported aggregates and hashes checked; absent logits/trace tensors prevent independent numerical recomputation.",
                       "Weights, exported GGUF and native binaries are absent from the reports-only bundle; remote hashes are corroborated, not recomputed locally.",
                       "Four short text prompts, not task accuracy, full-precision agreement, W5/W8 retained-model parity or long-context coverage.",
                       "Sampled process VRAM at context capacity 256, not a transient peak, hardware minimum, multimodal requirement or throughput benchmark."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    print(json.dumps(audit(parser.parse_args().bundle), indent=2))
