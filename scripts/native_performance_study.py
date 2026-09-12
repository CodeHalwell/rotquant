"""Extra gates, isolated profiles and matched controls after the reference pilot."""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path

from scripts.native_gpu_workflow import digest, fingerprint, write_json
from scripts.native_pilot_controls import phase_minutes


@contextlib.contextmanager
def kernel_environment(kernel="reference", *, profile=False):
    values = {"ROTQUANT_RQ3_KERNEL": kernel,
              "ROTQUANT_RQ3_PROFILE": "1" if profile else None,
              "GGML_CUDA_DISABLE_GRAPHS": "1" if profile else None}
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def comparison(reference, candidate):
    """Only comparable complete uninstrumented runs may produce a speed ratio."""
    for r in (reference, candidate):
        if r.get("passed") is not True or r.get("measurement_kind") != "throughput":
            raise ValueError("Partial or profiled runs cannot enter a throughput comparison")
        if r["summary"]["measured_repetitions"] != r["controls"]["repetitions"]:
            raise ValueError("Incomplete measurement repetitions")
        rows = r.get("rows", [])
        if (len(rows) != r["controls"]["repetitions"] + 1
                or any(not row.get("completed") or row.get("warmup") != (i == 0)
                       or len(row.get("decode_input_ids", [])) != r["controls"]["decode_steps"]
                       for i, row in enumerate(rows))):
            raise ValueError("Incomplete measurement rows")
        values = [r["summary"][k] for k in ("prefill_tokens_per_second", "decode_tokens_per_second")]
        values.append(r["memory"].get("sampled_peak_process_vram_mib"))
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Invalid measured rates or process memory")
        if r["settings"].get("rq3_profile") is not False or r["settings"].get("cuda_graphs_disabled") is not False:
            raise ValueError("Instrumented settings cannot enter a throughput comparison")
    for key in ("context", "context_capacity", "runtime_files", "backend", "prompt_ids_sha256", "controls"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Comparison mismatch: {key}")
    a, b = dict(reference["settings"]), dict(candidate["settings"])
    a.pop("rq3_kernel", None)
    b.pop("rq3_kernel", None)
    if a != b:
        raise ValueError("Comparison settings mismatch")
    tokens = reference["rows"][1]["decode_input_ids"]
    if any(row.get("decode_input_ids") != tokens for r in (reference, candidate) for row in r["rows"]):
        raise ValueError("Comparison decode tokens differ")
    return {"prefill_rate_ratio_to_reference": candidate["summary"]["prefill_tokens_per_second"] / reference["summary"]["prefill_tokens_per_second"],
            "decode_rate_ratio_to_reference": candidate["summary"]["decode_tokens_per_second"] / reference["summary"]["decode_tokens_per_second"],
            "sampled_vram_ratio_to_reference": candidate["memory"]["sampled_peak_process_vram_mib"] / reference["memory"]["sampled_peak_process_vram_mib"],
            "boundary": "Within-run fixed-token bridge rates. Not task accuracy, a pure-kernel rate, or an equal-quality/size claim."}


def run_study(workflow, command, *, arm, library, runtime, source, build_dir, exported,
              source_arm, parity, references, controls, environment, baselines, profile):
    def read(path):
        return json.loads(Path(path).read_text())
    def require(path, kernel):
        r = read(path)
        if (r.get("passed") is not True or r.get("runtime_files") != runtime
                or r.get("settings", {}).get("rq3_kernel") != kernel):
            raise ValueError(f"Wrong runtime/kernel or failed study gate: {path}")
        return r
    signature = {"runtime": runtime, "environment": environment, "arm": arm,
                 "export": exported["receipt"], "controls": controls}
    options = [item for key, value in controls.items() for item in ("--" + key.replace("_", "-"), value)]
    # Every invocation revalidates the candidate. Neither an old reference pass
    # nor a build cache hit authorizes executing an untested candidate on 4B.
    with kernel_environment("tiled4"):
        def operators(stage):
            report = stage.directory / "report.json"
            command(stage, "check_rq3_gpu.py", "--library", library, "--backend", "CUDA0", "--output", report)
            return require(report, "tiled4"), [report]
        workflow.stage("tiled4-operators-CUDA0", operators, signature=signature, minutes=5, reuse=False)
        for bits in (6, 8):
            def synthetic(stage, bits=bits):
                fixture = build_dir / f"tiled4-w{bits}-{fingerprint(str(stage.directory))[:12]}.gguf"
                report = stage.directory / "report.json"
                command(stage, "make_rq3_model_fixture.py", "--output", fixture, "--llama-dir", source,
                        "--vocabulary-bits", bits)
                command(stage, "check_rq3_model.py", "--library", library, "--model", fixture,
                        "--backend", "CUDA0", "--output", report)
                return require(report, "tiled4"), [report]
            workflow.stage(f"tiled4-whole-model-w{bits}", synthetic, signature=signature, minutes=5, reuse=False)
        def retained(stage):
            output = stage.directory / "retained"
            command(stage, "run_rq3_retained_gpu.py", "--library", library, "--backend", "CUDA0",
                    "--source-arm", source_arm, "--export", exported["directory"], "--output-dir", output)
            path = output / "report.json"
            return {**require(path, "tiled4"), "report_path": str(path)}, [path]
        candidate_gate = workflow.stage(f"tiled4-retained-{arm}", retained, signature=signature, minutes=10, reuse=False)

    result = {"protocol": "rq3-native-performance-study-v1", "arm": arm, "completed": False,
              "environment": environment, "comparisons": [], "profiles": [],
              "boundary": "No automatic kernel or recipe promotion. CUDA profiling is diagnostic only."}
    destination = workflow.root / f"comparison-{arm}.json"
    write_json(destination, result)
    for context, reference_path in references.items():
        reference = read(reference_path)
        with kernel_environment("tiled4"):
            def timing(stage, context=context, reference_path=reference_path):
                output = stage.directory / "pilot"
                command(stage, "run_rq3_performance_pilot.py", "--library", library,
                        "--export", exported["directory"], "--parity-report", candidate_gate["report_path"],
                        "--probes", source_arm / "packed_probes.safetensors", "--context", context,
                        "--replay-report", reference_path, "--output-dir", output, *options)
                path = output / "report.json"
                return {"path": str(path), "report": require(path, "tiled4")}, [path]
            measured = workflow.stage(f"tiled4-{arm}-ctx{context}", timing, signature=signature,
                minutes=phase_minutes(context, controls["decode_steps"], controls["repetitions"]), reuse=False)
        result["comparisons"].append({"label": "tiled4", "context": context,
            "reference_report_sha256": digest(reference_path), "candidate_report_sha256": digest(measured["path"]),
            **comparison(reference, measured["report"])})
        write_json(destination, result)

    if profile:
        # The smallest SELECTED context only; a targeted 2048 run remains 2048.
        context = min(references)
        for variant, gate in (("reference", parity), ("tiled4", candidate_gate)):
            with kernel_environment(variant, profile=True):
                def diagnostic(stage, variant=variant, gate=gate):
                    output = stage.directory / "profile"
                    command(stage, "run_rq3_performance_pilot.py", "--library", library,
                            "--export", exported["directory"], "--parity-report", gate["report_path"],
                            "--probes", source_arm / "packed_probes.safetensors", "--context", context,
                            "--replay-report", references[context], "--output-dir", output, *options)
                    path = output / "report.json"
                    r = require(path, variant)
                    if r.get("measurement_kind") != "diagnostic":
                        raise ValueError("Profiling was not enabled")
                    return {"path": str(path), "report_sha256": digest(path)}, [path]
                info = workflow.stage(f"profile-{variant}-{arm}-ctx{context}", diagnostic, signature=signature,
                    minutes=phase_minutes(context, controls["decode_steps"], controls["repetitions"], profiling=True), reuse=False)
                result["profiles"].append({"kernel": variant, "context": context, **info})
                write_json(destination, result)

    with kernel_environment():
        for baseline in baselines:
            def prepare(stage, baseline=baseline):
                path = stage.directory / "baseline.json"
                command(stage, "run_native_gguf_baseline.py", "prepare", "--baseline", baseline,
                        "--artifact-dir", build_dir.parent / "conventional-ggufs", "--output", path)
                receipt = read(path)
                if receipt.get("passed") is not True:
                    raise ValueError("Baseline preparation failed")
                return {"path": str(path)}, [path, Path(receipt["path"])]
            prepared = workflow.stage(f"prepare-{baseline}", prepare, signature=signature, minutes=12)
            for context, reference_path in references.items():
                def conventional(stage, baseline=baseline, reference_path=reference_path, prepared=prepared):
                    output = stage.directory / "pilot"
                    command(stage, "run_native_gguf_baseline.py", "run", "--baseline", baseline,
                            "--baseline-receipt", prepared["path"], "--library", library,
                            "--export", exported["directory"], "--parity-report", parity["report_path"],
                            "--probes", source_arm / "packed_probes.safetensors", "--reference-report", reference_path,
                            "--llama-dir", source, "--output-dir", output)
                    path = output / "report.json"
                    return {"path": str(path), "report": require(path, "reference")}, [path]
                info = workflow.stage(f"{baseline}-{arm}-ctx{context}", conventional, signature=signature,
                    minutes=phase_minutes(context, controls["decode_steps"], controls["repetitions"]), reuse=False)
                result["comparisons"].append({"label": baseline, "context": context,
                    "reference_report_sha256": digest(reference_path), "candidate_report_sha256": digest(info["path"]),
                    **comparison(read(reference_path), info["report"])})
                write_json(destination, result)
    result["completed"] = True
    write_json(destination, result)
