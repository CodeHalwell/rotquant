"""One bounded end-to-end native GPU validation run, with verified stage reuse.

Production Colab mode uses CUDA and the original retained checkpoint. Explicit
Metal/synthetic/current-environment switches exist only for local validation.
This runner never changes a saved model, weakens parity, or runs a task sweep.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_gpu_environment import prepare_environment
from scripts.native_gpu_workflow import Workflow, digest, fingerprint, write_json


def read(path):
    return json.loads(Path(path).read_text())


def passed(path, runtime=None):
    result = read(path)
    if result.get("passed") is not True:
        raise ValueError(f"Failed gate: {path}")
    if runtime is not None and result.get("runtime_files") != runtime:
        raise ValueError(f"Gate used a different runtime: {path}")
    return result


def runtime_files(library):
    return {p.name: digest(p) for p in sorted(Path(library).parent.iterdir()) if p.is_file()
            and (".so" in p.name or ".dylib" in p.name or p.name == "default.metallib")}


def repository_identity(allow_dirty=False):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty and not allow_dirty:
        raise ValueError("Repository has edits. Use a clean published revision, not manual notebook patches.")
    return {"revision": revision, "dirty_local_test": bool(dirty),
            "workflow_sources": {str(p.relative_to(ROOT)): digest(p) for p in [
                Path(__file__), ROOT / "scripts/native_gpu_workflow.py", ROOT / "scripts/native_hashing.py",
                ROOT / "scripts/native_gpu_environment.py",
                ROOT / "scripts/native_gpu_cache.py", ROOT / "scripts/run_rq3_performance_pilot.py",
                ROOT / "scripts/native_pilot_controls.py",
                ROOT / "scripts/run_rq3_retained_gpu.py", ROOT / "scripts/rq3_test_runtime.py",
                ROOT / "scripts/check_rq3_model.py", ROOT / "scripts/check_rq3_gpu.py",
                ROOT / "scripts/check_rq3_conversion.py", ROOT / "scripts/export_rotquant_gguf_v2.py",
                ROOT / "scripts/preflight_native_gpu.py",
                ROOT / "scripts/build_rq3_runtime.py", ROOT / "requirements/native-gpu.txt",
                ROOT / "integrations/llama.cpp/rotquant-native-v2.patch"]}}


def run_pipeline(args):
    cache_dir = getattr(args, "persistent_cache_dir", None)
    pilot = getattr(args, "performance_pilot", False)
    pilot_controls = {"decode_steps": getattr(args, "decode_steps", 32),
                      "repetitions": getattr(args, "repetitions", 3),
                      "min_decode_tps": getattr(args, "min_decode_tps", 2.),
                      "max_vram_mib": getattr(args, "max_vram_mib", 16384.)}
    if pilot and (args.synthetic_only or args.timing):
        raise ValueError("Performance pilot requires retained evidence and cannot combine with legacy --timing")
    if pilot:
        from scripts.native_pilot_controls import validate_controls
        validate_controls(128, *pilot_controls.values())
    if cache_dir and args.backend != "CUDA":
        raise ValueError("Persistent binary cache currently supports Linux CUDA only")
    if pilot or cache_dir:
        source_root = args.source_root.resolve()
        destinations = [args.output_dir, args.work_dir, *([cache_dir] if cache_dir else [])]
        for destination in destinations:
            resolved = destination.resolve()
            if resolved == source_root or source_root in resolved.parents or resolved in source_root.parents:
                raise ValueError("Results/work/cache paths must not overlap original saved evidence")
        if cache_dir:
            resolved_cache = cache_dir.resolve()
            for destination in (args.output_dir, args.work_dir):
                resolved = destination.resolve()
                if resolved == resolved_cache or resolved in resolved_cache.parents or resolved_cache in resolved.parents:
                    raise ValueError("Keep persistent cache separate from work/results directories")
    if args.backend == "CUDA" and not shutil.which("nvcc"):
        raise ValueError("Missing nvcc: select a CUDA Colab GPU runtime before starting")
    if args.backend == "Metal" and platform.system() != "Darwin":
        raise ValueError("Metal requires macOS")
    if not args.synthetic_only:
        for arm in args.arm:
            for item in ("checkpoint", "prepared.json", "preparation.json", "packed_probes.safetensors"):
                if not (args.source_root / arm / item).exists():
                    raise FileNotFoundError(f"Missing original saved evidence: {args.source_root / arm / item}")
    identity = repository_identity(args.allow_dirty_local_test)
    controls = {**identity, "source_root": str(args.source_root.resolve()), "arms": args.arm,
                "backend": args.backend, "synthetic_only": args.synthetic_only, "timing": args.timing,
                "work_dir": str(args.work_dir.resolve()), "current_environment_local_test": args.current_environment,
                "build_jobs": args.jobs,
                "metal_tensor_disable": os.environ.get("GGML_METAL_TENSOR_DISABLE")}
    if cache_dir or pilot:
        controls.update(persistent_cache_dir=str(cache_dir.resolve()) if cache_dir else None,
                        performance_pilot=pilot, pilot_controls=pilot_controls)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.work_dir / "llama-source"
    build_dir = args.work_dir / "llama-build"
    environment_dir = args.work_dir / "venv"
    python = Path(sys.executable) if args.current_environment else environment_dir / "bin/python"
    base_torch = importlib.metadata.version("torch")
    backend = "CUDA0" if args.backend == "CUDA" else "MTL0"
    old_path = os.environ.get("PATH", "")
    old_library_path = os.environ.get("LD_LIBRARY_PATH")
    try:
        with Workflow(args.output_dir, ROOT, controls, args.active_minutes) as workflow:
            def command(stage, name, *values, label=None):
                return stage.command([python, "-u", ROOT / "scripts" / name, *values], label or Path(name).stem)

            def dependencies(stage):
                if not args.current_environment:
                    prepare_environment(stage, ROOT, environment_dir, base_torch)
                    os.environ["PATH"] = str(environment_dir / "bin") + os.pathsep + old_path
                report = stage.directory / "environment.json"
                extra = ["--allow-unpinned-local-test"] if args.current_environment else []
                command(stage, "preflight_native_gpu.py", "environment", "--backend", args.backend,
                        "--expected-torch", base_torch, "--output", report, *extra)
                return passed(report), [report]
            # Recheck imports/hardware every invocation; an existing binary is
            # not proof that the current Python/CUDA environment still matches.
            environment = workflow.stage("environment", dependencies, minutes=12, reuse=False)

            source_reports = {}
            if not args.synthetic_only:
                for arm in args.arm:
                    def preflight(stage, arm=arm):
                        report = stage.directory / "source.json"
                        command(stage, "preflight_native_gpu.py", "source", "--source-arm", args.source_root / arm,
                                "--bits", "6" if "v6" in arm else "8", "--output", report)
                        return passed(report), [report]
                    source_reports[arm] = workflow.stage(f"source-{arm}", preflight, minutes=10, reuse=False)
            # Leave ample space for binaries, a retained export and staging.
            required = max(12 * 1024**3, 3 * sum(r["checkpoint_bytes"] for r in source_reports.values()))
            if shutil.disk_usage(args.work_dir).free < required:
                raise ValueError(f"Need at least {required / 1024**3:.1f} GiB free local disk before building/exporting")

            def native_build(stage):
                cache_report = stage.directory / "cache.json"
                if cache_dir:
                    command(stage, "native_gpu_cache.py", "runtime", "--source-dir", source,
                            "--build-dir", build_dir, "--cache-dir", cache_dir, "--jobs", args.jobs,
                            "--output", cache_report)
                    restored = read(cache_report)
                    receipt, library = restored["build_receipt"], Path(restored["library"])
                else:
                    command(stage, "build_rq3_runtime.py", "--source-dir", source, "--build-dir", build_dir,
                            "--backend", args.backend, "--jobs", args.jobs, "--repair-known-loader")
                    receipt = read(build_dir / "build-receipt.json")
                    library = Path(receipt["library"])
                if receipt.get("load_validated") is not True:
                    raise ValueError("Builder did not pass its native binding-load gate")
                saved = stage.directory / "build-receipt.json"
                write_json(saved, receipt)
                # Include binaries and patched sources so resume cannot trust
                # missing local cache or edited native source files.
                contract = read(ROOT / "integrations/llama.cpp/rotquant-native-v2-files.json")
                files = [saved, *[library.parent / p for p in runtime_files(library)],
                         *[source / p for p in contract["files_sha256"]]]
                if cache_dir:
                    files.append(cache_report)
                return {"library": str(library), "receipt": str(saved)}, files
            built = workflow.stage("native-build-and-load", native_build,
                                   signature=environment, minutes=45)
            library = Path(built["library"])
            build_dir.mkdir(parents=True, exist_ok=True)  # restored builds still need fixture space
            runtime = runtime_files(library)
            if cache_dir:
                # Relocated ELF dependencies must be found in this verified
                # local directory, not in an old Colab build path or Drive.
                os.environ["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + (old_library_path or "")

            # Always load the binding in a fresh process, even on a build cache hit.
            def load_gate(stage):
                stage.command([python, "-u", "-c",
                               "from pathlib import Path; import sys; from scripts.rq3_test_runtime import NativeTests; NativeTests(Path(sys.argv[1])); print('Binding load passed')",
                               library], "load-check")
                report = stage.directory / "load.json"
                write_json(report, {"passed": True, "runtime_files": runtime})
                return passed(report, runtime), [report]
            workflow.stage("binding-load", load_gate, minutes=2, reuse=False)

            def numerical(stage, script, extra):
                report = stage.directory / "report.json"
                command(stage, script, "--library", library, *extra(report))
                return passed(report, runtime), [report]
            gates = []
            for device in ("CPU", backend):
                gates.append(workflow.stage(f"operators-{device}",
                    lambda stage, device=device: numerical(stage, "check_rq3_gpu.py", lambda out: ["--backend", device, "--output", out]),
                    signature={"runtime": runtime, "environment": environment}, minutes=5, reuse=not (pilot or cache_dir)))
            for bits in (6, 8):
                def whole_model(stage, bits=bits):
                    fixture = build_dir / f"fixture-w{bits}-{fingerprint(str(stage.directory))[:12]}.gguf"
                    command(stage, "make_rq3_model_fixture.py", "--output", fixture, "--llama-dir", source,
                            "--vocabulary-bits", bits, label="make-fixture")
                    return numerical(stage, "check_rq3_model.py", lambda out: ["--model", fixture,
                                     "--backend", backend, "--output", out])
                gates.append(workflow.stage(f"whole-model-w{bits}", whole_model,
                    signature={"runtime": runtime, "environment": environment}, minutes=5, reuse=not (pilot or cache_dir)))
            gates.append(workflow.stage("hf-conversion", lambda stage: numerical(stage,
                "check_rq3_conversion.py", lambda out: ["--llama-dir", source, "--backend", backend,
                "--output-dir", build_dir / ("conversion-" + fingerprint(str(stage.directory))[:12]), "--report", out]),
                signature={"runtime": runtime, "environment": environment}, minutes=10, reuse=not (pilot or cache_dir)))
            if any(g.get("passed") is not True or g.get("runtime_files") != runtime for g in gates):
                raise ValueError("Synthetic gate failed; real model execution is forbidden")

            for arm, evidence in source_reports.items():
                def export(stage, arm=arm):
                    destination = args.work_dir / "exports" / fingerprint(str(stage.directory))[:16]
                    cache_report = stage.directory / "cache.json"
                    if cache_dir:
                        command(stage, "native_gpu_cache.py", "export", "--source-arm", args.source_root / arm,
                                "--destination", destination, "--source-dir", source,
                                "--cache-dir", cache_dir, "--output", cache_report)
                        destination = Path(read(cache_report)["directory"])
                    else:
                        command(stage, "export_rotquant_gguf_v2.py", args.source_root / arm / "checkpoint",
                                destination, "--llama-cpp-dir", source)
                    receipt = read(destination / "export.json")
                    saved = stage.directory / "export.json"
                    write_json(saved, receipt)
                    return {"directory": str(destination), "receipt": receipt}, [saved, destination / "export.json",
                            *([cache_report] if cache_dir else []),
                            *[destination / p for p in receipt["artifact_files"]]]
                exported = workflow.stage(f"export-{arm}", export, signature=evidence, minutes=15)
                def retained(stage, arm=arm, exported=exported):
                    report_dir = stage.directory / "retained"
                    extra = ["--timing"] if args.timing else []
                    command(stage, "run_rq3_retained_gpu.py", "--library", library, "--backend", backend,
                            "--source-arm", args.source_root / arm, "--export", exported["directory"],
                            "--output-dir", report_dir, *extra)
                    report = report_dir / "report.json"
                    return {**passed(report, runtime), "report_path": str(report)}, [report]
                parity = workflow.stage(f"retained-{arm}", retained,
                    signature={"runtime": runtime, "environment": environment, "source": evidence,
                               "export": exported["receipt"], "timing": args.timing}, minutes=20, reuse=not (pilot or cache_dir))
                if pilot:
                    for context in (128, 512, 2048):
                        def performance(stage, context=context, arm=arm, exported=exported, parity=parity):
                            output = stage.directory / "pilot"
                            options = [item for key, value in pilot_controls.items()
                                       for item in ("--" + key.replace("_", "-"), value)]
                            command(stage, "run_rq3_performance_pilot.py", "--library", library,
                                    "--export", exported["directory"], "--parity-report", parity["report_path"],
                                    "--probes", args.source_root / arm / "packed_probes.safetensors",
                                    "--context", context, "--output-dir", output, *options)
                            report = output / "report.json"
                            return passed(report, runtime), [report]
                        workflow.stage(f"pilot-{arm}-ctx{context}", performance,
                            signature={"runtime": runtime, "environment": environment, "source": evidence,
                                       "export": exported["receipt"], "controls": pilot_controls},
                            minutes=3, reuse=False)
    finally:
        os.environ["PATH"] = old_path
        if old_library_path is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = old_library_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=("b5_v6_s0", "b5_v8_s0"))
    parser.add_argument("--active-minutes", type=int, default=90)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--persistent-cache-dir", type=Path, help="Private Drive build/export cache; no cached GPU passes")
    parser.add_argument("--performance-pilot", action="store_true", help="Fresh parity, then three independently bounded timing contexts")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--min-decode-tps", type=float, default=2.)
    parser.add_argument("--max-vram-mib", type=float, default=16384.)
    parser.add_argument("--backend", choices=("CUDA", "Metal"), default="CUDA")
    parser.add_argument("--synthetic-only", action="store_true", help="local test only; not retained 4B validation")
    parser.add_argument("--current-environment", action="store_true", help="explicit local test; Colab always uses the managed venv")
    parser.add_argument("--allow-dirty-local-test", action="store_true")
    args = parser.parse_args()
    args.arm = args.arm or ["b5_v6_s0"]
    if len(args.arm) != len(set(args.arm)) or not 1 <= args.jobs <= 32:
        parser.error("unique arms and 1..32 build jobs required")
    # An outer notebook wrapper terminates this process group on interruption.
    # Raising here lets the inner live runner terminate its child group too.
    def terminate(signum, frame):
        raise KeyboardInterrupt("Validation interrupted; child processes will be stopped")
    signal.signal(signal.SIGTERM, terminate)
    run_pipeline(args)


if __name__ == "__main__":
    main()
