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
                ROOT / "scripts/preflight_native_gpu.py",
                ROOT / "scripts/build_rq3_runtime.py", ROOT / "requirements/native-gpu.txt",
                ROOT / "integrations/llama.cpp/rotquant-native-v2.patch"]}}


def run_pipeline(args):
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
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.work_dir / "llama-source"
    build_dir = args.work_dir / "llama-build"
    environment_dir = args.work_dir / "venv"
    python = Path(sys.executable) if args.current_environment else environment_dir / "bin/python"
    base_torch = importlib.metadata.version("torch")
    backend = "CUDA0" if args.backend == "CUDA" else "MTL0"
    old_path = os.environ.get("PATH", "")
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
                command(stage, "build_rq3_runtime.py", "--source-dir", source, "--build-dir", build_dir,
                        "--backend", args.backend, "--jobs", args.jobs, "--repair-known-loader")
                receipt = read(build_dir / "build-receipt.json")
                if receipt.get("load_validated") is not True:
                    raise ValueError("Builder did not pass its native binding-load gate")
                library = Path(receipt["library"])
                saved = stage.directory / "build-receipt.json"
                write_json(saved, receipt)
                # Include binaries and patched sources so resume cannot trust
                # missing local cache or edited native source files.
                contract = read(ROOT / "integrations/llama.cpp/rotquant-native-v2-files.json")
                files = [saved, *[library.parent / p for p in runtime_files(library)],
                         *[source / p for p in contract["files_sha256"]]]
                return {"library": str(library), "receipt": str(saved)}, files
            built = workflow.stage("native-build-and-load", native_build,
                                   signature=environment, minutes=45)
            library = Path(built["library"])
            runtime = runtime_files(library)

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
                    signature={"runtime": runtime, "environment": environment}, minutes=5))
            for bits in (6, 8):
                def whole_model(stage, bits=bits):
                    fixture = build_dir / f"fixture-w{bits}-{fingerprint(str(stage.directory))[:12]}.gguf"
                    command(stage, "make_rq3_model_fixture.py", "--output", fixture, "--llama-dir", source,
                            "--vocabulary-bits", bits, label="make-fixture")
                    return numerical(stage, "check_rq3_model.py", lambda out: ["--model", fixture,
                                     "--backend", backend, "--output", out])
                gates.append(workflow.stage(f"whole-model-w{bits}", whole_model,
                    signature={"runtime": runtime, "environment": environment}, minutes=5))
            gates.append(workflow.stage("hf-conversion", lambda stage: numerical(stage,
                "check_rq3_conversion.py", lambda out: ["--llama-dir", source, "--backend", backend,
                "--output-dir", build_dir / ("conversion-" + fingerprint(str(stage.directory))[:12]), "--report", out]),
                signature={"runtime": runtime, "environment": environment}, minutes=10))
            if any(g.get("passed") is not True or g.get("runtime_files") != runtime for g in gates):
                raise ValueError("Synthetic gate failed; real model execution is forbidden")

            for arm, evidence in source_reports.items():
                def export(stage, arm=arm):
                    destination = args.work_dir / "exports" / fingerprint(str(stage.directory))[:16]
                    command(stage, "export_rotquant_gguf_v2.py", args.source_root / arm / "checkpoint",
                            destination, "--llama-cpp-dir", source)
                    receipt = read(destination / "export.json")
                    saved = stage.directory / "export.json"
                    write_json(saved, receipt)
                    return {"directory": str(destination), "receipt": receipt}, [saved, destination / "export.json",
                            *[destination / p for p in receipt["artifact_files"]]]
                exported = workflow.stage(f"export-{arm}", export, signature=evidence, minutes=15)
                def retained(stage, arm=arm, exported=exported):
                    report_dir = stage.directory / "retained"
                    extra = ["--timing"] if args.timing else []
                    command(stage, "run_rq3_retained_gpu.py", "--library", library, "--backend", backend,
                            "--source-arm", args.source_root / arm, "--export", exported["directory"],
                            "--output-dir", report_dir, *extra)
                    report = report_dir / "report.json"
                    return passed(report, runtime), [report]
                workflow.stage(f"retained-{arm}", retained,
                    signature={"runtime": runtime, "environment": environment, "source": evidence,
                               "export": exported["receipt"], "timing": args.timing}, minutes=20)
    finally:
        os.environ["PATH"] = old_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=("b5_v6_s0", "b5_v8_s0"))
    parser.add_argument("--active-minutes", type=int, default=90)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timing", action="store_true")
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
