"""Opt-in bounded native runtime experiment. Never provisions or releases a GPU."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_gpu_workflow import Workflow, artifacts_match, digest, fingerprint, write_json
from scripts.native_overnight_candidates import (
    CANDIDATES,
    CONTROL,
    REGISTRY_VERSION,
    SCREEN_CANDIDATES,
)
from scripts.native_performance_study import comparison, kernel_environment
from scripts.native_pilot_controls import phase_minutes
from scripts.run_native_gpu_validation import repository_identity, runtime_files


def read(path):
    return json.loads(Path(path).read_text())


def child_stage(root, name):
    state = read(root / "workflow.json")
    if state.get("status") != "passed":
        raise ValueError("Bootstrap workflow did not pass")
    rows = [r for r in state["attempts"] if r["name"] == name]
    if not rows or rows[-1]["status"] != "passed" or not artifacts_match(rows[-1]["artifacts"]):
        raise ValueError(f"Missing or modified bootstrap artifacts: {name}")
    return rows[-1]["value"]


def numerical_rejection(record):
    """Only explicit numerical outcomes may continue. No catch-all exit-1 rule."""
    if record.get("status") == "numerical_rejected":
        return record.get("passed") is False
    guards = record.get("parity", {}).get("guards", record.get("guards"))
    return (record.get("passed") is False and isinstance(guards, dict) and bool(guards)
            and all(type(v) is bool for v in guards.values()) and not all(guards.values())
            and record.get("gpu_custom_ops", 0) > 0 and record.get("cpu_fallback_forbidden") is True)


@contextlib.contextmanager
def deadline_watchdog(deadline):
    """Interrupt Python and its run_live child cleanup before the wall deadline."""
    remaining = deadline - time.time() - 90
    if remaining <= 0:
        raise TimeoutError("Original overnight wall deadline exhausted")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError("Overnight wall deadline reached; preserving partial reports")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def plan(args):
    if not 10 <= args.active_minutes <= 480 or not 1 <= args.jobs <= 32:
        raise ValueError("Overnight permits 10..480 minutes and 1..32 build jobs")
    paths = [args.output_dir, args.work_dir, args.source_root, args.persistent_cache_dir]
    for i, path in enumerate(paths):
        for other in paths[i+1:]:
            a, b = path.resolve(), other.resolve()
            if a == b or a in b.parents or b in a.parents:
                raise ValueError("Keep evidence, work, results and private cache disjoint")
    return {"protocol": "rq3-overnight-v1", "registry": REGISTRY_VERSION,
            "candidate_specs": CANDIDATES, "screen_candidates": list(SCREEN_CANDIDATES),
            "control": CONTROL, "arm": "b5_v6_s0", "contexts": [128, 512, 2048],
            "maximum_backbone_finalists": 2, "head_separate": True,
            "active_minutes": args.active_minutes, "wall_minutes": args.active_minutes,
            "conventional_controls": not args.skip_conventional,
            "boundary": "No GPU is launched by planning. New CUDA candidates remain unvalidated until this explicit run."}


def run(args):
    controls = {**plan(args), **repository_identity(args.allow_dirty_local_test),
        "paths": {key: str(getattr(args, key).resolve()) for key in
                  ("output_dir", "work_dir", "source_root", "persistent_cache_dir")},
        "jobs": args.jobs,
        "overnight_sources": {p.name: digest(p) for p in [Path(__file__),
            *sorted((ROOT / "scripts").glob("*overnight*.py")),
            ROOT / "scripts/native_overnight_candidates.py"]}}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = {"protocol": "rq3-overnight-results-v1", "completed": False,
        "started_utc_epoch": time.time(),
        "candidates": [], "finalists": [], "comparisons": [], "conventional_comparisons": [],
        "boundary": "Runtime/numerical evidence for one unchanged W5/W6 checkpoint. No automatic promotion, FP16 teacher accuracy claim or all-quant sweep."}
    with Workflow(args.output_dir, ROOT, controls, args.active_minutes, overnight=True) as workflow:  # noqa: SIM117
        with deadline_watchdog(workflow.state["wall_deadline_epoch"]):
            def persist():
                result["active_minutes"] = workflow.state["active_seconds"] / 60
                result["wall_minutes_remaining"] = max(0., workflow.state["wall_deadline_epoch"] - time.time()) / 60
                write_json(workflow.root / "experiment.json", result)
            persist()
            arm = "b5_v6_s0"
            source_arm = args.source_root / arm
            def bootstrap(stage):
                child = stage.directory / "workflow"
                command = [sys.executable, "-u", ROOT / "scripts/run_native_gpu_validation.py",
                    "--output-dir", child, "--work-dir", args.work_dir,
                    "--source-root", args.source_root, "--persistent-cache-dir", args.persistent_cache_dir,
                    "--active-minutes", "120", "--jobs", args.jobs, "--arm", arm]
                if args.allow_dirty_local_test:
                    command.append("--allow-dirty-local-test")
                with kernel_environment(CONTROL):
                    stage.command(command, "bootstrap")
                built = child_stage(child, "native-build-and-load")
                exported = child_stage(child, f"export-{arm}")
                parity = child_stage(child, f"retained-{arm}")
                return {"library": built["library"], "exported": exported["directory"],
                        "parity": parity["report_path"], "child": str(child)}, [child / "workflow.json"]
            boot = workflow.stage("bootstrap", bootstrap, minutes=120, reuse=False)
            library, exported = Path(boot["library"]), Path(boot["exported"])
            runtime = runtime_files(library)
            python = args.work_dir / "venv/bin/python"
            # Actual checkpoint inventory is recorded separately from synthetic
            # timing fixtures. No extrapolated whole-model rate is computed.
            inventory = read(exported / "export.json").get("matrices", [])
            write_json(workflow.root / "matrix-inventory.json", inventory)
            old_ld = os.environ.get("LD_LIBRARY_PATH")
            os.environ["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + (old_ld or "")
            try:
                def command(stage, name, *values):
                    return stage.command([python, "-u", ROOT / "scripts" / name, *values], Path(name).stem)

                def gate(name, kernel, script, values, *, minutes=10, rejectable=False, filename="report.json"):
                    def action(stage):
                        path = stage.directory / filename
                        error = None
                        with kernel_environment(kernel):
                            try:
                                command(stage, script, *values(stage, path))
                            except subprocess.CalledProcessError as caught:
                                error = caught
                        if not path.is_file():
                            if error:
                                raise error
                            raise ValueError("Worker produced no receipt")
                        record = read(path)
                        recorded_runtime = record.get("runtime_files", record.get("identity", {}).get("runtime_files"))
                        recorded_kernel = record.get("candidate", record.get("settings", {}).get("rq3_kernel"))
                        if recorded_runtime != runtime or recorded_kernel != kernel:
                            raise ValueError("Worker runtime/kernel identity mismatch")
                        rejected = rejectable and numerical_rejection(record)
                        if error and not rejected:
                            raise error
                        if record.get("passed") is not True and not rejected:
                            raise ValueError("Failed or incomplete gate")
                        return {"path": str(path), "report": record, "rejected": bool(rejected)}, [path]
                    info = workflow.stage(name, action, minutes=minutes, reuse=False)
                    persist()
                    if info["rejected"]:
                        print(f"REJECTED {kernel} at {name}; checking shared baseline in a fresh process", flush=True)
                        gate(name + "-baseline-health", CONTROL, "check_rq3_gpu.py",
                            lambda s, p: ["--library", library, "--backend", "CUDA0", "--output", p])
                    return info

                def candidate_screen(kernel):
                    outcome = {"kernel": kernel, "operator_passed": False, "eligible": False}
                    result["candidates"].append(outcome)
                    for phase in ("operators", "speed"):
                        item = gate(f"{kernel}-{phase}", kernel, "run_rq3_overnight_screen.py",
                            lambda s, p, phase=phase: ["--library", library, "--candidate", kernel,
                                                      "--phase", phase, "--output", p], rejectable=True, minutes=12)
                        outcome[phase] = {"path": item["path"], "sha256": digest(item["path"])}
                        if item["rejected"]:
                            outcome["rejected_at"] = phase
                            persist()
                            return outcome
                        if phase == "operators":
                            outcome["operator_passed"] = True
                        else:
                            outcome.update(item["report"]["decision"])
                        persist()
                    return outcome

                for kernel in SCREEN_CANDIDATES:
                    candidate_screen(kernel)
                eligible = [r for r in result["candidates"] if r["eligible"]]
                backbone = sorted((r for r in eligible if CANDIDATES[r["kernel"]]["rows"]),
                                  key=lambda r: (-r["score"], r["kernel"]))[:2]
                finalists = [r["kernel"] for r in backbone]
                if any(r["kernel"] == "head-warp" for r in eligible):
                    finalists.append("head-warp")
                result["finalists"] = finalists.copy()
                persist()
                if not finalists:
                    result.update(completed=True, outcome="no_eligible_candidate; stop early")
                    persist()
                    return result

                def retained(kernel):
                    return gate(f"{kernel}-retained", kernel, "run_rq3_retained_gpu.py",
                        lambda s, p: ["--library", library, "--backend", "CUDA0", "--source-arm", source_arm,
                                      "--export", exported, "--output-dir", p.parent],
                        filename="retained/report.json", rejectable=kernel in CANDIDATES, minutes=15)

                def fidelity(kernel, reference=None):
                    return gate(f"{kernel}-fidelity", kernel, "run_rq3_overnight_fidelity.py",
                        lambda s, p: ["--library", library, "--source-arm", source_arm, "--export", exported,
                                      "--output-dir", p.parent, *(["--reference", reference] if reference else
                                       ["--logits-path", args.work_dir / f"control-logits-{fingerprint(str(s.directory))[:16]}.safetensors"])],
                        filename="fidelity/report.json", rejectable=kernel in CANDIDATES, minutes=30)

                def timing(kernel, parity, context, label, reference=None, profile=False):
                    def action(stage):
                        directory = stage.directory / "pilot"
                        options = ["--replay-report", reference, "--replay-kernel", CONTROL] if reference else []
                        with kernel_environment(kernel, profile=profile):
                            command(stage, "run_rq3_performance_pilot.py", "--library", library, "--export", exported,
                                "--parity-report", parity, "--probes", source_arm / "packed_probes.safetensors",
                                "--context", context, "--output-dir", directory, "--decode-steps", 32,
                                "--repetitions", 3, "--max-vram-mib", 16384, "--min-decode-tps", 2, *options)
                        path = directory / "report.json"
                        r = read(path)
                        if (r.get("passed") is not True or r.get("runtime_files") != runtime
                                or r.get("settings", {}).get("rq3_kernel") != kernel):
                            raise ValueError("Invalid model timing")
                        return {"path": str(path), "report": r}, [path]
                    return workflow.stage(label, action,
                        minutes=phase_minutes(context, profiling=profile), reuse=False)

                reference_fidelity = fidelity(CONTROL)["path"]
                references = {c: timing(CONTROL, boot["parity"], c, f"control-start-{c}") for c in (128, 512, 2048)}
                confirmed = {}
                # Reverse candidate order at alternate contexts; bracket every
                # group with a repeated identical-token control.
                for kernel in finalists:
                    good = True
                    for bits in (6, 8):
                        def tiny_values(stage, path, bits=bits, kernel=kernel):
                            fixture = args.work_dir / f"{kernel}-tiny-w{bits}-{fingerprint(str(stage.directory))[:12]}.gguf"
                            command(stage, "make_rq3_model_fixture.py", "--output", fixture,
                                    "--llama-dir", args.work_dir / "llama-source", "--vocabulary-bits", bits)
                            return ["--library", library, "--model", fixture, "--backend", "CUDA0", "--output", path]
                        item = gate(f"{kernel}-tiny-w{bits}", kernel, "check_rq3_model.py", tiny_values, rejectable=True)
                        if item["rejected"]:
                            good = False
                            break
                    if not good:
                        continue
                    item = retained(kernel)
                    if item["rejected"]:
                        continue
                    if fidelity(kernel, reference_fidelity)["rejected"]:
                        continue
                    confirmed[kernel] = item["path"]
                # A single frozen combination, only after BOTH components
                # independently clear retained + extended numerical gates.
                if "head-warp" in confirmed and "hgemm-512" in confirmed:
                    combined = candidate_screen("hgemm-512-head")
                    if combined["eligible"]:
                        item = retained("hgemm-512-head")
                        if not item["rejected"] and not fidelity("hgemm-512-head", reference_fidelity)["rejected"]:
                            confirmed["hgemm-512-head"] = item["path"]
                result["numerically_confirmed"] = list(confirmed)
                persist()
                for index, context in enumerate((128, 512, 2048)):
                    order = list(confirmed)
                    if index % 2:
                        order.reverse()
                    before = timing(CONTROL, boot["parity"], context, f"control-before-{context}", references[context]["path"])
                    for kernel in order:
                        item = timing(kernel, confirmed[kernel], context, f"{kernel}-ctx{context}", references[context]["path"])
                        result["comparisons"].append({"kernel": kernel, "context": context,
                            "control_path": before["path"], "candidate_path": item["path"],
                            **comparison(before["report"], item["report"])})
                        persist()
                    after = timing(CONTROL, boot["parity"], context, f"control-after-{context}", references[context]["path"])
                    result.setdefault("control_drift", []).append({"context": context, **comparison(before["report"], after["report"])})
                    persist()
                if confirmed and not args.skip_conventional:
                    def baselines(stage):
                        child = stage.directory / "workflow"
                        cmd = [sys.executable, "-u", ROOT / "scripts/run_native_gpu_validation.py",
                            "--output-dir", child, "--work-dir", args.work_dir,
                            "--source-root", args.source_root, "--persistent-cache-dir", args.persistent_cache_dir,
                            "--active-minutes", 120, "--jobs", args.jobs, "--arm", arm,
                            "--performance-study", "--study-candidate", "none", "--skip-profile"]
                        if args.allow_dirty_local_test:
                            cmd.append("--allow-dirty-local-test")
                        stage.command(cmd, "fresh-conventional-controls")
                        state = read(child / "workflow.json")
                        if state.get("status") != "passed":
                            raise ValueError("Conventional comparison workflow failed")
                        return {"root": str(child)}, [child / "workflow.json"]
                    fresh = workflow.stage("fresh-conventional", baselines, minutes=120, reuse=False)
                    for baseline in ("bf16", "ud_q4"):
                        for context in (128, 512, 2048):
                            value = child_stage(Path(fresh["root"]), f"{baseline}-{arm}-ctx{context}")
                            result["conventional_comparisons"].append({"baseline": baseline, "context": context,
                                "path": value["path"], **comparison(references[context]["report"], value["report"])})
                            persist()
                # One diagnostic pair, separate from all throughput readouts.
                if confirmed:
                    winner = max(result["comparisons"], key=lambda x: max(x["prefill_rate_ratio_to_reference"], x["decode_rate_ratio_to_reference"]))["kernel"]
                    for kernel, parity_path in ((CONTROL, boot["parity"]), (winner, confirmed[winner])):
                        timing(kernel, parity_path, 128, f"diagnostic-{kernel}", references[128]["path"], profile=True)
                result.update(completed=True, outcome="review_required; no automatic promotion")
                persist()
            finally:
                if old_ld is None:
                    os.environ.pop("LD_LIBRARY_PATH", None)
                else:
                    os.environ["LD_LIBRARY_PATH"] = old_ld
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "work-dir", "source-root", "persistent-cache-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--active-minutes", type=int, default=480)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--skip-conventional", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-dirty-local-test", action="store_true")
    return p


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Supervisor stopped run")))
    arguments = parser().parse_args()
    print(json.dumps(plan(arguments) if arguments.dry_run else run(arguments), indent=2), flush=True)
