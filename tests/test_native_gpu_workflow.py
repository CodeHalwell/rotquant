"""Workflow/Colab orchestration tests; fake stages never count as GPU evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from scripts import native_gpu_workflow as support
from scripts import run_native_gpu_validation as pipeline


def test_native_hashes_do_not_require_python_311(tmp_path, monkeypatch):
    from scripts import build_rq3_runtime, check_rq3_model, export_rotquant_gguf_v2
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    path = tmp_path / "fixture.so"
    path.write_bytes(b"native evidence" * 100_000)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    for helper in (support.digest, build_rq3_runtime.digest, export_rotquant_gguf_v2.file_digest):
        assert helper(path) == expected
    assert check_rq3_model.runtime_identity(path) == {path.name: expected}


def test_resume_checks_artifacts_and_retains_failed_attempts(tmp_path):
    root = tmp_path / "reports"
    calls = []
    def work(stage):
        calls.append(stage.directory)
        path = stage.directory / "report.json"
        support.write_json(path, {"passed": True})
        return {"answer": 1}, [path]
    with support.Workflow(root, tmp_path, {}, 1) as workflow:
        assert workflow.stage("test", work) == {"answer": 1}
    with support.Workflow(root, tmp_path, {}, 1) as workflow:
        workflow.stage("test", work)
    assert len(calls) == 1
    (calls[0] / "report.json").write_text("tampered")
    with support.Workflow(root, tmp_path, {}, 1) as workflow:
        workflow.stage("test", work)
    assert len(calls) == 2 and (calls[0] / "report.json").read_text() == "tampered"
    with pytest.raises(ValueError, match="different-runtime"), support.Workflow(root, tmp_path, {}, 1) as workflow:
        workflow.stage("test", lambda stage: (_ for _ in ()).throw(ValueError("different-runtime")), signature=2)
    assert json.loads((root / "summary.json").read_text())["status"] == "failed"
    assert list(tmp_path.glob("reports-reports-*.zip"))


def test_idle_time_does_not_consume_budget(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(support.time, "monotonic", lambda: clock[0])
    def action(stage):
        clock[0] += 12
        return {}, []
    with support.Workflow(tmp_path / "r", tmp_path, {}, 1) as workflow:
        workflow.stage("one", action)
    clock[0] += 10_000  # user idle time between invocations
    with support.Workflow(tmp_path / "r", tmp_path, {}, 1) as workflow:
        workflow.stage("two", action)
        assert workflow.state["active_seconds"] == 24


def test_timeout_stops_future_commands_and_is_recorded(tmp_path, monkeypatch):
    def timeout(*a, **k):
        raise TimeoutError("bounded child timed out")
    monkeypatch.setattr(support, "run_live", timeout)
    with pytest.raises(TimeoutError), support.Workflow(tmp_path / "r", tmp_path, {}, 1) as workflow:
        workflow.stage("one", lambda stage: stage.command(["false"], "test"))
        pytest.fail("must not proceed past failed stage")
    report = json.loads((tmp_path / "r/workflow.json").read_text())
    assert report["attempts"][0]["status"] == "failed"
    assert "timed out" in report["error"]


def test_real_timeout_reaps_the_subprocess(tmp_path):
    pid_file = tmp_path / "pid.txt"
    with pytest.raises(TimeoutError), support.Workflow(tmp_path / "r", tmp_path, {}, 1) as workflow:
        workflow.stage("bounded", lambda stage: stage.command([
            sys.executable, "-u", "-c",
            "import os, sys, time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)",
            pid_file], "sleeper"), minutes=0.02)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_controls_and_concurrent_runs_are_guarded(tmp_path):
    with (support.Workflow(tmp_path / "r", tmp_path, {"model": "one"}, 1), pytest.raises(BlockingIOError),
          support.Workflow(tmp_path / "r", tmp_path, {"model": "one"}, 1)):
        pytest.fail("concurrent run accepted")
    with (pytest.raises(ValueError, match="controls changed"),
          support.Workflow(tmp_path / "r", tmp_path, {"model": "two"}, 1)):
        pytest.fail("changed controls accepted")


def test_reports_archive_never_includes_weights(tmp_path):
    with support.Workflow(tmp_path / "r", tmp_path, {}, 1) as workflow:
        (workflow.root / "model.gguf").write_bytes(b"weights")
        (workflow.root / "model.safetensors").write_bytes(b"weights")
    archive = next(tmp_path.glob("r-reports-*.zip"))
    with zipfile.ZipFile(archive) as handle:
        assert "summary.json" in handle.namelist()
        assert all(not name.endswith(("gguf", "safetensors")) for name in handle.namelist())


def test_orphaned_attempt_is_bounded_and_preserved(tmp_path):
    root = tmp_path / "r"
    with support.Workflow(root, tmp_path, {}, 1) as workflow:
        workflow.state["attempts"].append({"name": "killed", "status": "running", "limit_seconds": 20,
                                           "started_utc_epoch": 0})
    with support.Workflow(root, tmp_path, {}, 1) as workflow:
        assert workflow.state["attempts"][0]["status"] == "interrupted"
        assert workflow.state["active_seconds"] == 20


def fake_pipeline(tmp_path, monkeypatch, fail=None):
    args = argparse.Namespace(backend="CUDA", source_root=tmp_path / "source", arm=["b5_v6_s0"],
        output_dir=tmp_path / "reports", work_dir=tmp_path / "work", current_environment=True,
        allow_dirty_local_test=True, synthetic_only=False, timing=False, active_minutes=90, jobs=2)
    arm = args.source_root / args.arm[0]
    (arm / "checkpoint").mkdir(parents=True)
    for name in ("prepared.json", "preparation.json", "packed_probes.safetensors"):
        (arm / name).write_bytes(b"immutable source")
    monkeypatch.setattr(pipeline, "repository_identity", lambda _: {"revision": "test"})
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/fake/nvcc")
    commands = []
    def command(stage, cmd, label):
        cmd = list(map(str, cmd))
        script = Path(cmd[2]).name
        cli = cmd[3:]
        commands.append((script, cli))
        def option(name):
            return Path(cli[cli.index(name) + 1])
        library = args.work_dir / "llama-build/bin/librotquant_ggml_test.so"
        if script == "build_rq3_runtime.py":
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(b"synthetic test library, never loaded")
            contract = pipeline.read(pipeline.ROOT / "integrations/llama.cpp/rotquant-native-v2-files.json")
            for name in contract["files_sha256"]:
                path = args.work_dir / "llama-source" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source")
            support.write_json(args.work_dir / "llama-build/build-receipt.json", {"load_validated": True, "library": str(library)})
        elif script == "preflight_native_gpu.py":
            support.write_json(option("--output"), {"passed": True, "checkpoint_bytes": 1})
        elif script == "make_rq3_model_fixture.py":
            option("--output").write_bytes(b"fixture")
        elif script == "export_rotquant_gguf_v2.py":
            destination = Path(cli[1])
            destination.mkdir(parents=True)
            (destination / "model.gguf").write_bytes(b"packed weights")
            support.write_json(destination / "export.json", {"artifact_files": {"model.gguf": {"bytes": 14}}})
        elif script == "-c":
            return  # binding smoke is explicitly mocked
        else:
            if fail and fail in script:
                raise subprocess.CalledProcessError(1, cmd)
            out = option("--output-dir") / "report.json" if script in {"run_rq3_retained_gpu.py", "run_rq3_performance_pilot.py"} else option("--report" if "--report" in cli else "--output")
            support.write_json(out, {"passed": True, "runtime_files": pipeline.runtime_files(library)})
    monkeypatch.setattr(support.Stage, "command", command)
    return args, commands


def test_full_pipeline_order_and_verified_resume(tmp_path, monkeypatch):
    args, commands = fake_pipeline(tmp_path, monkeypatch)
    pipeline.run_pipeline(args)
    names = [name for name, _ in commands]
    assert names.index("preflight_native_gpu.py") < names.index("build_rq3_runtime.py")
    assert names.index("check_rq3_conversion.py") < names.index("export_rotquant_gguf_v2.py") < names.index("run_rq3_retained_gpu.py")
    assert pipeline.read(args.output_dir / "summary.json")["status"] == "passed"
    commands.clear()
    pipeline.run_pipeline(args)
    assert all(name in {"preflight_native_gpu.py", "-c"} for name, _ in commands)
    assert (args.source_root / "b5_v6_s0/prepared.json").read_bytes() == b"immutable source"


def test_targeted_context_skips_other_timings_but_not_parity(tmp_path, monkeypatch):
    args, commands = fake_pipeline(tmp_path, monkeypatch)
    args.performance_pilot = True
    args.context = [2048]
    pipeline.run_pipeline(args)
    pilots = [cli for name, cli in commands if name == "run_rq3_performance_pilot.py"]
    assert len(pilots) == 1 and pilots[0][pilots[0].index("--context") + 1] == "2048"
    assert any(name == "run_rq3_retained_gpu.py" for name, _ in commands)
    assert pipeline.read(args.output_dir / "workflow.json")["controls"]["pilot_contexts"] == [2048]


def test_numerical_failure_prevents_export_and_retained_execution(tmp_path, monkeypatch):
    args, commands = fake_pipeline(tmp_path, monkeypatch, fail="check_rq3_gpu")
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args)
    assert not any(name in {"export_rotquant_gguf_v2.py", "run_rq3_retained_gpu.py"} for name, _ in commands)
    assert pipeline.read(args.output_dir / "summary.json")["status"] == "failed"


def test_managed_environment_setup_preserves_torch_and_restores_path(tmp_path, monkeypatch):
    args, _ = fake_pipeline(tmp_path, monkeypatch)
    args.current_environment = False
    saved_command = support.Stage.command
    old_path = os.environ.get("PATH", "")
    setup = []
    # Failed ensurepip can leave this executable; its existence must not skip setup.
    partial = args.work_dir / "venv/bin/python"
    partial.parent.mkdir(parents=True)
    partial.touch()
    def command(stage, cmd, label):
        words = list(map(str, cmd))
        if words[1:3] == ["-m", "venv"]:
            assert {"--without-pip", "--system-site-packages", "--copies"} <= set(words)
            assert "--clear" not in words
            setup.append("venv")
        elif words[1:3] == ["-m", "pip"]:
            assert words[0] == sys.executable
            assert words[words.index("--python") + 1] == str(partial)
            assert {"--isolated", "--require-virtualenv"} <= set(words)
            if "--version" in words:
                setup.append("installer")
                return
            if "-c" in words:
                constraint = Path(words[words.index("-c") + 1]).read_text()
                assert constraint == f"torch=={pipeline.importlib.metadata.version('torch')}\n"
                setup.append("pinned-dependencies")
            else:
                assert "--no-deps" in words and "-e" in words
                setup.append("editable-without-dependencies")
        elif Path(words[2]).name == "native_gpu_environment.py":
            assert words[words.index("--expected-torch-file") + 1].endswith("torch/__init__.py")
            setup.append(label)
        else:
            return saved_command(stage, cmd, label)
    monkeypatch.setattr(support.Stage, "command", command)
    pipeline.run_pipeline(args)
    assert setup == ["venv", "verify-target", "installer", "pinned-dependencies",
                     "editable-without-dependencies", "verify-installed-target"]
    assert os.environ["PATH"] == old_path


def test_changed_runtime_invalidates_all_numerical_gates(tmp_path, monkeypatch):
    args, commands = fake_pipeline(tmp_path, monkeypatch)
    pipeline.run_pipeline(args)
    library = args.work_dir / "llama-build/bin/librotquant_ggml_test.so"
    library.write_bytes(b"changed")
    commands.clear()
    pipeline.run_pipeline(args)
    assert any(name == "build_rq3_runtime.py" for name, _ in commands)
    # Our mocked rebuild is byte-identical to the original. Existing numerical
    # receipts may therefore be reused, but the changed binary was not trusted.


def test_pilot_requires_fresh_parity_and_caps_each_context(tmp_path, monkeypatch):
    args, commands = fake_pipeline(tmp_path, monkeypatch)
    args.performance_pilot = True
    pipeline.run_pipeline(args)
    names = [name for name, _ in commands]
    assert names.index("run_rq3_retained_gpu.py") < names.index("run_rq3_performance_pilot.py")
    pilots = [cli for name, cli in commands if name == "run_rq3_performance_pilot.py"]
    assert [cli[cli.index("--context") + 1] for cli in pilots] == ["128", "512", "2048"]
    rows = pipeline.read(args.output_dir / "workflow.json")["attempts"]
    assert [row["limit_seconds"] for row in rows if row["name"].startswith("pilot-")] == [240, 360, 720]
    commands.clear()
    pipeline.run_pipeline(args)
    assert sum(name == "check_rq3_gpu.py" for name, _ in commands) == 2
    assert sum(name == "run_rq3_retained_gpu.py" for name, _ in commands) == 1
    assert sum(name == "run_rq3_performance_pilot.py" for name, _ in commands) == 3
    assert not any(name == "build_rq3_runtime.py" for name, _ in commands)


def test_failed_pilot_stops_larger_contexts_preserving_parity(tmp_path, monkeypatch):
    args, commands = fake_pipeline(tmp_path, monkeypatch, fail="run_rq3_performance_pilot")
    args.performance_pilot = True
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(args)
    assert sum(name == "run_rq3_performance_pilot.py" for name, _ in commands) == 1
    rows = pipeline.read(args.output_dir / "workflow.json")["attempts"]
    assert next(row for row in rows if row["name"] == "retained-b5_v6_s0")["status"] == "passed"
    assert pipeline.read(args.output_dir / "summary.json")["status"] == "failed"


def test_cached_pipeline_wiring_and_library_environment(tmp_path, monkeypatch):
    args, _ = fake_pipeline(tmp_path, monkeypatch)
    args.performance_pilot = True
    args.persistent_cache_dir = tmp_path / "private-cache"
    monkeypatch.setenv("LD_LIBRARY_PATH", "original")
    saved = support.Stage.command
    cache_calls = []
    def command(stage, cmd, label):
        words = list(map(str, cmd))
        if len(words) > 2 and Path(words[2]).name == "native_gpu_cache.py":
            def option(name):
                return Path(words[words.index(name) + 1])
            kind = words[3]
            cache_calls.append(kind)
            if kind == "runtime":
                saved(stage, [words[0], "-u", "build_rq3_runtime.py"], "mock-build")
                receipt = pipeline.read(args.work_dir / "llama-build/build-receipt.json")
                result = {"library": receipt["library"], "build_receipt": receipt, "cache_hit": False}
            else:
                destination = option("--destination")
                saved(stage, [words[0], "-u", "export_rotquant_gguf_v2.py", args.source_root / "b5_v6_s0/checkpoint",
                              destination], "mock-export")
                result = {"directory": str(destination), "cache_hit": False}
            support.write_json(option("--output"), result)
            return
        if len(words) > 2 and Path(words[2]).name in {"check_rq3_gpu.py", "run_rq3_performance_pilot.py"}:
            assert os.environ["LD_LIBRARY_PATH"].startswith(str(args.work_dir / "llama-build/bin"))
        return saved(stage, cmd, label)
    monkeypatch.setattr(support.Stage, "command", command)
    pipeline.run_pipeline(args)
    assert cache_calls == ["runtime", "export"]
    assert os.environ["LD_LIBRARY_PATH"] == "original"
    assert pipeline.read(args.output_dir / "summary.json")["status"] == "passed"


@pytest.mark.parametrize("target", ["work_dir", "output_dir", "persistent_cache_dir"])
def test_pilot_never_writes_inside_original_evidence(tmp_path, monkeypatch, target):
    args, commands = fake_pipeline(tmp_path, monkeypatch)
    args.performance_pilot = True
    setattr(args, target, args.source_root / "must-not-write-here")
    with pytest.raises(ValueError, match="overlap original"):
        pipeline.run_pipeline(args)
    assert not commands


def test_notebook_cells_execute_in_order_with_mock_colab(tmp_path, monkeypatch):
    """Execute the actual cells; hardware/network/driver are explicitly mocked."""
    from scripts import colab_runtime
    from scripts.build_qwen35_native_gpu_notebook import build_notebook
    notebook = build_notebook()
    prefix = str(tmp_path / "content")
    arm = Path(prefix) / "drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f/b5_v6_s0"
    (arm / "checkpoint").mkdir(parents=True)
    for name in ("prepared.json", "preparation.json", "packed_probes.safetensors"):
        (arm / name).touch()
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/fake/tool")
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            scripts = Path(cmd[-1]) / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "run_native_gpu_validation.py").touch()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kwargs: "" if "status" in cmd else "a" * 40)
    def launch(cmd, label, *, repo_dir, log_root):
        calls.append(cmd)
        root = Path(cmd[cmd.index("--output-dir") + 1])
        support.write_json(root / "summary.json", {"status": "mocked-orchestration-only", "active_minutes": 0, "stages": []})
    monkeypatch.setattr(colab_runtime, "run_live", launch)
    scope = {}
    original_path = list(sys.path)
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source.replace("/content/", prefix + "/"), "notebook-cell", "exec"), scope)  # noqa: S102 - execute repository-owned notebook cells under mocks
    finally:
        sys.path[:] = original_path
    assert scope["COMMIT"] == "a" * 40
    assert sum("--active-minutes" in cmd for cmd in calls) == 1
    assert scope["RUN_TIMING"] is False


def test_dependency_pins_match_preflight():
    from scripts.preflight_native_gpu import PACKAGES
    requirements = {line.split("==")[0]: line.split("==")[1] for line in
                    (pipeline.ROOT / "requirements/native-gpu.txt").read_text().splitlines()
                    if line and not line.startswith("#")}
    assert PACKAGES == requirements
