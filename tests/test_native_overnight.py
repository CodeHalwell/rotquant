"""Offline arithmetic/orchestration tests. Mocks are never GPU evidence."""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import pytest

from scripts import native_gpu_workflow as support
from scripts import native_overnight_safety as safety
from scripts import run_native_overnight as pipeline
from scripts.build_qwen35_native_overnight_notebook import build_notebook
from scripts.native_overnight_candidates import (
    CANDIDATES,
    CONTROL,
    SCRATCH_LIMIT,
    experimental_dispatch,
    requested_scratch,
)
from scripts.run_rq3_overnight_fidelity import guards
from scripts.run_rq3_overnight_screen import NumericalRejection, parity, summarize_pairs
from tests.test_native_performance_study import receipt


def test_existing_budget_limit_is_preserved(tmp_path):
    with pytest.raises(ValueError, match="180"):
        support.Workflow(tmp_path, tmp_path, {}, 480)
    for minutes in (0, 481):
        with pytest.raises(ValueError):
            support.Workflow(tmp_path, tmp_path, {}, minutes, overnight=True)


def test_resume_does_not_renew_wall_deadline(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(support.time, "time", lambda: clock[0])
    with support.Workflow(tmp_path / "r", tmp_path, {}, 10, overnight=True) as w:
        deadline = w.state["wall_deadline_epoch"]
        assert deadline == 1600
    clock[0] = 1590
    with pytest.raises(TimeoutError), support.Workflow(tmp_path / "r", tmp_path, {}, 10, overnight=True) as w:
        assert w.state["wall_deadline_epoch"] == deadline
        w.stage("never", lambda _: pytest.fail("Expired run launched work"))
    with pytest.raises(ValueError, match="controls changed"), support.Workflow(tmp_path / "r", tmp_path, {}, 20, overnight=True):
        pass


def test_wall_watchdog_restores_handler():
    previous = signal.getsignal(signal.SIGALRM)
    with pipeline.deadline_watchdog(pipeline.time.time() + 120):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        with pytest.raises(TimeoutError):
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_scratch_and_dispatch(candidate):
    for rows, width, tokens in ((2560, 9216, 2048), (137, 256, 7), (5, 256, 1)):
        assert 0 <= requested_scratch(candidate, rows, width, tokens) <= SCRATCH_LIMIT
    assert not experimental_dispatch(candidate, 2, 7)
    assert not experimental_dispatch(candidate, 0, 3)
    if CANDIDATES[candidate]["rows"]:
        with pytest.raises(ValueError, match="scratch"):
            requested_scratch(candidate, 1, 2**25, 128)


@pytest.mark.parametrize("seed", range(8))
def test_warp_fwht_and_dot_preserve_original_order(seed):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(13, 128)).astype(np.float32)
    reference = source.copy()
    lanes = np.arange(128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        lo, hi = reference[:, lanes & ~stride], reference[:, lanes | stride]
        reference = np.where(lanes & stride, lo - hi, lo + hi)
    actual = source.reshape(-1, 4, 32).copy()
    lanes = np.arange(32)
    for stride in (1, 2, 4, 8, 16):
        other = actual[:, :, lanes ^ stride]
        actual = np.where(lanes & stride, other - actual, actual + other)
    a0, a1 = actual[:, 0] + actual[:, 1], actual[:, 0] - actual[:, 1]
    a2, a3 = actual[:, 2] + actual[:, 3], actual[:, 2] - actual[:, 3]
    actual = np.stack((a0+a2, a1+a3, a0-a2, a1-a3), axis=1).reshape(-1, 128)
    np.testing.assert_array_equal(actual, reference)
    x = rng.normal(size=actual.shape).astype(np.float16).astype(np.float32)
    products = actual.astype(np.float16).astype(np.float32) * x
    a = products.copy()
    for stride in (64, 32, 16, 8, 4, 2, 1):
        a[:, :stride] += a[:, stride:2*stride]
    v = products.reshape(-1, 4, 32)
    b = (v[:, 0] + v[:, 2]) + (v[:, 1] + v[:, 3])
    for stride in (16, 8, 4, 2, 1):
        b[:, :stride] += b[:, stride:2*stride]
    np.testing.assert_array_equal(a[:, 0], b[:, 0])


@pytest.mark.parametrize("chunk", [1, 3, 128, 512])
def test_gemm_column_major_mapping_and_half_boundary(chunk):
    rng = np.random.default_rng(7)
    # Near half rounding boundaries; weights/input rounded before, not after GEMM.
    w = rng.normal(size=(137, 256)).astype(np.float16).astype(np.float32)
    x = rng.normal(size=(17, 256)).astype(np.float16).astype(np.float32)
    rp = rng.permutation(137)
    expected = (x @ w[rp].T).astype(np.float16)
    actual = np.zeros((17, 137), dtype=np.float16)
    for first in range(0, 137, chunk):
        count = min(chunk, 137-first)
        # cuBLAS transposed column-major W(K,N), non-transposed X(K,T).
        a = w[rp[first:first+count]].ravel().reshape((256, count), order="F")
        b = x.ravel().reshape((256, 17), order="F")
        c = a.T @ b
        actual[:, first:first+count] = c.T.astype(np.float16)
    np.testing.assert_allclose(actual, expected, atol=.002, rtol=.002)


def test_numerical_rejection_is_narrow():
    assert pipeline.numerical_rejection({"passed": False, "status": "numerical_rejected"})
    assert not pipeline.numerical_rejection({"passed": False, "error": "CUDA lost"})
    assert not pipeline.numerical_rejection({"passed": False, "guards": {"precision": False}})
    with pytest.raises(NumericalRejection):
        parity(np.array([1.01]), np.array([1.]), False)
    with pytest.raises(NumericalRejection):
        parity(np.array([np.nan]), np.array([1.]), False)
    with pytest.raises(NumericalRejection):
        parity(np.array([1.00001]), np.array([1.]), True)
    metrics = {"max_abs_error": .01, "mean_abs_error": .001, "mean_kl": 1e-6, "top1_agreement": 1.}
    assert all(guards(metrics, [1], [1]).values())
    assert not all(guards(metrics, [1], [2]).values())


def test_microbench_bad_samples_and_order_fail_closed():
    rows = [{"outputs": 5, "width": 256, "tokens": 4, "mode": 0, "round": i,
             "control_seconds": [2.] * 3, "candidate_seconds": [1.] * 3} for i in (0, 1)]
    assert summarize_pairs(rows)["eligible"]
    with pytest.raises(ValueError):
        summarize_pairs(rows[:1])
    rows[1]["round"] = 0
    with pytest.raises(ValueError):
        summarize_pairs(rows)
    rows[1]["round"] = 1
    rows[0]["candidate_seconds"][0] = float("nan")
    with pytest.raises(ValueError):
        summarize_pairs(rows)


def archive_fixture(tmp_path):
    root = tmp_path / "run"
    with support.Workflow(root, tmp_path, {}, 10, overnight=True) as w:
        support.write_json(root / "evidence.json", {"example": True})
        w.stage("ok", lambda _: ({}, []))
    return root


def test_release_only_after_complete_verified_archive(tmp_path):
    root = archive_fixture(tmp_path)
    calls = []
    def release():
        assert json.loads((tmp_path / "run-release.json").read_text())["verified"]
        calls.append(1)
    result = safety.finish(root, True, release=release)
    assert calls == [1] and result["release_call_returned"]
    assert not result["billing_stop_confirmed"]
    support.write_json(root / "evidence.json", {"changed": True})
    with pytest.raises(ValueError, match="verified archive"):
        safety.finish(root, True, release=release)
    assert calls == [1]


def test_release_failure_is_reported_not_success(tmp_path):
    root = archive_fixture(tmp_path)
    def unavailable():
        raise RuntimeError("provider unavailable")
    receipt = safety.finish(root, True, release=unavailable)
    assert "provider unavailable" in receipt["release_error"]
    assert "release_call_returned" not in receipt


def test_supervisor_does_not_replenish_expired_run(tmp_path, monkeypatch):
    root = archive_fixture(tmp_path)
    state = json.loads((root / "workflow.json").read_text())
    state["wall_deadline_epoch"] = 1
    support.write_json(root / "workflow.json", state)
    monkeypatch.setattr(safety, "run_live", lambda *a, **kw: pytest.fail("Must not launch expired work"))
    with pytest.raises(TimeoutError):
        safety.supervise([], root, tmp_path, 10)


def fake_pipeline(tmp_path, monkeypatch, *, reject=None, broken=False, empty=False):
    args = pipeline.parser().parse_args(["--output-dir", str(tmp_path / "reports"), "--work-dir", str(tmp_path / "work"),
        "--source-root", str(tmp_path / "original"), "--persistent-cache-dir", str(tmp_path / "cache"), "--skip-conventional"])
    monkeypatch.setattr(pipeline, "repository_identity", lambda _: {"revision": "mock-not-GPU"})
    monkeypatch.setattr(pipeline, "runtime_files", lambda _: {"lib": "test"})
    exported = tmp_path / "exported"
    exported.mkdir()
    support.write_json(exported / "export.json", {"matrices": [{"rows": 2560, "columns": 2560}]})
    parity_path = tmp_path / "parity.json"
    support.write_json(parity_path, {"passed": True})
    library = tmp_path / "lib" / "lib.so"
    library.parent.mkdir()
    library.write_bytes(b"mock")
    calls = []
    def child(root, name):
        if name == "native-build-and-load":
            return {"library": str(library)}
        if name.startswith("export-"):
            return {"directory": str(exported)}
        if name.startswith("retained-"):
            return {"report_path": str(parity_path)}
        if name.startswith(("bf16-", "ud_q4-")):
            r = receipt("reference")
            r["context"] = int(name.split("ctx")[-1])
            r["baseline"] = "bf16" if name.startswith("bf16") else "ud_q4"
            path = root / (name + ".json")
            support.write_json(path, r)
            return {"path": str(path), "report": r}
        raise AssertionError(name)
    monkeypatch.setattr(pipeline, "child_stage", child)
    def command(stage, cmd, label):
        cmd = list(map(str, cmd))
        script = next(Path(c).name for c in cmd if c.endswith(".py"))
        kernel = os.environ.get("ROTQUANT_RQ3_KERNEL", "reference")
        calls.append((script, kernel, cmd))
        def option(name):
            return cmd[cmd.index(name) + 1]
        if script == "run_native_gpu_validation.py":
            support.write_json(Path(option("--output-dir")) / "workflow.json", {"status": "passed"})
            return
        if script == "make_rq3_model_fixture.py":
            return
        directory_mode = "--output-dir" in cmd
        path = Path(option("--output-dir")) / "report.json" if directory_mode else Path(option("--output"))
        r = {"passed": True, "runtime_files": {"lib": "test"}, "settings": {"rq3_kernel": kernel}}
        if script == "run_rq3_overnight_screen.py":
            r.update(candidate=kernel, decision={"eligible": not empty, "score": 2. if kernel == "hgemm-512" else 1.5})
            if kernel == reject:
                r.update(passed=False, status="failed" if broken else "numerical_rejected")
                support.write_json(path, r)
                raise subprocess.CalledProcessError(1 if broken else 2, cmd)
        if script == "run_rq3_performance_pilot.py":
            r = receipt(kernel)
            r["context"] = int(option("--context"))
            if kernel != CONTROL:
                r["summary"]["prefill_tokens_per_second"] *= 2
            if os.environ.get("ROTQUANT_RQ3_PROFILE") == "1":
                r["measurement_kind"] = "diagnostic"
        support.write_json(path, r)
    monkeypatch.setattr(support.Stage, "command", command)
    return args, calls


def test_mock_end_to_end_all_gates_then_timings(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch)
    result = pipeline.run(args)
    assert result["completed"]
    assert len(result["control_drift"]) == 3
    assert "hgemm-512-head" in result["numerically_confirmed"]
    assert len([x for x in result["finalists"] if x != "head-warp"]) == 2
    assert len(result["comparisons"]) == 3 * len(result["numerically_confirmed"])
    commands = [name for name, _, _ in calls]
    assert commands.index("run_rq3_overnight_screen.py") < commands.index("run_rq3_performance_pilot.py")
    assert commands.index("run_rq3_overnight_fidelity.py") < commands.index("run_rq3_performance_pilot.py")
    assert (args.output_dir / "summary.json").exists()


def test_mock_numerical_rejection_checks_health_and_continues(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch, reject="sgemm-128")
    result = pipeline.run(args)
    assert result["completed"]
    assert any(name == "check_rq3_gpu.py" and kernel == CONTROL for name, kernel, _ in calls)
    assert "sgemm-128" not in result["finalists"]


def test_mock_runtime_failure_is_fatal(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch, reject="sgemm-128", broken=True)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run(args)
    assert not any(name == "run_rq3_performance_pilot.py" for name, _, _ in calls)
    assert json.loads((args.output_dir / "summary.json").read_text())["status"] == "failed"


def test_mock_fresh_conventional_comparisons(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch)
    args.skip_conventional = False
    result = pipeline.run(args)
    assert len(result["conventional_comparisons"]) == 6
    assert any("--performance-study" in cmd for _, _, cmd in calls)


def test_mock_no_candidate_stops_early(tmp_path, monkeypatch):
    args, calls = fake_pipeline(tmp_path, monkeypatch, empty=True)
    result = pipeline.run(args)
    assert result["completed"] and not result["finalists"]
    assert not any(name == "run_rq3_performance_pilot.py" for name, _, _ in calls)


def test_mock_pipeline_resume_keeps_json_controls_and_budget(tmp_path, monkeypatch):
    args, _ = fake_pipeline(tmp_path, monkeypatch, empty=True)
    pipeline.run(args)
    first = json.loads((args.output_dir / "workflow.json").read_text())
    pipeline.run(args)
    second = json.loads((args.output_dir / "workflow.json").read_text())
    assert second["wall_deadline_epoch"] == first["wall_deadline_epoch"]
    assert second["active_seconds"] >= first["active_seconds"]
    assert len(second["attempts"]) == 2 * len(first["attempts"])


def test_mock_fidelity_capture_and_candidate_rejection(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    from scripts import run_rq3_overnight_fidelity as fidelity
    exported = tmp_path / "exported"
    exported.mkdir()
    support.write_json(exported / "export.json", {"test": "not GPU evidence"})
    probes = tmp_path / "probes.safetensors"
    save_file({"p0.input.input_ids": np.array([[1, 2, 3]], dtype=np.int64)}, str(probes))
    monkeypatch.setattr(fidelity, "verified_evidence", lambda *_: (probes, {}))
    monkeypatch.setattr(fidelity, "runtime_identity", lambda _: {"lib": "mock"})
    class Model:
        custom_ops = 1
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def evaluate(self, ids, reset=False, selected=1):
            logits = np.arange(25, dtype=np.float32)
            if os.environ["ROTQUANT_RQ3_KERNEL"] != CONTROL:
                logits = logits[::-1].copy()
            return np.tile(logits, (selected, 1)) if selected > 1 else logits
        def is_eog(self, token):
            return False
    monkeypatch.setattr(fidelity, "NativeTests", lambda _: types.SimpleNamespace(model=lambda *a: Model()))
    monkeypatch.setattr(fidelity, "ExperimentDiagnostics", lambda _: types.SimpleNamespace(snapshot=lambda _: {}, require=lambda *a: {}))
    args = types.SimpleNamespace(library=tmp_path / "library", source_arm=tmp_path / "source", exported=exported,
                                 output_dir=tmp_path / "control", reference=None, logits_path=tmp_path / "logits.safetensors")
    monkeypatch.setenv("ROTQUANT_RQ3_KERNEL", CONTROL)
    assert fidelity.run(args) == 0
    original = json.loads((args.output_dir / "report.json").read_text())
    assert len(original["rows"]) == 3 and all(r["positions"] == 16 for r in original["rows"])
    args.reference = args.output_dir / "report.json"
    args.output_dir = tmp_path / "candidate"
    monkeypatch.setenv("ROTQUANT_RQ3_KERNEL", "head-warp")
    assert fidelity.run(args) == 2
    result = json.loads((args.output_dir / "report.json").read_text())
    assert result["status"] == "numerical_rejected"
    assert not result["rows"][0]["guards"]["exact_generation"]


def test_notebook_mock_run_all_without_network_or_gpu(tmp_path, monkeypatch):
    notebook = build_notebook()
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None
    namespace = {}
    for module, value in {
        "google.colab": types.SimpleNamespace(drive=types.SimpleNamespace(mount=lambda _: None)),
        "IPython.display": types.SimpleNamespace(Markdown=lambda x: x, display=lambda _: None),
    }.items():
        monkeypatch.setitem(sys.modules, module, value)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kw: "" if "status" in command else "a"*40)
    monkeypatch.setattr("shutil.which", lambda _: "/mock")
    source = tmp_path / "source/b5_v6_s0"
    source.mkdir(parents=True)
    for name in ("checkpoint", "prepared.json", "preparation.json", "packed_probes.safetensors"):
        (source / name).touch()
    repo = tmp_path / "notebook/repository/scripts"
    repo.mkdir(parents=True)
    for name in ("run_native_overnight.py", "native_overnight_safety.py", "run_rq3_overnight_screen.py", "run_rq3_overnight_fidelity.py"):
        (repo / name).touch()
    invoked = []
    monkeypatch.setattr(safety, "supervise", lambda *a, **kw: invoked.append((a, kw)))
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        text = cell.source.replace("/content/rotquant-native-overnight", str(tmp_path / "notebook"))
        text = text.replace("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f", str(source.parent))
        text = text.replace("/content/drive/MyDrive/rotquant", str(tmp_path / "drive"))
        exec(compile(text, "<offline notebook mock>", "exec"), namespace)  # noqa: S102
    assert len(invoked) == 1 and invoked[0][1]["auto_release"] is True
    assert invoked[0][0][3] == 480
