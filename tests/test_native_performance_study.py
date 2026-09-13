"""Offline receipts/mocks only: no CUDA speed or conformance claims."""
from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import pytest

from scripts import native_performance_study as study
from scripts import run_native_gguf_baseline as baseline
from scripts import run_rq3_performance_pilot as pilot
from scripts.build_qwen35_native_followup_notebook import build_notebook as followup_notebook
from scripts.build_qwen35_native_optimization_notebook import build_notebook
from scripts.native_cuda_diagnostics import NAMES, difference
from scripts.native_gpu_workflow import digest, write_json
from scripts.native_performance_summary import tables
from scripts.native_pilot_controls import phase_minutes, selected_contexts


def receipt(kernel="reference"):
    return {"passed": True, "status": "passed", "measurement_kind": "throughput",
        "context": 128, "context_capacity": 256, "runtime_files": {"lib": "test"}, "backend": "CUDA0",
        "prompt_ids_sha256": "test", "export_sha256": "test", "probe_sha256": "test",
        "controls": {"repetitions": 2, "decode_steps": 8, "warmup_repetitions": 1,
                     "min_decode_tps": 2., "max_process_vram_mib": 16384.},
        "settings": {"rq3_kernel": kernel, "rq3_profile": False, "cuda_graphs_disabled": False},
        "rows": [{"completed": True, "warmup": i == 0, "decode_input_ids": [1] * 8} for i in range(3)],
        "summary": {"measured_repetitions": 2, "prefill_tokens_per_second": 36., "decode_tokens_per_second": 20.},
        "memory": {"sampled_peak_process_vram_mib": 3450.}}


def test_contexts_and_budgets():
    assert selected_contexts() == [128, 512, 2048]
    assert selected_contexts([2048]) == [2048]
    assert selected_contexts([512, 128]) == [128, 512]
    assert [phase_minutes(c) for c in selected_contexts()] == [4, 6, 12]
    assert [phase_minutes(c, profiling=True) for c in selected_contexts()] == [8, 12, 24]
    assert phase_minutes(2048, 64, 5, profiling=True) == 30
    for values in ([], [128, 128], [256]):
        with pytest.raises(ValueError):
            selected_contexts(values)


def test_environment_isolation_even_on_error(monkeypatch):
    monkeypatch.setenv("ROTQUANT_RQ3_KERNEL", "original")
    monkeypatch.setenv("ROTQUANT_RQ3_PROFILE", "previous")
    monkeypatch.delenv("GGML_CUDA_DISABLE_GRAPHS", raising=False)
    with pytest.raises(RuntimeError), study.kernel_environment("tiled4", profile=True):
        assert os.environ["ROTQUANT_RQ3_KERNEL"] == "tiled4"
        assert os.environ["GGML_CUDA_DISABLE_GRAPHS"] == "1"
        raise RuntimeError("mock stop")
    assert os.environ["ROTQUANT_RQ3_KERNEL"] == "original"
    assert os.environ["ROTQUANT_RQ3_PROFILE"] == "previous"
    assert "GGML_CUDA_DISABLE_GRAPHS" not in os.environ


def test_complete_matched_comparison():
    ref, candidate = receipt(), receipt("tiled4")
    candidate["summary"]["prefill_tokens_per_second"] *= 2
    assert study.comparison(ref, candidate)["prefill_rate_ratio_to_reference"] == 2
    assert pilot.reference_tokens(ref) == [1] * 8


@pytest.mark.parametrize("mutation", ["failed", "profile", "settings", "rows", "tokens", "partial", "context", "nan", "memory"])
def test_invalid_comparisons_are_rejected(mutation):
    ref, candidate = receipt(), receipt("tiled4")
    if mutation == "failed":
        candidate["passed"] = False
    elif mutation == "profile":
        candidate["measurement_kind"] = "diagnostic"
    elif mutation == "settings":
        candidate["settings"]["cuda_graphs_disabled"] = True
    elif mutation == "rows":
        candidate["rows"].pop()
    elif mutation == "tokens":
        candidate["rows"][1]["decode_input_ids"][0] = 2
    elif mutation == "partial":
        candidate["rows"][1]["completed"] = False
    elif mutation == "context":
        candidate["context"] = 512
    elif mutation == "nan":
        candidate["summary"]["decode_tokens_per_second"] = float("nan")
    else:
        candidate["memory"]["sampled_peak_process_vram_mib"] = None
    with pytest.raises(ValueError):
        study.comparison(ref, candidate)


@pytest.mark.parametrize("mutation", ["profile", "kernel", "rows", "tokens", "warmup", "negative"])
def test_reference_replay_rejects_incomplete_or_mismatched_receipts(mutation):
    ref = receipt()
    if mutation == "profile":
        ref["settings"]["rq3_profile"] = True
    elif mutation == "kernel":
        ref["settings"]["rq3_kernel"] = "tiled4"
    elif mutation == "rows":
        ref["rows"].pop()
    elif mutation == "tokens":
        ref["rows"][1]["decode_input_ids"][0] = 9
    elif mutation == "warmup":
        ref["rows"][0]["warmup"] = False
    else:
        for row in ref["rows"]:
            row["decode_input_ids"][0] = -1
    with pytest.raises(ValueError):
        pilot.reference_tokens(ref)


def test_replay_and_profiles_do_not_change_measured_call_counts():
    class Model:
        def __init__(self):
            self.calls = []
        def evaluate(self, ids, reset=False):
            self.calls.append((list(ids), reset))
            return np.array([2., 0.])  # greedy is 0; replay must feed 1
    class Memory:
        def report(self):
            return {"sampled_peak_process_vram_mib": 1000.}
    class Diagnostics:
        count = 0
        def snapshot(self):
            self.count += 1
            return {"operators": {name: {"milliseconds": self.count * 2., "host_dispatches": self.count} for name in NAMES},
                    "tiled_host_dispatches": self.count, "instrumented": True}
    ticks, saved = iter(range(1000)), []
    model = Model()
    rows = pilot.measure(model, np.array([0, 1]), context=128, decode=8, repetitions=2,
        minimum_tps=.1, maximum_vram=16384., memory=Memory(),
        persist=lambda rows, current: saved.append(copy.deepcopy(current)),
        clock=lambda: next(ticks), forced_decode=[1] * 8, diagnostics=Diagnostics())
    assert len(model.calls) == 27
    assert all(ids == [1] for ids, reset in model.calls if not reset)
    assert saved[0]["phase"] == "prefill" and "prefill_seconds" not in saved[0]
    assert rows[1]["prefill_profile"]["operators"]["matrix"]["milliseconds"] == 2.
    assert rows[1]["decode_input_ids"] == [1] * 8
    assert pilot.summarize(rows)["decode_tokens_per_second"] == 1.


def test_profile_difference():
    a = {"operators": {name: {"milliseconds": 2., "host_dispatches": 3} for name in NAMES}, "tiled_host_dispatches": 1}
    b = {"operators": {name: {"milliseconds": 8., "host_dispatches": 5} for name in NAMES}, "tiled_host_dispatches": 4, "instrumented": True}
    assert difference(a, b)["operators"]["vocabulary_head"] == {"milliseconds": 6., "host_dispatches": 2}


def test_pinned_download_and_corruption(tmp_path, monkeypatch):
    import huggingface_hub
    payload = b"fake GGUF, never executed"
    model = tmp_path / "fake.gguf"
    model.write_bytes(payload)
    release = {"name": model.name, "bytes": len(payload), "sha256": digest(model)}
    monkeypatch.setitem(baseline.RELEASES, "bf16", release)
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: calls.append(kw))
    args = types.SimpleNamespace(baseline="bf16", artifact_dir=tmp_path, output=tmp_path / "receipt.json")
    baseline.prepare(args)
    assert not calls
    assert json.loads(args.output.read_text())["revision"] == baseline.REVISION
    model.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="pinned release"):
        baseline.prepare(args)


def test_study_operator_failure_stops_candidate_model_execution(tmp_path):
    calls = []
    class Workflow:
        root = tmp_path
        def stage(self, name, action, **kwargs):
            calls.append(name)
            action(types.SimpleNamespace(directory=tmp_path / name))
    def command(stage, name, *args):
        path = Path(args[args.index("--output") + 1])
        write_json(path, {"passed": False})
    with pytest.raises(ValueError, match="failed study gate"):
        study.run_study(Workflow(), command, arm="test", library=tmp_path / "lib", runtime={},
            source=tmp_path, build_dir=tmp_path, exported={"receipt": {}}, source_arm=tmp_path,
            parity={}, references={}, controls={}, environment={}, baselines=[], profile=False)
    assert calls == ["tiled4-operators-CUDA0"]


@pytest.mark.parametrize("candidate", ["tiled4", "decode4", "none"])
def test_targeted_study_order_and_no_profile_rate_comparisons(tmp_path, candidate):
    from scripts.native_gpu_workflow import Workflow
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, receipt())
    commands = []
    def command(stage, name, *args):
        commands.append((name, os.environ.get("ROTQUANT_RQ3_KERNEL"), os.environ.get("ROTQUANT_RQ3_PROFILE")))
        def option(key):
            return Path(args[args.index(key) + 1])
        result = receipt(os.environ.get("ROTQUANT_RQ3_KERNEL", "reference"))
        if name == "make_rq3_model_fixture.py":
            return
        if name == "run_native_gguf_baseline.py" and args[0] == "prepare":
            fake = stage.directory / "test.gguf"
            fake.write_bytes(b"mock only")
            write_json(option("--output"), {"passed": True, "path": str(fake)})
            return
        if os.environ.get("ROTQUANT_RQ3_PROFILE"):
            result["measurement_kind"] = "diagnostic"
        out = option("--output-dir") / "report.json" if "--output-dir" in args else option("--output")
        write_json(out, result)
    controls = {"decode_steps": 8, "repetitions": 2, "min_decode_tps": 2., "max_vram_mib": 16384.}
    with Workflow(tmp_path / "reports", tmp_path, {}, 90) as workflow:
        study.run_study(workflow, command, arm="test", library=tmp_path / "lib", runtime={"lib": "test"},
            source=tmp_path, build_dir=tmp_path, exported={"receipt": {}, "directory": tmp_path},
            source_arm=tmp_path, parity={"report_path": reference_path}, references={128: reference_path},
            controls=controls, environment={}, baselines=["bf16", "ud_q4"],
            profile=candidate != "none", candidate=candidate)
    result = json.loads((tmp_path / "reports/comparison-test.json").read_text())
    assert result["completed"] and len(result["comparisons"]) == (2 if candidate == "none" else 3)
    assert len(result["profiles"]) == (0 if candidate == "none" else 2)
    assert all(r["context"] == 128 for r in result["comparisons"])
    assert commands[0][:2] == ("run_native_gguf_baseline.py", "reference")
    names = [name for name, _, _ in commands]
    if candidate == "none":
        assert set(names) == {"run_native_gguf_baseline.py"}
    else:
        assert ("check_rq3_gpu.py", candidate, None) in commands
        assert names.index("run_rq3_retained_gpu.py") < names.index("run_rq3_performance_pilot.py")
        assert max(i for i, n in enumerate(names) if n == "run_native_gguf_baseline.py") < names.index("check_rq3_gpu.py")


def test_conventional_run_same_bridge_replays_tokens(tmp_path, monkeypatch):
    import torch
    ref = receipt()
    exported, parity_path, probes = tmp_path / "export", tmp_path / "parity.json", tmp_path / "probes"
    parity_path.write_text("mock gate")
    ref["parity_report_sha256"] = digest(parity_path)
    ref["prompt_ids_sha256"] = baseline.hashlib.sha256(np.resize([0, 1], 128).astype("<i4").tobytes()).hexdigest()
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, ref)
    model_path = tmp_path / "fake.gguf"
    model_path.write_bytes(b"fake, never executed")
    release = {"name": model_path.name, "bytes": model_path.stat().st_size, "sha256": digest(model_path)}
    monkeypatch.setitem(baseline.RELEASES, "bf16", release)
    receipt_path = tmp_path / "baseline.json"
    write_json(receipt_path, {"passed": True, "baseline": "bf16", "release": release,
        "repo": baseline.REPO, "revision": baseline.REVISION, "path": str(model_path)})
    monkeypatch.setattr(baseline, "execution_settings", lambda: ref["settings"])
    monkeypatch.setattr(baseline, "verify_inputs", lambda *a: (ref, ref["runtime_files"]))
    monkeypatch.setattr(baseline, "token_identity", lambda *a: {"vocabulary_size": 2})
    monkeypatch.setattr(baseline, "load_file", lambda *a: {"p0.input.input_ids": torch.tensor([[0, 1]])})
    calls = []
    class Model:
        vocab, custom_ops = 2, 0
        def evaluate(self, ids, reset=False):
            calls.append((list(ids), reset))
            return np.array([2., 0.])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Memory:
        samples = (1000,)
        def report(self):
            return {"sampled_peak_process_vram_mib": 1000.}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(baseline, "NativeTests", lambda _: types.SimpleNamespace(model=lambda *a: Model()))
    monkeypatch.setattr(baseline, "ProcessVRAM", Memory)
    args = types.SimpleNamespace(output_dir=tmp_path / "result", baseline="bf16", exported=exported,
        library=tmp_path / "library", parity_report=parity_path, probes=probes, reference_report=reference_path,
        baseline_receipt=receipt_path, llama_dir=tmp_path)
    baseline.run(args)
    result = json.loads((args.output_dir / "report.json").read_text())
    assert result["passed"] and result["text_model_bytes"] == release["bytes"]
    assert len(calls) == 27 and all(ids == [1] for ids, reset in calls if not reset)
    assert study.comparison(ref, result)["decode_rate_ratio_to_reference"] > 0


def test_summary_keeps_latest_partial_and_profiles_separate(tmp_path):
    old, current = receipt(), receipt()
    current.update(passed=False, status="stopped", summary=None, rows=[])
    write_json(tmp_path / "stages/example/attempt-001/pilot/report.json", old)
    write_json(tmp_path / "stages/example/attempt-002/pilot/report.json", current)
    prof = receipt()
    prof["measurement_kind"] = "diagnostic"
    for row in prof["rows"]:
        for phase in ("prefill", "decode"):
            row[phase + "_profile"] = {"operators": {name: {"milliseconds": 7., "host_dispatches": 2} for name in NAMES}}
    write_json(tmp_path / "stages/profile/attempt-001/profile/report.json", prof)
    timing, diagnostic = tables(tmp_path)
    assert "stopped" in timing and "36.00" not in timing
    assert "matrix" in diagnostic and "7.00" in diagnostic


@pytest.mark.parametrize("followup", [False, True])
def test_optimization_notebook_executes_top_to_bottom_with_explicit_mocks(tmp_path, monkeypatch, followup):
    from scripts import colab_runtime
    notebook = followup_notebook() if followup else build_notebook()
    nbformat.validate(notebook)
    suffix = "followup" if followup else "optimization"
    run_name = "followup2" if followup else "study1"
    saved = nbformat.read(Path(__file__).resolve().parents[1] / f"notebooks/qwen35_4b_native_{suffix}_colab.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in notebook.cells] == [(c.cell_type, c.source) for c in saved.cells]
    content = str(tmp_path / "content")
    arm = Path(content) / "drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f/b5_v6_s0"
    (arm / "checkpoint").mkdir(parents=True)
    for name in ("prepared.json", "preparation.json", "packed_probes.safetensors"):
        (arm / name).touch()
    downloads, launches = [], []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda _: None)
    colab.files = types.SimpleNamespace(download=downloads.append)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ipython = types.ModuleType("IPython")
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.Markdown = lambda text: text
    ipython_display.display = lambda *objects, **kwargs: None
    ipython.display = ipython_display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    monkeypatch.setattr("shutil.which", lambda _: "/fake/tool")
    def command(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            scripts = Path(cmd[-1]) / "scripts"
            scripts.mkdir(parents=True)
            for path in (Path(__file__).resolve().parents[1] / "scripts").glob("*.py"):
                (scripts / path.name).touch()
    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: "" if "status" in cmd else "a" * 40)
    def launch(cmd, label, *, repo_dir, log_root):
        launches.append(cmd)
        root = Path(cmd[cmd.index("--output-dir") + 1])
        write_json(root / "summary.json", {"status": "mocked-only", "active_minutes": 0, "stages": []})
        if followup:
            write_json(root / "stages/conventional-preflight-q4_0/attempt-001/report.json", {
                "format": "q4_0", "passed": True, "gpu_executed": False,
                "cross_backend_numerics_role": "mock only, no GPU executed",
                "same_backend": {"CPU": {"passed": True, "metrics": {}}},
                "cross_backend": {"public_api": {"passed": False, "metrics": {}}},
            })
        (root.parent / f"{run_name}-reports-123.zip").touch()
    monkeypatch.setattr(colab_runtime, "run_live", launch)
    scope, old_path = {}, list(sys.path)
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                exec(compile(cell.source.replace("/content/", content + "/"), "mock-study", "exec"), scope)  # noqa: S102
    finally:
        sys.path[:] = old_path
    assert len(launches) == 1 and "--performance-study" in launches[0]
    assert launches[0].count("--context") == (2 if followup else 3) and launches[0].count("--baseline") == 2
    assert len(downloads) == 1 and downloads[0].endswith(f"{run_name}-reports-123.zip")
    if followup:
        assert "diagnostic" in notebook.cells[5].source
        assert "2c4037e76f71" in notebook.cells[5].source
        assert "--skip-profile" in launches[0]
        assert launches[0][launches[0].index("--study-candidate") + 1] == "decode4"


def test_decode4_preserves_128_lane_reduction_order_in_float32():
    # Arithmetic model, not execution of the CUDA kernel. Adversarial magnitudes
    # make a different association observable even when rounded inputs match.
    rng = np.random.default_rng(7041)
    products = (rng.standard_normal((19, 86, 128)) * rng.choice([.0001, 1., 10000.], (19, 86, 128))).astype(np.float32)
    accumulated = np.zeros((19, 128), dtype=np.float32)
    for group in range(products.shape[1]):
        accumulated += products[:, group]
    reference = accumulated.copy()
    for stride in (64, 32, 16, 8, 4, 2, 1):
        reference[:, :stride] += reference[:, stride:2 * stride]
    candidate = (accumulated[:, :32] + accumulated[:, 64:96]) + (accumulated[:, 32:64] + accumulated[:, 96:128])
    for stride in (16, 8, 4, 2, 1):
        candidate[:, :stride] += candidate[:, stride:2 * stride]
    np.testing.assert_array_equal(candidate[:, 0].view(np.uint32), reference[:, 0].view(np.uint32))


@pytest.mark.parametrize("baselines,profile", [([], False), (["bf16"], True)])
def test_baseline_only_rejects_unusable_controls(baselines, profile):
    with pytest.raises(ValueError, match="Baseline-only"):
        study.run_study(None, None, arm=None, library=None, runtime=None, source=None,
            build_dir=None, exported=None, source_arm=None, parity=None, references=None,
            controls=None, environment=None, baselines=baselines, profile=profile, candidate="none")
