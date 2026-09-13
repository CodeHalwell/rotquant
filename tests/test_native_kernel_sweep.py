"""CPU arithmetic, receipt and explicit orchestration mocks; never GPU evidence."""
from __future__ import annotations

import ast
import copy
import ctypes as C
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import pytest

from rotquant.native_v3 import NativeV3Matrix, decode_native_v3_rows
from scripts import native_kernel_sweep as sweep
from scripts import native_performance_study as study
from scripts import run_rq3_kernel_screen as screen
from scripts.build_qwen35_native_backbone_notebook import build_notebook
from scripts.check_rq3_dispatch import require_dispatch
from scripts.native_gpu_workflow import Workflow, digest, write_json
from scripts.native_kernel_candidates import W5_TILES
from scripts.native_performance_summary import tables
from scripts.rq3_test_runtime import NativeTests
from scripts.run_rq3_performance_pilot import reference_tokens
from tests.test_native_gpu_workflow import fake_pipeline, pipeline, support
from tests.test_native_performance_study import receipt


def rows_for_ratio(ratio=1.2):
    return [{"rows": r, "width": c, "tokens": t, "round": turn, "exact_decode4": True,
             "decode4_seconds": [.01] * screen.REPEATS, "candidate_seconds": [.01 / ratio] * screen.REPEATS}
            for r, c in screen.SHAPES for t in screen.TOKEN_COUNTS for turn in range(screen.ROUNDS)]


@pytest.mark.parametrize("ratio,eligible", [(1.2, True), (1., False), (.94, False)])
def test_screen_eligibility_is_not_automatic_promotion(ratio, eligible):
    result = screen.decision(rows_for_ratio(ratio))
    assert result["eligible"] is eligible
    assert result["score"] == pytest.approx(ratio)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "zero", "parity", "samples"])
def test_invalid_screen_fails_closed(fault):
    rows = rows_for_ratio()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault in ("nan", "zero"):
        rows[0]["candidate_seconds"][0] = float("nan") if fault == "nan" else 0.
    elif fault == "parity":
        rows[0]["exact_decode4"] = False
    else:
        rows[0]["decode4_seconds"].pop()
    with pytest.raises(ValueError):
        screen.decision(rows)


def test_one_regressing_case_blocks_good_average():
    rows = rows_for_ratio(2.)
    rows[0]["candidate_seconds"] = [.1] * screen.REPEATS
    assert not screen.decision(rows)["eligible"]


@pytest.mark.parametrize("candidates,maximum", [([], 1), (["w5s8", "w5s8"], 1), (["unknown"], 1), (["w5s8"], 3)])
def test_sweep_controls_are_bounded(candidates, maximum):
    with pytest.raises(ValueError):
        sweep.validate_candidates(candidates, maximum)


@pytest.mark.parametrize("rows,width", [(1, 128), (5, 256), (7, 512)])
def test_w5_byte_unpack_matches_canonical_words_and_scale_metadata(rows, width):
    matrix = screen.fixture(rows, width)
    assert NativeV3Matrix.from_bytes(matrix.to_bytes()).to_bytes() == matrix.to_bytes()
    data = matrix.words.view(np.uint8)
    decoded = np.empty((rows, width), dtype=np.float32)
    # Independent CPU model of the specialized CUDA byte/scale addressing.
    for r in range(rows):
        for c in range(width):
            byte, shift = ((r * width + c) * 5) >> 3, (c * 5) & 7
            code = int(data[byte]) >> shift
            if shift > 3:
                code |= int(data[byte + 1]) << (8 - shift)
            index = r * (width // 128) + c // 128
            scale = np.float32(matrix.offsets[index >> 8]) + (
                np.float32(matrix.scales.flat[index]) * np.float32(matrix.steps[index >> 8]))
            decoded[r, c] = matrix.codebook[code & 31] * scale
    np.testing.assert_array_equal(decoded, decode_native_v3_rows(matrix))


def test_w5_metadata_crosses_block256():
    matrix = screen.fixture(137, 512)
    indices = np.arange(matrix.layout.scale_count)
    values = matrix.offsets[indices >> 8].astype(np.float32) + (
        matrix.scales.ravel().astype(np.float32) * matrix.steps[indices >> 8].astype(np.float32))
    np.testing.assert_array_equal(values, matrix.decoded_scales(0, len(values)))


@pytest.mark.parametrize("kernel", W5_TILES)
@pytest.mark.parametrize("tokens", [1, 3, 4, 7, 8, 9, 15, 16, 17])
def test_candidate_requires_its_own_fresh_tile_counter(kernel, tokens):
    active = tokens == 1 or tokens >= 4
    delta = {"operators": {"matrix": {"host_dispatches": 1}},
             "decode_host_dispatches": int(tokens == 1), "tiled_host_dispatches": int(tokens >= 4),
             "w5_host_dispatches": {str(t): int(active and t == W5_TILES[kernel]) for t in (4, 8, 16)}}
    require_dispatch(delta, kernel, tokens)
    bad = copy.deepcopy(delta)
    bad["w5_host_dispatches"] = {}
    with pytest.raises(ValueError, match="Missing W5"):
        require_dispatch(bad, kernel, tokens)
    if active:
        delta["w5_host_dispatches"][str(W5_TILES[kernel])] = 0
        with pytest.raises(ValueError, match="Wrong W5"):
            require_dispatch(delta, kernel, tokens)


@pytest.mark.parametrize("fault", [None, "output", "timing", "counter"])
def test_resident_screen_pairs_order_and_persists_failure(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(screen, "SHAPES", ((5, 256),))
    state = {"operators": {n: {"milliseconds": 0., "host_dispatches": 0} for n in (
        "rotation", "matrix", "vocabulary_head", "embedding")},
        "decode_host_dispatches": 0, "tiled_host_dispatches": 0,
        "w5_host_dispatches": {str(t): 0 for t in (4, 8, 16)}, "instrumented": False}
    calls = []
    class Counters:
        def __init__(self):
            self.binding = {"mock": True}
        def snapshot(self):
            return copy.deepcopy(state)
    def operator(backend, matrix, signs, inputs, benchmark):
        kernel = os.environ["ROTQUANT_RQ3_KERNEL"]
        calls.append((kernel, len(inputs)))
        assert backend == "CUDA0" and benchmark == (screen.REPEATS, screen.ITERATIONS)
        assert os.environ["GGML_CUDA_DISABLE_GRAPHS"] == "1"
        assert "ROTQUANT_RQ3_PROFILE" not in os.environ
        state["operators"]["matrix"]["host_dispatches"] += 31
        state["decode_host_dispatches" if len(inputs) == 1 else "tiled_host_dispatches"] += 31
        if kernel in W5_TILES and fault != "counter":
            state["w5_host_dispatches"][str(W5_TILES[kernel])] += 31
        output = np.zeros((len(inputs), 5), dtype=np.float32)
        if kernel in W5_TILES and fault == "output":
            output[0, 0] = 1
        seconds = .001 if kernel == "decode4" else .0005
        if fault == "timing":
            seconds = float("nan")
        return output, [seconds] * screen.REPEATS
    monkeypatch.setattr(screen, "NativeTests", lambda _: types.SimpleNamespace(operator=operator))
    monkeypatch.setattr(screen, "CudaDiagnostics", lambda _: Counters())
    monkeypatch.setattr(screen, "runtime_identity", lambda _: {"mock": "runtime"})
    output = tmp_path / "report.json"
    monkeypatch.setenv("ROTQUANT_RQ3_KERNEL", "before")
    if fault:
        with pytest.raises((ValueError, AssertionError)):
            screen.check(tmp_path / "lib", "w5s8", output)
        assert json.loads(output.read_text())["status"] == "failed"
    else:
        result = screen.check(tmp_path / "lib", "w5s8", output)
        assert result["passed"] and result["decision"]["eligible"]
        assert calls[:4] == [("decode4", 1), ("w5s8", 1), ("w5s8", 1), ("decode4", 1)]
        assert len(result["rows"]) == 4
    assert os.environ["ROTQUANT_RQ3_KERNEL"] == "before"


def test_explicit_decode4_replay_not_silently_accepted_as_original_reference():
    r = receipt("decode4")
    with pytest.raises(ValueError):
        reference_tokens(r)
    assert reference_tokens(r, "decode4") == [1] * 8
    r["settings"]["rq3_profile"] = True
    with pytest.raises(ValueError):
        reference_tokens(r, "decode4")


@pytest.mark.parametrize("key", ["export_sha256", "probe_sha256"])
def test_paired_model_timing_requires_same_artifact_and_probes(key):
    a, b = receipt("decode4"), receipt("w5s8")
    b[key] = "different"
    with pytest.raises(ValueError, match=key):
        study.comparison(a, b)


def test_native_benchmark_binding_returns_samples_not_allocation_time():
    native = NativeTests.__new__(NativeTests)
    def evaluate(*args):
        raise AssertionError("wrong entry point")
    evaluate.argtypes = [C.c_char_p] + [C.c_void_p] * 11
    def benchmark(*args):
        assert args[-3:-1] == (3, 10)
        assert len(args) == 15 and args[11] > 4096
        output = np.ctypeslib.as_array(C.cast(args[11], C.POINTER(C.c_float)), shape=(5,))
        output[:] = .25
        samples = np.ctypeslib.as_array(C.cast(args[-1], C.POINTER(C.c_double)), shape=(3,))
        samples[:] = [.001, .002, .003]
        return 0
    native.lib = types.SimpleNamespace(rq3_test_eval=evaluate, rq3_test_benchmark=benchmark)
    output, seconds = native.operator("CPU", screen.fixture(5, 256), np.ones(256, dtype=np.int8),
                                      np.zeros((1, 256), dtype=np.float32), benchmark=(3, 10))
    assert seconds == [.001, .002, .003]
    assert (output == .25).all()
    with pytest.raises(ValueError):
        native.operator("CPU", screen.fixture(5, 256), np.ones(256, dtype=np.int8),
                        np.zeros((1, 256), dtype=np.float32), benchmark=(0, 10))


@pytest.mark.parametrize("eligible", [True, False])
def test_shortlist_caps_full_model_work_and_keeps_all_screen_evidence(tmp_path, eligible):
    commands = []
    def command(stage, name, *args):
        commands.append(name)
        path = Path(args[args.index("--output") + 1])
        if name == "check_rq3_gpu.py":
            result = {"passed": True, "runtime_files": {"mock": 1},
                      "settings": {"rq3_kernel": os.environ["ROTQUANT_RQ3_KERNEL"]}}
        else:
            candidate = args[args.index("--candidate") + 1]
            rows = rows_for_ratio(1.2 if eligible else 1.)
            result = {"passed": True, "runtime_files": {"mock": 1}, "candidate": candidate,
                      "baseline": "decode4", "rows": rows, "decision": screen.decision(rows)}
        write_json(path, result)
    with Workflow(tmp_path / "reports", tmp_path, {}, 90) as workflow:
        finalists = sweep.run_screen(workflow, command, tmp_path / "lib", {"mock": 1}, list(W5_TILES), 2)
    assert len(finalists) == (2 if eligible else 0)
    result = json.loads((tmp_path / "reports/kernel-shortlist.json").read_text())
    assert result["completed"] and len(result["candidates"]) == 3
    assert commands == ["check_rq3_gpu.py", *[x for _ in range(3) for x in ("check_rq3_gpu.py", "run_rq3_kernel_screen.py")]]


def test_screen_failure_prevents_later_candidates(tmp_path):
    calls = []
    def command(stage, name, *args):
        calls.append(name)
        write_json(Path(args[args.index("--output") + 1]), {"passed": False})
    with pytest.raises(ValueError), Workflow(tmp_path / "reports", tmp_path, {}, 90) as workflow:
        sweep.run_screen(workflow, command, tmp_path / "lib", {}, list(W5_TILES), 2)
    assert calls == ["check_rq3_gpu.py"]
    shortlist = json.loads((tmp_path / "reports/kernel-shortlist.json").read_text())
    assert not shortlist["completed"] and not shortlist["finalists"]


def test_resumed_readout_excludes_older_screen_model_timings(tmp_path):
    write_json(tmp_path / "stages/old/attempt-001/pilot/report.json", receipt("w5s8"))
    write_json(tmp_path / "stages/current/attempt-001/pilot/report.json", receipt("decode4"))
    # Directories in a downloaded archive retain their original Colab paths.
    write_json(tmp_path / "workflow.json", {"attempts": [
        {"name": name, "directory": f"/content/reports/stages/{name}/attempt-001", "started_utc_epoch": epoch}
        for name, epoch in (("old", 1), ("current", 3))]})
    timing, _ = tables(tmp_path, since_epoch=2)
    assert "current" in timing and "old" not in timing
    timing, _ = tables(tmp_path, since_epoch=4)
    assert "current" not in timing and "old" not in timing


@pytest.mark.parametrize("promising", [False, True])
def test_pipeline_screens_before_model_and_uses_decode4_baseline(tmp_path, monkeypatch, promising):
    args, commands = fake_pipeline(tmp_path, monkeypatch)
    args.kernel_sweep, args.context = True, [128]
    studies, events = [], []
    original = support.Stage.command
    def command(stage, cmd, label):
        events.append((stage.directory.parent.name, os.environ.get("ROTQUANT_RQ3_KERNEL")))
        return original(stage, cmd, label)
    monkeypatch.setattr(support.Stage, "command", command)
    def shortlist(*args):
        assert not any(name == "export_rotquant_gguf_v2.py" for name, _ in commands)
        events.append(("screen", os.environ["ROTQUANT_RQ3_KERNEL"]))
        write_json(args[0].root / "kernel-shortlist.json", {"mock": True})
        return ["w5s8"] if promising else []
    monkeypatch.setattr(sweep, "run_screen", shortlist)
    monkeypatch.setattr(study, "run_study", lambda *a, **kw: studies.append(kw))
    monkeypatch.setenv("ROTQUANT_RQ3_KERNEL", "original")
    pipeline.run_pipeline(args)
    assert os.environ["ROTQUANT_RQ3_KERNEL"] == "original"
    names = [name for name, _ in commands]
    if promising:
        assert len(studies) == 1 and studies[0]["baseline_kernel"] == "decode4"
        assert not studies[0]["profile"] and not studies[0]["baselines"]
        assert all(kernel == "decode4" for name, kernel in events if name.startswith(("retained-", "pilot-")))
    else:
        assert not studies
        assert "export_rotquant_gguf_v2.py" not in names and "run_rq3_retained_gpu.py" not in names


@pytest.mark.parametrize("candidate", W5_TILES)
def test_model_confirmation_is_paired_with_decode4(tmp_path, candidate):
    refpath = tmp_path / "decode4.json"
    write_json(refpath, receipt("decode4"))
    commands = []
    def command(stage, name, *args):
        commands.append((name, args))
        if name == "make_rq3_model_fixture.py":
            return
        result = receipt(os.environ["ROTQUANT_RQ3_KERNEL"])
        path = Path(args[args.index("--output-dir") + 1]) / "report.json" if "--output-dir" in args else Path(args[args.index("--output") + 1])
        write_json(path, result)
    with Workflow(tmp_path / "reports", tmp_path, {}, 90) as workflow:
        study.run_study(workflow, command, arm="test", library=tmp_path / "lib", runtime={"lib": "test"},
            source=tmp_path, build_dir=tmp_path, exported={"receipt": {}, "directory": tmp_path},
            source_arm=tmp_path, parity={"report_path": refpath}, references={128: refpath},
            controls={"decode_steps": 8, "repetitions": 2, "min_decode_tps": 2., "max_vram_mib": 16384.},
            environment={}, baselines=[], profile=False, candidate=candidate, baseline_kernel="decode4")
    report = json.loads((tmp_path / f"reports/comparison-test-{candidate}.json").read_text())
    assert report["completed"] and report["baseline_kernel"] == "decode4"
    assert report["comparisons"][0]["reference_report_sha256"] == digest(refpath)
    names = [name for name, _ in commands]
    assert names.index("run_rq3_retained_gpu.py") < names.index("run_rq3_performance_pilot.py")
    for name, args in commands:
        if name == "run_rq3_performance_pilot.py":
            assert args[args.index("--replay-kernel") + 1] == "decode4"


def test_backbone_notebook_executes_top_to_bottom_with_explicit_mocks(tmp_path, monkeypatch):
    from scripts import colab_runtime
    notebook = build_notebook()
    nbformat.validate(notebook)
    saved = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_backbone_colab.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in notebook.cells] == [(c.cell_type, c.source) for c in saved.cells]
    content = str(tmp_path / "content")
    arm = Path(content) / "drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f/b5_v6_s0"
    (arm / "checkpoint").mkdir(parents=True)
    for name in ("prepared.json", "preparation.json", "packed_probes.safetensors"):
        (arm / name).touch()
    downloads, launches, rendered = [], [], []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda _: None)
    colab.files = types.SimpleNamespace(download=downloads.append)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    display = types.ModuleType("IPython.display")
    display.Markdown = lambda text: text
    display.display = rendered.append
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr("shutil.which", lambda _: "/fake/tool")
    def command(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            scripts = Path(cmd[-1]) / "scripts"
            scripts.mkdir(parents=True)
            for path in (Path(__file__).resolve().parents[1] / "scripts").glob("*.py"):
                (scripts / path.name).touch()
    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: "" if "status" in cmd else "a" * 40)
    def launch(cmd, label, **kwargs):
        launches.append(cmd)
        root = Path(cmd[cmd.index("--output-dir") + 1])
        write_json(root / "summary.json", {"status": "mock-only", "active_minutes": 0, "stages": []})
        write_json(root / "kernel-shortlist.json", {"completed": True, "outcome": "mock-only", "finalists": ["w5s8"], "started_utc_epoch": 1,
            "candidates": [{"kernel": "w5s8", **screen.decision(rows_for_ratio())}]})
        (root.parent / "backbone1-reports-123.zip").touch()
    monkeypatch.setattr(colab_runtime, "run_live", launch)
    scope, original_path = {}, list(sys.path)
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                exec(compile(cell.source.replace("/content/", content + "/"), "mock-backbone", "exec"), scope)  # noqa: S102
    finally:
        sys.path[:] = original_path
    assert len(launches) == 1 and "--kernel-sweep" in launches[0]
    assert "--performance-study" not in launches[0] and "--baseline" not in launches[0]
    assert launches[0].count("--kernel-candidate") == 3 and launches[0].count("--context") == 1
    assert len(downloads) == 1 and len(rendered) == 2
    assert scope["MAX_FINALISTS"] == 2 and scope["ACTIVE_BUDGET_MINUTES"] == 90
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in saved.cells if c.cell_type == "code")
