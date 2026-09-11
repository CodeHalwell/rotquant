"""No paid GPU required: cache, timing arithmetic, guards and notebook tests."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import pytest

from scripts import native_gpu_workflow as workflow
from scripts import run_rq3_performance_pilot as pilot
from scripts.build_qwen35_native_gpu_pilot_notebook import build_notebook
from scripts.native_gpu_cache import ArtifactCache, safe_name


def test_cache_roundtrip_and_compatibility_miss(tmp_path):
    cache = ArtifactCache(tmp_path / "drive")
    source = tmp_path / "library.so"
    source.write_bytes(b"fake library - never execute")
    request = {"gpu": "test", "driver": "1", "source": "abc"}
    assert cache.restore("runtime", request, tmp_path / "restore") is None
    entry = cache.publish("runtime", request, {"bin/library.so": source}, {"kind": "fixture"})
    target, record = cache.restore("runtime", request, tmp_path / "restore")
    assert (target / "bin/library.so").read_bytes() == source.read_bytes()
    assert record["metadata"]["kind"] == "fixture"
    assert cache.restore("runtime", {**request, "driver": "2"}, tmp_path / "restore") is None
    assert entry.is_dir() and (entry / "manifest.json").exists()
    assert target != entry


@pytest.mark.parametrize("corruption", ["bytes", "extra", "symlink", "missing", "request", "traversal"])
def test_cache_rejects_corruption_before_restore(tmp_path, corruption):
    cache = ArtifactCache(tmp_path / "cache")
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    entry = cache.publish("runtime", {"source": 1}, {"binary": source}, {})
    manifest = json.loads((entry / "manifest.json").read_text())
    if corruption == "bytes":
        (entry / "binary").write_bytes(b"changed")
    elif corruption == "extra":
        (entry / "extra").touch()
    elif corruption == "symlink":
        (entry / "link").symlink_to(source)
    elif corruption == "missing":
        (entry / "binary").unlink()
    elif corruption == "request":
        manifest["request"] = {"source": 2}
    else:
        manifest["files"]["../escape"] = manifest["files"].pop("binary")
    (entry / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        cache.restore("runtime", {"source": 1}, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()
    assert entry.exists()  # do not delete corrupted original evidence


@pytest.mark.parametrize("name", ["", "/abs", "../file", "a/../../file", "./file", "a//file", "a\\file", "manifest.json"])
def test_cache_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_cache_preserves_soname_aliases_as_regular_files(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    source = tmp_path / "libtest.so.1"
    source.write_bytes(b"test")
    alias = tmp_path / "libtest.so"
    alias.symlink_to(source.name)
    cache.publish("runtime", {}, {source.name: source, alias.name: alias}, {})
    target, _ = cache.restore("runtime", {}, tmp_path / "local")
    assert (target / source.name).read_bytes() == (target / alias.name).read_bytes()
    assert not (target / alias.name).is_symlink()


def test_partial_cache_is_not_reused_and_export_collision_is_preserved(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    partial = cache.root / "runtime/.incomplete-test"
    partial.mkdir(parents=True)
    (partial / "binary").write_bytes(b"partial")
    assert cache.restore("runtime", {}, tmp_path / "local") is None
    source = tmp_path / "export.gguf"
    source.write_bytes(b"one")
    entry = cache.publish("export", {}, {"model.gguf": source}, {})
    source.write_bytes(b"two")
    with pytest.raises(ValueError, match="immutable export"):
        cache.publish("export", {}, {"model.gguf": source}, {})
    assert (entry / "model.gguf").read_bytes() == b"one"


def test_runtime_cache_survives_new_work_directory_and_reloads_binding(tmp_path, monkeypatch):
    from scripts import build_rq3_runtime as builder
    from scripts import native_gpu_cache as cache
    calls = []
    monkeypatch.setattr(cache, "runtime_request", lambda: {"source": "test", "hardware": "test"})
    monkeypatch.setattr(builder, "prepare_source", lambda *a, **k: None)
    def build(source, directory, backend, jobs, **kwargs):
        calls.append("build")
        library = directory / "bin/librotquant_ggml_test.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"mock only, not an executable library")
        result = {"load_validated": True, "library": str(library), "library_sha256": workflow.digest(library)}
        workflow.write_json(directory / "build-receipt.json", result)
        return result
    monkeypatch.setattr(builder, "build", build)
    loads = []
    monkeypatch.setattr(builder, "verify_library_load", lambda path: loads.append(path))
    args = argparse.Namespace(cache_dir=tmp_path / "drive", source_dir=tmp_path / "first/source",
                              build_dir=tmp_path / "first/build", jobs=2)
    cold = cache.prepare_runtime(args)
    args.source_dir, args.build_dir = tmp_path / "second/source", tmp_path / "second/build"
    # Preserve the real environment: this helper runs in its own child in production.
    monkeypatch.setenv("LD_LIBRARY_PATH", "original")
    warm = cache.prepare_runtime(args)
    assert not cold["cache_hit"] and warm["cache_hit"]
    assert calls == ["build"] and len(loads) == 1
    assert str(tmp_path / "second") in warm["library"]
    assert workflow.digest(warm["library"]) == workflow.digest(cold["library"])


class FakeModel:
    def __init__(self):
        self.calls = []
    def evaluate(self, ids, reset=False):
        self.calls.append((len(ids), reset))
        return np.array([0., 1.], dtype=np.float32)


class FakeMemory:
    def __init__(self, peak=1000.):
        self.peak = peak
    def report(self):
        return {"sampled_peak_process_vram_mib": self.peak}


def run_fake(*, clock_step=.01, min_tps=2., peak=1000., persist=None):
    ticks = iter(np.arange(1000) * clock_step)
    model = FakeModel()
    saved = []
    def save(rows, current):
        saved.append(copy.deepcopy((rows, current)))
        if persist:
            persist(rows, current)
    rows = pilot.measure(model, np.array([0, 1]), context=128, decode=8, repetitions=2,
                         minimum_tps=min_tps, maximum_vram=16384., memory=FakeMemory(peak),
                         persist=save, clock=lambda: float(next(ticks)))
    return model, rows, saved


def test_measurement_denominators_warmup_and_partial_progress():
    model, rows, saved = run_fake()
    assert len(rows) == 3 and [r["warmup"] for r in rows] == [True, False, False]
    assert sum(reset for _, reset in model.calls) == 3
    assert sum(count == 1 for count, _ in model.calls) == 24
    summary = pilot.summarize(rows)
    assert summary["measured_repetitions"] == 2
    assert summary["decode_tokens_per_second"] == pytest.approx(100.)
    assert summary["prefill_tokens_per_second"] == pytest.approx(12800.)
    assert summary["median_decode_step_ms"] == pytest.approx(10.)
    assert any(current and current["decode_steps"] == 3 for _, current in saved)
    assert saved[-1][1] is None


def test_cost_guards_preserve_completed_measurements_and_stop():
    saved = []
    with pytest.raises(RuntimeError, match="below"):
        run_fake(clock_step=1., persist=lambda rows, current: saved.append(copy.deepcopy(rows)))
    assert len(saved[-1]) == 2  # excluded warmup plus first measured slow rep
    assert pilot.summarize(saved[-1])["decode_tokens_per_second"] == 1.
    with pytest.raises(RuntimeError, match="VRAM"):
        run_fake(peak=20000.)


def test_pilot_rejects_failed_or_changed_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "runtime_identity", lambda _: {"lib.so": "abc"})
    exported = tmp_path / "export"
    exported.mkdir()
    model = exported / "model.gguf"
    model.write_bytes(b"test model")
    probe = tmp_path / "probe"
    probe.write_bytes(b"test probes")
    files = {"model.gguf": {"sha256": workflow.digest(model), "bytes": model.stat().st_size}}
    workflow.write_json(exported / "export.json", {"artifact_files": files, "checkpoint_manifest_sha256": "checkpoint"})
    gate = {"passed": True, "parity": {"passed": True}, "runtime_files": {"lib.so": "abc"},
            "backend": "CUDA0", "cpu_fallback_forbidden": True, "gpu_custom_ops": 1,
            "export_sha256": workflow.digest(exported / "export.json"), "probe_sha256": workflow.digest(probe),
            "checkpoint_manifest_sha256": "checkpoint", "export_files": files}
    gate_path = tmp_path / "gate.json"
    workflow.write_json(gate_path, gate)
    pilot.verify_inputs(tmp_path / "lib.so", exported, gate_path, probe)
    for key, value in (("passed", False), ("runtime_files", {}), ("gpu_custom_ops", 0), ("probe_sha256", "bad")):
        workflow.write_json(gate_path, {**gate, key: value})
        with pytest.raises(ValueError, match="matching retained"):
            pilot.verify_inputs(tmp_path / "lib.so", exported, gate_path, probe)
    workflow.write_json(gate_path, gate)
    model.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="artifact changed"):
        pilot.verify_inputs(tmp_path / "lib.so", exported, gate_path, probe)


def test_notebook_is_valid_and_executes_top_to_bottom_under_mocks(tmp_path, monkeypatch):
    from scripts import colab_runtime
    notebook = build_notebook()
    nbformat.validate(notebook)
    saved = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_gpu_pilot_colab.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in notebook.cells] == [(c.cell_type, c.source) for c in saved.cells]
    content = str(tmp_path / "content")
    arm = Path(content) / "drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f/b5_v6_s0"
    (arm / "checkpoint").mkdir(parents=True)
    for name in ("prepared.json", "preparation.json", "packed_probes.safetensors"):
        (arm / name).touch()
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda _: None)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr("shutil.which", lambda _: "/fake/tool")
    def command(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            scripts = Path(cmd[-1]) / "scripts"
            scripts.mkdir(parents=True)
            for name in ("run_native_gpu_validation.py", "native_gpu_cache.py", "run_rq3_performance_pilot.py", "native_pilot_controls.py"):
                (scripts / name).touch()
    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: "" if "status" in cmd else "a" * 40)
    launches = []
    def launch(cmd, label, *, repo_dir, log_root):
        launches.append(cmd)
        root = Path(cmd[cmd.index("--output-dir") + 1])
        workflow.write_json(root / "summary.json", {"status": "mocked-only", "active_minutes": 0, "stages": []})
    monkeypatch.setattr(colab_runtime, "run_live", launch)
    scope = {}
    old_path = list(sys.path)
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                exec(compile(cell.source.replace("/content/", content + "/"), "pilot-notebook", "exec"), scope)  # noqa: S102
    finally:
        sys.path[:] = old_path
    assert len(launches) == 1 and "--performance-pilot" in launches[0]
    assert "--persistent-cache-dir" in launches[0]
    assert scope["ARMS"] == ("b5_v6_s0",)
    assert scope["MEASURED_REPETITIONS"] == 3


@pytest.mark.parametrize("values", [(128, 7, 3, 2., 16384.), (128, 32, 0, 2., 16384.),
                                      (8192, 32, 3, 2., 16384.), (128, 32, 3, float("nan"), 16384.)])
def test_pilot_controls_are_bounded(values):
    with pytest.raises(ValueError):
        pilot.validate_controls(*values)
