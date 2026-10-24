"""Small real file receipts exercise continuation without model/GPU execution."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.native_gpu_workflow import artifacts, fingerprint, write_json
from scripts.run_qwen35_ternary_research import TernaryStudy
from scripts.ternary_research_continuation import (
    MODULE,
    PARITY_WORKER,
    SOURCE_CHANGES,
    HashVerifier,
    continuation,
    runtime_provenance,
)
from scripts.ternary_research_plan import PROTOCOL, arm_record, arms


def read(path):
    return json.loads(path.read_text())


def measurement(label):
    quality = {"summary": {"ppl": 2., "mean_teacher_kl": 0.}, "rows": [{"id": "doc"}]}
    return {"status": "complete", "label": label, "backend": "CUDA0/reference/chunk16",
            "data_fingerprint": "frozen", "runtime": {"libnative.so": "original-binary"},
            "quality": {"test": quality, "long_test": quality},
            "tasks": {"summary": {"math": {"accuracy": 1., "correct": 1, "items": 1}},
                      "rows": [{"id": "task", "benchmark": "math", "input_hash": "fixed",
                                "max_new_tokens": 32, "task_success": 1.}]},
            "speed": {"completed": True, "prefill_chunk_tokens": 16, "rows": [],
                      "summary": {"128": {"decode_tokens_per_second": 5.}}}}


@pytest.fixture
def parent_run(tmp_path):
    root = tmp_path / "parent"
    config = {"research": {"task_margin": .01}, "bonsai": {"packing": "PTQ1_0"}}
    old = {"protocol": PROTOCOL, "config": config,
           "runtime": {"gpu": "A100", "python": "3.13.15", "versions": {"torch": "2.11.0+cu130"}},
           "source": {**dict.fromkeys(SOURCE_CHANGES - {MODULE}, "old"), "rotquant/quantize.py": "unchanged"}}
    identity = fingerprint(old)
    write_json(root / "identity.json", old)
    write_json(root / "config.json", config)
    write_json(root / "arms/w5_v6.json", arm_record(arms()[0]))
    defaults = {"config": root / "config.json", "root": root, "identity": identity}
    state = {"protocol": "rq3-gpu-e2e-v2", "status": "failed", "attempts": [],
             "controls": {"identity": identity, "active_budget_minutes": 480, "overnight_wall_minutes": 480},
             "active_seconds": 8240.559252754007, "wall_deadline_epoch": time.time() - 100}

    def record(name, value, **options):
        directory = root / "stages" / name / "attempt-001"
        directory.mkdir(parents=True)
        value = {**value, "status": "complete"}
        for key in ("data_path", "source", "teacher_dir", "scorer_dir", "probes", "checkpoint", "native"):
            if key in value:
                target = directory / value[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"payload-{key}")
                value[key] = str(target)
        write_json(directory / "report.json", value)
        row = {"name": name, "status": "passed", "directory": str(directory), "value": value,
               "signature": fingerprint({k: str(v) for k, v in {**defaults, **options}.items()}),
               "artifacts": artifacts([p for p in directory.rglob("*") if p.is_file()])}
        state["attempts"].append(row)
        return value

    data = record("freeze", {"data_path": "data.json", "fingerprint": "frozen"})
    defaults["data"] = data["data_path"]
    native = {"library": str(tmp_path / "gone/build/native.so"), "source": str(tmp_path / "gone/llama")}
    state["attempts"].append({"name": "runtime", "status": "passed", "signature": fingerprint(identity),
                              "value": native, "artifacts": {native["library"]: {"bytes": 1, "sha256": "gone"}}})
    teacher = record("teacher", {"source": "source", "teacher_dir": "logits", "scorer_dir": "scorer",
                                  "probes": "probes.npz", "artifact_bytes": 1000})
    prepared = record("prepare-w5_v6", {"checkpoint": "checkpoint", "probes": "probes.npz",
                                        "selection": {}, "artifact_bytes": 400, "identity": identity},
                      arm=root / "arms/w5_v6.json", teacher_dir=teacher["teacher_dir"])
    record("reload-w5_v6", {"parity": [{"passed": True}]},
           checkpoint=prepared["checkpoint"], probes=prepared["probes"])
    w5 = record("export-w5_v6", {"native": "native.gguf", "artifact_bytes": 300},
                checkpoint=prepared["checkpoint"], llama_dir=native["source"])
    fp16 = record("export-fp16", {"native": "native.gguf", "artifact_bytes": 1000},
                  source=teacher["source"], quant_export=w5["native"], llama_dir=native["source"])
    for label, exported, probes in (("fp16", fp16, teacher["probes"]), ("w5_v6", w5, prepared["probes"])):
        record("measure-" + label, measurement(label), native=exported["native"], probes=probes,
               library=native["library"], source=teacher["source"], teacher_dir=teacher["teacher_dir"],
               scorer_dir=teacher["scorer_dir"], label=label)
    write_json(root / "workflow.json", state)
    current = copy.deepcopy(old)
    current["source"].update(dict.fromkeys(SOURCE_CHANGES, "new"))
    return root, current


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_full_verified_import_preserves_parent_and_original_provenance(parent_run, tmp_path):
    parent, current = parent_run
    before = snapshot(parent)
    receipt = continuation(parent, tmp_path / "child", current, 480)
    assert snapshot(parent) == before
    assert receipt["budget_minutes"] == 342
    assert receipt["parent_active_seconds"] == 8240.559252754007
    assert len(receipt["imports"]) == 8
    assert receipt["source_changes"][MODULE] == {"old": None, "new": "new"}
    assert receipt["imports"]["measure-fp16"]["value"]["runtime"] == {"libnative.so": "original-binary"}
    assert receipt["parent_identity"] != receipt["child_identity"]
    assert not Path(receipt["original_runtime"]["value"]["library"]).exists()


@pytest.mark.parametrize("change", ["config", "gpu", "software", "numerics", "extra-file", "removed-file"])
def test_rejects_incompatible_config_runtime_or_unreviewed_source(parent_run, tmp_path, change):
    parent, current = parent_run
    if change == "config":
        current["config"]["seed"] = 1
    elif change == "gpu":
        current["runtime"]["gpu"] = "different GPU"
    elif change == "software":
        current["runtime"]["versions"]["torch"] = "other"
    elif change == "numerics":
        current["source"]["rotquant/quantize.py"] = "modified"
    elif change == "extra-file":
        current["source"]["scripts/new_backend.py"] = "added"
    else:
        del current["source"]["scripts/ternary_research_bonsai.py"]
    with pytest.raises(ValueError):
        continuation(parent, tmp_path / "child", current, 480)
    assert not (tmp_path / "child/continuation.json").exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt", "signature", "latest-failed", "running", "report"])
def test_rejects_unverified_parent_stages_without_modifying_parent(parent_run, tmp_path, damage):
    parent, current = parent_run
    state = read(parent / "workflow.json")
    row = next(r for r in state["attempts"] if r["name"] == "teacher")
    payload = Path(row["value"]["source"])
    if damage == "missing":
        payload.unlink()
    elif damage == "corrupt":
        payload.write_text("changed")
    elif damage == "signature":
        row["signature"] = "unrelated"
    elif damage == "latest-failed":
        state["attempts"].append({**row, "status": "failed"})
    elif damage == "running":
        state["status"] = "running"
    else:
        row["value"]["artifact_bytes"] = 999
    write_json(parent / "workflow.json", state)
    before = snapshot(parent)
    with pytest.raises(ValueError):
        continuation(parent, tmp_path / "child", current, 480)
    assert snapshot(parent) == before


@pytest.mark.parametrize("change", ["backend", "data_fingerprint", "label", "speed"])
def test_complete_measurement_requires_original_common_execution(parent_run, tmp_path, change):
    parent, current = parent_run
    state = read(parent / "workflow.json")
    row = next(r for r in state["attempts"] if r["name"] == "measure-fp16")
    if change == "speed":
        row["value"]["speed"]["prefill_chunk_tokens"] = 8192
    else:
        row["value"][change] = "other"
    report = Path(row["directory"]) / "report.json"
    write_json(report, row["value"])
    row["artifacts"] = artifacts([report])
    write_json(parent / "workflow.json", state)
    with pytest.raises(ValueError, match="execution binding"):
        continuation(parent, tmp_path / "child", current, 480)


def make_study(parent, current, child, monkeypatch, *, fresh_budget=False):
    monkeypatch.setattr("scripts.ternary_research_plan.source_files", lambda: current["source"])
    monkeypatch.setattr("scripts.run_qwen35_vocabulary_budget.runtime_identity", lambda: current["runtime"])
    args = SimpleNamespace(output_dir=child, work_dir=child.parent / "new-work", budget_minutes=480, resume_from=parent, fresh_budget=fresh_budget)
    return TernaryStudy(args, current["config"])


def test_missing_runtime_rebuild_failure_retains_measured_controls(parent_run, tmp_path, monkeypatch):
    parent, current = parent_run
    before = snapshot(parent)
    child = tmp_path / "child"
    study = make_study(parent, current, child, monkeypatch)

    def fail_build():
        assert set(study.measured) == {"fp16", "w5_v6"}
        raise RuntimeError("native compiler unavailable")

    study.build_runtime = fail_build
    with pytest.raises(RuntimeError, match="compiler"):
        study.run()
    report = read(child / "comparison.json")
    measured = {r["label"]: r for r in report["rows"] if r["status"] == "measured"}
    assert set(measured) == {"fp16", "w5_v6"}
    assert all(row["origin"]["identity"] == study.continuation["parent_identity"] for row in measured.values())
    assert "imports" not in report["continuation"]
    state = read(child / "workflow.json")
    assert state["controls"]["active_budget_minutes"] == 342
    assert state["wall_deadline_epoch"] > time.time()
    assert snapshot(parent) == before
    resumed = make_study(parent, current, child, monkeypatch)
    with resumed.workflow:
        assert resumed.workflow.state["wall_deadline_epoch"] == state["wall_deadline_epoch"]


def test_imported_controls_are_never_automatically_reexecuted(parent_run, tmp_path, monkeypatch):
    parent, current = parent_run
    study = make_study(parent, current, tmp_path / "child", monkeypatch)
    calls = []
    study.build_runtime = lambda: setattr(study, "runtime", {"library": "new.so", "source": "new-llama"})

    def worker(name, *args, **kwargs):
        calls.append(name)
        raise AssertionError("completed control was executed")

    def bonsai():
        calls.append("bonsai")
        raise RuntimeError("stop before unfinished arms")

    study.worker, study.bonsai = worker, bonsai
    with pytest.raises(RuntimeError, match="unfinished arms"):
        study.run()
    assert calls == ["bonsai"]


def test_continuation_commands_disable_writes_to_parent_scorer_bytecode(parent_run, tmp_path, monkeypatch):
    parent, current = parent_run
    study = make_study(parent, current, tmp_path / "child", monkeypatch)
    commands = []
    stage = SimpleNamespace(command=lambda command, label: commands.append(command))
    study.command(stage, "ternary_research_worker.py", "bonsai")
    assert commands[0][1:3] == ["-B", "-u"]


def test_parent_cannot_be_its_own_child_or_contain_it(parent_run, monkeypatch):
    parent, current = parent_run
    before = snapshot(parent)
    for child in (parent, parent / "child", parent.parent):
        with pytest.raises(ValueError, match="separate child"):
            make_study(parent, current, child, monkeypatch)
    assert snapshot(parent) == before


def test_changed_existing_child_provenance_is_rejected(parent_run, tmp_path):
    parent, current = parent_run
    child = tmp_path / "child"
    receipt = continuation(parent, child, current, 480)
    assert continuation(parent, child, current, 480) == receipt
    before = snapshot(child)
    with pytest.raises(ValueError, match="provenance changed"):
        continuation(parent, child, current, 300)
    assert snapshot(child) == before


def test_import_requires_the_same_original_native_runtime_for_both_controls(parent_run, tmp_path):
    parent, current = parent_run
    state = read(parent / "workflow.json")
    row = next(r for r in state["attempts"] if r["name"] == "measure-w5_v6")
    row["value"]["runtime"]["libnative.so"] = "unmatched-binary"
    report = Path(row["directory"]) / "report.json"
    write_json(report, row["value"])
    row["artifacts"] = artifacts([report])
    write_json(parent / "workflow.json", state)
    with pytest.raises(ValueError, match="different native runtimes"):
        continuation(parent, tmp_path / "child", current, 480)


def test_payload_cannot_be_removed_from_the_hash_receipt(parent_run, tmp_path):
    parent, current = parent_run
    state = read(parent / "workflow.json")
    row = next(r for r in state["attempts"] if r["name"] == "teacher")
    del row["artifacts"][row["value"]["source"]]
    write_json(parent / "workflow.json", state)
    with pytest.raises(ValueError, match="not fully hashed"):
        continuation(parent, tmp_path / "child", current, 480)


def test_no_child_allowance_after_parent_active_budget_is_consumed(parent_run, tmp_path):
    parent, current = parent_run
    state = read(parent / "workflow.json")
    state["active_seconds"] = 480 * 60
    write_json(parent / "workflow.json", state)
    with pytest.raises(ValueError, match="allowance is exhausted"):
        continuation(parent, tmp_path / "child", current, 480)


def child_record(study, name, signature, value):
    path = study.root / "stages" / name / "attempt-001/report.json"
    write_json(path, value)
    row = {"name": name, "signature": fingerprint(signature), "status": "passed", "value": value,
           "artifacts": artifacts([path]), "directory": str(path.parent)}
    study.workflow.state["attempts"].append(row)
    return row


def test_child_hydrates_bonsai_without_rerunning_imported_fp16_export(parent_run, tmp_path, monkeypatch):
    parent, current = parent_run
    study = make_study(parent, current, tmp_path / "child", monkeypatch)
    with study.workflow:
        study.hydrate()
        runtime = {"library": str(tmp_path / "new.so"), "source": str(tmp_path / "new-source")}
        child_record(study, "runtime", study.identity, runtime)
        child_record(study, "bonsai-assets", study.config["bonsai"],
                     {"complete_model_payload_bytes": 200, "language_model_bytes": 100, "scope": "complete"})
        bonsai = {"library": str(tmp_path / "bonsai.so")}
        child_record(study, "bonsai-runtime", study.identity, bonsai)
        signature = {k: str(v) for k, v in study.worker_options(**study.bonsai_options(bonsai)).items()}
        child_record(study, "measure-bonsai2_4b", signature, measurement("bonsai2_4b"))
        study.hydrate()
        assert set(study.measured) == {"fp16", "w5_v6", "bonsai2_4b"}


def test_runtime_provenance_does_not_hide_failed_latest_preflight(parent_run, tmp_path, monkeypatch):
    parent, current = parent_run
    study = make_study(parent, current, tmp_path / "child", monkeypatch)
    with study.workflow:
        study.runtime = {"library": str(tmp_path / "new.so"), "source": str(tmp_path / "new-source")}
        child_record(study, "runtime", study.identity, study.runtime)
        child_record(study, "native-preflight", study.runtime, {"passed": True})
        row = child_record(study, "native-preflight", study.runtime, {"passed": False})
        row["status"] = "failed"
        child_record(study, "ternary-native-preflight", study.runtime, {"passed": True})
        report = runtime_provenance(study)
        assert report["preflight"] == {"native-preflight": False, "ternary-native-preflight": True}
        assert not report["same_library_binary"]


def worker_record(study, name, value, **options):
    directory = study.root / "stages" / name / "attempt-001"
    for key in ("checkpoint", "probes", "native"):
        if key in value:
            path = directory / value[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(key)
            value[key] = str(path)
    signature = {key: str(value) for key, value in study.worker_options(**options).items()}
    row = child_record(study, name, signature, {"status": "complete", **value})
    row["artifacts"] = artifacts([path for path in directory.rglob("*") if path.is_file()])
    return row["value"]


@pytest.fixture
def second_run(parent_run, tmp_path, monkeypatch):
    parent, current = parent_run
    child = tmp_path / "first-child"
    study = make_study(parent, current, child, monkeypatch)
    monkeypatch.setattr("scripts.ternary_research_continuation.verify_native_source", lambda row: None)
    with study.workflow:
        study.hydrate()
        library = tmp_path / "work/libnative.so"
        library.parent.mkdir()
        library.write_text("same-runtime")
        study.runtime = {"library": str(library), "source": str(tmp_path / "llama")}
        row = child_record(study, "runtime", study.identity, study.runtime)
        row["artifacts"].update(artifacts([library]))
        for name in ("native-preflight", "ternary-native-preflight"):
            child_record(study, name, study.runtime, {"passed": True})
        assets = study.work / "bonsai2_4b/artifacts"
        assets.mkdir(parents=True)
        for name in ("model.gguf", "vision.gguf"):
            (assets / name).write_text(name)
        row = child_record(study, "bonsai-assets", study.config["bonsai"],
                            {"model": str(assets / "model.gguf"), "vision": str(assets / "vision.gguf"),
                             "complete_model_payload_bytes": 200, "language_model_bytes": 100, "scope": "complete"})
        row["artifacts"].update(artifacts(assets.glob("*.gguf")))
        bonsai = {"library": str(library)}
        row = child_record(study, "bonsai-runtime", study.identity, bonsai)
        row["artifacts"].update(artifacts([library]))
        value = {**measurement("bonsai2_4b"), "backend": "CUDA0/Prism/chunk16"}
        worker_record(study, "measure-bonsai2_4b", value, **study.bonsai_options(bonsai))
        arm = child / "arms/w6_v6.json"
        write_json(arm, arm_record(arms()[1]))
        prepared = worker_record(study, "prepare-w6_v6",
                                 {"checkpoint": "checkpoint", "probes": "probes.npz", "selection": {}},
                                 arm=arm, teacher_dir=study.teacher["teacher_dir"])
        worker_record(study, "reload-w6_v6", {"parity": [{"passed": True}]},
                      checkpoint=prepared["checkpoint"], probes=prepared["probes"])
        worker_record(study, "export-w6_v6", {"native": "native.gguf", "artifact_bytes": 400},
                      checkpoint=prepared["checkpoint"], llama_dir=study.runtime["source"])
        study.workflow.state["active_seconds"] = 5799.3911806649985
    state = read(child / "workflow.json")
    state["status"] = "failed"
    state["attempts"].append({"name": "measure-w6_v6", "status": "failed", "signature": "failed-only"})
    write_json(child / "workflow.json", state)
    next_source = copy.deepcopy(current)
    next_source["source"][MODULE] = "second-generation"
    return parent, child, next_source


def test_second_generation_preserves_three_measurements_and_w6_without_reexecution(second_run, tmp_path, monkeypatch):
    parent, child, current = second_run
    before = snapshot(parent), snapshot(child)
    study = make_study(child, current, tmp_path / "second-child", monkeypatch)
    assert study.continuation["budget_minutes"] == 245
    assert study.continuation["protocol"] == "ternary-continuation-v2"
    calls = []

    def worker(name, *args, **kwargs):
        calls.append(name)
        raise RuntimeError("stop at first unfinished measurement")

    study.worker = worker
    with pytest.raises(RuntimeError, match="first unfinished"):
        study.run()
    assert calls == ["measure-w6_v6"]
    assert set(study.prepared) == {"w5_v6", "w6_v6"}
    assert set(study.measured) == {"fp16", "w5_v6", "bonsai2_4b"}
    report = read(study.root / "comparison.json")
    rows = {row["label"]: row for row in report["rows"]}
    assert rows["fp16"]["origin"]["root"] == str(parent)
    assert rows["w5_v6"]["origin"]["root"] == str(parent)
    assert rows["bonsai2_4b"]["origin"]["root"] == str(child)
    assert rows["w6_v6"]["preparation_origin"]["root"] == str(child)
    assert "origin" not in rows["w6_v6"]
    assert report["continuation"]["native_runtime"]["runtime_reused"]
    assert (snapshot(parent), snapshot(child)) == before
    assert not study.workflow.state["attempts"]


def test_second_generation_resume_restores_its_completed_w6_measurement(second_run, tmp_path, monkeypatch):
    _, child, current = second_run
    grandchild = tmp_path / "second-child"
    study = make_study(child, current, grandchild, monkeypatch)
    with study.workflow:
        study.hydrate()
        worker_record(study, "measure-w6_v6", measurement("w6_v6"),
                      **study.native_options("w6_v6", study.prepared["w6_v6"]["probes"]))
    resumed = make_study(child, current, grandchild, monkeypatch)
    with resumed.workflow:
        resumed.hydrate()
        assert set(resumed.measured) == {"fp16", "w5_v6", "bonsai2_4b", "w6_v6"}
        resumed.summarize()
        row = next(row for row in read(grandchild / "comparison.json")["rows"] if row["label"] == "w6_v6")
        assert row["origin"] == {"root": str(grandchild), "identity": resumed.identity, "imported": False}
        assert row["preparation_origin"]["root"] == str(child)


@pytest.mark.parametrize("damage", ["ancestor", "continuation", "runtime", "latest-preflight", "w6-export"])
def test_second_generation_rejects_changed_lineage_runtime_and_payloads(second_run, tmp_path, damage):
    parent, child, current = second_run
    state = read(child / "workflow.json")
    if damage == "ancestor":
        (parent / "identity.json").write_text("{}")
    elif damage == "continuation":
        (child / "continuation.json").write_text("{}")
    elif damage == "runtime":
        row = next(r for r in state["attempts"] if r["name"] == "runtime")
        Path(row["value"]["library"]).write_text("wrong-runtime")
    elif damage == "latest-preflight":
        row = next(r for r in state["attempts"] if r["name"] == "native-preflight")
        state["attempts"].append({**row, "status": "failed"})
        write_json(child / "workflow.json", state)
    else:
        row = next(r for r in state["attempts"] if r["name"] == "export-w6_v6")
        Path(row["value"]["native"]).write_text("wrong-gguf")
    with pytest.raises((ValueError, KeyError)):
        continuation(child, tmp_path / "second-child", current, 480)


def test_hashes_are_deduplicated_only_while_metadata_and_expected_hash_agree(tmp_path, monkeypatch):
    import scripts.ternary_research_continuation as module
    path = tmp_path / "payload"
    path.write_text("first")
    files = artifacts([path])
    real_digest, calls = module.digest, []

    def counted(value):
        calls.append(str(value))
        return real_digest(value)

    monkeypatch.setattr(module, "digest", counted)
    verify = HashVerifier()
    assert verify(files) and verify(files)
    assert calls == [str(path)]
    assert not verify({str(path): {**files[str(path)], "sha256": "wrong"}})
    path.write_text("other")
    assert not verify(files)
    assert calls == [str(path), str(path)]


def test_worker_change_requires_exact_hashed_diagnostic_binding(parent_run, tmp_path):
    parent, current = parent_run
    # Add the existing worker to both source snapshots before creating a run.
    previous = read(parent / "identity.json")
    previous["source"][PARITY_WORKER] = "worker-old"
    # Exercise the correction validator directly; parent signatures stay immutable.
    current["source"][PARITY_WORKER] = "worker-new"
    from scripts.ternary_research_continuation import correction_receipt, source_changes
    identity = fingerprint(previous)
    changes = {PARITY_WORKER: {"old": "worker-old", "new": "worker-new"}}
    validation = tmp_path / "validation.json"
    write_json(validation, {"passed": True, "parent_identity": identity, "source_changes": changes})
    path = tmp_path / "correction.json"
    receipt = {"protocol": "ternary-parity-correction-v1", "parent_identity": identity,
               "source_changes": changes, "diagnostics": artifacts([validation]),
               "validation": {"passed": True, "report": str(validation)}}
    write_json(path, receipt)
    result = correction_receipt(path, previous, current, identity, HashVerifier())
    assert result["source_changes"] == changes
    assert source_changes(previous["source"], current["source"], parity_correction=changes)[PARITY_WORKER] == changes[PARITY_WORKER]
    with pytest.raises(ValueError, match="does not permit"):
        source_changes(previous["source"], current["source"])
    validation.write_text('{"passed": false}')
    with pytest.raises(ValueError, match="missing or changed"):
        correction_receipt(path, previous, current, identity, HashVerifier())
