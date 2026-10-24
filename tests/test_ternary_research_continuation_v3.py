"""Fresh continuation preserves six accepted arms and immutable ancestry."""
import copy
from pathlib import Path

import pytest
from test_ternary_research_continuation import (  # noqa: F401
    make_study,
    measurement,
    parent_run,
    read,
    second_run,
    snapshot,
    worker_record,
)

from scripts.native_gpu_workflow import write_json
from scripts.ternary_research_continuation import MODULE, continuation
from scripts.ternary_research_plan import arm_record, arms


@pytest.fixture
def exhausted_run(request, tmp_path, monkeypatch):
    ancestor, parent, current = request.getfixturevalue("second_run")
    root = tmp_path / "second-child"
    study = make_study(parent, current, root, monkeypatch)
    with study.workflow:
        study.hydrate()
        worker_record(study, "measure-w6_v6", measurement("w6_v6"),
                      **study.native_options("w6_v6", study.prepared["w6_v6"]["probes"]))
        for arm in arms()[2:5]:
            path = root / "arms" / f"{arm.name}.json"
            write_json(path, arm_record(arm))
            options = {"parent": study.prepared[arm.parent]["checkpoint"]} if arm.parent else {}
            prepared = worker_record(study, "prepare-" + arm.name,
                {"checkpoint": "checkpoint", "probes": "probes.npz", "selection": {}},
                arm=path, teacher_dir=study.teacher["teacher_dir"], **options)
            study.prepared[arm.name] = prepared
            worker_record(study, "reload-" + arm.name, {"parity": [{"passed": True}]},
                          checkpoint=prepared["checkpoint"], probes=prepared["probes"])
            study.exported[arm.name] = worker_record(study, "export-" + arm.name,
                {"native": "native.gguf", "artifact_bytes": 400},
                checkpoint=prepared["checkpoint"], llama_dir=study.runtime["source"])
            if arm.name != "ternary_ptq":
                worker_record(study, "measure-" + arm.name, measurement(arm.name),
                              **study.native_options(arm.name, prepared["probes"]))
    state = read(root / "workflow.json")
    state.update(status="failed", active_seconds=14579.127193439002)
    state["attempts"].append({"name": "measure-ternary_ptq", "status": "failed", "signature": "partial-only"})
    write_json(root / "workflow.json", state)
    source = copy.deepcopy(current)
    source["source"][MODULE] = "third-generation"
    return ancestor, parent, root, source


def test_fresh_child_reuses_six_arms_and_starts_only_unfinished_measurement(exhausted_run, tmp_path, monkeypatch):
    ancestor, parent, root, current = exhausted_run
    before = [snapshot(p) for p in (ancestor, parent, root)]
    study = make_study(root, current, tmp_path / "third-child", monkeypatch, fresh_budget=True)
    assert study.continuation["protocol"] == "ternary-continuation-v3"
    assert study.continuation["budget_minutes"] == 480
    calls = []

    def stop(name, *args, **kwargs):
        calls.append(name)
        raise RuntimeError("first unfinished measurement")

    study.worker = stop
    with pytest.raises(RuntimeError, match="first unfinished"):
        study.run()
    assert calls == ["measure-ternary_ptq"]
    assert set(study.measured) == {"fp16", "w5_v6", "w6_v6", "w4_v6", "w4_v6_recovered", "bonsai2_4b"}
    assert set(study.prepared) == {"w5_v6", "w6_v6", "w4_v6", "w4_v6_recovered", "ternary_ptq"}
    rows = {r["label"]: r for r in read(study.root / "comparison.json")["rows"]}
    assert rows["fp16"]["origin"]["root"] == str(ancestor)
    assert rows["bonsai2_4b"]["origin"]["root"] == str(parent)
    for label in ("w6_v6", "w4_v6", "w4_v6_recovered"):
        assert rows[label]["origin"]["root"] == str(root)
    assert rows["ternary_ptq"]["preparation_origin"]["root"] == str(root)
    assert "origin" not in rows["ternary_ptq"]
    assert [snapshot(p) for p in (ancestor, parent, root)] == before


def test_fresh_child_reconnect_cannot_renew_or_change_its_allowance(exhausted_run, tmp_path, monkeypatch):
    _, _, root, current = exhausted_run
    child = tmp_path / "third-child"
    study = make_study(root, current, child, monkeypatch, fresh_budget=True)
    with study.workflow:
        study.hydrate()
    deadline = read(child / "workflow.json")["wall_deadline_epoch"]
    resumed = make_study(root, current, child, monkeypatch, fresh_budget=True)
    with resumed.workflow:
        assert resumed.workflow.state["wall_deadline_epoch"] == deadline
    before = snapshot(child)
    with pytest.raises(ValueError, match="provenance changed"):
        continuation(root, child, current, 300, fresh_budget=True)
    assert snapshot(child) == before


@pytest.mark.parametrize("damage", ["ancestor", "runtime", "ternary-export"])
def test_fresh_child_rejects_corrupted_ancestry_and_reused_payloads(exhausted_run, tmp_path, damage):
    ancestor, parent, root, current = exhausted_run
    if damage == "ancestor":
        (ancestor / "workflow.json").write_text("{}")
    elif damage == "runtime":
        row = read(parent / "workflow.json")["attempts"]
        runtime = next(r for r in row if r["name"] == "runtime")
        Path(runtime["value"]["library"]).write_text("changed runtime")
    else:
        row = read(root / "workflow.json")["attempts"]
        exported = next(r for r in row if r["name"] == "export-ternary_ptq")
        Path(exported["value"]["native"]).write_text("changed export")
    with pytest.raises((ValueError, KeyError)):
        continuation(root, tmp_path / "third-child", current, 480, fresh_budget=True)
