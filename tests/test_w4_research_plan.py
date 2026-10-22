import copy
import json

import pytest
import torch
import yaml

from scripts import run_qwen35_w4_research as driver
from scripts import w4_research_data as data
from scripts import w4_research_plan as plan
from scripts.native_gpu_workflow import Workflow, fingerprint, write_json
from scripts.w4_research_results import comparison


@pytest.fixture
def config():
    return yaml.safe_load(plan.DEFAULT_CONFIG.read_text())


def test_arm_configuration_is_independent_and_preserves_controls(config):
    original = copy.deepcopy(config)
    arms = plan.initial_arms(plan.validate_config(config))
    assert [(a.bits, a.group_size) for a in arms] == [(5, 128), (4, 128), (4, 128), (4, 64), (4, 32)]
    for arm in arms:
        configured = arm.config(config)
        assert configured["vocabulary"] == original["vocabulary"]
        assert configured["patch"] == original["patch"]
        assert configured["quant"]["group_size"] == arm.group_size
    assert config == original


@pytest.mark.parametrize("path,value", [
    (("model_revision",), "main"), (("calib_dataset_revision",), "main"),
    (("dtype",), "bfloat16"), (("research", "groups"), [128]),
    (("quant", "scale_bits"), 16), (("vocabulary", "bits"), 8),
    (("recovery", "distill_train_scales"), False),
    (("recovery", "distill_train_rotations"), True),
    (("recovery", "distill_ce_weight"), .1),
    (("recovery", "distill_code_refresh_interval"), 20),
    (("recovery", "distill_train_batches"), 1),
    (("research", "speed_contexts"), [128, 9000]),
    (("research", "public_samples_per_benchmark"), 600),
])
def test_protocol_rejects_uncontrolled_or_impossible_runs(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        plan.validate_config(config)


def test_identity_rejects_changed_code_config_or_runtime(tmp_path, monkeypatch, config):
    source = {"core.py": "abc"}
    monkeypatch.setattr(plan, "source_files", lambda: source)
    first = plan.bind_run(tmp_path, config, {"torch": "same"})
    assert plan.bind_run(tmp_path, config, {"torch": "same"}) == first
    with pytest.raises(ValueError, match="changed"):
        plan.bind_run(tmp_path, config, {"torch": "other"})
    source["core.py"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        plan.bind_run(tmp_path, config, {"torch": "same"})


def candidate(arm, kl, size=100):
    return {"arm": plan.arm_record(arm), "selection": {"mean_teacher_kl": kl},
            "artifact_bytes": size, "final_test_score": 100 - kl}


def test_candidate_selection_uses_outer_selection_only():
    initial = candidate(plan.Arm("initial"), .02)
    recovered = candidate(plan.Arm("recovered", recovery=True), .01)
    recovered["final_test_score"] = -100
    assert plan.choose_candidate([initial, recovered]) == recovered
    recovered["selection"]["mean_teacher_kl"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        plan.choose_candidate([initial, recovered])


def test_code_and_recovery_ablations_share_initial_winner(tmp_path, config):
    study = driver.ResearchStudy.__new__(driver.ResearchStudy)
    study.config, study.root, study.prepared = config, tmp_path, {}
    def prepare(arm):
        kl = .001 if arm.name == "w4_codes" else .1 / arm.group_size
        study.prepared[arm.name] = candidate(arm, kl)
    study.prepare = prepare
    study.research()
    original = study.prepared["w4_g128_h"]["arm"]
    scale = study.prepared["w4_recovered"]["arm"]
    combined = study.prepared["w4_codes_recovered"]["arm"]
    assert scale["group_size"] == combined["group_size"] == original["group_size"]
    assert scale["code_refit_passes"] == 0 and combined["code_refit_passes"] == 2
    assert scale["recovery"] and combined["recovery"]


def test_document_partitions_exclude_history_and_each_other(monkeypatch, config):
    config.update(n_calib=2, calib_seq_len=20)
    config["research"].update(recovery_train_documents=2, recovery_validation_documents=1,
                              recovery_selection_documents=1, selection_documents=2,
                              test_documents=2, long_documents=1, long_context=32)
    monkeypatch.setattr(data, "prior_rows", lambda: ({0, 2, 4}, "history"))
    def loader(tokenizer, count, length, device, **kwargs):
        excluded = kwargs.get("exclude_source_rows") or set()
        rows = [row for row in range(100) if row not in excluded][:count]
        return [data.experiment.TokenBatch(torch.full((1, length), row + 1), row) for row in rows]
    monkeypatch.setattr(data.experiment, "build_calib_loader", loader)
    documents, history = data.collect_documents(config, None)
    assert history == "history" and [x["source_row"] for x in documents["calibration"]] == [0, 1]
    evaluation = [x["source_row"] for key, values in documents.items() if key != "calibration" for x in values]
    assert len(evaluation) == len(set(evaluation))
    assert not set(evaluation) & {0, 1, 2, 4}


def manifest():
    documents = {part: [{"id": part, "source_row": i, "input_ids": [i, i + 1, i + 2]}]
                 for i, part in enumerate(("calibration", "selection", "test", "long_test"))}
    value = {"documents": documents, "public_items": [], "speed_inputs": {
        "2": {"prompt_ids": [3, 4], "decode_ids": [2]}}}
    value["fingerprint"] = fingerprint(value)
    return value


@pytest.mark.parametrize("mutation", ["fingerprint", "row", "tokens", "speed"])
def test_manifest_rejects_overlap_and_changed_frozen_inputs(mutation):
    value = manifest()
    data.validate_manifest(value)
    if mutation == "fingerprint":
        value["fingerprint"] = "bad"
    else:
        if mutation == "row":
            value["documents"]["test"][0]["source_row"] = 0
        if mutation == "tokens":
            value["documents"]["test"][0]["input_ids"] = [0, 1, 2]
        if mutation == "speed":
            value["speed_inputs"]["2"]["prompt_ids"] = [5, 5]
        value["fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        data.validate_manifest(value)


def test_worker_stage_serializes_paths_and_reuses_verified_result(tmp_path):
    study = driver.ResearchStudy.__new__(driver.ResearchStudy)
    study.root, study.config_path, study.identity, study.data = tmp_path, tmp_path / "config.json", "same", None
    study.workflow = Workflow(tmp_path, plan.ROOT, {}, 10)
    commands = []
    def command(stage, script, *arguments):
        commands.append([script, *map(str, arguments)])
        write_json(stage.directory / "report.json", {"status": "complete"})
    study.command = command
    with study.workflow:
        study.worker("freeze", "freeze")
        study.worker("freeze", "freeze")
    assert len(commands) == 1
    assert "--config" in commands[0] and str(study.config_path) in commands[0]


def test_preflight_conversion_uses_real_cli_and_failure_stops_stage(tmp_path):
    study = driver.ResearchStudy.__new__(driver.ResearchStudy)
    study.runtime = {"library": "lib.so", "source": "llama"}
    commands = []
    study.workflow = Workflow(tmp_path, plan.ROOT, {}, 10)
    def command(stage, script, *arguments):
        commands.append((script, list(map(str, arguments))))
        if script == "check_rq3_conversion.py":
            raise RuntimeError("native numerical failure")
    study.command = command
    with pytest.raises(RuntimeError, match="native numerical"), study.workflow:
        study.preflight()
    script, args = commands[-1]
    assert script == "check_rq3_conversion.py"
    assert "--output-dir" in args and "--report" in args and "--output" not in args
    assert json.loads((tmp_path / "workflow.json").read_text())["attempts"][-1]["status"] == "failed"


def test_incomplete_comparison_leaves_absent_measurements_missing():
    result = comparison({}, {}, {}, None, expected=["fp16", "w5_v6", "w4_codes"])
    assert not result["complete"]
    for row in result["rows"]:
        assert row["status"] == "pending"
        assert row["native_payload_bytes"] is row["quality"] is row["speed"] is None


def test_run_records_partial_report_before_propagating_failure(tmp_path, config):
    study = driver.ResearchStudy.__new__(driver.ResearchStudy)
    study.config, study.root, study.teacher = config, tmp_path, None
    study.prepared, study.exported, study.measured = {}, {}, {}
    study.workflow = Workflow(tmp_path, plan.ROOT, {}, 10)
    study.worker = lambda *args, **kwargs: {"data_path": "frozen.json"}
    study.build_runtime = lambda: (_ for _ in ()).throw(RuntimeError("build failed"))
    with pytest.raises(RuntimeError, match="build failed"):
        study.run()
    assert not json.loads((tmp_path / "comparison.json").read_text())["complete"]
