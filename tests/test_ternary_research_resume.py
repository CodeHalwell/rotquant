"""Receipt-driven resume and useful partial reports without model/GPU work."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.native_gpu_workflow import Workflow, artifacts, fingerprint, write_json
from scripts.run_qwen35_ternary_research import TernaryStudy
from scripts.ternary_research_plan import LABELS, arms
from scripts.ternary_research_results import comparison, partial_measurement
from scripts.ternary_research_worker import TernaryWorker


def quality():
    return {"rows": [{"id": "document"}], "summary": {"mean_teacher_kl": 0.0, "ppl": 2.0}}


def tasks():
    return {"rows": [{"id": "task", "benchmark": "math", "input_hash": "frozen",
                      "max_new_tokens": 32, "task_success": 1.0}],
            "summary": {"math": {"items": 1, "correct": 1, "accuracy": 1.0}}}


def measurement(label):
    return {"status": "complete", "label": label, "backend": "CUDA0/reference",
            "quality": {"test": quality(), "long_test": quality()}, "tasks": tasks(),
            "speed": {"completed": True, "rows": [], "summary": {"128": {"decode_tokens_per_second": 5.0}}}}


def study_at(root):
    study = TernaryStudy.__new__(TernaryStudy)
    study.root, study.work = root, root / "work"
    study.identity, study.config_path = "identity", root / "config.json"
    study.config = {"research": {"task_margin": .01}, "bonsai": {"packing": "PTQ1_0"}}
    study.data, study.runtime, study.teacher = None, None, None
    study.prepared, study.exported, study.measured, study.failures = {}, {}, {}, {}
    study.workflow = Workflow(root, Path.cwd(), {"identity": study.identity}, 480, overnight=True)
    return study


def record(study, name, signature, value, *, status="passed"):
    previous = [row for row in study.workflow.state["attempts"] if row["name"] == name]
    directory = study.root / "stages" / name / f"attempt-{len(previous) + 1:03d}"
    path = directory / "report.json"
    write_json(path, value)
    row = {"name": name, "signature": fingerprint(signature), "status": status,
           "value": value, "artifacts": artifacts([path]), "directory": str(directory)}
    study.workflow.state["attempts"].append(row)
    return row


def record_worker(study, name, value, **options):
    signature = {key: str(value) for key, value in study.worker_options(**options).items()}
    return record(study, name, signature, value)


def completed_ledger(root):
    """Write actual hashed receipts with the same request chain as the driver."""
    study = study_at(root)
    study.workflow.state = {"protocol": "rq3-gpu-e2e-v2", "controls": study.workflow.controls,
                            "active_seconds": 1., "attempts": [], "status": "passed",
                            "wall_deadline_epoch": time.time() - 1}
    data = {"data_path": str(root / "data.json"), "fingerprint": "data-fingerprint"}
    record_worker(study, "freeze", data)
    study.data = data
    study.runtime = {"library": str(root / "runtime.so"), "source": str(root / "llama")}
    record(study, "runtime", study.identity, study.runtime)
    study.teacher = {"teacher_dir": str(root / "teacher"), "source": str(root / "source"),
                     "scorer_dir": str(root / "scorer"), "probes": str(root / "source-probes"),
                     "artifact_bytes": 1000}
    record_worker(study, "teacher", study.teacher)
    for arm in arms():
        prepared = {"checkpoint": str(root / arm.name / "checkpoint"),
                    "probes": str(root / arm.name / "probes"), "selection": {}, "artifact_bytes": 400}
        options = {"parent": study.prepared[arm.parent]["checkpoint"]} if arm.parent else {}
        record_worker(study, "prepare-" + arm.name, prepared,
                      arm=root / "arms" / f"{arm.name}.json", teacher_dir=study.teacher["teacher_dir"], **options)
        record_worker(study, "reload-" + arm.name, {"parity": True},
                      checkpoint=prepared["checkpoint"], probes=prepared["probes"])
        study.prepared[arm.name] = prepared
        exported = {"native": str(root / arm.name / "native"), "artifact_bytes": 300,
                    "language_payload_bytes": 200}
        record_worker(study, "export-" + arm.name, exported,
                      checkpoint=prepared["checkpoint"], llama_dir=study.runtime["source"])
        study.exported[arm.name] = exported
        record_worker(study, "measure-" + arm.name, measurement(arm.name),
                      **study.native_options(arm.name, prepared["probes"]))
    study.exported["fp16"] = {"native": str(root / "fp16/native"), "artifact_bytes": 1000}
    record_worker(study, "export-fp16", study.exported["fp16"], source=study.teacher["source"],
                  quant_export=study.exported["w5_v6"]["native"], llama_dir=study.runtime["source"])
    record_worker(study, "measure-fp16", measurement("fp16"), **study.native_options("fp16", study.teacher["probes"]))
    record(study, "bonsai-assets", study.config["bonsai"],
           {"complete_model_payload_bytes": 200, "language_model_bytes": 100, "scope": "complete"})
    bonsai_runtime = {"library": str(root / "bonsai.so")}
    record(study, "bonsai-runtime", study.identity, bonsai_runtime)
    record_worker(study, "measure-bonsai2_4b", measurement("bonsai2_4b"), **study.bonsai_options(bonsai_runtime))
    study.workflow.save()
    return study


def forbidden(*args, **kwargs):
    raise AssertionError("a completed resume must not start GPU work")


def test_completed_expired_resume_restores_all_rows_without_gpu_work(tmp_path):
    completed_ledger(tmp_path)
    study = study_at(tmp_path)
    study.worker, study.build_runtime = forbidden, forbidden
    study.run()
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert result["complete"]
    assert {row["label"] for row in result["rows"]} == set(LABELS)
    assert all(row["status"] == "measured" and row["native_payload_bytes"] > 0 for row in result["rows"])


def test_expired_incomplete_resume_preserves_previously_measured_rows(tmp_path):
    previous = completed_ledger(tmp_path)
    previous.workflow.state["attempts"] = [row for row in previous.workflow.state["attempts"]
                                           if row["name"] != "measure-ternary_recovered"]
    previous.workflow.save()
    study = study_at(tmp_path)
    study.worker = forbidden
    study.build_runtime = lambda: study.workflow.stage("native-preflight", forbidden, reuse=False)
    with pytest.raises(TimeoutError):
        study.run()
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert not result["complete"]
    assert sum(row["status"] == "measured" for row in result["rows"]) == 8
    assert result["error"]["setup"].startswith("TimeoutError")


@pytest.mark.parametrize("invalidate", ["artifact", "signature", "latest-failed"])
def test_hydration_rejects_changed_artifacts_requests_and_failed_latest_attempts(tmp_path, invalidate):
    previous = completed_ledger(tmp_path)
    row = next(row for row in previous.workflow.state["attempts"] if row["name"] == "export-w4_v6")
    if invalidate == "artifact":
        Path(next(iter(row["artifacts"]))).write_text('{"changed": true}')
    elif invalidate == "signature":
        row["signature"] = "different-parent-or-runtime"
    else:
        previous.workflow.state["attempts"].append({**row, "status": "failed"})
    previous.workflow.save()
    study = study_at(tmp_path)
    with study.workflow:
        study.hydrate()
    assert "w4_v6" not in study.exported and "w4_v6" not in study.measured
    assert "w5_v6" in study.measured


def write_partial(directory, study, label="w5_v6"):
    context = {"protocol": "ternary-measurement-context-v1", "identity": study.identity,
               "data_fingerprint": study.data["fingerprint"], "label": label, "backend": "CUDA0/reference"}
    write_json(directory / "measurement-context.json", context)
    write_json(directory / "test/quality.json", quality())
    write_json(directory / "long_test/quality.json", quality())
    write_json(directory / "tasks/tasks.json", tasks())
    write_json(directory / "speed.json", {"completed": False, "rows": [{"completed": False}], "summary": {}})
    return context


def test_failed_measurement_keeps_completed_metrics_and_hash_bound_partial_receipt(tmp_path):
    study = study_at(tmp_path)
    study.data = {"data_path": str(tmp_path / "data.json"), "fingerprint": "data"}
    def command(stage, *args):
        write_partial(stage.directory, study)
        raise TimeoutError("decode budget expired")
    study.command = command
    with study.workflow:
        with pytest.raises(TimeoutError):
            study.worker("measure-w5_v6", "native", label="w5_v6")
        row = study.workflow.state["attempts"][-1]
        assert row["status"] == "failed" and row["partial"]["artifacts"]
        saved = study.saved_worker("measure-w5_v6", label="w5_v6", partial=True)
        assert saved["status"] == "failed" and saved["quality"]["test"]["summary"]["mean_teacher_kl"] == 0
        assert saved["tasks"]["summary"]["math"]["accuracy"] == 1
        assert saved["speed"]["completed"] is False
        result = comparison({}, {}, {"fp16": measurement("fp16"), "w5_v6": saved}, None)
        failed = next(row for row in result["rows"] if row["label"] == "w5_v6")
        assert failed["status"] == "failed" and not result["complete"]
        assert "w5_v6_vs_fp16" in result["paired_tasks"]
        Path(row["directory"], "test/quality.json").write_text('{"tampered": true}')
        assert study.saved_worker("measure-w5_v6", label="w5_v6", partial=True) is None


def test_partial_measurements_need_current_attempt_deployment_binding(tmp_path):
    study = study_at(tmp_path)
    study.data = {"fingerprint": "data"}
    context = write_partial(tmp_path, study)
    assert partial_measurement(tmp_path, {**context, "identity": "different-source"}, "failed") is None
    (tmp_path / "measurement-context.json").unlink()
    assert partial_measurement(tmp_path, context, "failed") is None


def test_unclean_exit_recovers_only_its_bound_attempt_and_then_requires_hashes(tmp_path):
    previous = completed_ledger(tmp_path)
    row = next(row for row in previous.workflow.state["attempts"] if row["name"] == "measure-w5_v6")
    write_partial(Path(row["directory"]), previous)
    row.update(status="running", started_utc_epoch=time.time() - 10, limit_seconds=60)
    row.pop("value")
    row.pop("artifacts")
    previous.workflow.save()
    study = study_at(tmp_path)
    with study.workflow:
        study.hydrate()
        assert study.measured["w5_v6"]["status"] == "failed"
        assert study.measured["w5_v6"]["tasks"]["summary"]["math"]["accuracy"] == 1
        receipt = next(row for row in study.workflow.state["attempts"] if row["name"] == "measure-w5_v6")
        assert receipt["status"] == "interrupted" and receipt["partial"]["artifacts"]
    Path(receipt["directory"], "test/quality.json").write_text('{"changed": true}')
    resumed = study_at(tmp_path)
    with resumed.workflow:
        resumed.hydrate()
        assert "w5_v6" not in resumed.measured


def test_missing_measurement_sections_never_complete_or_crash_pairing():
    results = {label: {"status": "complete"} for label in LABELS}
    report = comparison({}, {}, results, None)
    assert not report["complete"] and report["paired_tasks"] == {}
    assert all(row["status"] == "pending" for row in report["rows"])


def test_comparison_retains_training_compute_and_numerical_policy():
    metrics = {"recovery": {"optimiser_seen_tokens": 1024, "wall_seconds_this_attempt": 2.0,
                             "peak_cuda_allocated_bytes": None,
                             "numerics": {"policy": "model-scoped"}}}
    report = comparison({"ternary_recovered": {"metrics": metrics}}, {}, {}, None)
    row = next(row for row in report["rows"] if row["label"] == "ternary_recovered")
    assert row["training_metrics"] == metrics


def test_deployment_format_gate_precedes_partial_measurement_context(tmp_path):
    worker = TernaryWorker.__new__(TernaryWorker)
    worker.args = SimpleNamespace(phase="native", label="w5_v6", identity="run")
    worker.output = tmp_path
    worker.data = {"fingerprint": "data"}
    with pytest.raises(ValueError, match="dispatch"):
        worker.native_results(SimpleNamespace(custom_ops=False), None, None, None)
    assert not (tmp_path / "measurement-context.json").exists()
