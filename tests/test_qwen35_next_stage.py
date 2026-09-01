"""The focused Qwen3.5 ladder and Colab must remain matched and resumable."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen35_4b_optimization_stage_colab.ipynb"


def _load_runner():
    path = ROOT / "scripts" / "run_qwen35_next_stage.py"
    spec = importlib.util.spec_from_file_location("run_qwen35_next_stage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_w4_ladder_has_source_and_matched_codebook_controls():
    runner = _load_runner()
    trials = runner.stage_trials("w4")
    assert [trial.arm for trial in trials] == [
        "source_fp16",
        "gaussian_w4_control",
        "calibrated_w4_control",
        "gptq_gaussian_w4",
        "gptq_calibrated_w4",
    ]
    controls = {trial.arm: dict(trial.overrides) for trial in trials}
    assert controls["gaussian_w4_control"]["quant.error_comp"] == "none"
    assert controls["gptq_gaussian_w4"]["quant.error_comp"] == "gptq"
    assert controls["calibrated_w4_control"]["quant.codebook"] == "calibrated"
    assert controls["gptq_calibrated_w4"]["quant.codebook"] == "calibrated"


def test_w4a8_ladder_anchors_each_composition_step_to_its_predecessor():
    runner = _load_runner()
    trials = runner.stage_trials("w4a8")
    assert [trial.arm for trial in trials] == [
        "source_fp16",
        "promoted_w4",
        "optimized_w4",
        "w4a8",
        "w4a8_e8",
    ]
    overrides = {trial.arm: dict(trial.overrides) for trial in trials}
    promoted = overrides["promoted_w4"]
    assert promoted["patch.rotation"] == "fwht"
    assert promoted["patch.train_rotation"] is None
    assert promoted["patch.share_rotations"] is False
    assert promoted["patch.activation_bits"] is None
    assert promoted["quant.scale_bits"] == 16
    assert promoted["quant.bias_correction"] == "none"
    resolved = runner._resolved_trial_config(trials[1], seed=0)
    assert resolved["n_calib"] == 128
    assert resolved["calib_seq_len"] == 512
    assert resolved["quant"]["codebook"] == "gaussian"
    assert resolved["quant"]["error_comp"] == "gptq"
    assert resolved["quant"]["group_size"] == 128
    w4a8 = overrides["w4a8"]
    assert w4a8["patch.rotation"] == "fwht"
    assert w4a8["patch.train_rotation"] is None
    assert w4a8["patch.activation_bits"] == 8
    assert overrides["w4a8_e8"]["patch.train_rotation"] is None
    assert runner.PAIRED_ARMS["w4a8"] == (
        ("optimized_w4", "promoted_w4"),
        ("w4a8", "promoted_w4"),
        ("w4a8_e8", "w4a8"),
    )


def test_summary_uses_same_stage_seed_source_ppl():
    runner = _load_runner()
    source, candidate = runner.stage_trials("w4")[:2]
    source_payload = {
        "run_id": "source",
        "metrics": {"ppl_wikitext2": 10.0, "ppl_c4": 20.0},
    }
    candidate_payload = {
        "run_id": "candidate",
        "metrics": {"ppl_wikitext2": 11.0, "ppl_c4": 18.0},
    }
    rows = runner.summarize_results([
        (source, 0, source_payload, 1.0, False),
        (candidate, 0, candidate_payload, 2.0, False),
    ])
    assert rows[1]["ppl_wikitext2_relative_to_source"] == pytest.approx(0.1)
    assert rows[1]["ppl_c4_relative_to_source"] == pytest.approx(-0.1)


def test_paired_comparison_uses_matched_windows_and_prompt_hashes():
    runner = _load_runner()
    trials = {trial.arm: trial for trial in runner.stage_trials("w4")}

    def payload(run_id, offset):
        return {
            "run_id": run_id,
            "metrics": {
                "ppl_wikitext2_details": {
                    "window_hashes": ["a", "b"],
                    "window_mean_nll": [1.0 + offset, 2.0 + offset],
                },
                "logit_fidelity": {"prompt_metrics": [{
                    "input_hash": "prompt",
                    "mean_teacher_kl": 0.1 + offset,
                    "top1_agreement": 0.9 - offset,
                    "nll_delta": 0.2 + offset,
                }]},
                "trajectory": {"prompt_metrics": [{
                    "input_hash": "prompt",
                    "token_agreement": 0.8 - offset,
                    "exact_trajectory_rate": 1.0 - offset,
                    "mean_matching_prefix": 20.0 - offset,
                }]},
            },
        }

    comparisons = runner.paired_comparisons([
        (trials["gaussian_w4_control"], 0, payload("control", 0.0), 1.0, False),
        (trials["gptq_gaussian_w4"], 0, payload("candidate", 0.1), 1.0, False),
    ], draws=20)
    report = comparisons[0]
    assert report["candidate_arm"] == "gptq_gaussian_w4"
    assert report["metrics"]["wikitext2_mean_nll"]["mean_delta"] == pytest.approx(0.1)
    assert report["metrics"]["logit_fidelity.mean_teacher_kl"][
        "mean_delta"
    ] == pytest.approx(0.1)


def test_next_stage_notebook_is_valid_compilable_and_fail_safe():
    payload = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    assert payload["nbformat"] == 4
    assert payload["metadata"]["kernelspec"]["name"] == "python3"
    assert all(cell.get("id") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")

    assert 'REPO_REF = "main"' in source
    assert 'run_git(["fetch", "--force", "origin", REPO_REF]' in source
    assert 'run_git(["checkout", "--detach", "FETCH_HEAD"]' in source
    assert "RUN_W4 = True" in source
    assert "RUN_RECOVERY = False" in source
    assert "REQUIRE_FAST_HADAMARD = True" in source
    assert "bounded source build" in source
    assert "--dry-run" in source
    assert 'sys.executable, "-u"' in source
    assert "subprocess.Popen" in source
    assert "stderr=subprocess.STDOUT" in source
    assert 'child_env["PYTHONUNBUFFERED"] = "1"' in source
    assert "ppl_wikitext2_relative_to_source" in source
    assert "trajectory_token_agreement" in source
    assert "paired_comparisons" in source
    assert "300-prompt" in source
