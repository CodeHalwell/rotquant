"""The focused Qwen3.5 ladder and Colab must remain matched and resumable."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
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


def _load_selector():
    path = ROOT / "scripts" / "select_qwen35_ablation_finalists.py"
    spec = importlib.util.spec_from_file_location(
        "select_qwen35_ablation_finalists", path)
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


def test_failed_optimization_is_decomposed_into_nested_factor_ablation():
    runner = _load_runner()
    trials = runner.stage_trials("ablation")
    assert [trial.arm for trial in trials] == [
        "promoted_w4",
        "scale8_w4",
        "mean_bias_w4",
        "shared_fwht_w4",
        "butterfly_control_w4",
        "butterfly_hessian_w4",
        "butterfly_hessian_signs_w4",
        "shared_butterfly_hessian_signs_w4",
    ]
    resolved = {
        trial.arm: runner._resolved_trial_config(trial, 0) for trial in trials
    }
    promoted = resolved["promoted_w4"]
    assert promoted["quant"]["error_comp"] == "gptq"
    assert promoted["quant"]["scale_bits"] == 16
    assert promoted["quant"]["bias_correction"] == "none"
    assert promoted["patch"]["rotation"] == "fwht"
    assert promoted["patch"]["share_rotations"] is False
    assert resolved["scale8_w4"]["quant"]["scale_bits"] == 8
    assert resolved["mean_bias_w4"]["quant"]["bias_correction"] == "mean"
    assert resolved["shared_fwht_w4"]["patch"]["share_rotations"] is True
    assert resolved["butterfly_hessian_signs_w4"]["patch"][
        "train_rotation"
    ]["learn_signs"] is True
    assert resolved["shared_butterfly_hessian_signs_w4"]["patch"][
        "share_rotations"
    ] is True
    assert promoted["eval"]["fail_fast"] == {
        "mean_teacher_kl_max": 0.25,
        "top1_agreement_min": 0.75,
    }
    assert ("butterfly_hessian_w4", "promoted_w4") in runner.PAIRED_ARMS[
        "ablation"
    ]
    assert (
        "shared_butterfly_hessian_signs_w4", "promoted_w4"
    ) in runner.PAIRED_ARMS["ablation"]


def test_sign_replication_and_dynamic_exact_byte_stages_are_registered():
    runner = _load_runner()
    signs = {trial.arm: runner._resolved_trial_config(trial, 0)
             for trial in runner.stage_trials("signs")}
    assert list(signs) == [
        "promoted_w4", "learned_signs_fp32_w4", "learned_signs_fp16_w4"
    ]
    assert signs["learned_signs_fp32_w4"]["patch"][
        "rotation_storage_dtype"
    ] == "float32"
    assert signs["learned_signs_fp16_w4"]["patch"][
        "rotation_storage_dtype"
    ] == "float16"

    dynamic = {trial.arm: runner._resolved_trial_config(trial, 0)
               for trial in runner.stage_trials("dynamic")}
    assert list(dynamic) == [
        "source_fp16", "uniform_w3", "uniform_scale8_w4",
        "random_mixed_fwht", "dynamic_mixed_unrotated",
        "dynamic_mixed_fwht", "dynamic_mixed_signs_fp16",
    ]
    search = dynamic["dynamic_mixed_fwht"]["patch"]["dynamic"]
    assert search["candidate_bits"] == [2, 3, 4, 5, 6, 8]
    assert search["target_complete_bytes"] == 3_584_533_344
    assert search["require_target_match"] is True
    assert dynamic["random_mixed_fwht"]["patch"]["dynamic"][
        "allocation"
    ] == "random"
    assert dynamic["dynamic_mixed_signs_fp16"]["patch"][
        "train_rotation"
    ]["learn_signs"] is True


def test_ablation_finalist_selection_requires_guards_and_a_material_signal():
    selector = _load_selector()

    def row(arm, **updates):
        payload = {
            "stage": "ablation",
            "arm": arm,
            "seed": 0,
            "evaluation_halted": False,
            "mean_teacher_kl": 0.020,
            "top1_agreement": 0.92,
            "ppl_wikitext2": 15.0,
            "ppl_c4": 17.0,
            "trajectory_token_agreement": 0.90,
            "complete_persistent_model_bytes": 4_000_000_000,
        }
        payload.update(updates)
        return payload

    summary = {"code_revision": "abc", "rows": [
        row("promoted_w4"),
        row("scale8_w4", complete_persistent_model_bytes=3_900_000_000),
        row("neutral_w4"),
        row("regressed_w4", mean_teacher_kl=0.022),
        row("halted_w4", evaluation_halted=True, mean_teacher_kl=0.001),
    ]}
    result = selector.select_finalists(summary)

    assert result["finalists"] == ["scale8_w4"]
    assert result["confirmation_arms"] == ["promoted_w4", "scale8_w4"]
    decisions = {row["arm"]: row for row in result["decisions"]}
    assert decisions["scale8_w4"]["signals"][
        "complete_persistent_model_bytes"
    ]
    assert not decisions["neutral_w4"]["selected"]
    assert not decisions["regressed_w4"]["guards"]["mean_teacher_kl"]
    assert not decisions["halted_w4"]["selected"]


def test_next_stage_dry_run_filters_to_requested_arms(tmp_path):
    output = subprocess.check_output([
        sys.executable,
        str(ROOT / "scripts" / "run_qwen35_next_stage.py"),
        "--output-dir", str(tmp_path),
        "--stage", "ablation",
        "--seed", "1",
        "--arm", "scale8_w4",
        "--dry-run",
    ], cwd=ROOT, text=True)
    assert '"total_trials": 1' in output
    assert '"arm": "scale8_w4"' in output
    assert '"seed": 1' in output
    assert "mean_bias_w4" not in output


def test_runner_persists_progress_and_partial_summary_after_each_arm(
    tmp_path, monkeypatch
):
    runner = _load_runner()

    def fake_run_trial(trial, **kwargs):
        assert kwargs["position"] == 1
        assert kwargs["total"] == 1
        assert kwargs["heartbeat_seconds"] == 0
        return {"run_id": "fake", "metrics": {}}, 12.0, False

    monkeypatch.setattr(runner, "_run_trial", fake_run_trial)
    monkeypatch.setattr(sys, "argv", [
        "run_qwen35_next_stage.py",
        "--output-dir", str(tmp_path),
        "--stage", "ablation",
        "--seed", "0",
        "--arm", "promoted_w4",
        "--heartbeat-seconds", "0",
    ])
    runner.main()

    progress = json.loads((tmp_path / "next_stage_progress.json").read_text())
    assert progress["state"] == "complete"
    assert progress["completed_trials"] == 1
    assert progress["completed"][0]["arm"] == "promoted_w4"
    partial = json.loads(
        (tmp_path / "next_stage_partial_summary.json").read_text())
    final = json.loads((tmp_path / "next_stage_summary.json").read_text())
    assert partial["complete"] is False
    assert final["complete"] is True


def test_long_kv_uses_promoted_w4_and_two_bit_e8p_only():
    runner = _load_runner()
    trials = runner.stage_trials("long-kv")
    assert [trial.arm for trial in trials] == [
        "source_fp16_e8", "promoted_w4_e8", "w4a8_e8"
    ]
    resolved = {
        trial.arm: runner._resolved_trial_config(trial, 0) for trial in trials
    }
    promoted = resolved["promoted_w4_e8"]
    assert promoted["quant"]["error_comp"] == "gptq"
    assert promoted["quant"]["scale_bits"] == 16
    assert promoted["quant"]["bias_correction"] == "none"
    assert promoted["patch"]["rotation"] == "fwht"
    assert promoted["patch"]["share_rotations"] is False
    assert promoted["patch"].get("activation_bits") is None
    assert promoted["eval"]["kv_cache"]["key_bits"] == 2
    assert promoted["eval"]["kv_cache"]["value_bits"] == 2
    assert promoted["eval"]["kv_cache"]["codebook"] == "e8p"
    assert promoted["eval"]["kv_cache"]["prompt_len"] == 8192
    assert resolved["w4a8_e8"]["patch"]["activation_bits"] == 8


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
    assert "RUN_ABLATION_SCREEN = True" in source
    assert "RUN_ABLATION_CONFIRM = True" in source
    assert "RUN_LONG_CONTEXT_KV = True" in source
    assert "RUN_UNSLOTH_KL = True" in source
    assert "CONFIRM_SEEDS = (0, 1, 2)" in source
    assert "REQUIRE_FAST_HADAMARD = True" in source
    assert "bounded source build" in source
    assert "--dry-run" in source
    assert "--heartbeat-seconds" in source
    assert "next_stage_progress.json" in source
    assert "Persistent log:" in source
    assert "notebook heartbeat" in source
    assert 'sys.executable, "-u"' in source
    assert "subprocess.Popen" in source
    assert "stderr=subprocess.STDOUT" in source
    assert 'child_env["PYTHONUNBUFFERED"] = "1"' in source
    assert "select_qwen35_ablation_finalists.py" in source
    assert "FINALIST_ARMS" in source
    assert "run_unsloth_qwen35_4b_kl.py" in source
    assert "UD-Q4_K_XL" in source
    assert "llama_cpp_python" in source
    assert '"diskcache==5.6.3"' in source
    assert "import diskcache, llama_cpp" in source
    assert "ppl_wikitext2" in source
    assert "kl_vs_control" in source
    assert "trajectory_token_agreement" in source
    assert "paired_comparisons" in source
    assert "kv_endpoint_passed" in source
    assert "full_experiment_validation.json" in source
    assert "300-prompt" in source
