"""Format-aware allocator must remain distinct, paired, and Colab-runnable."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen35_4b_allocator_v4_colab.ipynb"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(arm: str, seed: int = 0, signature: str | None = None, **updates):
    row = {
        "stage": "allocator-v4",
        "arm": arm,
        "seed": seed,
        "evaluation_halted": False,
        "mean_teacher_kl": 0.030,
        "top1_agreement": 0.91,
        "ppl_wikitext2": 10.0,
        "ppl_c4": 14.5,
        "diverse_mean_teacher_kl": 0.080,
        "diverse_top1_agreement": 0.85,
        "diverse_trajectory_token_agreement": 0.37,
        "complete_persistent_model_bytes": 3_567_200_000,
        "dynamic_estimated_artifact_bytes": 3_587_000_000,
        "dynamic_target_artifact_bytes": 3_584_533_344,
        "dynamic_actual_target_match": True,
        "dynamic_scoring_matches_deployed": True,
        "dynamic_allocation_fingerprint": signature or arm,
        "dynamic_counts_by_format": {"gaussian_w3_g128": 80},
    }
    row.update(updates)
    return row


def test_allocator_v4_stage_is_bounded_and_has_matched_controls():
    runner = _load_script("run_qwen35_next_stage")
    trials = runner.stage_trials("allocator-v4")
    assert [trial.arm for trial in trials] == [
        "source_fp16",
        "uniform_scale8_w4",
        "bits_only_pareto",
        "format_random_exact",
        "format_pareto_all",
        "format_pareto_gaussian",
        "format_pareto_g128",
    ]
    resolved = {trial.arm: runner._resolved_trial_config(trial, 0) for trial in trials}
    palette = resolved["format_pareto_all"]["patch"]["dynamic"]["candidate_formats"]
    assert len(palette) == 6
    assert {item["bits"] for item in palette} == {3, 4, 5}
    assert {item["codebook"] for item in palette} == {"gaussian", "calibrated"}
    assert resolved["bits_only_pareto"]["patch"]["dynamic"]["candidate_formats"] == []
    assert resolved["format_random_exact"]["patch"]["dynamic"]["allocation"] == "random_pareto"
    assert (
        "format_pareto_all", "bits_only_pareto"
    ) in runner.PAIRED_ARMS["allocator-v4"]


def test_allocator_v4_selector_requires_improvement_over_bits_only():
    selector = _load_script("select_qwen35_allocator_v4_finalists")
    rows = [
        _row("source_fp16", mean_teacher_kl=0.0, top1_agreement=1.0),
        _row("uniform_scale8_w4", mean_teacher_kl=0.017),
        _row("bits_only_pareto", mean_teacher_kl=0.031),
        _row("format_random_exact", mean_teacher_kl=0.11),
        _row(
            "format_pareto_all", signature="same", mean_teacher_kl=0.028,
            diverse_mean_teacher_kl=0.072,
        ),
        _row(
            "format_pareto_gaussian", signature="same", mean_teacher_kl=0.029,
            diverse_mean_teacher_kl=0.074,
        ),
        _row(
            "format_pareto_g128", signature="g128", mean_teacher_kl=0.032,
            diverse_mean_teacher_kl=0.082,
        ),
    ]
    result = selector.select_finalists({
        "complete": True,
        "seeds": [0],
        "code_revision": "abc",
        "rows": rows,
    })
    assert result["finalists"] == ["format_pareto_all"]
    assert result["deduplicated_recipes"] == {
        "format_pareto_gaussian": "format_pareto_all"
    }
    assert result["confirmation_arms"] == [
        "uniform_scale8_w4", "bits_only_pareto", "format_random_exact",
        "format_pareto_all",
    ]


def _confirmation_inputs():
    rows = []
    pairs = []
    for seed in (0, 1, 2):
        rows.extend([
            _row("bits_only_pareto", seed, mean_teacher_kl=0.031),
            _row("format_random_exact", seed, mean_teacher_kl=0.11),
            _row(
                "format_pareto_all", seed, mean_teacher_kl=0.028,
                top1_agreement=0.915,
            ),
        ])
        for baseline in ("bits_only_pareto", "format_random_exact"):
            pairs.append({
                "stage": "allocator-v4",
                "candidate_arm": "format_pareto_all",
                "baseline_arm": baseline,
                "seed": seed,
                "metrics": {
                    "logit_fidelity.mean_teacher_kl": {
                        "bootstrap_95_ci": [-0.01, -0.001],
                        "paired_samples": 24, "interval_reliable": True,
                    }
                },
            })
    summary = {
        "complete": True,
        "seeds": [0, 1, 2],
        "code_revision": "abc",
        "rows": rows,
        "paired_comparisons": pairs,
    }
    selection = {
        "code_revision": "abc",
        "finalists": ["format_pareto_all"],
        "bits_baseline_arm": "bits_only_pareto",
        "random_control_arm": "format_random_exact",
    }
    comparison = {
        "rotquant_code_revision": "abc",
        "prompt_hashes_match": True,
        "byte_tolerance_fraction": 0.01,
        "comparisons": [{
            "arm": "format_pareto_all",
            "within_byte_gate": True,
            "byte_basis": "measured_artifact",
            "baseline_bytes": 3_584_533_344,
            "exported_seeds": [0],
            "candidate_metrics": {
                "mean_teacher_kl": 0.028, "top1_agreement": 0.915,
            },
            "unsloth_metrics": {
                "mean_teacher_kl": 0.012, "top1_agreement": 0.941,
            },
        }],
    }
    identity = {"code_revision": "abc", "seed": 0,
                "stage": "allocator-v4", "arm": "format_pareto_all",
                "trial_fingerprint": "trial", "allocation_fingerprint": "format_pareto_all"}
    rows[2]["packed_artifact_identity"] = identity
    rows[2]["trial_fingerprint"] = "trial"
    rows[2]["packed_manifest_sha256"] = "a" * 64
    rows[2]["packed_artifact_bytes"] = 3_584_533_344
    comparison["comparisons"][0]["artifact_measurements"] = [
        {"seed": 0, "identity": identity, "manifest_sha256": "a" * 64,
         "bytes": 3_584_533_344}
    ]
    return summary, selection, comparison


def test_allocator_v4_assessor_requires_paired_bits_and_random_wins():
    assessor = _load_script("assess_qwen35_allocator_v4")
    summary, selection, comparison = _confirmation_inputs()
    result = assessor.assess(summary, selection, comparison)
    assert result["format_allocator_winners"] == ["format_pareto_all"]
    assert result["provider_competitive_winners"] == []


@pytest.mark.parametrize("metric,value", [
    ("top1_agreement", 0.80), ("diverse_mean_teacher_kl", 0.16),
    ("diverse_top1_agreement", 0.50), ("diverse_trajectory_token_agreement", 0.0),
    ("ppl_wikitext2", float("nan")), ("ppl_c4", float("inf")),
    ("mean_teacher_kl", None),
])
def test_confirmation_reapplies_all_quality_guards(metric, value):
    assessor = _load_script("assess_qwen35_allocator_v4")
    summary, selection, comparison = _confirmation_inputs()
    # A small secondary win cannot cancel an unrelated severe regression.
    for row in summary["rows"]:
        if row["arm"] == "format_pareto_all":
            row["ppl_wikitext2"] = 9.9
            if row["seed"] != 0:
                row[metric] = value
    result = assessor.assess(summary, selection, comparison)
    assert result["format_allocator_winners"] == []
    assert not result["decisions"][0]["gates"]["no_quality_regressions_all_seeds"]


@pytest.mark.parametrize("field,value", [
    ("interval_reliable", False), ("paired_samples", 2),
    ("paired_samples", None), ("bootstrap_95_ci", [float("nan"), -0.01]),
    ("bootstrap_95_ci", [-0.001, -0.01]),
])
def test_confirmation_requires_reliable_finite_paired_evidence(field, value):
    assessor = _load_script("assess_qwen35_allocator_v4")
    summary, selection, comparison = _confirmation_inputs()
    summary["paired_comparisons"][2]["metrics"]["logit_fidelity.mean_teacher_kl"][field] = value
    assert not assessor.assess(summary, selection, comparison)["format_allocator_winners"]


@pytest.mark.parametrize("failure", ["estimate", "missing", "stale", "wrong_seed"])
def test_confirmation_requires_matching_measured_seed_zero_export(failure):
    assessor = _load_script("assess_qwen35_allocator_v4")
    summary, selection, comparison = copy.deepcopy(_confirmation_inputs())
    exported = comparison["comparisons"][0]
    if failure == "estimate":
        exported["byte_basis"] = "persistent_tensor_estimate"
    elif failure == "missing":
        exported["artifact_measurements"] = []
    elif failure == "stale":
        exported["artifact_measurements"][0]["identity"] = {"code_revision": "old"}
    else:
        exported["exported_seeds"] = [1]
    assert not assessor.assess(summary, selection, comparison)["format_allocator_winners"]


def test_allocator_v4_notebook_matches_builder_and_compiles():
    builder = _load_script("build_qwen35_allocator_v4_notebook")
    generated = builder.build_notebook()
    committed = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(generated)
    nbformat.validate(committed)
    assert [cell.source for cell in generated.cells] == [
        cell.source for cell in committed.cells
    ]
    assert generated.metadata == committed.metadata
    source = "\n".join(cell.source for cell in committed.cells)
    for expected in (
        '"--stage", "allocator-v4"',
        "select_qwen35_allocator_v4_finalists.py",
        "assess_qwen35_allocator_v4.py",
        "bits_only_pareto",
        "format_random_exact",
        "dynamic_counts_by_format",
        "preflight_allocator_v4.py",
    ):
        assert expected in source
    for index, cell in enumerate(committed.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")


def test_allocator_v4_synthetic_preflight():
    preflight = _load_script("preflight_allocator_v4")
    result = preflight.preflight("cpu")
    assert result["passed"] is True
    assert len(result["formats"]) == 6


def test_allocator_v4_dry_run_is_runnable(tmp_path):
    output = subprocess.check_output([
        sys.executable,
        str(ROOT / "scripts" / "run_qwen35_next_stage.py"),
        "--output-dir", str(tmp_path),
        "--stage", "allocator-v4",
        "--seed", "0",
        "--arm", "format_pareto_all",
        "--dry-run",
    ], cwd=ROOT, text=True)
    assert '"total_trials": 1' in output
    assert '"arm": "format_pareto_all"' in output
