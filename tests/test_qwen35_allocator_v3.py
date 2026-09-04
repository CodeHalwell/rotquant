"""Allocator-v3 must be exact-size, distinct, paired, and Colab-runnable."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen35_4b_allocator_v3_colab.ipynb"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(arm: str, signature: str | None = None, **updates):
    row = {
        "stage": "allocator-v3",
        "arm": arm,
        "seed": 0,
        "evaluation_halted": False,
        "mean_teacher_kl": 0.10,
        "top1_agreement": 0.85,
        "ppl_wikitext2": 10.5,
        "ppl_c4": 15.0,
        "diverse_mean_teacher_kl": 0.20,
        "diverse_top1_agreement": 0.75,
        "diverse_trajectory_token_agreement": 0.25,
        "complete_persistent_model_bytes": 3_563_000_000,
        "dynamic_estimated_artifact_bytes": 3_583_750_000,
        "dynamic_target_artifact_bytes": 3_584_533_344,
        "dynamic_actual_target_match": True,
        "dynamic_scoring_matches_deployed": True,
        "dynamic_allocation_fingerprint": signature,
    }
    row.update(updates)
    return row


def test_allocator_v3_stage_has_binding_high_precision_and_exact_random():
    runner = _load_script("run_qwen35_next_stage")
    trials = runner.stage_trials("allocator-v3")
    assert [trial.arm for trial in trials] == [
        "source_fp16",
        "uniform_scale8_w4",
        "random_broad_exact",
        "pareto_global",
        "pareto_global_refined",
        "pareto_w6_top5_refined",
        "pareto_w8_top1_refined",
        "pareto_w8_top2p5_refined",
        "pareto_w8_top5_refined",
    ]
    resolved = {trial.arm: runner._resolved_trial_config(trial, 0) for trial in trials}
    base = resolved["pareto_global"]["patch"]["dynamic"]
    assert base["target_artifact_bytes"] == 3_584_533_344
    assert base["artifact_overhead_bytes"] == 20_750_000
    assert base["target_tolerance_fraction"] == 0.001
    assert resolved["random_broad_exact"]["patch"]["dynamic"]["allocation"] == "random_pareto"
    assert resolved["pareto_global_refined"]["patch"]["dynamic"]["refinement_passes"] == 8
    assert resolved["pareto_w8_top2p5_refined"]["patch"]["dynamic"]["protect_min_bits"] == 8
    assert (
        "pareto_w8_top2p5_refined",
        "random_broad_exact",
    ) in runner.PAIRED_ARMS["allocator-v3"]


def test_allocator_v3_selector_deduplicates_deployed_recipes():
    selector = _load_script("select_qwen35_allocator_v3_finalists")
    rows = [
        _row("uniform_scale8_w4", mean_teacher_kl=0.02),
        _row("random_broad_exact", "random"),
        _row("pareto_global", "same", mean_teacher_kl=0.09, top1_agreement=0.86),
        _row("pareto_global_refined", "same", mean_teacher_kl=0.08, top1_agreement=0.87),
        _row("pareto_w6_top5_refined", "w6", mean_teacher_kl=0.085, top1_agreement=0.86),
        _row("pareto_w8_top1_refined", "w8-1", mean_teacher_kl=0.12),
        _row("pareto_w8_top2p5_refined", "w8-2", mean_teacher_kl=0.12),
        _row("pareto_w8_top5_refined", "w8-5", mean_teacher_kl=0.12),
    ]
    result = selector.select_finalists(
        {
            "code_revision": "abc",
            "seeds": [0],
            "rows": rows,
            "complete": True,
        },
        limit=3,
    )
    assert result["finalists"] == ["pareto_global_refined", "pareto_w6_top5_refined"]
    assert result["deduplicated_recipes"] == {"pareto_global": "pareto_global_refined"}
    assert result["confirmation_arms"] == [
        "uniform_scale8_w4",
        "random_broad_exact",
        "pareto_global_refined",
        "pareto_w6_top5_refined",
    ]


def test_allocator_v3_assessor_requires_direct_paired_random_evidence():
    assessor = _load_script("assess_qwen35_allocator_v3")
    rows = []
    paired = []
    for seed in (0, 1, 2):
        rows.extend(
            [
                _row("random_broad_exact", f"random-{seed}", seed=seed),
                _row(
                    "pareto_global_refined",
                    f"winner-{seed}",
                    seed=seed,
                    mean_teacher_kl=0.08,
                    top1_agreement=0.87,
                    ppl_wikitext2=10.2,
                    ppl_c4=14.8,
                    diverse_mean_teacher_kl=0.17,
                    diverse_top1_agreement=0.77,
                    diverse_trajectory_token_agreement=0.28,
                ),
            ]
        )
        paired.append(
            {
                "stage": "allocator-v3",
                "seed": seed,
                "candidate_arm": "pareto_global_refined",
                "baseline_arm": "random_broad_exact",
                "metrics": {"logit_fidelity.mean_teacher_kl": {"bootstrap_95_ci": [-0.03, -0.01]}},
            }
        )
    summary = {
        "code_revision": "abc",
        "seeds": [0, 1, 2],
        "rows": rows,
        "paired_comparisons": paired,
        "complete": True,
    }
    selection = {
        "code_revision": "abc",
        "finalists": ["pareto_global_refined"],
        "random_control_arm": "random_broad_exact",
    }
    comparison = {
        "rotquant_code_revision": "abc",
        "prompt_hashes_match": True,
        "comparisons": [
            {
                "arm": "pareto_global_refined",
                "within_byte_gate": True,
                "candidate_metrics": {
                    "mean_teacher_kl": 0.08,
                    "top1_agreement": 0.87,
                },
                "unsloth_metrics": {
                    "mean_teacher_kl": 0.01,
                    "top1_agreement": 0.94,
                },
            }
        ],
    }
    result = assessor.assess(summary, selection, comparison)
    assert result["allocator_winners"] == ["pareto_global_refined"]
    assert result["provider_competitive_winners"] == []


def test_allocator_v3_notebook_matches_builder_and_compiles():
    builder = _load_script("build_qwen35_allocator_v3_notebook")
    generated = builder.build_notebook()
    committed = nbformat.read(NOTEBOOK, as_version=4)
    assert [cell.source for cell in generated.cells] == [cell.source for cell in committed.cells]
    assert generated.metadata == committed.metadata
    source = "\n".join(cell.source for cell in committed.cells if cell.cell_type == "code")
    for expected in (
        '"--stage", "allocator-v3"',
        "select_qwen35_allocator_v3_finalists.py",
        "assess_qwen35_allocator_v3.py",
        "missing direct paired random comparisons",
        "excluded_suffixes",
    ):
        assert expected in source
    for index, cell in enumerate(committed.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")


def test_allocator_v3_dry_run_is_runnable(tmp_path):
    output = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_qwen35_next_stage.py"),
            "--output-dir",
            str(tmp_path),
            "--stage",
            "allocator-v3",
            "--seed",
            "0",
            "--arm",
            "pareto_w8_top1_refined",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
    )
    assert '"total_trials": 1' in output
    assert '"arm": "pareto_w8_top1_refined"' in output
