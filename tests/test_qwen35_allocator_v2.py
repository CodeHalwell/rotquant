"""Allocator-v2 must stay faithful, resumable, and statistically guarded."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen35_4b_allocator_v2_colab.ipynb"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(arm: str, **updates):
    row = {
        "stage": "allocator-v2",
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
        "complete_persistent_model_bytes": 3_584_533_344,
        "dynamic_actual_target_match": True,
        "dynamic_scoring_matches_deployed": True,
        "dynamic_proxy_rank_correlation": 0.5,
    }
    row.update(updates)
    return row


def test_allocator_v2_stage_reuses_one_faithful_broad_screen():
    runner = _load_script("run_qwen35_next_stage")
    trials = runner.stage_trials("allocator-v2")
    assert [trial.arm for trial in trials] == [
        "source_fp16",
        "uniform_w3",
        "uniform_scale8_w4",
        "random_adjacent_w34",
        "pareto_adjacent_local",
        "pareto_adjacent_global",
        "pareto_broad_local",
        "pareto_broad_global",
        "pareto_broad_global_protected",
    ]
    resolved = {
        trial.arm: runner._resolved_trial_config(trial, 0) for trial in trials
    }
    global_recipe = resolved["pareto_broad_global"]["patch"]["dynamic"]
    assert global_recipe["scoring_error_comp"] == "inherit"
    assert global_recipe["scoring_scale"] == "inherit"
    assert global_recipe["global_kl_batches"] == 2
    assert global_recipe["allocation"] == "pareto"
    assert global_recipe["target_complete_bytes"] == 3_584_533_344
    adjacent = resolved["pareto_adjacent_global"]["patch"]["dynamic"]
    assert adjacent["allocation_min_bits"] == 3
    assert adjacent["allocation_max_bits"] == 4
    protected = resolved["pareto_broad_global_protected"]["patch"]["dynamic"]
    assert protected["protect_top_fraction"] == 0.10
    assert protected["protect_min_bits"] == 4


def test_allocator_v2_selector_requires_fidelity_and_another_signal():
    selector = _load_script("select_qwen35_allocator_v2_finalists")
    rows = [
        _row("uniform_scale8_w4", mean_teacher_kl=0.02),
        _row("random_adjacent_w34"),
        _row(
            "pareto_adjacent_local",
            mean_teacher_kl=0.095,
            ppl_wikitext2=10.3,
        ),
        _row("pareto_adjacent_global", mean_teacher_kl=0.095),
        _row(
            "pareto_broad_local",
            mean_teacher_kl=0.095,
            ppl_c4=14.8,
            dynamic_scoring_matches_deployed=False,
        ),
        _row(
            "pareto_broad_global",
            mean_teacher_kl=0.12,
            ppl_c4=14.0,
        ),
        _row(
            "pareto_broad_global_protected",
            mean_teacher_kl=0.09,
            diverse_mean_teacher_kl=0.18,
            top1_agreement=0.86,
        ),
    ]
    result = selector.select_finalists({"code_revision": "abc", "rows": rows})
    assert result["finalists"] == [
        "pareto_broad_global_protected", "pareto_adjacent_local"
    ]
    decisions = {entry["arm"]: entry for entry in result["decisions"]}
    assert not decisions["pareto_adjacent_global"]["eligible"]
    assert not decisions["pareto_broad_local"]["guards"]["faithful_scoring"]
    assert not decisions["pareto_broad_global"]["guards"]["primary_kl"]


def test_allocator_v2_notebook_matches_builder_and_compiles():
    builder = _load_script("build_qwen35_allocator_v2_notebook")
    generated = builder.build_notebook()
    committed = nbformat.read(NOTEBOOK, as_version=4)
    assert [cell.cell_type for cell in generated.cells] == [
        cell.cell_type for cell in committed.cells
    ]
    assert [cell.source for cell in generated.cells] == [
        cell.source for cell in committed.cells
    ]
    assert generated.metadata == committed.metadata
    source = "\n".join(
        cell.source for cell in committed.cells if cell.cell_type == "code"
    )
    for expected in (
        '"--stage", "allocator-v2"',
        "select_qwen35_allocator_v2_finalists.py",
        "dynamic_proxy_rank_correlation",
        "notebook heartbeat",
        "RUN_UNSLOTH_KL = True",
    ):
        assert expected in source
    for index, cell in enumerate(committed.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")


def test_allocator_v2_dry_run_is_runnable(tmp_path):
    output = subprocess.check_output([
        sys.executable,
        str(ROOT / "scripts" / "run_qwen35_next_stage.py"),
        "--output-dir", str(tmp_path),
        "--stage", "allocator-v2",
        "--seed", "0",
        "--arm", "pareto_broad_global_protected",
        "--dry-run",
    ], cwd=ROOT, text=True)
    assert '"total_trials": 1' in output
    assert '"arm": "pareto_broad_global_protected"' in output
