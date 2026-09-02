"""The exact-byte dynamic workflow must remain reproducible and comparable."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen35_4b_dynamic_mixed_precision_colab.ipynb"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dynamic_row(arm: str, *, seed: int = 0, **updates):
    row = {
        "stage": "dynamic",
        "arm": arm,
        "seed": seed,
        "evaluation_halted": False,
        "mean_teacher_kl": 0.020,
        "p95_teacher_kl": 0.070,
        "top1_agreement": 0.930,
        "nll_delta": 0.010,
        "ppl_wikitext2": 9.8,
        "ppl_c4": 14.2,
        "diverse_mean_teacher_kl": 0.030,
        "diverse_top1_agreement": 0.900,
        "diverse_trajectory_token_agreement": 0.55,
        "complete_persistent_model_bytes": 3_584_533_344,
        "dynamic_actual_target_match": True,
        "logit_fidelity_input_hashes": ["prompt-a", "prompt-b"],
    }
    row.update(updates)
    return row


def test_dynamic_selector_requires_exact_bytes_and_keeps_random_control():
    selector = _load_script("select_qwen35_dynamic_finalists")
    summary = {"code_revision": "abc", "rows": [
        _dynamic_row("uniform_scale8_w4"),
        _dynamic_row("random_mixed_fwht", mean_teacher_kl=0.0198),
        _dynamic_row("dynamic_mixed_unrotated", mean_teacher_kl=0.0190),
        _dynamic_row(
            "dynamic_mixed_fwht",
            mean_teacher_kl=0.0185,
            dynamic_actual_target_match=False,
        ),
        _dynamic_row(
            "dynamic_mixed_signs_fp16",
            diverse_trajectory_token_agreement=None,
            evaluation_halted=True,
        ),
    ]}
    result = selector.select_finalists(summary)

    assert result["finalists"] == ["dynamic_mixed_unrotated"]
    assert result["confirmation_arms"] == [
        "uniform_scale8_w4", "random_mixed_fwht",
        "dynamic_mixed_unrotated",
    ]
    decisions = {decision["arm"]: decision for decision in result["decisions"]}
    assert not decisions["dynamic_mixed_fwht"]["guards"]["target_bytes"]
    assert not decisions["dynamic_mixed_fwht"]["selected"]
    assert decisions["dynamic_mixed_signs_fp16"]["evaluation_halted"] is True
    assert decisions["dynamic_mixed_signs_fp16"]["missing_metrics"] == [
        "diverse_trajectory_token_agreement"
    ]


def test_unsloth_comparison_requires_prompt_match_and_reports_byte_gate():
    comparison = _load_script("compare_qwen35_dynamic_to_unsloth")
    summary = {"code_revision": "abc", "rows": [
        _dynamic_row(
            "dynamic_mixed_fwht", seed=0,
            packed_artifact_bytes=3_584_533_344,
        ),
        _dynamic_row(
            "dynamic_mixed_fwht", seed=1,
            mean_teacher_kl=0.018,
            packed_artifact_bytes=3_584_533_344,
        ),
    ]}
    unsloth = {
        "candidate": {"complete_artifact_bytes": 3_584_533_344},
        "metrics": {
            "input_hashes": ["prompt-a", "prompt-b"],
            "mean_teacher_kl": 0.01294,
            "p95_teacher_kl": 0.03614,
            "top1_agreement": 0.95,
            "nll_delta": 0.004,
        },
        "collection_fingerprint": "collection",
        "prompt_manifest_fingerprint": "prompts",
    }
    result = comparison.compare(summary, unsloth, ["dynamic_mixed_fwht"])
    row = result["comparisons"][0]
    assert row["within_byte_gate"] is True
    assert row["candidate_metrics"]["mean_teacher_kl"] == pytest.approx(0.019)
    assert row["relative_kl_delta"] > 0

    summary["rows"][1]["logit_fidelity_input_hashes"] = ["different"]
    with pytest.raises(ValueError, match="input hashes differ"):
        comparison.compare(summary, unsloth, ["dynamic_mixed_fwht"])


def test_generated_notebook_matches_builder_and_compiles(tmp_path):
    builder = _load_script("build_qwen35_dynamic_mixed_notebook")
    generated = builder.build_notebook()
    committed = nbformat.read(NOTEBOOK, as_version=4)
    # nbformat creates fresh cell IDs on each build; source and metadata are
    # the stable generated contract.
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
        "RUN_SIGN_REPLICATION = True",
        "RUN_DYNAMIC_SCREEN = True",
        "RUN_DYNAMIC_CONFIRMATION = True",
        "RUN_UNSLOTH_KL = True",
        "ROTQUANT_TOKEN_CACHE_DIR",
        "--stage", "dynamic",
        "select_qwen35_dynamic_finalists.py",
        "compare_qwen35_dynamic_to_unsloth.py",
    ):
        assert expected in source
    for index, cell in enumerate(committed.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook-cell-{index}", "exec")


def test_dynamic_workflow_dry_run_is_runnable(tmp_path):
    output = subprocess.check_output([
        sys.executable,
        str(ROOT / "scripts" / "run_qwen35_next_stage.py"),
        "--output-dir", str(tmp_path),
        "--stage", "dynamic",
        "--seed", "0",
        "--arm", "dynamic_mixed_fwht",
        "--dry-run",
    ], cwd=ROOT, text=True)
    assert '"total_trials": 1' in output
    assert '"arm": "dynamic_mixed_fwht"' in output
