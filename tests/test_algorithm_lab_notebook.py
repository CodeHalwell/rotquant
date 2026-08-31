"""The algorithm-lab Colab must remain structurally runnable and fail-safe."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "rotquant_algorithm_lab_colab.ipynb"


def _notebook_source(payload: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_algorithm_lab_notebook_is_valid_and_code_cells_compile():
    payload = json.loads(NOTEBOOK.read_text())

    assert payload["nbformat"] == 4
    assert payload["metadata"]["kernelspec"]["name"] == "python3"
    assert all(cell.get("id") for cell in payload["cells"])

    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{NOTEBOOK}:cell-{index}",
            )


def test_algorithm_lab_notebook_covers_every_track_and_requires_confirmation():
    payload = json.loads(NOTEBOOK.read_text())
    source = _notebook_source(payload)

    assert "CONFIRM_EXPENSIVE_RUN = False" in source
    assert "algorithmic_trial_matrix" in source
    assert "MIN_ALLOCATION_BYTE_SAVING" in source
    assert "RUN_CROSS_FAMILY = True" in source
    assert "RUN_TRAJECTORY_VALIDATION = True" in source
    assert "RUN_LOGIT_FIDELITY = True" in source
    assert "RUN_SCREEN_LAYER_DRIFT = True" in source
    assert "RUN_LAYER_DRIFT = True" in source
    assert "REQUIRE_FAST_HADAMARD = True" in source
    assert 'stage not in {\n        "validation", "cross_family"' in source
    assert "trajectory_token_agreement" in source
    assert "paired_window_statistics" in source
    assert "MATCHED_CONTROL_MAP" in source
    assert "complete_persistent_model_bytes" in source
    assert "quarantined incomplete record" in source
    assert "Divergence-300-style" in source
    assert "oracle_value_retrieval_curve" in source
    assert '"packed_key_recall_measured": False' in source

    for heading in (
        "Verify the CUDA runtime",
        "synthetic exact-rate preflight",
        "content-addressed, resumable model runner",
        "source/W4 fail-fast sentinel",
        "prioritized primary-model seed-0 screen",
        "Promote only bounded, matched-control Pareto candidates",
        "Validate promoted profiles across three full primary-model seeds",
        "Confirm promoted profiles on a second model family",
        "Collect real attention probabilities and cache values",
        "Measure source and quantized-value retrieval curves",
        "Persist the audit trail and compact archive",
    ):
        assert heading in source
