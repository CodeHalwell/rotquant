"""Shared, fail-closed quality guards for screening and confirmation."""
from __future__ import annotations

import math
from typing import Any

QUALITY_GUARDS = {
    "primary_kl": ("mean_teacher_kl", "ratio", 1.02),
    "primary_top1": ("top1_agreement", "delta", -0.01),
    "wikitext2": ("ppl_wikitext2", "ratio", 1.02),
    "c4": ("ppl_c4", "ratio", 1.02),
    "diverse_kl": ("diverse_mean_teacher_kl", "ratio", 1.02),
    "diverse_top1": ("diverse_top1_agreement", "delta", -0.01),
    "trajectory": ("diverse_trajectory_token_agreement", "delta", -0.02),
}


def finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def quality_guards(candidate: dict, baseline: dict) -> dict[str, bool]:
    guards = {}
    for name, (metric, kind, limit) in QUALITY_GUARDS.items():
        value, reference = candidate.get(metric), baseline.get(metric)
        valid = finite_number(value) and finite_number(reference)
        if valid and kind == "ratio":
            valid = reference > 0 and value >= 0 and value <= reference * limit
        elif valid:
            valid = 0 <= reference <= 1 and 0 <= value <= 1 and value >= reference + limit
        guards[name] = bool(valid)
    return guards


def dynamic_row_valid(row: dict) -> bool:
    return (
        row.get("dynamic_actual_target_match") is True
        and row.get("dynamic_scoring_matches_deployed") is True
        and row.get("evaluation_halted") is False
        and isinstance(row.get("dynamic_allocation_fingerprint"), str)
        and bool(row["dynamic_allocation_fingerprint"])
        and finite_number(row.get("complete_persistent_model_bytes"))
        and row["complete_persistent_model_bytes"] > 0
        and all(quality_guards(row, row).values())
    )
