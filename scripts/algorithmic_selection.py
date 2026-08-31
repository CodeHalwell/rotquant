"""Pure selection and reporting policy for the Algorithm Lab.

The Colab runner owns model execution and metric collection.  This module keeps
the comparatively small, high-impact decision policy importable and unit
testable without pandas, CUDA, or notebook state.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical(value: Any) -> str:
    """Return a stable representation for nested result metadata."""

    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def outcome_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify screen rows that produced the same deployed outcome.

    Exact equality is intentional.  Approximate matches may represent genuinely
    different recipes and must survive to confirmation.  Dynamic allocation
    counts plus quality and drift prevent a shared headline PPL from collapsing
    unrelated candidates.
    """

    return (
        row.get("track"),
        _finite_number(row.get("complete_persistent_model_bytes")),
        _finite_number(row.get("ppl")),
        _canonical(row.get("dynamic_counts")),
        _finite_number(row.get("mean_layer_nmse")),
        _finite_number(row.get("worst_layer_nmse")),
    )


def pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the storage/PPL Pareto frontier, ordered by quality then bytes."""

    frontier: list[Mapping[str, Any]] = []
    for row in rows:
        row_bytes = _finite_number(row.get("complete_persistent_model_bytes"))
        row_ppl = _finite_number(row.get("ppl"))
        if row_bytes is None or row_ppl is None:
            continue
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_bytes = _finite_number(other.get("complete_persistent_model_bytes"))
            other_ppl = _finite_number(other.get("ppl"))
            if other_bytes is None or other_ppl is None:
                continue
            if (
                other_bytes <= row_bytes
                and other_ppl <= row_ppl
                and (other_bytes < row_bytes or other_ppl < row_ppl)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda row: (
            float(row["ppl"]),
            float(row["complete_persistent_model_bytes"]),
            str(row["profile"]),
        ),
    )


def select_promoted_profiles(
    rows: Sequence[Mapping[str, Any]],
    *,
    matched_controls: Mapping[str, str],
    always_include: Sequence[str] = ("gaussian_w4_mse",),
    outcome_preference: Sequence[str] = (),
    max_per_track: int = 2,
    max_relative_ppl: float = 1.0,
    min_allocation_byte_saving: float = 0.01,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply confidence gates before bounded Pareto promotion.

    Matched controls are validation dependencies, not promotion candidates, so
    they cannot consume a track slot.  A matched candidate must be exact-rate
    and its paired 95% interval must exclude zero before it reaches the Pareto
    cap.  Identical deployed outcomes are collapsed using ``outcome_preference``.
    """

    if max_per_track < 1:
        raise ValueError("max_per_track must be positive")
    by_name = {str(row["profile"]): row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("screen rows must have unique profile names")

    decisions: dict[str, dict[str, Any]] = {
        name: {"profile": name, "eligible": False, "selected": False, "reason": ""}
        for name in by_name
    }
    control_names = set(matched_controls.values())
    eligible: list[Mapping[str, Any]] = []

    for name, row in by_name.items():
        reason: str | None = None
        relative_ppl = _finite_number(row.get("relative_ppl"))
        if name == "source_fp16":
            reason = "source_control"
        elif name in control_names:
            reason = "matched_control"
        elif not bool(row.get("target_reached", False)):
            reason = "rate_target_missed"
        elif bool(row.get("ppl_stopped_early", False)):
            reason = "catastrophic_early_stop"
        elif relative_ppl is None or relative_ppl > max_relative_ppl:
            reason = "screen_quality_gate_failed"
        elif name in matched_controls:
            ci_high = _finite_number(row.get("matched_control_ci_high"))
            if not bool(row.get("matched_rate", False)):
                reason = "matched_rate_gate_failed"
            elif ci_high is None or ci_high >= 0:
                reason = "matched_confidence_gate_failed"
        if reason is not None:
            decisions[name]["reason"] = reason
            continue
        decisions[name]["eligible"] = True
        decisions[name]["reason"] = "eligible"
        eligible.append(row)

    preference_rank = {name: index for index, name in enumerate(outcome_preference)}
    signature_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in eligible:
        signature_groups.setdefault(outcome_signature(row), []).append(row)

    deduplicated: list[Mapping[str, Any]] = []
    for group in signature_groups.values():
        winner = min(
            group,
            key=lambda row: (
                preference_rank.get(str(row["profile"]), len(preference_rank)),
                str(row["profile"]),
            ),
        )
        deduplicated.append(winner)
        for row in group:
            name = str(row["profile"])
            if row is not winner:
                decisions[name]["eligible"] = False
                decisions[name]["reason"] = f"equivalent_to:{winner['profile']}"

    selected: set[str] = set()
    tracks = sorted({str(row["track"]) for row in deduplicated})
    for track in tracks:
        group = [row for row in deduplicated if str(row["track"]) == track]
        frontier = pareto_frontier(group)
        frontier_names = {str(row["profile"]) for row in frontier}
        for row in group:
            name = str(row["profile"])
            if name not in frontier_names:
                decisions[name]["reason"] = "pareto_dominated"
        for index, row in enumerate(frontier):
            name = str(row["profile"])
            if index < max_per_track:
                selected.add(name)
            else:
                decisions[name]["reason"] = "track_confirmation_cap"

    for name in always_include:
        row = by_name.get(name)
        if row is not None and not bool(row.get("ppl_stopped_early", False)):
            selected.add(name)

    baseline = by_name.get("gaussian_w4_mse")
    baseline_bytes = (
        _finite_number(baseline.get("complete_persistent_model_bytes"))
        if baseline is not None
        else None
    )
    if baseline_bytes:
        for name in selected.copy():
            row = by_name[name]
            if str(row.get("track")) != "allocation":
                continue
            candidate_bytes = _finite_number(row.get("complete_persistent_model_bytes"))
            saving = 1.0 - candidate_bytes / baseline_bytes if candidate_bytes else -math.inf
            if saving < min_allocation_byte_saving:
                selected.remove(name)
                decisions[name]["reason"] = "allocation_byte_saving_gate_failed"

    for name in selected:
        decisions[name]["selected"] = True
        decisions[name]["reason"] = "selected"
    return sorted(selected), [decisions[name] for name in sorted(decisions)]


def validation_status(
    *,
    research_only: bool,
    control_only: bool,
    primary_quality_passed: bool,
    cross_family_available: bool,
    cross_family_quality_passed: bool,
    primary_matched_control_passed: bool = True,
    cross_family_matched_control_passed: bool = True,
) -> str:
    """Classify evidence without turning a placeholder into a release claim."""

    if control_only:
        return "control_only"
    if not primary_quality_passed:
        return "research_quality_failed" if research_only else "primary_quality_failed"
    if not primary_matched_control_passed:
        return "matched_control_failed"
    if not cross_family_available:
        return "cross_family_missing"
    if not cross_family_quality_passed:
        return "research_cross_family_failed" if research_only else "cross_family_failed"
    if not cross_family_matched_control_passed:
        return "cross_family_matched_control_failed"
    return "research_confirmed" if research_only else "runtime_candidate"


__all__ = [
    "outcome_signature",
    "pareto_frontier",
    "select_promoted_profiles",
    "validation_status",
]
