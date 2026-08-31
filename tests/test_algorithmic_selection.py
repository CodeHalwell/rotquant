"""Confidence-first promotion policy for the Algorithm Lab."""

from scripts.algorithmic_selection import (
    select_promoted_profiles,
    validation_status,
)


def _row(profile, *, track="scalar", size=100, ppl=10.0, **overrides):
    row = {
        "profile": profile,
        "track": track,
        "complete_persistent_model_bytes": size,
        "ppl": ppl,
        "relative_ppl": 0.1,
        "target_reached": True,
        "ppl_stopped_early": False,
        "dynamic_counts": None,
        "mean_layer_nmse": 0.01,
        "worst_layer_nmse": 0.02,
    }
    row.update(overrides)
    return row


def test_matched_controls_do_not_consume_slots_and_confidence_is_required():
    rows = [
        _row("source_fp16", size=200, ppl=9.0),
        _row("gaussian_w4_mse", size=120, ppl=10.0),
        _row("random", track="allocation", size=90, ppl=11.0),
        _row(
            "teacher",
            track="allocation",
            size=88,
            ppl=10.5,
            matched_rate=True,
            matched_control_ci_high=-0.01,
        ),
        _row(
            "uncertain",
            track="allocation",
            size=80,
            ppl=10.2,
            matched_rate=True,
            matched_control_ci_high=0.01,
        ),
    ]

    selected, decisions = select_promoted_profiles(
        rows,
        matched_controls={"teacher": "random", "uncertain": "random"},
        max_per_track=1,
    )
    reasons = {row["profile"]: row["reason"] for row in decisions}

    assert selected == ["gaussian_w4_mse", "teacher"]
    assert reasons["random"] == "matched_control"
    assert reasons["uncertain"] == "matched_confidence_gate_failed"


def test_equivalent_outcomes_use_explicit_preference():
    rows = [
        _row("gaussian_w4_mse", size=120),
        _row("teacher", track="allocation", size=90, ppl=10.4),
        _row("guarded", track="allocation", size=90, ppl=10.4),
    ]

    selected, decisions = select_promoted_profiles(
        rows,
        matched_controls={},
        outcome_preference=("teacher", "guarded"),
    )
    reasons = {row["profile"]: row["reason"] for row in decisions}

    assert selected == ["gaussian_w4_mse", "teacher"]
    assert reasons["guarded"] == "equivalent_to:teacher"


def test_allocation_must_save_complete_persistent_bytes():
    rows = [
        _row("gaussian_w4_mse", size=100),
        _row("teacher", track="allocation", size=100, ppl=9.5),
    ]

    selected, decisions = select_promoted_profiles(
        rows,
        matched_controls={},
        min_allocation_byte_saving=0.01,
    )
    reasons = {row["profile"]: row["reason"] for row in decisions}

    assert selected == ["gaussian_w4_mse"]
    assert reasons["teacher"] == "allocation_byte_saving_gate_failed"


def test_validation_status_never_promotes_missing_evidence():
    common = {
        "research_only": False,
        "control_only": False,
        "primary_quality_passed": True,
        "cross_family_available": True,
        "cross_family_quality_passed": True,
    }
    assert validation_status(**common) == "runtime_candidate"
    assert validation_status(**(common | {"control_only": True})) == "control_only"
    assert validation_status(
        **(common | {"primary_matched_control_passed": False})
    ) == "matched_control_failed"
    assert validation_status(
        **(common | {"cross_family_available": False})
    ) == "cross_family_missing"
    assert validation_status(
        **(common | {"cross_family_quality_passed": False})
    ) == "cross_family_failed"
    assert validation_status(
        **(common | {"cross_family_matched_control_passed": False})
    ) == "cross_family_matched_control_failed"
    assert validation_status(**(common | {"research_only": True})) == "research_confirmed"
