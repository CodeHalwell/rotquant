#!/usr/bin/env python3
"""Produce the fail-closed three-seed format-aware allocator decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.eval.promotion import dynamic_row_valid, finite_number, quality_guards
from rotquant.utils import write_result

PROTOCOL = "qwen35-allocator-v4-confirmation-decision-v1"
STAGE = "allocator-v4"
EXPECTED_SEEDS = (0, 1, 2)


def _paired_kl_clear(report: dict[str, Any]) -> bool:
    try:
        metric = report["metrics"]["logit_fidelity.mean_teacher_kl"]
        interval = metric["bootstrap_95_ci"]
        return (
            len(interval) == 2 and all(finite_number(v) for v in interval)
            and interval[0] <= interval[1] < 0
            and metric.get("interval_reliable") is True
            and finite_number(metric.get("paired_samples"))
            and metric["paired_samples"] >= 20
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def assess(
    summary: dict[str, Any],
    selection: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    finalists = tuple(map(str, selection.get("finalists", ())))
    seeds = tuple(map(int, summary.get("seeds", ())))
    bits_arm = str(selection.get("bits_baseline_arm"))
    random_arm = str(selection.get("random_control_arm"))
    entries = [row for row in summary.get("rows", ()) if row.get("stage") == STAGE]
    rows = {
        (str(row.get("arm")), int(row.get("seed", -1))): row
        for row in entries
    }
    pair_entries = [
        report for report in summary.get("paired_comparisons", ())
        if report.get("stage") == STAGE
    ]
    paired = {
        (
            str(report.get("candidate_arm")),
            str(report.get("baseline_arm")),
            int(report.get("seed", -1)),
        ): report
        for report in pair_entries
    }
    provider = {
        str(row.get("arm")): row
        for row in comparison.get("comparisons", ())
    }
    revision = summary.get("code_revision")
    errors = []
    if summary.get("complete") is not True:
        errors.append("confirmation summary is incomplete")
    if seeds != EXPECTED_SEEDS:
        errors.append(f"confirmation seeds must be {EXPECTED_SEEDS}, found {seeds}")
    if selection.get("code_revision") != revision:
        errors.append("selection and confirmation revisions differ")
    if comparison.get("rotquant_code_revision") != revision:
        errors.append("provider comparison and confirmation revisions differ")
    if comparison.get("prompt_hashes_match") is not True:
        errors.append("provider comparison prompt hashes do not match")
    tolerance = comparison.get("byte_tolerance_fraction")
    if not finite_number(tolerance) or not 0 <= tolerance <= 0.01:
        errors.append("exported-byte tolerance must be explicitly at most 1%")
    if not isinstance(revision, str) or not revision:
        errors.append("confirmation revision is missing")
    if bits_arm != "bits_only_pareto" or random_arm != "format_random_exact":
        errors.append("confirmation controls do not match the registered protocol")
    if len(set(finalists)) != len(finalists):
        errors.append("duplicate finalist arms")
    if len(provider) != len(comparison.get("comparisons", ())):
        errors.append("duplicate provider comparisons")
    if len(rows) != len(entries):
        errors.append("confirmation contains duplicate arm/seed rows")
    if len(paired) != len(pair_entries):
        errors.append("confirmation contains duplicate paired comparisons")

    decisions = []
    for arm in finalists:
        required_rows = [
            (name, seed)
            for seed in EXPECTED_SEEDS
            for name in (arm, bits_arm, random_arm)
        ]
        missing_rows = [identity for identity in required_rows if identity not in rows]
        required_pairs = [
            (arm, baseline, seed)
            for seed in EXPECTED_SEEDS
            for baseline in (bits_arm, random_arm)
        ]
        missing_pairs = [identity for identity in required_pairs if identity not in paired]
        arm_rows = [
            rows[(arm, seed)] for seed in EXPECTED_SEEDS
            if (arm, seed) in rows
        ]

        def better_all(
            metric: str,
            baseline: str,
            direction: str,
            *,
            candidate_arm: str = arm,
            candidate_rows: list[dict[str, Any]] = arm_rows,
        ) -> bool:
            return len(candidate_rows) == len(EXPECTED_SEEDS) and all(
                (baseline, seed) in rows
                and finite_number(rows[(candidate_arm, seed)].get(metric))
                and finite_number(rows[(baseline, seed)].get(metric))
                and
                (
                    float(rows[(candidate_arm, seed)][metric])
                    < float(rows[(baseline, seed)][metric])
                    if direction == "lower"
                    else float(rows[(candidate_arm, seed)][metric])
                    > float(rows[(baseline, seed)][metric])
                )
                for seed in EXPECTED_SEEDS
            )

        per_seed_guards = {
            str(seed): quality_guards(rows.get((arm, seed), {}),
                                     rows.get((bits_arm, seed), {}))
            for seed in EXPECTED_SEEDS
        }
        no_regressions = all(all(g.values()) for g in per_seed_guards.values())
        valid_rows = not missing_rows and all(
            dynamic_row_valid(rows[key]) for key in required_rows
        )

        target_pass = len(arm_rows) == len(EXPECTED_SEEDS) and all(
            row.get("dynamic_actual_target_match") is True for row in arm_rows
        )
        not_halted = len(arm_rows) == len(EXPECTED_SEEDS) and all(
            not bool(row.get("evaluation_halted", False)) for row in arm_rows
        )
        kl_better_bits = better_all("mean_teacher_kl", bits_arm, "lower")
        kl_better_random = better_all("mean_teacher_kl", random_arm, "lower")
        paired_bits_clear = not missing_pairs and all(
            _paired_kl_clear(paired[(arm, bits_arm, seed)])
            for seed in EXPECTED_SEEDS
        )
        paired_random_clear = not missing_pairs and all(
            _paired_kl_clear(paired[(arm, random_arm, seed)])
            for seed in EXPECTED_SEEDS
        )
        secondary = {
            metric: better_all(metric, bits_arm, direction)
            for metric, direction in {
                "top1_agreement": "higher",
                "ppl_wikitext2": "lower",
                "ppl_c4": "lower",
                "diverse_mean_teacher_kl": "lower",
                "diverse_top1_agreement": "higher",
                "diverse_trajectory_token_agreement": "higher",
            }.items()
        }
        provider_row = provider.get(arm)
        byte_gate = bool(
            provider_row and provider_row.get("within_byte_gate") is True
            and provider_row.get("byte_basis") == "measured_artifact"
            and 0 in provider_row.get("exported_seeds", ())
        )
        # An export must describe this recipe and trial, not an older run that
        # happened to have approximately the right size. Only seed 0 is required
        # by the registered export policy; never imply all seeds were exported.
        measurements = (provider_row or {}).get("artifact_measurements", ())
        measured_seeds = [item.get("seed") for item in measurements]
        byte_gate = byte_gate and 0 in measured_seeds and (
            len(set(measured_seeds)) == len(measured_seeds)
            and set(measured_seeds) == set(provider_row.get("exported_seeds", ()))
        )
        byte_gate = byte_gate and bool(measurements) and all(
            finite_number(measurement.get("bytes"))
            and measurement["bytes"] > 0
            and measurement["bytes"] == rows.get(
                (arm, measurement.get("seed")), {}
            ).get("packed_artifact_bytes")
            and finite_number(provider_row.get("baseline_bytes"))
            and provider_row["baseline_bytes"] > 0
            and finite_number(tolerance)
            and abs(measurement["bytes"] / provider_row["baseline_bytes"] - 1) <= tolerance
            and
            measurement.get("identity") == rows.get(
                (arm, measurement.get("seed")), {}
            ).get("packed_artifact_identity")
            and isinstance(measurement.get("identity"), dict)
            and measurement["identity"].get("code_revision") == revision
            and measurement["identity"].get("seed") == measurement.get("seed")
            and measurement["identity"].get("arm") == arm
            and measurement["identity"].get("stage") == STAGE
            and bool(measurement["identity"].get("trial_fingerprint"))
            and measurement["identity"]["trial_fingerprint"] == rows.get(
                (arm, measurement.get("seed")), {}
            ).get("trial_fingerprint")
            and isinstance(measurement.get("manifest_sha256"), str)
            and len(measurement["manifest_sha256"]) == 64
            and measurement["manifest_sha256"] == rows.get(
                (arm, measurement.get("seed")), {}
            ).get("packed_manifest_sha256")
            and measurement["identity"].get("allocation_fingerprint") == rows.get(
                (arm, measurement.get("seed")), {}
            ).get("dynamic_allocation_fingerprint")
            for measurement in measurements
        )
        candidate_metrics = (provider_row or {}).get("candidate_metrics") or {}
        provider_metrics = (provider_row or {}).get("unsloth_metrics") or {}
        provider_competitive = all(
            finite_number(metrics.get(key))
            for metrics in (candidate_metrics, provider_metrics)
            for key in ("mean_teacher_kl", "top1_agreement")
        ) and (
            candidate_metrics["mean_teacher_kl"] <= provider_metrics["mean_teacher_kl"]
            and candidate_metrics["top1_agreement"] >= provider_metrics["top1_agreement"]
        )
        promoted = (
            not errors
            and not missing_rows
            and not missing_pairs
            and valid_rows
            and no_regressions
            and target_pass
            and not_halted
            and kl_better_bits
            and paired_bits_clear
            and kl_better_random
            and paired_random_clear
            and any(secondary.values())
            and byte_gate
        )
        decisions.append({
            "arm": arm,
            "format_allocator_promoted": promoted,
            "provider_competitive": promoted and provider_competitive,
            "missing_rows": [list(value) for value in missing_rows],
            "missing_paired_comparisons": [list(value) for value in missing_pairs],
            "gates": {
                "protocol_integrity": not errors,
                "valid_candidate_and_control_rows": valid_rows,
                "no_quality_regressions_all_seeds": no_regressions,
                "target_bytes_all_seeds": target_pass,
                "not_halted_all_seeds": not_halted,
                "kl_better_than_bits_all_seeds": kl_better_bits,
                "paired_bits_kl_ci_below_zero_all_seeds": paired_bits_clear,
                "kl_better_than_random_all_seeds": kl_better_random,
                "paired_random_kl_ci_below_zero_all_seeds": paired_random_clear,
                "secondary_win_over_bits_all_seeds": any(secondary.values()),
                "within_exported_byte_gate": byte_gate,
            },
            "secondary_wins_over_bits": secondary,
            "quality_guards_by_seed": per_seed_guards,
            "exported_seeds": (provider_row or {}).get("exported_seeds", []),
            "provider_comparison": provider_row,
        })
    return {
        "protocol": PROTOCOL,
        "code_revision": revision,
        "seeds": list(seeds),
        "bits_baseline_arm": bits_arm,
        "random_control_arm": random_arm,
        "protocol_errors": errors,
        "decisions": decisions,
        "format_allocator_winners": [
            row["arm"] for row in decisions
            if row["format_allocator_promoted"]
        ],
        "provider_competitive_winners": [
            row["arm"] for row in decisions if row["provider_competitive"]
        ],
        "boundary": (
            "Internal development decision only. Public claims still require "
            "the registered engine-neutral 300-prompt evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(
        json.loads(args.summary.read_text(encoding="utf-8")),
        json.loads(args.selection.read_text(encoding="utf-8")),
        json.loads(args.comparison.read_text(encoding="utf-8")),
    )
    write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
