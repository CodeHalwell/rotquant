#!/usr/bin/env python3
"""Produce the fail-closed three-seed allocator-v3 decision record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.utils import write_result

PROTOCOL = "qwen35-allocator-v3-confirmation-decision-v1"
STAGE = "allocator-v3"
EXPECTED_SEEDS = (0, 1, 2)


def assess(
    summary: dict[str, Any],
    selection: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    finalists = tuple(map(str, selection.get("finalists", ())))
    seeds = tuple(map(int, summary.get("seeds", ())))
    random_arm = str(selection.get("random_control_arm"))
    row_entries = [row for row in summary.get("rows", ()) if row.get("stage") == STAGE]
    rows = {(str(row.get("arm")), int(row.get("seed", -1))): row for row in row_entries}
    paired_entries = [
        report for report in summary.get("paired_comparisons", ()) if report.get("stage") == STAGE
    ]
    paired = {
        (
            str(report.get("candidate_arm")),
            str(report.get("baseline_arm")),
            int(report.get("seed", -1)),
        ): report
        for report in paired_entries
    }
    provider = {str(row.get("arm")): row for row in comparison.get("comparisons", ())}
    revision = summary.get("code_revision")
    protocol_errors = []
    if summary.get("complete") is not True:
        protocol_errors.append("confirmation summary is incomplete")
    if seeds != EXPECTED_SEEDS:
        protocol_errors.append(f"confirmation seeds must be {EXPECTED_SEEDS}, found {seeds}")
    if selection.get("code_revision") != revision:
        protocol_errors.append("selection and confirmation revisions differ")
    if comparison.get("rotquant_code_revision") != revision:
        protocol_errors.append("provider comparison and confirmation revisions differ")
    if comparison.get("prompt_hashes_match") is not True:
        protocol_errors.append("provider comparison prompt hashes do not match")
    if len(rows) != len(row_entries):
        protocol_errors.append("confirmation contains duplicate arm/seed rows")
    if len(paired) != len(paired_entries):
        protocol_errors.append("confirmation contains duplicate paired comparisons")
    protocol_valid = not protocol_errors
    decisions = []
    for arm in finalists:
        missing_rows = [
            seed for seed in seeds if (arm, seed) not in rows or (random_arm, seed) not in rows
        ]
        missing_pairs = [seed for seed in seeds if (arm, random_arm, seed) not in paired]
        arm_rows = [rows[(arm, seed)] for seed in seeds if (arm, seed) in rows]
        random_rows = [rows[(random_arm, seed)] for seed in seeds if (random_arm, seed) in rows]
        paired_rows = [
            paired[(arm, random_arm, seed)] for seed in seeds if (arm, random_arm, seed) in paired
        ]
        target_pass = bool(arm_rows) and all(
            row.get("dynamic_actual_target_match") is True for row in arm_rows
        )
        not_halted = bool(arm_rows) and all(
            not bool(row.get("evaluation_halted", False)) for row in arm_rows
        )
        kl_better = (
            bool(arm_rows)
            and len(arm_rows) == len(random_rows)
            and all(
                float(candidate["mean_teacher_kl"]) < float(control["mean_teacher_kl"])
                for candidate, control in zip(arm_rows, random_rows)
            )
        )
        paired_kl_clear = bool(paired_rows) and all(
            float(report["metrics"]["logit_fidelity.mean_teacher_kl"]["bootstrap_95_ci"][1]) < 0
            for report in paired_rows
        )
        secondary_metrics = {
            "top1_agreement": "higher",
            "ppl_wikitext2": "lower",
            "ppl_c4": "lower",
            "diverse_mean_teacher_kl": "lower",
            "diverse_top1_agreement": "higher",
            "diverse_trajectory_token_agreement": "higher",
        }
        secondary_wins = {}
        for metric, direction in secondary_metrics.items():
            secondary_wins[metric] = bool(arm_rows) and all(
                (
                    float(candidate[metric]) < float(control[metric])
                    if direction == "lower"
                    else float(candidate[metric]) > float(control[metric])
                )
                for candidate, control in zip(arm_rows, random_rows)
            )
        provider_row = provider.get(arm)
        within_exported_byte_gate = bool(
            provider_row and provider_row.get("within_byte_gate") is True
        )
        provider_competitive = bool(provider_row) and (
            float(provider_row["candidate_metrics"]["mean_teacher_kl"])
            <= float(provider_row["unsloth_metrics"]["mean_teacher_kl"])
            and float(provider_row["candidate_metrics"]["top1_agreement"])
            >= float(provider_row["unsloth_metrics"]["top1_agreement"])
        )
        allocator_promoted = (
            protocol_valid
            and not missing_rows
            and not missing_pairs
            and target_pass
            and not_halted
            and kl_better
            and paired_kl_clear
            and any(secondary_wins.values())
            and within_exported_byte_gate
        )
        decisions.append(
            {
                "arm": arm,
                "allocator_promoted": allocator_promoted,
                "provider_competitive": allocator_promoted and provider_competitive,
                "missing_seeds": missing_rows,
                "missing_paired_random_comparisons": missing_pairs,
                "gates": {
                    "protocol_integrity": protocol_valid,
                    "target_bytes_all_seeds": target_pass,
                    "not_halted_all_seeds": not_halted,
                    "kl_better_than_random_all_seeds": kl_better,
                    "paired_kl_ci_below_zero_all_seeds": paired_kl_clear,
                    "secondary_win_all_seeds": any(secondary_wins.values()),
                    "within_exported_byte_gate": within_exported_byte_gate,
                },
                "secondary_wins": secondary_wins,
                "provider_comparison": provider_row,
            }
        )
    return {
        "protocol": PROTOCOL,
        "code_revision": summary.get("code_revision"),
        "seeds": list(seeds),
        "random_control_arm": random_arm,
        "protocol_errors": protocol_errors,
        "decisions": decisions,
        "allocator_winners": [row["arm"] for row in decisions if row["allocator_promoted"]],
        "provider_competitive_winners": [
            row["arm"] for row in decisions if row["provider_competitive"]
        ],
        "boundary": (
            "Internal development decision only. A public provider claim still "
            "requires the registered 300-prompt engine-neutral evaluation."
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
