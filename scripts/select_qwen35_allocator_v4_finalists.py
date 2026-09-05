#!/usr/bin/env python3
"""Select distinct format-aware recipes over the proven bits-only allocator."""

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

PROTOCOL = "qwen35-allocator-v4-finalist-selection-v1"
STAGE = "allocator-v4"
QUALITY_CEILING = "uniform_scale8_w4"
BITS_BASELINE = "bits_only_pareto"
RANDOM_CONTROL = "format_random_exact"
CANDIDATE_ARMS = (
    "format_pareto_all",
    "format_pareto_gaussian",
    "format_pareto_g128",
)
REQUIRED = (
    "mean_teacher_kl",
    "top1_agreement",
    "ppl_wikitext2",
    "ppl_c4",
    "diverse_mean_teacher_kl",
    "diverse_top1_agreement",
    "diverse_trajectory_token_agreement",
    "complete_persistent_model_bytes",
)


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not finite_number(value):
        raise TypeError(f"{row.get('arm')} is missing numeric {key}")
    return float(value)


def _missing(row: dict[str, Any]) -> list[str]:
    return [
        key for key in REQUIRED
        if not finite_number(row.get(key))
    ]


def _fingerprint(row: dict[str, Any]) -> str | None:
    value = row.get("dynamic_allocation_fingerprint")
    return value if isinstance(value, str) and value else None


def _audit_dynamic_control(row: dict[str, Any], arm: str, seed: int) -> None:
    failures = []
    if _missing(row):
        failures.append(f"missing metrics {_missing(row)}")
    if bool(row.get("evaluation_halted", False)):
        failures.append("evaluation halted")
    if row.get("dynamic_actual_target_match") is not True:
        failures.append("artifact estimate missed")
    if row.get("dynamic_scoring_matches_deployed") is not True:
        failures.append("candidate scoring was not faithful")
    if _fingerprint(row) is None:
        failures.append("allocation fingerprint missing")
    if not dynamic_row_valid(row):
        failures.append("invalid quality/deployment evidence")
    if failures:
        raise ValueError(f"{arm}/seed-{seed}: " + "; ".join(failures))


def select_finalists(
    summary: dict[str, Any], *, seed: int = 0, limit: int = 2
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if summary.get("complete") is not True:
        raise ValueError("allocator-v4 screen summary is incomplete")
    if tuple(map(int, summary.get("seeds", ()))) != (seed,):
        raise ValueError("allocator-v4 screen must contain only the requested seed")
    entries = [
        row for row in summary.get("rows", ())
        if row.get("stage") == STAGE and int(row.get("seed", -1)) == seed
    ]
    rows = {str(row.get("arm")): row for row in entries}
    if len(rows) != len(entries):
        raise ValueError("allocator-v4 summary contains duplicate arm/seed rows")
    for arm in (QUALITY_CEILING, BITS_BASELINE, RANDOM_CONTROL):
        if arm not in rows:
            raise ValueError(f"allocator-v4 summary is missing {arm}/seed-{seed}")
    _audit_dynamic_control(rows[BITS_BASELINE], BITS_BASELINE, seed)
    _audit_dynamic_control(rows[RANDOM_CONTROL], RANDOM_CONTROL, seed)
    if _missing(rows[QUALITY_CEILING]) or bool(
        rows[QUALITY_CEILING].get("evaluation_halted", False)
    ):
        raise ValueError(f"{QUALITY_CEILING}/seed-{seed} is incomplete")

    baseline = {key: _number(rows[BITS_BASELINE], key) for key in REQUIRED}
    decisions: list[dict[str, Any]] = []
    viable: list[tuple[float, int, str, str]] = []
    for order, arm in enumerate(CANDIDATE_ARMS):
        row = rows.get(arm)
        if row is None:
            decisions.append({
                "arm": arm, "selected": False, "missing": True,
                "missing_metrics": list(REQUIRED),
            })
            continue
        missing = _missing(row)
        fingerprint = _fingerprint(row)
        if fingerprint is None:
            missing.append("dynamic_allocation_fingerprint")
        if missing:
            decisions.append({
                "arm": arm, "selected": False, "missing": False,
                "eligible": False, "missing_metrics": missing,
            })
            continue
        values = {key: _number(row, key) for key in REQUIRED}
        guards = {
            "target_bytes": row.get("dynamic_actual_target_match") is True,
            "faithful_scoring": row.get("dynamic_scoring_matches_deployed") is True,
            "not_halted": not bool(row.get("evaluation_halted", False)),
            **quality_guards(values, baseline),
        }
        signals = {
            "primary_kl": values["mean_teacher_kl"]
            <= baseline["mean_teacher_kl"] * 0.99,
            "primary_top1": values["top1_agreement"]
            >= baseline["top1_agreement"] + 0.005,
            "wikitext2": values["ppl_wikitext2"]
            <= baseline["ppl_wikitext2"] * 0.995,
            "c4": values["ppl_c4"] <= baseline["ppl_c4"] * 0.995,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= baseline["diverse_mean_teacher_kl"] * 0.99,
            "diverse_top1": values["diverse_top1_agreement"]
            >= baseline["diverse_top1_agreement"] + 0.005,
            "trajectory": values["diverse_trajectory_token_agreement"]
            >= baseline["diverse_trajectory_token_agreement"] + 0.01,
        }
        fidelity_signal = signals["primary_kl"] or signals["diverse_kl"]
        eligible = all(guards.values()) and fidelity_signal and any(signals.values())
        score = (
            values["mean_teacher_kl"] / baseline["mean_teacher_kl"]
            + values["diverse_mean_teacher_kl"]
            / baseline["diverse_mean_teacher_kl"]
            + values["ppl_wikitext2"] / baseline["ppl_wikitext2"]
            + values["ppl_c4"] / baseline["ppl_c4"]
            - values["top1_agreement"]
            - values["diverse_top1_agreement"]
            - values["diverse_trajectory_token_agreement"]
        )
        if eligible:
            viable.append((score, order, arm, str(fingerprint)))
        decisions.append({
            "arm": arm,
            "selected": False,
            "missing": False,
            "missing_metrics": [],
            "eligible": eligible,
            "score": score,
            "guards": guards,
            "signals": signals,
            "metrics": values,
            "allocation_fingerprint": fingerprint,
            "counts_by_format": row.get("dynamic_counts_by_format"),
        })

    finalists: list[str] = []
    owners: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for _score, _order, arm, fingerprint in sorted(viable):
        if fingerprint in owners:
            duplicates[arm] = owners[fingerprint]
            continue
        owners[fingerprint] = arm
        if len(finalists) < limit:
            finalists.append(arm)
    for decision in decisions:
        arm = str(decision["arm"])
        decision["selected"] = arm in finalists
        if arm in duplicates:
            decision["duplicate_of"] = duplicates[arm]

    confirmation = [QUALITY_CEILING, BITS_BASELINE, RANDOM_CONTROL, *finalists]
    return {
        "protocol": PROTOCOL,
        "code_revision": summary.get("code_revision"),
        "screen_seed": seed,
        "quality_ceiling_arm": QUALITY_CEILING,
        "bits_baseline_arm": BITS_BASELINE,
        "random_control_arm": RANDOM_CONTROL,
        "finalists": finalists,
        "confirmation_arms": list(dict.fromkeys(confirmation)),
        "decisions": decisions,
        "deduplicated_recipes": duplicates,
        "interpretation": (
            "Seed-0 format screen only. A finalist must improve held-out KL "
            "over the exact-size bits-only allocator, then reproduce against "
            "both bits-only and same-palette random controls at seeds 0/1/2."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    result = select_finalists(
        json.loads(args.summary.read_text(encoding="utf-8")),
        seed=args.seed,
        limit=args.limit,
    )
    write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
