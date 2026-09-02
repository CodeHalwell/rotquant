#!/usr/bin/env python3
"""Select seed-0 exact-byte mixed-precision finalists for replication."""
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

PROTOCOL = "qwen35-dynamic-finalist-selection-v1"
BASELINE_ARM = "uniform_scale8_w4"
RANDOM_CONTROL = "random_mixed_fwht"
CANDIDATE_ARMS = (
    "dynamic_mixed_unrotated",
    "dynamic_mixed_fwht",
    "dynamic_mixed_signs_fp16",
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
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{row.get('arm')} is missing numeric {key}")
    return float(value)


def _missing_numeric(row: dict[str, Any]) -> list[str]:
    return [
        key for key in REQUIRED
        if not isinstance(row.get(key), (int, float))
        or isinstance(row.get(key), bool)
    ]


def select_finalists(summary: dict[str, Any], *, seed: int = 0,
                     limit: int = 2) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = [
        row for row in summary.get("rows", [])
        if row.get("stage") == "dynamic" and int(row.get("seed", -1)) == seed
    ]
    by_arm = {str(row.get("arm")): row for row in rows}
    if len(by_arm) != len(rows):
        raise ValueError("dynamic summary contains duplicate arm/seed rows")
    baseline = by_arm.get(BASELINE_ARM)
    if baseline is None:
        raise ValueError(f"dynamic summary is missing {BASELINE_ARM}/seed-{seed}")
    baseline_missing = _missing_numeric(baseline)
    if baseline_missing:
        raise ValueError(
            f"{BASELINE_ARM}/seed-{seed} is missing {baseline_missing}"
        )
    base = {key: _number(baseline, key) for key in REQUIRED}

    decisions = []
    viable: list[tuple[float, str]] = []
    for arm in CANDIDATE_ARMS:
        row = by_arm.get(arm)
        if row is None:
            decisions.append({
                "arm": arm,
                "selected": False,
                "missing": True,
                "missing_metrics": list(REQUIRED),
            })
            continue
        missing_metrics = _missing_numeric(row)
        if missing_metrics:
            decisions.append({
                "arm": arm,
                "selected": False,
                "missing": False,
                "eligible": False,
                "evaluation_halted": bool(row.get("evaluation_halted", False)),
                "missing_metrics": missing_metrics,
            })
            continue
        values = {key: _number(row, key) for key in REQUIRED}
        target_match = row.get("dynamic_actual_target_match") is True
        guards = {
            "target_bytes": target_match,
            "not_halted": not bool(row.get("evaluation_halted", False)),
            "primary_kl": values["mean_teacher_kl"]
            <= base["mean_teacher_kl"] * 1.05,
            "primary_top1": values["top1_agreement"]
            >= base["top1_agreement"] - 0.01,
            "wikitext2": values["ppl_wikitext2"]
            <= base["ppl_wikitext2"] * 1.02,
            "c4": values["ppl_c4"] <= base["ppl_c4"] * 1.02,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= base["diverse_mean_teacher_kl"] * 1.05,
            "diverse_top1": values["diverse_top1_agreement"]
            >= base["diverse_top1_agreement"] - 0.015,
            "diverse_trajectory": values["diverse_trajectory_token_agreement"]
            >= base["diverse_trajectory_token_agreement"] - 0.03,
        }
        signals = {
            "primary_kl": values["mean_teacher_kl"]
            <= base["mean_teacher_kl"] * 0.99,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= base["diverse_mean_teacher_kl"] * 0.99,
            "primary_top1": values["top1_agreement"]
            >= base["top1_agreement"] + 0.005,
            "diverse_top1": values["diverse_top1_agreement"]
            >= base["diverse_top1_agreement"] + 0.005,
            "diverse_trajectory": values["diverse_trajectory_token_agreement"]
            >= base["diverse_trajectory_token_agreement"] + 0.01,
        }
        eligible = all(guards.values()) and any(signals.values())
        score = (
            values["mean_teacher_kl"] / base["mean_teacher_kl"]
            + values["diverse_mean_teacher_kl"]
            / base["diverse_mean_teacher_kl"]
            + values["ppl_wikitext2"] / base["ppl_wikitext2"]
            + values["ppl_c4"] / base["ppl_c4"]
            - (values["top1_agreement"] - base["top1_agreement"])
            - (values["diverse_top1_agreement"]
               - base["diverse_top1_agreement"])
            - (values["diverse_trajectory_token_agreement"]
               - base["diverse_trajectory_token_agreement"])
        )
        if eligible:
            viable.append((score, arm))
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
        })

    finalists = [arm for _score, arm in sorted(viable)[:limit]]
    for decision in decisions:
        decision["selected"] = decision.get("arm") in finalists
    confirmation = [BASELINE_ARM]
    if RANDOM_CONTROL in by_arm:
        confirmation.append(RANDOM_CONTROL)
    confirmation.extend(finalists)
    return {
        "protocol": PROTOCOL,
        "code_revision": summary.get("code_revision"),
        "screen_seed": seed,
        "baseline_arm": BASELINE_ARM,
        "random_control_arm": RANDOM_CONTROL,
        "finalists": finalists,
        "confirmation_arms": list(dict.fromkeys(confirmation)),
        "decisions": decisions,
        "interpretation": (
            "Seed-0 exact-byte screen only. Selected recipes must beat the "
            "matched-format random control across seeds 1/2 before promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_finalists(summary, seed=args.seed, limit=args.limit)
    write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
