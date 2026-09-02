#!/usr/bin/env python3
"""Select W4 ablation finalists under pre-registered seed-0 guardrails.

The screen is deliberately permissive enough to keep plausible improvements,
but it does not promote an arm merely for avoiding the catastrophic fail-fast
gate. A finalist must stay close to the promoted W4 control on every core
quality metric and improve at least one quality or exact-byte objective by a
material amount. Seeds 1/2 then test whether that seed-0 signal replicates.
"""
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

PROTOCOL = "qwen35-w4-ablation-finalist-selection-v1"
BASELINE_ARM = "promoted_w4"
REQUIRED_METRICS = (
    "mean_teacher_kl",
    "top1_agreement",
    "ppl_wikitext2",
    "ppl_c4",
    "trajectory_token_agreement",
    "complete_persistent_model_bytes",
)
THRESHOLDS = {
    "guard_mean_teacher_kl_ratio_max": 1.05,
    "guard_top1_delta_min": -0.01,
    "guard_ppl_ratio_max": 1.02,
    "guard_trajectory_delta_min": -0.02,
    "guard_bytes_ratio_max": 1.005,
    "signal_mean_teacher_kl_ratio_max": 0.99,
    "signal_top1_delta_min": 0.005,
    "signal_ppl_ratio_max": 0.995,
    "signal_trajectory_delta_min": 0.01,
    "signal_bytes_ratio_max": 0.995,
}


def _number(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _rows_for_seed(summary: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    return [
        row for row in summary.get("rows", [])
        if row.get("stage") == "ablation" and int(row.get("seed", -1)) == seed
    ]


def select_finalists(summary: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    rows = _rows_for_seed(summary, seed)
    by_arm = {str(row.get("arm")): row for row in rows}
    if len(by_arm) != len(rows):
        raise ValueError("ablation summary contains duplicate arm/seed rows")
    baseline = by_arm.get(BASELINE_ARM)
    if baseline is None:
        raise ValueError(f"ablation summary is missing {BASELINE_ARM}/seed-{seed}")
    missing_baseline = [
        metric for metric in REQUIRED_METRICS
        if _number(baseline, metric) is None
    ]
    if missing_baseline:
        raise ValueError(
            "baseline is missing required metrics: " + ", ".join(missing_baseline)
        )

    base = {metric: _number(baseline, metric) for metric in REQUIRED_METRICS}
    assert all(value is not None for value in base.values())
    decisions = []
    finalists = []
    for arm, row in by_arm.items():
        if arm == BASELINE_ARM:
            continue
        metrics = {metric: _number(row, metric) for metric in REQUIRED_METRICS}
        missing = [name for name, value in metrics.items() if value is None]
        guards: dict[str, bool] = {}
        signals: dict[str, bool] = {}
        if not missing and not row.get("evaluation_halted", False):
            guards = {
                "mean_teacher_kl": metrics["mean_teacher_kl"]
                <= base["mean_teacher_kl"]
                * THRESHOLDS["guard_mean_teacher_kl_ratio_max"],
                "top1_agreement": metrics["top1_agreement"]
                >= base["top1_agreement"]
                + THRESHOLDS["guard_top1_delta_min"],
                "ppl_wikitext2": metrics["ppl_wikitext2"]
                <= base["ppl_wikitext2"] * THRESHOLDS["guard_ppl_ratio_max"],
                "ppl_c4": metrics["ppl_c4"]
                <= base["ppl_c4"] * THRESHOLDS["guard_ppl_ratio_max"],
                "trajectory_token_agreement": metrics[
                    "trajectory_token_agreement"
                ] >= base["trajectory_token_agreement"]
                + THRESHOLDS["guard_trajectory_delta_min"],
                "complete_persistent_model_bytes": metrics[
                    "complete_persistent_model_bytes"
                ] <= base["complete_persistent_model_bytes"]
                * THRESHOLDS["guard_bytes_ratio_max"],
            }
            signals = {
                "mean_teacher_kl": metrics["mean_teacher_kl"]
                <= base["mean_teacher_kl"]
                * THRESHOLDS["signal_mean_teacher_kl_ratio_max"],
                "top1_agreement": metrics["top1_agreement"]
                >= base["top1_agreement"]
                + THRESHOLDS["signal_top1_delta_min"],
                "ppl_wikitext2": metrics["ppl_wikitext2"]
                <= base["ppl_wikitext2"] * THRESHOLDS["signal_ppl_ratio_max"],
                "ppl_c4": metrics["ppl_c4"]
                <= base["ppl_c4"] * THRESHOLDS["signal_ppl_ratio_max"],
                "trajectory_token_agreement": metrics[
                    "trajectory_token_agreement"
                ] >= base["trajectory_token_agreement"]
                + THRESHOLDS["signal_trajectory_delta_min"],
                "complete_persistent_model_bytes": metrics[
                    "complete_persistent_model_bytes"
                ] <= base["complete_persistent_model_bytes"]
                * THRESHOLDS["signal_bytes_ratio_max"],
            }
        selected = (
            not missing
            and not row.get("evaluation_halted", False)
            and all(guards.values())
            and any(signals.values())
        )
        if selected:
            finalists.append(arm)
        decisions.append({
            "arm": arm,
            "selected": selected,
            "evaluation_halted": bool(row.get("evaluation_halted", False)),
            "missing_metrics": missing,
            "guards": guards,
            "signals": signals,
            "metrics": metrics,
        })

    return {
        "protocol": PROTOCOL,
        "code_revision": summary.get("code_revision"),
        "screen_seed": seed,
        "baseline_arm": BASELINE_ARM,
        "thresholds": THRESHOLDS,
        "baseline_metrics": base,
        "decisions": decisions,
        "finalists": finalists,
        "confirmation_arms": [BASELINE_ARM, *finalists],
        "interpretation": (
            "Seed-0 screen only. A selected arm is a replication candidate, "
            "not a promoted recipe; seeds 1/2 and paired evidence decide next."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_finalists(summary, seed=args.seed)
    write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
