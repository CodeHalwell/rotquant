#!/usr/bin/env python3
"""Select allocator-v2 recipes that beat the matched random W3/W4 control."""
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

PROTOCOL = "qwen35-allocator-v2-finalist-selection-v1"
STAGE = "allocator-v2"
QUALITY_CEILING = "uniform_scale8_w4"
RANDOM_CONTROL = "random_adjacent_w34"
CANDIDATE_ARMS = (
    "pareto_adjacent_local",
    "pareto_adjacent_global",
    "pareto_broad_local",
    "pareto_broad_global",
    "pareto_broad_global_protected",
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


def _missing(row: dict[str, Any]) -> list[str]:
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
        if row.get("stage") == STAGE and int(row.get("seed", -1)) == seed
    ]
    by_arm = {str(row.get("arm")): row for row in rows}
    if len(by_arm) != len(rows):
        raise ValueError("allocator-v2 summary contains duplicate arm/seed rows")
    control = by_arm.get(RANDOM_CONTROL)
    ceiling = by_arm.get(QUALITY_CEILING)
    if control is None or ceiling is None:
        raise ValueError("allocator-v2 summary is missing registered controls")
    for name, row in ((RANDOM_CONTROL, control), (QUALITY_CEILING, ceiling)):
        missing = _missing(row)
        if missing:
            raise ValueError(f"{name}/seed-{seed} is missing {missing}")
    base = {key: _number(control, key) for key in REQUIRED}
    ceiling_values = {key: _number(ceiling, key) for key in REQUIRED}

    decisions: list[dict[str, Any]] = []
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
        missing_metrics = _missing(row)
        if missing_metrics:
            decisions.append({
                "arm": arm,
                "selected": False,
                "missing": False,
                "eligible": False,
                "missing_metrics": missing_metrics,
            })
            continue
        values = {key: _number(row, key) for key in REQUIRED}
        guards = {
            "target_bytes": row.get("dynamic_actual_target_match") is True,
            "faithful_scoring": row.get("dynamic_scoring_matches_deployed") is True,
            "not_halted": not bool(row.get("evaluation_halted", False)),
            "primary_kl": values["mean_teacher_kl"]
            <= base["mean_teacher_kl"] * 1.02,
            "primary_top1": values["top1_agreement"]
            >= base["top1_agreement"] - 0.01,
            "wikitext2": values["ppl_wikitext2"]
            <= base["ppl_wikitext2"] * 1.02,
            "c4": values["ppl_c4"] <= base["ppl_c4"] * 1.02,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= base["diverse_mean_teacher_kl"] * 1.02,
            "diverse_top1": values["diverse_top1_agreement"]
            >= base["diverse_top1_agreement"] - 0.01,
            "diverse_trajectory": values["diverse_trajectory_token_agreement"]
            >= base["diverse_trajectory_token_agreement"] - 0.02,
        }
        signals = {
            "primary_kl": values["mean_teacher_kl"]
            <= base["mean_teacher_kl"] * 0.98,
            "primary_top1": values["top1_agreement"]
            >= base["top1_agreement"] + 0.005,
            "wikitext2": values["ppl_wikitext2"]
            <= base["ppl_wikitext2"] * 0.995,
            "c4": values["ppl_c4"] <= base["ppl_c4"] * 0.995,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= base["diverse_mean_teacher_kl"] * 0.98,
            "diverse_top1": values["diverse_top1_agreement"]
            >= base["diverse_top1_agreement"] + 0.005,
            "diverse_trajectory": values["diverse_trajectory_token_agreement"]
            >= base["diverse_trajectory_token_agreement"] + 0.01,
        }
        fidelity_signal = signals["primary_kl"] or signals["diverse_kl"]
        eligible = (
            all(guards.values())
            and fidelity_signal
            and sum(signals.values()) >= 2
        )
        # Rank by matched-control quality, then use distance to the larger W4
        # ceiling only as a deterministic secondary preference.
        score = (
            values["mean_teacher_kl"] / base["mean_teacher_kl"]
            + values["diverse_mean_teacher_kl"]
            / base["diverse_mean_teacher_kl"]
            + values["ppl_wikitext2"] / base["ppl_wikitext2"]
            + values["ppl_c4"] / base["ppl_c4"]
            - values["top1_agreement"]
            - values["diverse_top1_agreement"]
            - values["diverse_trajectory_token_agreement"]
            + 0.05 * (
                values["mean_teacher_kl"]
                / ceiling_values["mean_teacher_kl"]
            )
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
            "proxy_rank_correlation": row.get(
                "dynamic_proxy_rank_correlation"
            ),
        })

    finalists = [arm for _score, arm in sorted(viable)[:limit]]
    for decision in decisions:
        decision["selected"] = decision.get("arm") in finalists
    confirmation = [QUALITY_CEILING, RANDOM_CONTROL, *finalists]
    return {
        "protocol": PROTOCOL,
        "code_revision": summary.get("code_revision"),
        "screen_seed": seed,
        "quality_ceiling_arm": QUALITY_CEILING,
        "random_control_arm": RANDOM_CONTROL,
        "finalists": finalists,
        "confirmation_arms": list(dict.fromkeys(confirmation)),
        "decisions": decisions,
        "interpretation": (
            "Seed-0 faithful exact-byte screen. A recipe must beat the matched "
            "random W3/W4 allocation on fidelity plus another registered metric; "
            "seeds 1/2 remain mandatory before promotion."
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
