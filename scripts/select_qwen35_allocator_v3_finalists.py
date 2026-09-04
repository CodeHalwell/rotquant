#!/usr/bin/env python3
"""Select distinct allocator-v3 recipes against the exact broad random control."""

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

PROTOCOL = "qwen35-allocator-v3-finalist-selection-v1"
STAGE = "allocator-v3"
QUALITY_CEILING = "uniform_scale8_w4"
RANDOM_CONTROL = "random_broad_exact"
CANDIDATE_ARMS = (
    "pareto_global",
    "pareto_global_refined",
    "pareto_w6_top5_refined",
    "pareto_w8_top1_refined",
    "pareto_w8_top2p5_refined",
    "pareto_w8_top5_refined",
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
        key
        for key in REQUIRED
        if not isinstance(row.get(key), (int, float)) or isinstance(row.get(key), bool)
    ]


def _signature(row: dict[str, Any]) -> str | None:
    value = row.get("dynamic_allocation_fingerprint")
    return value if isinstance(value, str) and value else None


def select_finalists(summary: dict[str, Any], *, seed: int = 0, limit: int = 3) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if summary.get("complete") is not True:
        raise ValueError("allocator-v3 screen summary is incomplete")
    if tuple(map(int, summary.get("seeds", ()))) != (seed,):
        raise ValueError("allocator-v3 screen must contain only the requested seed")
    rows = [
        row
        for row in summary.get("rows", [])
        if row.get("stage") == STAGE and int(row.get("seed", -1)) == seed
    ]
    by_arm = {str(row.get("arm")): row for row in rows}
    if len(by_arm) != len(rows):
        raise ValueError("allocator-v3 summary contains duplicate arm/seed rows")
    control = by_arm.get(RANDOM_CONTROL)
    ceiling = by_arm.get(QUALITY_CEILING)
    if control is None or ceiling is None:
        raise ValueError("allocator-v3 summary is missing registered controls")
    for name, row in ((RANDOM_CONTROL, control), (QUALITY_CEILING, ceiling)):
        missing = _missing(row)
        if missing:
            raise ValueError(f"{name}/seed-{seed} is missing {missing}")
        if bool(row.get("evaluation_halted", False)):
            raise ValueError(f"{name}/seed-{seed} halted during evaluation")
    random_requirements = {
        "target match": control.get("dynamic_actual_target_match") is True,
        "artifact accounting": isinstance(
            control.get("dynamic_estimated_artifact_bytes"), (int, float)
        )
        and not isinstance(control.get("dynamic_estimated_artifact_bytes"), bool),
        "faithful scoring": control.get("dynamic_scoring_matches_deployed") is True,
        "allocation fingerprint": _signature(control) is not None,
    }
    failed_random = [name for name, passed in random_requirements.items() if not passed]
    if failed_random:
        raise ValueError(f"{RANDOM_CONTROL}/seed-{seed} failed " + ", ".join(failed_random))
    base = {key: _number(control, key) for key in REQUIRED}
    ceiling_values = {key: _number(ceiling, key) for key in REQUIRED}

    decisions: list[dict[str, Any]] = []
    viable: list[tuple[float, int, str, str]] = []
    for order, arm in enumerate(CANDIDATE_ARMS):
        row = by_arm.get(arm)
        if row is None:
            decisions.append(
                {
                    "arm": arm,
                    "selected": False,
                    "missing": True,
                    "missing_metrics": list(REQUIRED),
                }
            )
            continue
        missing_metrics = _missing(row)
        signature = _signature(row)
        if signature is None:
            missing_metrics.append("dynamic_allocation_fingerprint")
        if missing_metrics:
            decisions.append(
                {
                    "arm": arm,
                    "selected": False,
                    "missing": False,
                    "eligible": False,
                    "missing_metrics": missing_metrics,
                }
            )
            continue
        values = {key: _number(row, key) for key in REQUIRED}
        estimated_artifact = row.get("dynamic_estimated_artifact_bytes")
        target_artifact = row.get("dynamic_target_artifact_bytes")
        artifact_accounted = (
            isinstance(estimated_artifact, (int, float))
            and not isinstance(estimated_artifact, bool)
            and isinstance(target_artifact, (int, float))
            and not isinstance(target_artifact, bool)
        )
        guards = {
            "target_bytes": row.get("dynamic_actual_target_match") is True,
            "artifact_accounted": artifact_accounted,
            "faithful_scoring": row.get("dynamic_scoring_matches_deployed") is True,
            "not_halted": not bool(row.get("evaluation_halted", False)),
            "primary_kl": values["mean_teacher_kl"] <= base["mean_teacher_kl"] * 1.02,
            "primary_top1": values["top1_agreement"] >= base["top1_agreement"] - 0.01,
            "wikitext2": values["ppl_wikitext2"] <= base["ppl_wikitext2"] * 1.02,
            "c4": values["ppl_c4"] <= base["ppl_c4"] * 1.02,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= base["diverse_mean_teacher_kl"] * 1.02,
            "diverse_top1": values["diverse_top1_agreement"]
            >= base["diverse_top1_agreement"] - 0.01,
            "diverse_trajectory": values["diverse_trajectory_token_agreement"]
            >= base["diverse_trajectory_token_agreement"] - 0.02,
        }
        signals = {
            "primary_kl": values["mean_teacher_kl"] <= base["mean_teacher_kl"] * 0.98,
            "primary_top1": values["top1_agreement"] >= base["top1_agreement"] + 0.005,
            "wikitext2": values["ppl_wikitext2"] <= base["ppl_wikitext2"] * 0.995,
            "c4": values["ppl_c4"] <= base["ppl_c4"] * 0.995,
            "diverse_kl": values["diverse_mean_teacher_kl"]
            <= base["diverse_mean_teacher_kl"] * 0.98,
            "diverse_top1": values["diverse_top1_agreement"]
            >= base["diverse_top1_agreement"] + 0.005,
            "diverse_trajectory": values["diverse_trajectory_token_agreement"]
            >= base["diverse_trajectory_token_agreement"] + 0.01,
        }
        fidelity_signal = signals["primary_kl"] or signals["diverse_kl"]
        eligible = all(guards.values()) and fidelity_signal and sum(signals.values()) >= 2
        score = (
            values["mean_teacher_kl"] / base["mean_teacher_kl"]
            + values["diverse_mean_teacher_kl"] / base["diverse_mean_teacher_kl"]
            + values["ppl_wikitext2"] / base["ppl_wikitext2"]
            + values["ppl_c4"] / base["ppl_c4"]
            - values["top1_agreement"]
            - values["diverse_top1_agreement"]
            - values["diverse_trajectory_token_agreement"]
            + 0.05 * values["mean_teacher_kl"] / ceiling_values["mean_teacher_kl"]
        )
        if eligible:
            viable.append((score, order, arm, signature))
        decisions.append(
            {
                "arm": arm,
                "selected": False,
                "missing": False,
                "missing_metrics": [],
                "eligible": eligible,
                "score": score,
                "guards": guards,
                "signals": signals,
                "metrics": values,
                "allocation_fingerprint": signature,
                "estimated_artifact_bytes": estimated_artifact,
                "target_artifact_bytes": target_artifact,
                "proxy_rank_correlation": row.get("dynamic_proxy_rank_correlation"),
            }
        )

    finalists: list[str] = []
    fingerprint_owner: dict[str, str] = {}
    duplicate_of: dict[str, str] = {}
    for _score, _order, arm, signature in sorted(viable):
        owner = fingerprint_owner.get(signature)
        if owner is not None:
            duplicate_of[arm] = owner
            continue
        fingerprint_owner[signature] = arm
        if len(finalists) < limit:
            finalists.append(arm)
    for decision in decisions:
        arm = str(decision.get("arm"))
        decision["selected"] = arm in finalists
        if arm in duplicate_of:
            decision["duplicate_of"] = duplicate_of[arm]

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
        "deduplicated_recipes": duplicate_of,
        "interpretation": (
            "Seed-0 exact-export-byte screen. Finalists have distinct deployed "
            "allocation fingerprints and must beat the same-palette random "
            "control in paired evidence at seeds 0/1/2 before promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_finalists(summary, seed=args.seed, limit=args.limit)
    write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
