#!/usr/bin/env python3
"""Compare confirmed RotQuant rows with the prompt-matched Unsloth Q4 anchor."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.utils import write_result

PROTOCOL = "qwen35-rotquant-vs-unsloth-q4-v1"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in values):
        raise ValueError(f"candidate rows are missing numeric {key}")
    return float(statistics.fmean(float(value) for value in values))


def compare(summary: dict[str, Any], unsloth: dict[str, Any],
            arms: list[str], *, byte_tolerance: float = 0.01) -> dict[str, Any]:
    if not 0 <= byte_tolerance <= 0.25:
        raise ValueError("byte_tolerance must be in [0, 0.25]")
    unsloth_metrics = unsloth.get("metrics") or {}
    baseline_bytes = int(unsloth["candidate"]["complete_artifact_bytes"])
    baseline_hashes = list(unsloth_metrics.get("input_hashes") or [])
    if not baseline_hashes:
        raise ValueError("Unsloth result contains no input hashes")

    comparisons = []
    for arm in arms:
        rows = [
            row for row in summary.get("rows", [])
            if row.get("stage") == "dynamic" and row.get("arm") == arm
        ]
        if not rows:
            raise ValueError(f"summary contains no dynamic rows for {arm}")
        for row in rows:
            if list(row.get("logit_fidelity_input_hashes") or []) != baseline_hashes:
                raise ValueError(f"{arm}/seed-{row.get('seed')} input hashes differ")
        artifact_sizes = [
            int(row["packed_artifact_bytes"]) for row in rows
            if isinstance(row.get("packed_artifact_bytes"), (int, float))
        ]
        candidate_bytes = (
            artifact_sizes[0]
            if artifact_sizes
            else round(_mean(rows, "complete_persistent_model_bytes"))
        )
        if any(size != candidate_bytes for size in artifact_sizes):
            raise ValueError(f"{arm} exported artifact sizes differ across seeds")
        byte_ratio = candidate_bytes / baseline_bytes
        candidate_metrics = {
            "mean_teacher_kl": _mean(rows, "mean_teacher_kl"),
            "p95_teacher_kl": _mean(rows, "p95_teacher_kl"),
            "top1_agreement": _mean(rows, "top1_agreement"),
            "nll_delta": _mean(rows, "nll_delta"),
        }
        comparisons.append({
            "arm": arm,
            "seeds": sorted(int(row["seed"]) for row in rows),
            "candidate_bytes": candidate_bytes,
            "baseline_bytes": baseline_bytes,
            "byte_ratio": byte_ratio,
            "byte_delta_fraction": byte_ratio - 1.0,
            "within_byte_gate": abs(byte_ratio - 1.0) <= byte_tolerance,
            "candidate_metrics": candidate_metrics,
            "unsloth_metrics": {
                key: float(unsloth_metrics[key]) for key in (
                    "mean_teacher_kl", "p95_teacher_kl",
                    "top1_agreement", "nll_delta"
                )
            },
            "candidate_minus_unsloth": {
                key: candidate_metrics[key] - float(unsloth_metrics[key])
                for key in candidate_metrics
            },
            "relative_kl_delta": (
                candidate_metrics["mean_teacher_kl"]
                / float(unsloth_metrics["mean_teacher_kl"]) - 1.0
            ),
            "relative_p95_kl_delta": (
                candidate_metrics["p95_teacher_kl"]
                / float(unsloth_metrics["p95_teacher_kl"]) - 1.0
            ),
        })
    return {
        "protocol": PROTOCOL,
        "rotquant_code_revision": summary.get("code_revision"),
        "unsloth_collection_fingerprint": unsloth.get("collection_fingerprint"),
        "prompt_manifest_fingerprint": unsloth.get(
            "prompt_manifest_fingerprint"
        ),
        "prompt_hashes_match": True,
        "byte_tolerance_fraction": byte_tolerance,
        "comparisons": comparisons,
        "interpretation": (
            "Cross-engine development anchor. Provider-quality claims require "
            "within-byte-gate rows and the registered engine-neutral protocol."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--unsloth", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--byte-tolerance", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.summary.read_text(encoding="utf-8")),
        json.loads(args.unsloth.read_text(encoding="utf-8")),
        list(dict.fromkeys(args.arm)),
        byte_tolerance=args.byte_tolerance,
    )
    write_result(str(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
