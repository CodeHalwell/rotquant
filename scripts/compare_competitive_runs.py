#!/usr/bin/env python3
"""Produce size-matched paired deltas for two completed run reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rotquant.eval.competition import CompetitiveEvalProtocol
from rotquant.eval.competitive_run import compare_run_reports
from rotquant.utils import write_result


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--max-size-delta", type=float, default=0.01)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_run_reports(
        candidate_report=_load_json(args.candidate),
        baseline_report=_load_json(args.baseline),
        protocol=CompetitiveEvalProtocol.from_manifest(_load_json(args.protocol)),
        max_size_delta_fraction=args.max_size_delta,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_result(str(args.output), comparison)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": comparison["status"],
                "artifact_comparison": comparison["artifact_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
