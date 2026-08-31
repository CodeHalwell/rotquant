#!/usr/bin/env python3
"""Aggregate engine-neutral prompt observations into a competitive run report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rotquant.eval.competition import CompetitiveEvalProtocol
from rotquant.eval.competitive_run import (
    PromptObservation,
    RunFailure,
    RunMetadata,
    aggregate_competitive_run,
)
from rotquant.eval.data_manifest import read_dataset_manifest
from rotquant.utils import write_result


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {error}") from error
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = CompetitiveEvalProtocol.from_manifest(_load_json(args.protocol))
    metadata = RunMetadata.from_manifest(_load_json(args.metadata))
    observations = tuple(
        PromptObservation.from_manifest(record) for record in _load_jsonl(args.observations)
    )
    failures = (
        tuple(RunFailure.from_manifest(record) for record in _load_jsonl(args.failures))
        if args.failures
        else ()
    )
    report = aggregate_competitive_run(
        protocol=protocol,
        prompt_manifest=read_dataset_manifest(args.prompt_manifest),
        metadata=metadata,
        observations=observations,
        failures=failures,
    )
    write_result(str(args.output), report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "observed_prompt_count": report["observed_prompt_count"],
                "failure_counts": report["failure_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
