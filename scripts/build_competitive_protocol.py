#!/usr/bin/env python3
"""Bind pinned calibration and held-out manifests into one eval protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rotquant.eval.data_manifest import protocol_from_manifests, read_dataset_manifest
from rotquant.utils import write_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--include-auxiliary-heads", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = protocol_from_manifests(
        model_id=args.model_id,
        model_revision=args.model_revision,
        calibration=read_dataset_manifest(args.calibration_manifest),
        evaluation=read_dataset_manifest(args.evaluation_manifest),
        include_auxiliary_heads=args.include_auxiliary_heads,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    write_result(str(args.output), protocol.manifest())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_fingerprint": protocol.fingerprint,
                "prompt_count": protocol.prompt_count,
                "generation_tokens": protocol.generation_tokens,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
