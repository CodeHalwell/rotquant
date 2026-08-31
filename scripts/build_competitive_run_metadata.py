#!/usr/bin/env python3
"""Hash and count every file in a deployed competitive-eval artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rotquant.eval.competition import CompetitiveEvalProtocol
from rotquant.eval.competitive_run import (
    RunMetadata,
    artifact_identity,
    inspect_artifact_files,
)
from rotquant.utils import write_result


def _artifact_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LOGICAL_NAME=PATH")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("artifact must be LOGICAL_NAME=PATH")
    return name, Path(path)


def _load_protocol(path: Path) -> CompetitiveEvalProtocol:
    with path.open(encoding="utf-8") as handle:
        return CompetitiveEvalProtocol.from_manifest(json.load(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-revision", required=True)
    parser.add_argument(
        "--artifact",
        type=_artifact_argument,
        action="append",
        required=True,
        help="Repeat LOGICAL_NAME=PATH for weights, projectors, MTP, and other files",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = inspect_artifact_files(tuple(args.artifact))
    metadata = RunMetadata(
        name=args.name,
        format=args.format,
        artifact_sha256=artifact_identity(files),
        artifact_bytes=sum(file.bytes for file in files),
        artifact_files=files,
        engine=args.engine,
        engine_revision=args.engine_revision,
        protocol_fingerprint=_load_protocol(args.protocol).fingerprint,
    )
    write_result(str(args.output), metadata.manifest())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": metadata.artifact_sha256,
                "artifact_bytes": metadata.artifact_bytes,
                "artifact_files": len(metadata.artifact_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
