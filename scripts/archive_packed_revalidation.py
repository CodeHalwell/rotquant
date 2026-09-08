#!/usr/bin/env python3
"""Preserve compact revalidation JSON byte-for-byte; never copy absent weights."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.vocabulary import file_digest
from scripts.run_qwen35_packed_validation import read_record
from scripts.run_qwen35_vocabulary_budget import digest


def archive(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("archive and original must not overlap")
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix in (".json", ".sha256"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("evidence cannot contain symlinks")
    checked = 0
    for p in paths:
        if p.suffix == ".sha256":
            if read_record(p.with_suffix(".json")) is None:
                raise ValueError("orphan checksum")
            checked += 1
    summary = json.loads((source / "summary.json").read_text())
    if not summary["complete"] or summary["artifact_validated_arms"] != ["b5_v6", "b5_v8"]:
        raise ValueError("incomplete/failed revalidation")
    for arm in ("b5_v6", "b5_v8"):
        record = read_record(source / f"{arm}_s0/validation.json")
        original = source / f"original/{arm}_s0"
        prepared = read_record(original / "prepared.json")
        intent = read_record(original / "preparation.json")
        if not record["passed"] or record["preparation_identity"] != prepared["identity"]:
            raise ValueError("validation lineage mismatch")
        if record["identity"]["revalidation"]["prepared_sha256"] != file_digest(
            original / "prepared.json"
        ):
            raise ValueError("prepared hash mismatch")
        manifest_path = original / "checkpoint/rotquant_config.json"
        manifest = json.loads(manifest_path.read_text())
        if file_digest(manifest_path) != record["manifest_sha256"] or manifest["deployment"][
            "experiment_identity"
        ] != digest(prepared["identity"]):
            raise ValueError("manifest binding mismatch")
        if (
            manifest["deployment"]["preparation_sha256"]
            != file_digest(original / "preparation.json")
            or intent["identity"] != prepared["identity"]
        ):
            raise ValueError("preparation binding mismatch")
        if sum(record["artifact"]["files"].values()) != record["size"]["measured_bytes"]:
            raise ValueError("artifact ledger does not add up")
    files = {
        str(p.relative_to(source)): {"bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in paths
    }
    index = {
        "protocol": "packed-revalidation-archive-v1",
        "source_bundle": source.name,
        "checksum_pairs": checked,
        "files": files,
        "original_bytes_preserved": True,
        "boundary": "Compact JSON/manifest evidence only. Logs and absent checkpoint/probe tensors are not archived; no independent CUDA rerun.",
    }
    for name, entry in files.items():
        if (output / name).exists() and file_digest(output / name) != entry["sha256"]:
            raise ValueError(f"archive collision: {name}")
    index_path = output / "evidence_index.json"
    if index_path.exists() and json.loads(index_path.read_text()) != index:
        raise ValueError("archive index collision")
    for name, entry in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(source / name, path)
        if file_digest(path) != entry["sha256"]:
            raise ValueError("copied hash mismatch")
    if not index_path.exists():
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = archive(args.input_dir, args.output_dir)
    print(json.dumps({"files": len(result["files"]), "checksum_pairs": result["checksum_pairs"]}))
