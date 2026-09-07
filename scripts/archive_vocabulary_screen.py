#!/usr/bin/env python3
"""Archive the compact, verified seed-0 screen without modifying original bytes."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.utils import write_result
from rotquant.vocabulary import file_digest
from scripts.assess_qwen35_vocabulary_budget import assess
from scripts.run_qwen35_vocabulary_budget import load_completed


def archive(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and archive must be separate, non-nested directories")
    trials = sorted((source / "model_trials").glob("b*_v*_s0.json"))
    records = []
    for path in trials:
        raw = json.loads(path.read_text())
        records.append(load_completed(path, raw["identity"]))
    if any(record is None for record in records):
        raise ValueError("screen has incomplete records")
    summary = assess(records, seeds=(0,))
    if not summary["complete"] or json.loads(json.dumps(summary)) != json.loads(
            (source / "screen_summary.json").read_text()):
        raise ValueError("saved screen summary does not reproduce")
    paths = [*trials, *(path.with_suffix(".sha256") for path in trials)]
    paths.extend(path for path in source.glob("*.json")
                 if path.is_file() and path.name != "evidence_index.json")
    files = {str(path.relative_to(source)): {"sha256": file_digest(path), "bytes": path.stat().st_size}
             for path in sorted(paths)}
    # Check every collision before creating anything, and never overwrite evidence.
    for name, record in files.items():
        existing = output / name
        if existing.exists() and file_digest(existing) != record["sha256"]:
            raise ValueError(f"archive collision: {name}")
    for name in files:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source / name, destination)
        if file_digest(destination) != files[name]["sha256"]:
            raise ValueError(f"archive copy mismatch: {name}")
    index = {"protocol": "rotquant-vocabulary-screen-evidence-v1", "source_bundle": source.name,
             "original_bytes_preserved": True, "trial_records": len(trials), "files": files,
             "omissions": "Logs, binary tensors, model/tokenizer files and caches are not archived. No packed full-Qwen artifact is implied."}
    write_result(str(output / "evidence_index.json"), index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/results/raw/qwen35_vocabulary_budget_ce6c8ec861a2")
    args = parser.parse_args()
    result = archive(args.input_dir, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "trial_records": result["trial_records"],
                      "files": len(result["files"])}, indent=2))


if __name__ == "__main__":
    main()
