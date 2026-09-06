#!/usr/bin/env python3
"""Copy a bounded v4 evidence bundle and index its unmodified JSON records."""
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


def archive(source: Path, destination: Path):
    filenames = [
        "environment.json", "runtime_preflight.json", "allocator_v4_validation.json",
        "allocator_v4_finalists.json", "allocator_v4_vs_unsloth.json", "allocator_v4_decision.json",
        "phase_summaries/allocator_v4_screen.json", "phase_summaries/allocator_v4_confirm.json",
        "unsloth_kl/unsloth_ud_q4_kl.json",
    ]
    filenames += [str(path.relative_to(source)) for path in (source / "model_trials").glob("qwen35*.json")]
    filenames += [str(path.relative_to(source)) for path in (source / "artifacts").rglob("rotquant_config.json")]
    if len([name for name in filenames if name.startswith("model_trials/")]) != 17:
        raise ValueError("expected exactly 17 unique completed v4 trials")
    decision = json.loads((source / "allocator_v4_decision.json").read_text())
    hashes = {}
    for name in sorted(filenames):
        origin, target = source / name, destination / name
        fingerprint = file_digest(origin)
        if target.exists() and file_digest(target) != fingerprint:
            raise ValueError(f"refusing to overwrite different archived evidence: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(origin, target)
        hashes[name] = fingerprint
    write_result(str(destination / "evidence_index.json"), {
        "protocol": "allocator-v4-evidence-archive-v1", "source_bundle": source.name,
        "code_revision": decision["code_revision"], "files_sha256": hashes,
        "boundary": "Original JSON copied unchanged. Export manifests included; tensor/tokenizer binaries omitted.",
    })
    return {"files": len(hashes), "destination": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.input_dir, args.output_dir)))


if __name__ == "__main__":
    main()
