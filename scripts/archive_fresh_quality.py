#!/usr/bin/env python3
"""Audit and preserve the completed nine-arm fresh-quality compact evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_qwen35_fresh_eval as fresh


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(source):
    root = source / "fresh"
    manifest = fresh.read_record(root / "manifest.json")
    fresh.validate_manifest(manifest)
    summary = fresh.read_record(root / "summary.json")
    expected = {"source_fp16", "gguf_bf16_bridge", "unsloth_ud_q4"} | {
        f"b5_v{v}_s{s}" for s in (0, 1, 2) for v in (6, 8)}
    require(summary["complete"] and not summary["missing"] and summary["reuse"] is None,
            "only the completed, non-reused nine-arm bundle is supported")
    require(set(summary["expected_labels"]) == expected and len(summary["expected_labels"]) == 9,
            "missing/duplicate registered arms")
    require(summary["manifest"] == manifest["fingerprint"], "summary manifest changed")
    collections, count = {}, 0
    for label in summary["expected_labels"]:
        directory = root / "runs" / label
        identity = fresh.read_record(directory / "identity.json")
        marker = fresh.read_record(directory / "complete.json")
        require(marker["complete"] and marker["identity"] == identity, "completion mismatch")
        require(identity["manifest"] == manifest["fingerprint"] and
                identity["source"] == manifest["identity"]["source"] and
                identity["runtime"] == manifest["identity"]["runtime"], "collection provenance mismatch")
        values = []
        for item in manifest["items"]:
            row = fresh.read_record(directory / (item["id"] + ".json"))
            require(row["id"] == item["id"] and row["input_hash"] == item["input_hash"] and
                    row["manifest"] == manifest["fingerprint"] and row["collection"] == fresh.fingerprint(identity),
                    "prompt pairing mismatch")
            require(row["tokens"] > 0 and all(math.isfinite(row[k]) for k in
                    ("mean_teacher_kl", "top1_agreement", "source_nll", "candidate_nll")), "nonfinite evidence")
            if item["kind"] == "task":
                scored = fresh.score_task(row["output_text"], item["expected"], truncated=row["truncated"])
                require(all(row[k] == v for k, v in scored.items()), "stored task oracle differs")
                count += 1
            values.append(row)
        if label.startswith("b5_"):
            parity = fresh.read_record(directory / "reload_probe_report.json")
            require(parity["identity"] == identity and parity["parity"]["passed"], "reload parity failed")
            residency = marker["residency"]
            require(residency["passed"] and residency["one_shared_vocabulary_owner"] and
                    residency["backbone_fallback_cache_bytes"] == 0, "residency failed")
        elif label != "source_fp16":
            gate = fresh.read_record(directory / "tokenizer_audit.json")
            require(gate["identity"] == identity and gate["passed"] and gate["policy"] == "frozen-hf"
                    and all(all(r["checks"].values()) for r in gate["records"]), "token audit failed")
        collections[label] = values
    for row in summary["rows"]:
        subset = [v for v in collections[row["label"]] if v["domain"] == row["domain"]]
        tokens = sum(v["tokens"] for v in subset)
        require(row["prompts"] == len(subset) and row["tokens"] == tokens, "summary counts differ")
        for key in ("mean_teacher_kl", "top1_agreement", "source_nll", "candidate_nll"):
            value = sum(v[key] * v["tokens"] for v in subset) / tokens
            require(math.isclose(value, row[key], abs_tol=1e-12), f"summary metric differs: {key}")
        if row["domain"] != "c4":
            for key in ("task_success", "json_valid", "truncated", "source_task_success",
                        "trajectory_token_agreement", "exact_trajectory"):
                require(math.isclose(sum(v[key] for v in subset) / len(subset), row[key], abs_tol=1e-12),
                        f"summary task metric differs: {key}")
    for contrast in summary["paired_contrasts"]:
        left, right = [[v for v in collections[contrast[label]] if v["domain"] == contrast["domain"]]
                       for label in ("left", "right")]
        computed = fresh.paired_family_interval(left, right, contrast["field"])
        require(all(computed[k] == contrast[k] for k in computed), "paired contrast differs")
    return {"collections": len(collections), "prompt_records": sum(map(len, collections.values())),
            "oracle_rescored_tasks": count, "summary_rows_recomputed": len(summary["rows"]),
            "paired_contrasts_recomputed": len(summary["paired_contrasts"])}


def archive(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    fresh.separate(source, output)
    require(not any(p.is_symlink() for p in source.rglob("*")), "symlinks not allowed in evidence")
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix in (".json", ".sha256"))
    checked = 0
    for path in paths:
        if path.suffix == ".sha256":
            require(fresh.read_record(path.with_suffix(".json")) is not None, "orphan checksum")
            checked += 1
    checks = audit(source)
    files = {str(p.relative_to(source)): {"bytes": p.stat().st_size, "sha256": fresh.file_digest(p)} for p in paths}
    index = {"protocol": "fresh-quality-archive-v1", "source_bundle": source.name,
             "checks": {**checks, "checksum_pairs": checked}, "files": files,
             "original_bytes_preserved": True,
             "boundary": "Compact JSON/manifest evidence only. Checkpoint/probe tensors and reference NPZ arrays are absent; raw KL and full-model execution were not independently rerun. Logs are not copied. Earlier strict task scores remain unchanged."}
    for name, entry in files.items():
        path = output / name
        require(not path.exists() or (not path.is_symlink() and fresh.file_digest(path) == entry["sha256"]),
                f"archive collision: {name}")
    target = output / "evidence_index.json"
    require(not target.exists() or json.loads(target.read_text()) == index, "archive index collision")
    for name, entry in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(source / name, path)
        require(fresh.file_digest(path) == entry["sha256"], "copied hash mismatch")
    if not target.exists():
        target.write_text(json.dumps(index, indent=2) + "\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = archive(args.input_dir, args.output_dir)
    print(json.dumps(result["checks"], indent=2))
