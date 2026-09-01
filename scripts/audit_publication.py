#!/usr/bin/env python3
"""Validate paper claims and optionally reconcile them with local artifacts.

The publication manifest is the single machine-readable source for numerical
claims in the paper. This command fails closed when a reported aggregate cannot
be reproduced from its constituent measurements or when a local artifact does
not match the recorded revision, size, structure, or digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _close(actual: float, expected: float, tolerance: float, label: str) -> dict[str, Any]:
    difference = abs(actual - expected)
    passed = difference <= tolerance
    return {
        "check": label,
        "passed": passed,
        "actual": actual,
        "expected": expected,
        "absolute_difference": difference,
        "tolerance": tolerance,
    }


def _exact(actual: Any, expected: Any, label: str) -> dict[str, Any]:
    return {
        "check": label,
        "passed": actual == expected,
        "actual": actual,
        "expected": expected,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_numerical_claims(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported publication manifest schema")

    checks: list[dict[str, Any]] = []
    joint = manifest["results"]["qwen_joint"]
    ppls = [float(value) for value in joint["candidate_ppl_by_seed"]]
    source_ppl = float(joint["source_ppl"])
    mean_ppl = sum(ppls) / len(ppls)
    mean_relative = 100.0 * (mean_ppl / source_ppl - 1.0)
    worst_relative = 100.0 * (max(ppls) / source_ppl - 1.0)
    checks.extend(
        [
            _close(mean_ppl, float(joint["reported_mean_ppl"]), 5e-5, "joint mean PPL"),
            _close(
                mean_relative,
                float(joint["reported_mean_relative_ppl_pct"]),
                0.005,
                "joint mean relative PPL",
            ),
            _close(
                worst_relative,
                float(joint["reported_worst_relative_ppl_pct"]),
                0.005,
                "joint worst relative PPL",
            ),
        ]
    )

    transfer = manifest["results"]["cache_transfer"]
    short_reduction = 100.0 * (
        1.0 - float(transfer["mixed_short_kl"]) / float(transfer["uniform_short_kl"])
    )
    long_reduction = 100.0 * (
        1.0 - float(transfer["mixed_long_kl"]) / float(transfer["uniform_long_kl"])
    )
    checks.extend(
        [
            _close(
                short_reduction,
                float(transfer["reported_short_reduction_pct"]),
                0.05,
                "short-context cache KL reduction",
            ),
            _close(
                long_reduction,
                float(transfer["reported_long_reduction_pct"]),
                0.05,
                "long-context cache KL reduction",
            ),
        ]
    )

    matched = manifest["results"]["matched_cache"]
    candidate = [float(value) for value in matched["candidate_kl_by_seed"]]
    control = [float(value) for value in matched["control_k4v4_kl_by_seed"]]
    if len(candidate) != len(control) or not candidate:
        raise ValueError("matched cache arrays must be non-empty and have equal length")
    ratios = [left / right for left, right in zip(candidate, control, strict=True)]
    reductions = [100.0 * (1.0 - ratio) for ratio in ratios]
    reported_reductions = [float(value) for value in matched["reported_reduction_pct_by_seed"]]
    if len(reductions) != len(reported_reductions):
        raise ValueError("reported matched-cache reductions have the wrong length")
    for seed, (actual, expected) in enumerate(zip(reductions, reported_reductions, strict=True)):
        checks.append(_close(actual, expected, 0.05, f"matched cache seed {seed} reduction"))
    mean_ratio = sum(ratios) / len(ratios)
    worst_ratio = max(ratios)
    checks.extend(
        [
            _close(mean_ratio, float(matched["reported_mean_ratio"]), 5e-6, "matched cache mean ratio"),
            _close(worst_ratio, float(matched["reported_worst_ratio"]), 5e-6, "matched cache worst ratio"),
        ]
    )

    models = manifest["models"]
    native = manifest["artifacts"]["native_joint_gguf"]
    checks.extend(
        [
            _exact(
                native["producer_model_id"],
                models["rotquant"]["model_id"],
                "native producer model id",
            ),
            _exact(
                native["producer_revision"],
                models["rotquant"]["revision"],
                "native producer revision",
            ),
            _exact(
                int(native["verified_projections"]),
                int(models["rotquant"]["quantized_modules"]),
                "native verified projection count",
            ),
            _exact(native["packed_conformance"], "pass", "native packed conformance status"),
        ]
    )
    source_tensor_bytes = int(models["source"]["tensor_bytes"])
    rotquant_tensor_bytes = int(models["rotquant"]["tensor_bytes"])
    source_artifact_bytes = int(models["source"]["complete_snapshot_bytes"])
    rotquant_artifact_bytes = int(models["rotquant"]["complete_snapshot_bytes"])
    # Like-for-like storage: the source index counts an MTP head that the
    # Transformers model never loads and the export never stores.  Compare
    # against the loaded tensors only, and check the reported figure.
    loaded_source_bytes = int(
        models["source"].get("tensor_bytes_excluding_mtp", source_tensor_bytes)
    )
    like_for_like = 100.0 * (1.0 - rotquant_tensor_bytes / loaded_source_bytes)
    if "reported_like_for_like_tensor_reduction_pct" in joint:
        checks.append(_close(
            like_for_like,
            float(joint["reported_like_for_like_tensor_reduction_pct"]),
            0.005,
            "like-for-like tensor storage reduction",
        ))
    derived = {
        "like_for_like_tensor_storage_reduction_pct": like_for_like,
        "loaded_source_tensor_bytes": loaded_source_bytes,
        "joint_mean_ppl": mean_ppl,
        "joint_mean_relative_ppl_pct": mean_relative,
        "joint_worst_relative_ppl_pct": worst_relative,
        "short_cache_kl_reduction_pct": short_reduction,
        "long_cache_kl_reduction_pct": long_reduction,
        "matched_cache_mean_ratio": mean_ratio,
        "matched_cache_worst_ratio": worst_ratio,
        "actual_tensor_storage_reduction_pct": 100.0
        * (1.0 - rotquant_tensor_bytes / source_tensor_bytes),
        "actual_snapshot_storage_reduction_pct": 100.0
        * (1.0 - rotquant_artifact_bytes / source_artifact_bytes),
    }
    return checks, derived


def audit_source_checkpoint(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"source tensor index not found: {index_path}")
    index = json.loads(index_path.read_text())
    tensor_bytes = int(index["metadata"]["total_size"])
    snapshot_bytes = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
    return [
        _exact(tensor_bytes, int(entry["tensor_bytes"]), "source tensor bytes"),
        _exact(snapshot_bytes, int(entry["complete_snapshot_bytes"]), "source snapshot bytes"),
    ]


def audit_rotquant_checkpoint(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = path / "rotquant_config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RotQuant manifest not found: {manifest_path}")
    checkpoint = json.loads(manifest_path.read_text())
    tensor_files = [str(name) for name in entry["tensor_files"]]
    tensor_bytes = sum((path / name).stat().st_size for name in tensor_files)
    snapshot_bytes = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
    lora_ranks = [int(module.get("lora_rank", 0)) for module in checkpoint["quantized_modules"]]
    return [
        _exact(tensor_bytes, int(entry["tensor_bytes"]), "RotQuant tensor bytes"),
        _exact(snapshot_bytes, int(entry["complete_snapshot_bytes"]), "RotQuant snapshot bytes"),
        _exact(
            len(checkpoint["quantized_modules"]),
            int(entry["quantized_modules"]),
            "RotQuant quantized module count",
        ),
        _exact(max(lora_ranks, default=0), 0, "RotQuant retained LoRA rank"),
    ]


def audit_native_gguf(path: Path, entry: dict[str, Any], verify_hash: bool) -> list[dict[str, Any]]:
    checks = [_exact(path.stat().st_size, int(entry["bytes"]), "native GGUF bytes")]
    if verify_hash:
        checks.append(_exact(_sha256(path), str(entry["sha256"]), "native GGUF SHA-256"))
    return checks


def _tex_number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def render_tex_macros(manifest: dict[str, Any], derived: dict[str, float]) -> str:
    joint = manifest["results"]["qwen_joint"]
    transfer = manifest["results"]["cache_transfer"]
    models = manifest["models"]
    native = manifest["artifacts"]["native_joint_gguf"]
    lines = [
        "% Generated by scripts/audit_publication.py; do not edit by hand.",
        f"\\newcommand{{\\QwenSourcePPL}}{{{_tex_number(float(joint['source_ppl']), 4)}}}",
        f"\\newcommand{{\\QwenJointMeanPPL}}{{{_tex_number(derived['joint_mean_ppl'], 4)}}}",
        f"\\newcommand{{\\QwenJointMeanRelativePct}}{{{_tex_number(derived['joint_mean_relative_ppl_pct'], 2)}}}",
        f"\\newcommand{{\\QwenJointWorstRelativePct}}{{{_tex_number(derived['joint_worst_relative_ppl_pct'], 2)}}}",
        f"\\newcommand{{\\EstimatedWeightReductionPct}}{{{_tex_number(float(joint['estimated_complete_weight_reduction_pct']), 2)}}}",
        f"\\newcommand{{\\ActualTensorReductionPct}}{{{_tex_number(derived['actual_tensor_storage_reduction_pct'], 2)}}}",
        f"\\newcommand{{\\ActualSnapshotReductionPct}}{{{_tex_number(derived['actual_snapshot_storage_reduction_pct'], 2)}}}",
        f"\\newcommand{{\\SourceTensorGB}}{{{int(models['source']['tensor_bytes']) / 1e9:.4f}}}",
        f"\\newcommand{{\\SourceTensorExclMtpGB}}{{{derived['loaded_source_tensor_bytes'] / 1e9:.4f}}}",
        f"\\newcommand{{\\LikeForLikeTensorReductionPct}}{{{_tex_number(derived['like_for_like_tensor_storage_reduction_pct'], 2)}}}",
        f"\\newcommand{{\\RotQuantTensorGB}}{{{int(models['rotquant']['tensor_bytes']) / 1e9:.4f}}}",
        f"\\newcommand{{\\CacheShortReductionPct}}{{{_tex_number(float(transfer['reported_short_reduction_pct']), 1)}}}",
        f"\\newcommand{{\\CacheLongReductionPct}}{{{_tex_number(float(transfer['reported_long_reduction_pct']), 1)}}}",
        f"\\newcommand{{\\MatchedCacheMeanRatio}}{{{_tex_number(derived['matched_cache_mean_ratio'], 3)}}}",
        f"\\newcommand{{\\MatchedCacheWorstRatio}}{{{_tex_number(derived['matched_cache_worst_ratio'], 3)}}}",
        f"\\newcommand{{\\NativeGGUFGB}}{{{int(native['bytes']) / 1e9:.4f}}}",
        f"\\newcommand{{\\SourceRevision}}{{\\texttt{{{models['source']['revision'][:12]}}}}}",
        f"\\newcommand{{\\RotQuantRevision}}{{\\texttt{{{models['rotquant']['revision'][:12]}}}}}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("paper/data/publication_results.json"))
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--rotquant-checkpoint", type=Path)
    parser.add_argument("--native-gguf", type=Path)
    parser.add_argument("--skip-native-hash", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("paper/generated/audit_report.json"))
    parser.add_argument("--tex-output", type=Path, default=Path("paper/generated/results.tex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    checks, derived = validate_numerical_claims(manifest)
    if args.source_checkpoint is not None:
        checks.extend(audit_source_checkpoint(args.source_checkpoint, manifest["models"]["source"]))
    if args.rotquant_checkpoint is not None:
        checks.extend(audit_rotquant_checkpoint(args.rotquant_checkpoint, manifest["models"]["rotquant"]))
    if args.native_gguf is not None:
        checks.extend(
            audit_native_gguf(
                args.native_gguf,
                manifest["artifacts"]["native_joint_gguf"],
                verify_hash=not args.skip_native_hash,
            )
        )

    failed = [check for check in checks if not check["passed"]]
    report = {
        "manifest": str(args.manifest),
        "passed": not failed,
        "checks": checks,
        "derived": derived,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(render_tex_macros(manifest, derived))
    if failed:
        labels = ", ".join(check["check"] for check in failed)
        raise SystemExit(f"publication audit failed: {labels}")
    print(json.dumps({"passed": True, "checks": len(checks), "derived": derived}, indent=2))


if __name__ == "__main__":
    main()
