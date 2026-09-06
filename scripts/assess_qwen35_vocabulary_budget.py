#!/usr/bin/env python3
"""Fail-closed screen selection: quality candidates, never artifact promotion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from rotquant.eval.promotion import QUALITY_GUARDS, finite_number, quality_guards
from rotquant.utils import write_result

PROTOCOL = "qwen35-vocabulary-budget-assessment-v1"
EXPECTED_ARMS = {f"b{b}_v{v}" for b in (16, 4, 5) for v in (16, 8, 6)}


def row_for(payload):
    metrics = payload["metrics"]
    primary = metrics.get("logit_fidelity", {})
    diverse = metrics.get("logit_fidelity_suites", {}).get("diverse", {})
    trajectory = metrics.get("trajectory_suites", {}).get("diverse", {})
    return {
        "arm": payload["arm"], "seed": payload["seed"],
        "mean_teacher_kl": primary.get("mean_teacher_kl"),
        "p95_teacher_kl": primary.get("p95_teacher_kl"),
        "top1_agreement": primary.get("top1_agreement"),
        "ppl_wikitext2": metrics.get("ppl_wikitext2"), "ppl_c4": metrics.get("ppl_c4"),
        "diverse_mean_teacher_kl": diverse.get("mean_teacher_kl"),
        "diverse_top1_agreement": diverse.get("top1_agreement"),
        "diverse_trajectory_token_agreement": trajectory.get("token_agreement"),
        "projected_artifact_bytes": metrics.get("byte_ledger", {}).get("projected_artifact_bytes"),
        "measured_artifact_bytes": None,
        "evaluation_halted": metrics.get("evaluation_halted", False),
    }


def paired_contrast(payloads, terms, *, metric="mean_teacher_kl", draws=4000):
    """Prompt-cluster interval; terms allow a paired difference-in-differences."""
    rows = [payloads[arm]["metrics"]["logit_fidelity"]["prompt_metrics"] for arm in terms]
    hashes = [[item["input_hash"] for item in arm] for arm in rows]
    if any(value != hashes[0] for value in hashes[1:]) or len(set(hashes[0])) != len(hashes[0]):
        raise ValueError("paired contrast requires unique identical prompt identities")
    weights = np.array([item["tokens"] for item in rows[0]], dtype=float)
    if any([item["tokens"] for item in arm] != weights.tolist() for arm in rows[1:]):
        raise ValueError("paired contrast token denominators differ")
    values = sum(coefficient * np.array([item[metric] for item in arm])
                 for coefficient, arm in zip(terms.values(), rows))
    if not len(values) or not np.isfinite(values).all() or (weights <= 0).any():
        raise ValueError("paired contrast requires finite nonempty measurements")
    rng = np.random.default_rng(20260906)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    boot = (values[indices] * weights[indices]).sum(1) / weights[indices].sum(1)
    return {"metric": metric, "terms": terms, "paired_prompts": len(values),
            "delta": float(np.average(values, weights=weights)),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "draws": draws, "unit": "paired prompt", "exploratory_screen": True}


def assess(results, *, seeds=(0,)):
    rows, issues, lookup = [], [], {}
    for payload in results:
        key = (payload["seed"], payload["arm"])
        if key in lookup:
            raise ValueError(f"duplicate arm/seed: {key}")
        if key[0] not in seeds or key[1] not in EXPECTED_ARMS:
            raise ValueError(f"unexpected arm/seed: {key}")
        lookup[key] = payload
        rows.append(row_for(payload))
    expected = {(seed, arm) for seed in seeds for arm in EXPECTED_ARMS}
    missing = sorted(expected - lookup.keys())
    decisions, contrasts = [], []
    for seed in seeds:
        base = lookup.get((seed, "b4_v16"))
        if base is None:
            continue
        # Config/runtime identities and actual prompt identities must agree, not
        # just suite labels. Different reference vocabularies invalidate pairing.
        for (entry_seed, arm), payload in lookup.items():
            if entry_seed != seed:
                continue
            if payload["identity"].get("source") != base["identity"].get("source"):
                issues.append(f"{seed}/{arm}: implementation mismatch")
            for key in ("config", "runtime", "prompt_files"):
                if payload["identity"].get(key) != base["identity"].get(key):
                    issues.append(f"{seed}/{arm}: {key} mismatch")
            if (payload["metrics"].get("teacher_source_vocab_digest")
                    != base["metrics"].get("teacher_source_vocab_digest")
                    or payload["metrics"].get("teacher_captured_before_mutation") is not True):
                issues.append(f"{seed}/{arm}: teacher identity mismatch")
            for path in (("logit_fidelity",), ("logit_fidelity_suites", "diverse"),
                         ("trajectory_suites", "diverse")):
                actual, reference = payload["metrics"], base["metrics"]
                for name in path:
                    actual, reference = actual.get(name, {}), reference.get(name, {})
                if (not payload["metrics"].get("evaluation_halted", False)
                        and (not actual.get("input_hashes")
                             or actual.get("input_hashes") != reference.get("input_hashes"))):
                    issues.append(f"{seed}/{arm}: {'/'.join(path)} prompt mismatch")
        for arm in ("b5_v8", "b5_v6"):
            candidate = lookup.get((seed, arm))
            if candidate is None:
                continue
            row, baseline = row_for(candidate), row_for(base)
            guards = quality_guards(row, baseline)
            kl, control = row["mean_teacher_kl"], baseline["mean_teacher_kl"]
            selected = (all(guards.values()) and not row["evaluation_halted"]
                        and finite_number(kl) and finite_number(control) and kl < control)
            decisions.append({"seed": seed, "arm": arm, "quality_candidate": selected,
                              "guards": guards, "primary_kl_improved": (
                                  finite_number(kl) and finite_number(control) and kl < control),
                              "artifact_promoted": False})
        if all((seed, arm) in lookup for arm in EXPECTED_ARMS):
            by_arm = {arm: lookup[seed, arm] for arm in EXPECTED_ARMS}
            for vocabulary in (8, 6):
                for label, terms in (
                    ("vocabulary_only", {f"b16_v{vocabulary}": 1, "b16_v16": -1}),
                    ("combined_vs_w4", {f"b5_v{vocabulary}": 1, "b4_v16": -1}),
                    ("interaction", {f"b5_v{vocabulary}": 1, "b5_v16": -1,
                                     f"b4_v{vocabulary}": -1, "b4_v16": 1}),
                ):
                    try:
                        contrast = paired_contrast(by_arm, terms)
                        contrasts.append({"seed": seed, "vocabulary_bits": vocabulary,
                                          "contrast": label, **contrast})
                    except (KeyError, ValueError) as exc:
                        issues.append(f"{seed}/{label}/v{vocabulary}: {exc}")
    complete = not missing and not issues
    finalists = [arm for arm in ("b5_v8", "b5_v6") if complete and all(
        any(d["seed"] == seed and d["arm"] == arm and d["quality_candidate"]
            for d in decisions) for seed in seeds)]
    return {"protocol": PROTOCOL, "seeds": list(seeds), "complete": complete,
            "missing": missing, "issues": issues, "rows": sorted(rows, key=lambda r: (r["seed"], r["arm"])),
            "quality_guards": QUALITY_GUARDS, "decisions": decisions,
            "contrasts": contrasts, "finalists": finalists, "provider_competitive": False,
            "artifact_promoted": False,
            "next_step": ("Implement/verify packed export, then independent confirmation."
                          if finalists else "Do not launch allocation/recovery; inspect screen diagnostics."),
            "interpretation": "Development screen only. Dense vocabulary and projected bytes cannot promote an artifact."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    from scripts.run_qwen35_vocabulary_budget import load_completed
    results = []
    for path in sorted(args.input_dir.glob("b*_v*_s*.json")):
        raw = json.loads(path.read_text())
        verified = load_completed(path, raw["identity"])
        if verified is not None:
            results.append(verified)
    summary = assess(results, seeds=tuple(args.seed or [0]))
    write_result(str(args.output), summary)
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise SystemExit("Screen incomplete or inconsistent; no candidate may advance.")


if __name__ == "__main__":
    main()
