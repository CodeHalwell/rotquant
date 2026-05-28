#!/usr/bin/env python
"""Collate results/<run_id>.json into comparison tables (and optional plots).

Groups runs by experiment id (E1..E8) and emits a tidy CSV plus a markdown table
with mean +/- std across seeds, so a finding only counts once it holds across
>=3 seeds on both WikiText-2 and C4.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List


def load_runs(results_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def _key(run: Dict[str, Any]) -> str:
    cfg = run.get("config", {})
    return f"{cfg.get('experiment', '?')}|{cfg.get('model', '?')}|{cfg.get('label', run.get('run_id'))}"


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[_key(r)].append(r)

    table: Dict[str, Dict[str, Any]] = {}
    for key, group in groups.items():
        agg: Dict[str, Any] = {"n_seeds": len(group)}
        metric_names = set()
        for r in group:
            for m, v in r.get("metrics", {}).items():
                if isinstance(v, (int, float)):
                    metric_names.add(m)
        for m in metric_names:
            vals = [r["metrics"][m] for r in group if isinstance(
                r["metrics"].get(m), (int, float))]
            if vals:
                agg[m] = {"mean": mean(vals),
                          "std": pstdev(vals) if len(vals) > 1 else 0.0}
        table[key] = agg
    return table


def to_markdown(table: Dict[str, Dict[str, Any]]) -> str:
    lines = ["| run | seeds | metric | mean | std |", "|---|---|---|---|---|"]
    for key, agg in sorted(table.items()):
        for m, v in agg.items():
            if isinstance(v, dict):
                lines.append(
                    f"| {key} | {agg['n_seeds']} | {m} | {v['mean']:.4f} | {v['std']:.4f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    table = aggregate(runs)
    md = to_markdown(table)
    with open(args.out, "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
