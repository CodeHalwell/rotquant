#!/usr/bin/env python3
"""Render or execute the pinned paper benchmark matrix.

Dry-run is the default. Pass ``--execute`` only on the intended GPU host after
reviewing ``paper/generated/publication_commands.json``. Commands use argument
arrays rather than a shell, and every Hub model is pinned to an immutable
revision in ``paper/benchmark_matrix.yaml``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _yaml_value(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("...").strip()


def _protocol_overrides(protocol: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "eval.ppl.seq_len": int(protocol["ppl_seq_len"]),
        "eval.ppl.max_samples": protocol.get("ppl_max_samples"),
        "eval.zeroshot": bool(protocol["zeroshot"]),
    }
    return overrides


def build_commands(
    matrix: dict[str, Any],
    protocol_name: str,
    results_dir: Path,
    model_filter: set[str] | None = None,
    method_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    if matrix.get("schema_version") != 1:
        raise ValueError("unsupported publication benchmark schema")
    protocol = matrix["protocols"][protocol_name]
    records: list[dict[str, Any]] = []
    for model_name, model in matrix["models"].items():
        if model_filter and model_name not in model_filter:
            continue
        for method_name in model["methods"]:
            if method_filter and method_name not in method_filter:
                continue
            method = matrix["methods"][method_name]
            for seed in method.get("seeds", [0]):
                if method["runner"] == "rotquant":
                    command = [
                        "uv",
                        "run",
                        "python",
                        "scripts/run_experiment.py",
                        model["config"],
                        "--output-dir",
                        str(results_dir),
                        "--model",
                        model["model_id"],
                        "--device",
                        model["device"],
                        "--seed",
                        str(seed),
                        "--set",
                        f"model_revision={model['revision']}",
                    ]
                    overrides = dict(method.get("overrides", {}))
                    overrides.update(_protocol_overrides(protocol))
                    for key, value in overrides.items():
                        command.extend(["--set", f"{key}={_yaml_value(value)}"])
                elif method["runner"] == "baseline":
                    command = [
                        "uv",
                        "run",
                        "python",
                        "baselines/run_baseline.py",
                        "--backend",
                        method["backend"],
                        "--model",
                        model["model_id"],
                        "--revision",
                        model["revision"],
                        "--bits",
                        str(method["bits"]),
                        "--group-size",
                        str(method["group_size"]),
                        "--device",
                        model["device"],
                        "--output-dir",
                        str(results_dir),
                        "--ppl-seq-len",
                        str(protocol["ppl_seq_len"]),
                    ]
                    if protocol.get("ppl_max_samples") is not None:
                        command.extend(["--ppl-max-samples", str(protocol["ppl_max_samples"])])
                    if protocol["zeroshot"]:
                        command.append("--zeroshot")
                        if protocol.get("zeroshot_limit") is not None:
                            command.extend(["--zeroshot-limit", str(protocol["zeroshot_limit"])])
                else:
                    raise ValueError(f"unknown runner {method['runner']!r}")
                records.append(
                    {
                        "model": model_name,
                        "model_id": model["model_id"],
                        "revision": model["revision"],
                        "method": method_name,
                        "seed": seed,
                        "protocol": protocol_name,
                        "command": command,
                    }
                )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("paper/benchmark_matrix.yaml"))
    parser.add_argument("--protocol", choices=("smoke", "publication"), default="smoke")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--results-dir", type=Path, default=Path("results/publication"))
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("paper/generated/publication_commands.json"),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = yaml.safe_load(args.matrix.read_text())
    records = build_commands(
        matrix,
        args.protocol,
        args.results_dir,
        model_filter=set(args.models) if args.models else None,
        method_filter=set(args.methods) if args.methods else None,
    )
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(
            f"[{record['protocol']}] {record['model']} / {record['method']} / "
            f"seed {record['seed']}: {' '.join(record['command'])}"
        )
        if args.execute:
            subprocess.run(record["command"], check=True)
    print(f"wrote {len(records)} commands to {args.plan_output}")


if __name__ == "__main__":
    main()
