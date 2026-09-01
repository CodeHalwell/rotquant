from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_publication_suite.py"
SPEC = importlib.util.spec_from_file_location("run_publication_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def test_publication_matrix_is_pinned_and_complete() -> None:
    matrix = yaml.safe_load((ROOT / "paper" / "benchmark_matrix.yaml").read_text())
    commands = suite.build_commands(matrix, "publication", Path("results/test"))
    assert len(commands) == 16
    assert all(len(record["revision"]) == 40 for record in commands)
    qwen35_methods = {record["method"] for record in commands if record["model"] == "qwen35_4b"}
    assert qwen35_methods == {"source", "rotquant_w4"}
    assert all("--execute" not in record["command"] for record in commands)


def test_smoke_protocol_is_bounded() -> None:
    matrix = yaml.safe_load((ROOT / "paper" / "benchmark_matrix.yaml").read_text())
    commands = suite.build_commands(
        matrix,
        "smoke",
        Path("results/test"),
        model_filter={"qwen3_4b"},
        method_filter={"source", "rotquant_w4", "gptq_w4"},
    )
    assert len(commands) == 5
    rendered = [" ".join(record["command"]) for record in commands]
    assert all("256" in command for command in rendered)
    assert all("--zeroshot" not in record["command"] for record in commands)
