"""Offline checks for the versioned ternary notebook and measured-result display."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from scripts import build_qwen35_ternary_research_notebook as ternary
from scripts import build_qwen35_w4_research_notebook as w4


def _report():
    return {"complete": False, "selection": {"chosen": "ternary_blockqat"},
            "scope": "Synthetic fixture for presentation tests; not model measurements.",
            "rows": [
                {"label": "fp16", "status": "measured", "backend": "native_cuda_reference",
                 "precision_scope": "full precision", "checkpoint_bytes": 8_100_000_000,
                 "native_payload_bytes": 8_200_000_000, "language_payload_bytes": 7_500_000_000,
                 "quality": {"test": {"summary": {"mean_teacher_kl": 0, "ppl": 7.5,
                    "top1_agreement": 1, "documents": 24, "tokens": 12000,
                    "confidence_intervals": {"mean_teacher_kl": [0, 0]}, "ppl_interval": [7.2, 7.8]}}},
                 "tasks": {"summary": {"math": {"items": 16, "correct": 8, "accuracy": 0.5, "truncated": 1}}},
                 "speed": {"completed": True, "summary": {"128": {
                    "prefill_tokens_per_second": 120, "decode_tokens_per_second": 24,
                    "measured_repetitions": 3}}}},
                {"label": "ternary_blockqat", "status": "measured", "backend": "native_cuda_reference",
                 "precision_scope": "mlp", "native_payload_bytes": 3_150_000_000,
                 "quality": {"test": {"summary": {"mean_teacher_kl": 0.025, "ppl": 8.2}}},
                 "tasks": {"summary": {"math": {"items": 16, "correct": 0, "accuracy": 0.0, "truncated": 3}}},
                 "speed": {"completed": False, "summary": {"128": {
                    "prefill_tokens_per_second": 130, "decode_tokens_per_second": None,
                    "measured_repetitions": 3}}}},
                {"label": "bonsai2_4b", "status": "failed", "backend": "official_bonsai_cuda",
                 "error": "fixture incomplete quality", "native_payload_bytes": 1_500_000_000,
                 "language_payload_bytes": 950_000_000,
                 "quality": {"test": {"summary": {"mean_teacher_kl": float("nan")}}},
                 "speed": {"completed": False, "summary": {"128": {
                    "prefill_tokens_per_second": None, "decode_tokens_per_second": 31,
                    "measured_repetitions": 3}}}},
                {"label": "ternary_recovered", "status": "pending", "backend": "native_cuda_reference"},
            ], "paired_tasks": {"ternary_blockqat_vs_fp16": {"math": {
                "items": 16, "mean_accuracy_delta": -0.5, "bootstrap_interval": [-0.75, -0.25],
                "hoeffding_interval": [-1.0, 0.18], "candidate_wins": 0, "candidate_losses": 8}}}}


def test_artifact_matches_generator_and_code_compiles_without_saved_claims():
    notebook = nbformat.read(ternary.OUTPUT, as_version=4)
    nbformat.validate(notebook)
    assert notebook == ternary.build_notebook()
    assert notebook.metadata.rotquant.study == "ternary-v1"
    for cell in notebook.cells:
        assert cell.id.startswith("ternary-v1-")
        if cell.cell_type == "code":
            compile(cell.source, f"{ternary.OUTPUT}:{cell.id}", "exec")
            assert cell.execution_count is None and cell.outputs == []
    assignment = ast.parse(next(cell.source for cell in notebook.cells
                                if cell.source.startswith("PRESENTATION_SCRIPT ="))).body[0]
    compile(ast.literal_eval(assignment.value), "presentation-script", "exec")


def test_working_w4_setup_and_runtime_are_preserved_in_separate_notebook():
    before = w4.build_notebook()
    notebook = ternary.build_notebook()
    assert w4.build_notebook() == before
    assert ternary.OUTPUT != w4.OUTPUT
    assert notebook.cells[4].source == before.cells[4].source  # ZIP checksum/extraction helpers
    assert notebook.cells[7].source == before.cells[7].source  # Isolated pip, pins, torch checks, CUDA smoke
    source = "\n".join(cell.source for cell in notebook.cells)
    for required in ('SOURCE_MODE = "upload"', 'SEEDS = [0]', 'PUBLIC_SAMPLES = 16',
                     'BUDGET_MINUTES = 480', 'REQUIRE_FAST_HADAMARD = True',
                     'r"[0-9a-fA-F]{40}"', '"--system-site-packages"',
                     '"--expected-torch-file", TORCH_FILE', '"rotquant/ternary_train.py"',
                     'configs/qwen35_4b_ternary_research_cuda.yaml', 'STUDY_VERSION = "ternary-v1"'):
        assert required in source
    assert '[sys.executable, "-m", "pip", "--isolated", "--python", str(PYTHON)' in source
    assert '"--require-virtualenv", "--disable-pip-version-check", "install"' in source
    assert '"--constraint", str(constraints)' in source
    assert '[uv, "pip"' not in source
    assert "scripts/run_qwen35_w4_research.py" not in source
    assert "rotquant/w4_research" not in source
    assert all(label in source for label in ternary.LABELS)


def test_plan_uses_new_driver_and_budget_source_config_identity(tmp_path):
    calls = []
    notebook = ternary.build_notebook()
    namespace = {}
    exec(notebook.cells[2].source, namespace)  # noqa: S102 - authored settings only
    namespace.update(SEEDS=[0, 1], SOURCE_ID="a" * 64, REPO_DIR=tmp_path / "repo",
                     RESULT_BASE=tmp_path / "results", LOCAL_BASE=tmp_path / "local",
                     PYTHON=tmp_path / "venv/python", subprocess=SimpleNamespace(
                         check_output=lambda *args, **kwargs: json.dumps({
                             "seed": 0, "research": {}, "ternary": {"quant": {"seed": 0}}})),
                     _run_live=lambda command, label, **kwargs: calls.append(command))
    source = next(cell.source for cell in notebook.cells if cell.source.startswith("config_reader ="))
    exec(source, namespace)  # noqa: S102 - external calls mocked
    original_roots = {run["root"] for run in namespace["RUNS"]}
    for command, seed in zip(calls, [0, 1]):
        assert command[-1] == "--dry-run"
        assert Path(command[2]).name == "run_qwen35_ternary_research.py"
        assert command[command.index("--budget-minutes") + 1] == "240"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert config["seed"] == seed
        assert config["ternary"]["quant"]["seed"] == seed
        assert config["research"]["public_samples_per_benchmark"] == 16
    identity = json.loads((next(iter(original_roots)) / "notebook-identity.json").read_text())
    assert identity["study"] == "ternary-v1"
    namespace["PUBLIC_SAMPLES"] = 32
    exec(source, namespace)  # noqa: S102 - changed settings must not reuse old outputs
    assert original_roots.isdisjoint(run["root"] for run in namespace["RUNS"])


def test_partial_receipts_keep_unknown_distinct_from_zero_and_keep_scope():
    tables = ternary.comparison_tables(_report())
    assert tables["models"][0]["Native GB"] == 8.2
    assert tables["models"][0]["Language GB"] == 7.5
    assert tables["models"][1]["Precision scope"] == "mlp"
    assert tables["models"][2]["Teacher KL"] is None
    assert tables["models"][2]["Status"] == "failed"
    assert tables["models"][2]["Error"] == "fixture incomplete quality"
    assert tables["models"][3]["Native GB"] is None
    assert tables["tasks"][0]["Accuracy (%)"] == 50.0
    assert tables["tasks"][1]["Accuracy (%)"] == 0.0
    assert tables["speed"][1]["Decode tokens/s"] is None
    assert tables["paired"][0]["Accuracy delta (fraction)"] == -0.5


def test_plots_use_complete_bytes_and_separate_speed_backends():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    tables = ternary.comparison_tables(_report())
    quality = ternary.quality_figure(tables["models"])
    points = [tuple(collection.get_offsets()[0]) for collection in quality.axes[0].collections
              if collection.__class__.__name__ == "PathCollection"]
    assert points == [(8.2, 0.0), (3.15, 0.025)]
    task = ternary.task_figure(tables["models"], tables["tasks"])
    assert [tuple(collection.get_offsets()[0]) for collection in task.axes[0].collections] == [(8.2, 50.0), (3.15, 0.0)]
    speed = ternary.speed_figure(tables["speed"])
    assert len(speed.axes) == 4
    assert [line.get_ydata().tolist() for line in speed.axes[1].lines] == [[24.0]]
    assert [line.get_ydata().tolist() for line in speed.axes[3].lines] == [[31.0]]
    for figure in (quality, task, speed):
        plt.close(figure)


def test_embedded_presentation_executes_and_matches_local_result(tmp_path):
    pytest.importorskip("matplotlib")
    notebook = ternary.build_notebook()
    assignment = ast.parse(next(cell.source for cell in notebook.cells
                                if cell.source.startswith("PRESENTATION_SCRIPT ="))).body[0]
    namespace = {"__name__": "presentation_test"}
    exec(ast.literal_eval(assignment.value), namespace)  # noqa: S102 - authored renderer, offline fixture
    result = namespace["render_comparison"](_report(), tmp_path)
    assert result["tables"] == ternary.comparison_tables(_report())
    assert result["complete"] is False
    assert len(result["figures"]) == 3
    assert all(Path(path).stat().st_size > 1000 for path in result["figures"])
    assert json.loads((tmp_path / "readout.json").read_text())["scope"].startswith("Synthetic")


@pytest.mark.parametrize("report", [{}, {"rows": [{"label": "bonsai2_4b", "status": "pending"}]}])
def test_unmeasured_report_has_no_fabricated_plot(report, tmp_path):
    pytest.importorskip("matplotlib")
    result = ternary.render_comparison(report, tmp_path)
    assert result["complete"] is False
    assert result["figures"] == []
    assert result["tables"]["tasks"] == []
    assert result["tables"]["speed"] == []
