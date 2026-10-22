"""Offline checks for the W4 notebook source, resume contract and presentation."""
from __future__ import annotations

import ast
import hashlib
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from scripts.build_qwen35_w4_research_notebook import (
    OUTPUT,
    build_notebook,
    comparison_tables,
    quality_figure,
    render_comparison,
    speed_figure,
    unpack_source_bundle,
)


def _bundle(path, files=None, extras=None):
    files = files or {"pyproject.toml": b"[project]\nname='fixture'", "scripts/runner.py": b"print('fixture')"}
    manifest = {"protocol": "rotquant-source-bundle-v1", "revision": "a" * 40,
                "files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("SOURCE_BUNDLE.json", json.dumps(manifest))
        for name, content in (extras or {}).items():
            archive.writestr(name, content)
    return path


def _report():
    return {
        "complete": False, "selection": {"chosen": "w4_g64_h"}, "rows": [
            {"label": "fp16", "checkpoint_bytes": 8_100_000_000, "native_payload_bytes": 8_200_000_000,
             "quality": {"test": {"summary": {"mean_teacher_kl": 0, "ppl": 7.5, "top1_agreement": 1,
                         "documents": 24, "tokens": 12000,
                         "confidence_intervals": {"mean_teacher_kl": [0, 0]}, "ppl_interval": [7.2, 7.8]}}},
             "tasks": {"summary": {"math": {"items": 16, "correct": 8, "accuracy": 0.5, "truncated": 1}}},
             "speed": {"completed": True, "summary": {"128": {"prefill_tokens_per_second": 120,
                 "decode_tokens_per_second": 24, "measured_repetitions": 3}}}},
            {"label": "w4_g64_h", "checkpoint_bytes": 3_050_000_000, "native_payload_bytes": 3_150_000_000,
             "quality": {"test": {"summary": {"mean_teacher_kl": 0.025, "ppl": 8.2,
                         "confidence_intervals": {"mean_teacher_kl": [0.015, 0.035]}, "ppl_interval": [7.8, 8.7]}}},
             "tasks": {"summary": {"math": {"items": 16, "correct": 0, "accuracy": 0.0, "truncated": 3}}},
             "speed": {"completed": False, "summary": {"128": {"prefill_tokens_per_second": 130,
                 "decode_tokens_per_second": None, "measured_repetitions": 3}}}},
            {"label": "not_evaluated", "checkpoint_bytes": None, "native_payload_bytes": None,
             "quality": {"test": {"summary": {"mean_teacher_kl": float("nan")}}}, "tasks": None, "speed": None},
        ], "paired_tasks": {"w4_g64_h_vs_fp16": {"math": {"items": 16, "mean_accuracy_delta": -0.5,
              "bootstrap_interval": [-0.75, -0.25], "hoeffding_interval": [-1.0, 0.18],
              "candidate_wins": 0, "candidate_losses": 8}}},
    }


def test_generated_notebook_matches_artifact_and_every_code_cell_parses():
    expected = build_notebook()
    actual = nbformat.read(OUTPUT, as_version=4)
    nbformat.validate(actual)
    assert actual == expected
    for cell in actual.cells:
        assert cell.id
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{OUTPUT}:{cell.id}")
            assert cell.execution_count is None and cell.outputs == []
    assignment = ast.parse(next(cell.source for cell in actual.cells
                                if cell.source.startswith("PRESENTATION_SCRIPT ="))).body[0]
    ast.parse(ast.literal_eval(assignment.value))


def test_notebook_defaults_use_upload_isolated_pip_and_existing_torch_checks():
    source = "\n".join(cell.source for cell in build_notebook().cells)
    assert 'SOURCE_MODE = "upload"' in source
    assert 'SEEDS = [0]' in source and 'PUBLIC_SAMPLES = 16' in source
    assert 'BUDGET_MINUTES = 480' in source and 'REQUIRE_FAST_HADAMARD = True' in source
    assert 'r"[0-9a-fA-F]{40}"' in source
    assert '"--system-site-packages"' in source
    assert '"--expected-torch-file", TORCH_FILE' in source
    assert '"torch-before.json"' in source and '"torch-final.json"' in source
    assert '[sys.executable, "-m", "pip", "--isolated", "--python", str(PYTHON)' in source
    assert '"--require-virtualenv", "--disable-pip-version-check", "install"' in source
    assert '"--constraint", str(constraints)' in source
    assert '[uv, "pip"' not in source
    assert 'https://astral.sh/uv/{UV_VERSION}/install.sh' in source
    assert 'fast-hadamard-transform.git@v1.1.0.post2' in source


def test_source_upload_verifies_files_and_reuses_unchanged_extraction(tmp_path):
    archive = _bundle(tmp_path / "source.zip")
    repository, identity = unpack_source_bundle(archive, tmp_path / "cache")
    assert len(identity) == 64
    assert unpack_source_bundle(archive, tmp_path / "cache") == (repository, identity)
    (repository / "scripts/__pycache__").mkdir()
    (repository / "scripts/__pycache__/runner.pyc").write_bytes(b"cached")
    assert unpack_source_bundle(archive, tmp_path / "cache") == (repository, identity)
    (repository / "scripts/runner.py").write_text("modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        unpack_source_bundle(archive, tmp_path / "cache")
    assert (repository / "scripts/runner.py").read_text() == "modified"


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py", "folder/../../outside.py", "folder\\outside.py", ".git/config"])
def test_zip_paths_cannot_escape_source_root(tmp_path, name):
    archive = _bundle(tmp_path / "source.zip", extras={name: b"bad"})
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_source_bundle(archive, tmp_path / "cache")
    assert not (tmp_path / "outside.py").exists()


def test_zip_rejects_symlinks_and_unlisted_files(tmp_path):
    archive = _bundle(tmp_path / "source.zip")
    with zipfile.ZipFile(archive, "a") as handle:
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(link, "../../outside")
    with pytest.raises(ValueError, match="non-regular"):
        unpack_source_bundle(archive, tmp_path / "cache")
    archive = _bundle(tmp_path / "other.zip", extras={"unexpected.py": b"pass"})
    with pytest.raises(ValueError, match="unlisted"):
        unpack_source_bundle(archive, tmp_path / "cache")


def test_cached_source_rejects_symlink_substitution(tmp_path):
    archive = _bundle(tmp_path / "source.zip")
    repository, _ = unpack_source_bundle(archive, tmp_path / "cache")
    outside = tmp_path / "outside.py"
    outside.write_bytes((repository / "scripts/runner.py").read_bytes())
    (repository / "scripts/runner.py").unlink()
    (repository / "scripts/runner.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        unpack_source_bundle(archive, tmp_path / "cache")


def test_seed_plan_divides_budget_and_changed_settings_get_new_roots(tmp_path):
    calls = []
    notebook = build_notebook()
    namespace = {}
    exec(notebook.cells[2].source, namespace)  # noqa: S102 - authored notebook settings only
    namespace.update(SEEDS=[0, 1, 2], PUBLIC_SAMPLES=16, SOURCE_ID="a" * 64,
                     REPO_DIR=tmp_path / "repo", RESULT_BASE=tmp_path / "results",
                     LOCAL_BASE=tmp_path / "local", PYTHON=tmp_path / "venv/python",
                     subprocess=SimpleNamespace(check_output=lambda *args, **kwargs: json.dumps({"seed": 0, "research": {}})),
                     _run_live=lambda command, label, **kwargs: calls.append(command))
    source = next(cell.source for cell in notebook.cells if cell.source.startswith("config_reader ="))
    exec(source, namespace)  # noqa: S102 - authored plan cell with subprocess calls mocked
    original_roots = [run["root"] for run in namespace["RUNS"]]
    assert len(calls) == 3
    for command, seed in zip(calls, [0, 1, 2]):
        assert command[-1] == "--dry-run"
        assert command[command.index("--budget-minutes") + 1] == "160"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert config["seed"] == seed
        assert config["research"]["public_samples_per_benchmark"] == 16
    namespace["PUBLIC_SAMPLES"] = 32
    exec(source, namespace)  # noqa: S102 - repeat the same mocked cell to test resume identity
    assert set(original_roots).isdisjoint(run["root"] for run in namespace["RUNS"])
    assert all(root.exists() for root in original_roots)


def test_partial_results_preserve_missing_values_and_real_zero_scores():
    tables = comparison_tables(_report())
    assert tables["models"][0]["Checkpoint GB"] == 8.1
    assert tables["models"][0]["Native GB"] == 8.2
    assert tables["models"][0]["Teacher KL"] == 0.0
    assert tables["models"][2]["Native GB"] is None
    assert tables["models"][2]["Teacher KL"] is None
    assert tables["tasks"][1]["Accuracy"] == 0.0
    assert tables["speed"][1]["Decode tokens/s"] is None
    assert tables["speed"][1]["All contexts complete"] is False
    assert tables["paired"][0]["Accuracy delta"] == -0.5
    assert tables["paired"][0]["Hoeffding 95% interval"] == [-1.0, 0.18]


def test_mock_result_figures_bind_actual_bytes_and_do_not_plot_missing_speed(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    tables = comparison_tables(_report())
    quality = quality_figure(tables["models"])
    points = [tuple(collection.get_offsets()[0]) for collection in quality.axes[0].collections
              if len(collection.get_offsets()) and collection.__class__.__name__ == "PathCollection"]
    assert points == [(8.2, 0.0), (3.15, 0.025)]
    speed = speed_figure(tables["speed"])
    assert len(speed.axes[0].lines) == 2
    assert len(speed.axes[1].lines) == 1
    assert speed.axes[1].lines[0].get_ydata().tolist() == [24.0]
    plt.close(quality)
    plt.close(speed)
    rendered = render_comparison(_report(), tmp_path / "figures")
    assert rendered["complete"] is False
    assert all(Path(path).stat().st_size > 1000 for path in rendered["figures"])
    assert json.loads((tmp_path / "figures/readout.json").read_text())["tables"] == tables


def test_empty_results_render_without_fabricated_measurements(tmp_path):
    pytest.importorskip("matplotlib")
    result = render_comparison({}, tmp_path)
    assert result["complete"] is False
    assert result["figures"] == []
    assert result["tables"] == {"models": [], "tasks": [], "speed": [], "paired": []}
