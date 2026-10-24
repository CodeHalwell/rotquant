"""Build the parity-first ternary follow-up without changing the completed v1 study."""
from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_qwen35_ternary_research_notebook as v1
from scripts import build_qwen35_w4_research_notebook as w4
from scripts.build_qwen35_packed_validation_notebook import code, md

OUTPUT = ROOT / "notebooks/qwen35_4b_ternary_research_v2_colab.ipynb"
HISTORICAL_ROOT = "/content/drive/MyDrive/rotquant/ternary_research_v1/f1561b8f83766a73-s0"
legacy_comparison_tables = v1.comparison_tables


def row_identity(row, phase):
    model = row.get("model", row["label"])
    profile = row.get("profile", "prism" if model == "bonsai2_4b" else "reference")
    if isinstance(profile, dict):
        profile = json.dumps(profile, sort_keys=True)
    return f"{model} / {profile} / {row.get('phase', phase)}"


def comparison_group(rows, phase):
    result = []
    for row in rows:
        item = copy.deepcopy(row)
        item["label"] = row_identity(row, phase)
        result.append(item)
    labels = [row["label"] for row in result]
    if len(labels) != len(set(labels)):
        raise ValueError("comparison contains duplicate model/profile/phase identities")
    return result


def comparison_tables(report):
    tables = legacy_comparison_tables(report)
    for displayed, source in zip(tables["models"], report.get("rows", [])):
        behaviour = source.get("behaviour") or {}
        metrics = source.get("training_metrics") or {}
        global_metrics = metrics.get("ternary_global", {})
        displayed.update({"Completion rate": behaviour.get("completion_rate"),
                          "Repetition rate": behaviour.get("repetition_rate"),
                          "Format rate": behaviour.get("format_rate"),
                          "Behaviour examples": behaviour.get("examples"),
                          "Behaviour data ID": behaviour.get("data_id"),
                          "Training accepted": global_metrics.get("accepted"),
                          "Best training step": global_metrics.get("best_step")})
    return tables


def render_v2_results(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for phase, key in (("historical", "historical_rows"), ("fresh", "rows")):
        group = {**report, "rows": comparison_group(report.get(key, []), phase)}
        group["paired_tasks"] = report.get("paired_tasks", {}) if phase == "fresh" else {}
        group["complete"] = report.get("complete") is True if phase == "fresh" else (
            bool(group["rows"]) and all(x.get("status") == "measured" for x in group["rows"]))
        groups[phase] = render_group(group, output / phase)
    result = {"complete": report.get("complete") is True,
              "parity_complete": report.get("parity_complete") is True,
              "stage": report.get("stage"), "groups": groups,
              "gates": report.get("gates", {}), "historical_import": report.get("historical_import"),
              "error": report.get("error")}
    (output / "readout.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def render_group(report, output):
    import types
    renderer_globals = {**v1.render_comparison.__globals__, "comparison_tables": comparison_tables}
    renderer = types.FunctionType(v1.render_comparison.__code__, renderer_globals)
    return renderer(report, output)


def port_cells(cells):
    replacements = {
        "scripts/run_qwen35_w4_research.py": "scripts/run_qwen35_ternary_research_v2.py",
        "configs/qwen35_4b_w4_research_cuda.yaml": "configs/qwen35_4b_ternary_research_v2_cuda.yaml",
        "rotquant/w4_research": "rotquant/ternary_research_v2",
        "rotquant-w4-research": "rotquant-ternary-research-v2",
        "Source is missing W4 research support": "Source is missing ternary v2 support",
        '"rotquant/activation_refit.py",': ('"rotquant/teacher_targets.py", '
            '"scripts/ternary_chat_data.py", "scripts/ternary_behaviour.py",'),
        "w4-reports-": "ternary-v2-reports-",
    }
    for cell in cells:
        for old, new in replacements.items():
            cell.source = cell.source.replace(old, new)
    return cells


def setup_cells():
    cells = port_cells(w4.setup_cells())
    cells[0] = md('''
        # RotQuant ternary follow-up · version 2

        Preserve the completed nine-arm comparison and investigate cached chat
        decoding before training. Compare complete-turn scale recovery and global
        assignment-changing ternary QAT with **FP16, W4, W5/W6, W6 and Bonsai 2 4B**.
        Measure complete payload bytes, quality, termination, repetition, and speed.

        Use an **A100 80 GB GPU** and high host RAM. Upload the new source ZIP.
        The default `STAGE = "parity"` runs bounded correctness diagnostics only.
        Review the saved gates before deliberately setting `STAGE = "full"`.
        This notebook contains no newly measured results.
    ''')
    cells[1] = md('''
        ## Settings
        Keep the source ZIP, historical root, config and seed unchanged to resume.
        The historical v1 root is read-only input. New receipts, checkpoints and
        reports go to `rotquant/ternary_research_v2` in a separate hashed directory.
        Reconnecting does not renew the persisted execution budget.
    ''')
    cells[2].source = cells[2].source.replace('SEEDS = [0]',
        'STUDY_VERSION = "ternary-v2"\nSTAGE = "parity"  # Set "full" after inspecting the parity gates.\n'
        f'HISTORICAL_ROOT = Path({HISTORICAL_ROOT!r})\n'
        'EXPERIMENTAL_PROFILES = []  # Optional names from the config.\nSEEDS = [0]')
    cells[2].source = '\n'.join(line for line in cells[2].source.splitlines()
                                 if 'PUBLIC_SAMPLES' not in line)
    cells[2].source += '\nassert STAGE in {"parity", "full"}\nassert SEEDS == [0], "The historical seed-0 comparison is immutable."'
    cells[5].source = cells[5].source.replace('>= 35 * 2**30', '>= 70 * 2**30')
    cells[5].source = cells[5].source.replace('Choose an A100 40 GB-class GPU.',
                                            'Choose an A100 80 GB GPU for global QAT.')
    return cells


def experiment_cells():
    return [md('''
        ## Review the next experiment
        First validate cached, uncached and native chat decoding on frozen
        diagnostics. Failed gates stop the run. The full stage adds complete
        teacher turns, response-only union KL/CE targets, held-out behavioural
        selection and assignment-changing global QAT. Runtime profiles use frozen
        weights and token traces; numerical changes remain separate profiles.

        The default profiles are reference, decode4 and ternary-g128.
        `EXPERIMENTAL_PROFILES` can enable named W2A8 or fused Tensor Core
        research profiles; each must pass its own synthetic and cached gates.

        The original nine measured rows remain visible with their provenance.
        Fresh model/profile/phase identities are separate, so an old measurement
        cannot be presented as a new CUDA result. Public task labels are excluded
        from teacher fitting and checkpoint selection.
    '''), code('''
        assert HISTORICAL_ROOT.is_dir(), "Mount the existing completed v1 result folder."
        historical_path = HISTORICAL_ROOT / "comparison.json"
        historical_report = json.loads(historical_path.read_text())
        assert historical_report.get("complete") is True, "Historical comparison must be complete."
        historical_sha = file_sha256(historical_path)
        config_reader = "import json,yaml,sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"
        base_config = json.loads(subprocess.check_output([str(PYTHON), "-c", config_reader,
            str(REPO_DIR / "configs/qwen35_4b_ternary_research_v2_cuda.yaml")], text=True))
        optional = base_config["v2"].get("optional_inference_profiles", {})
        assert isinstance(EXPERIMENTAL_PROFILES, list) and all(isinstance(x, str) for x in EXPERIMENTAL_PROFILES)
        assert len(EXPERIMENTAL_PROFILES) == len(set(EXPERIMENTAL_PROFILES)), "Duplicate experimental profiles."
        assert set(EXPERIMENTAL_PROFILES) <= optional.keys(), "Select a named optional profile from the config."
        for name in EXPERIMENTAL_PROFILES:
            assert name not in base_config["v2"]["inference_profiles"], "Profile is already enabled."
            base_config["v2"]["inference_profiles"][name] = json.loads(json.dumps(optional[name]))
        RUNS = []
        minutes_per_seed = BUDGET_MINUTES // len(SEEDS)
        for seed in SEEDS:
            config = json.loads(json.dumps(base_config))
            config["seed"] = seed
            identity = {"study": STUDY_VERSION, "source": SOURCE_ID, "config": config,
                        "historical_root": str(HISTORICAL_ROOT), "historical_comparison_sha256": historical_sha,
                        "budget_minutes": minutes_per_seed, "fast_hadamard_required": REQUIRE_FAST_HADAMARD}
            run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            root = RESULT_BASE / f"{run_id}-s{seed}"
            assert HISTORICAL_ROOT.resolve() != root.resolve()
            assert HISTORICAL_ROOT.resolve() not in root.resolve().parents
            root.mkdir(parents=True, exist_ok=True)
            config_path = root / "config.yaml"
            text = json.dumps(config, indent=2) + "\\n"
            if config_path.exists():
                assert config_path.read_text() == text, "Stored v2 settings differ."
            else:
                config_path.write_text(text)
            notebook_identity = root / "notebook-identity.json"
            identity_text = json.dumps(identity, indent=2) + "\\n"
            if notebook_identity.exists():
                assert notebook_identity.read_text() == identity_text, "Stored notebook identity differs."
            else:
                notebook_identity.write_text(identity_text)
            command = [str(PYTHON), "-u", str(REPO_DIR / "scripts/run_qwen35_ternary_research_v2.py"),
                       "--config", str(config_path), "--historical-root", str(HISTORICAL_ROOT),
                       "--output-dir", str(root), "--work-dir", str(LOCAL_BASE / "work" / run_id),
                       "--budget-minutes", str(minutes_per_seed), "--stage", STAGE]
            RUNS.append({"seed": seed, "root": root, "command": command})
            _run_live([*command, "--dry-run"], "v2-plan", repo_dir=REPO_DIR,
                      log_root=root / "launch-logs", timeout_seconds=120)
        print("Requested stage:", STAGE, "· persisted execution cap:", BUDGET_MINUTES, "minutes")
        print("Historical comparison SHA256:", historical_sha)
    '''), md('''
        ## Run the requested stage
        A parity-only pass does not mean the full comparison is complete.
        Inspect its cached-decoding reports and gates before choosing `full`.
        A failed gate leaves diagnostics available and prevents training.
        The full stage runs within the same persisted allowance and preserves
        verified historical measurements. Avoid concurrent copies of this run.
    '''), code('''
        if RUN_EXPERIMENT:
            for run in RUNS:
                try:
                    _run_live(run["command"], f"v2-{STAGE}-s{run['seed']}", repo_dir=REPO_DIR,
                              log_root=run["root"] / "launch-logs")
                except (subprocess.CalledProcessError, TimeoutError) as error:
                    print("Stage stopped:", STAGE, run["seed"], str(error))
                    print("Saved diagnostics:", run["root"])
        else:
            print("Plan only; no GPU experiment was launched.")
    ''')]


def presentation_script():
    functions = [w4.measured_number, v1.backend_label, v1.model_style, v1.model_comparison_row,
                 v1.comparison_tables, v1.figure_legend, v1.quality_points, v1.quality_figure,
                 v1.task_points, v1.task_figure, v1.speed_lines, v1.speed_axis, v1.speed_figure,
                 v1.render_comparison, row_identity, comparison_group, render_v2_results]
    script = "import copy, json, math, sys\nfrom pathlib import Path\n"
    script += "\n\n".join(inspect.getsource(function) for function in functions)
    script += "\n\nlegacy_comparison_tables = comparison_tables\n"
    script += inspect.getsource(comparison_tables)
    script += "\nrender_group = render_comparison\n"
    return script + ('\nif __name__ == "__main__":\n'
                     '    render_v2_results(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])\n')


def result_cells():
    cells = port_cells(w4.result_cells())
    cells[0] = md('''
        ## Results and diagnostic gates
        Historical and fresh measurements are shown separately. A fresh row
        remains pending until measured; missing metrics stay empty. Every plot
        includes the model, runtime profile and phase. Complete payload sizes
        include auxiliary tensors. Speed plots retain backend identity.

        Read completion/repetition/format gates alongside PPL and task accuracy.
        Fixed-token throughput is separate from latency to a completed answer.
        The `parity_complete` flag only clears the diagnostic stage.
    ''')
    cells[1] = code("PRESENTATION_SCRIPT = " + repr(presentation_script()))
    display_setup = w4.result_cells()[2].source.split('for run in RUNS:')[0]
    cells[2] = code(display_setup + '''
for run in RUNS:
    report_path = run["root"] / "comparison.json"
    print("Seed", run["seed"], "·", run["root"])
    resume_error = run["root"] / "resume-error.json"
    if resume_error.exists():
        print("Saved-result revalidation failed; the previous comparison was preserved:",
              json.dumps(json.loads(resume_error.read_text()), indent=2))
    if not report_path.exists():
        print("Comparison unavailable; inspect launch-logs and saved stage reports.")
        continue
    script = LOCAL_BASE / "render-ternary-v2-results.py"
    script.write_text(PRESENTATION_SCRIPT)
    figures = run["root"] / "figures"
    _run_live([str(PYTHON), str(script), str(report_path), str(figures)], "render-v2-results",
              repo_dir=REPO_DIR, log_root=run["root"] / "launch-logs", timeout_seconds=120)
    readout = json.loads((figures / "readout.json").read_text())
    print("Full comparison complete:", readout["complete"])
    print("Cached-decoding parity complete:", readout["parity_complete"])
    print("Gates:", json.dumps(readout["gates"], indent=2))
    if readout.get("error"):
        print("Stopped:", readout["error"])
    for phase, result in readout["groups"].items():
        print("HISTORICAL VERIFIED MEASUREMENTS" if phase == "historical" else "FRESH MEASUREMENTS")
        for title, key in (("Quality and complete bytes", "models"), ("Public tasks", "tasks"),
                           ("Speed by backend and runtime profile", "speed"), ("Paired tasks", "paired")):
            display_table(title, result["tables"][key])
        for path in result["figures"]:
            display(Image(filename=path))
    print("Historical import:", json.dumps(readout["historical_import"], indent=2))
''')
    return cells


def build_notebook():
    cells = setup_cells() + w4.environment_cells() + experiment_cells() + result_cells()
    for index, cell in enumerate(cells):
        cell["id"] = f"ternary-v2-{index:02d}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "rotquant": {"study": "ternary-v2", "default_stage": "parity"}})


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)
