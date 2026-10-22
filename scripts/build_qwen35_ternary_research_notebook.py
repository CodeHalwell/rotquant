"""Build the separate, versioned ternary QAT Colab study from the W4 setup."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_qwen35_w4_research_notebook as w4
from scripts.build_qwen35_packed_validation_notebook import code, md

OUTPUT = ROOT / "notebooks/qwen35_4b_ternary_research_colab.ipynb"
LABELS = ("fp16", "w6_v6", "w5_v6", "w4_v6", "w4_v6_recovered", "ternary_ptq",
          "ternary_blockqat", "ternary_recovered", "bonsai2_4b")
measured_number = w4.measured_number


def backend_label(row):
    value = (row.get("speed") or {}).get("backend") or row.get("backend")
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value) if value else "Unreported backend"


def model_style(label):
    labels = ["fp16", "w6_v6", "w5_v6", "w4_v6", "w4_v6_recovered", "ternary_ptq",
              "ternary_blockqat", "ternary_recovered", "bonsai2_4b"]
    index = labels.index(label) if label in labels else sum(map(ord, label))
    colors = ["#494949", "#245ca6", "#a96919", "#526c38", "#9c4f77"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    return colors[index % len(colors)], markers[index % len(markers)]


def model_comparison_row(row):
    quality = row.get("quality") or {}
    short = (quality.get("test") or {}).get("summary") or {}
    long = (quality.get("long_test") or {}).get("summary") or {}
    result = {"Model": row["label"], "Status": row.get("status", "unreported"),
              "Backend": backend_label(row), "Precision scope": row.get("precision_scope"),
              "Error": row.get("error")}
    for label, key in (("Checkpoint GB", "checkpoint_bytes"), ("Native GB", "native_payload_bytes"),
                       ("Language GB", "language_payload_bytes")):
        size = measured_number(row.get(key))
        result[label] = None if size is None else size / 1e9
    result.update({"Teacher KL": measured_number(short.get("mean_teacher_kl")),
                   "Teacher KL 95% interval": (short.get("confidence_intervals") or {}).get("mean_teacher_kl"),
                   "Perplexity": measured_number(short.get("ppl")),
                   "Perplexity 95% interval": short.get("ppl_interval"),
                   "Top-1 agreement": measured_number(short.get("top1_agreement")),
                   "Long-context KL": measured_number(long.get("mean_teacher_kl")),
                   "Test documents": short.get("documents"), "Test tokens": short.get("tokens")})
    return result


def comparison_tables(report):
    models, tasks, speed, paired = [], [], [], []
    for row in report.get("rows", []):
        models.append(model_comparison_row(row))
        for benchmark, values in ((row.get("tasks") or {}).get("summary") or {}).items():
            accuracy = measured_number(values.get("accuracy"))
            tasks.append({"Model": row["label"], "Benchmark": benchmark,
                          "Accuracy (%)": None if accuracy is None else 100 * accuracy,
                          "Items": values.get("items"), "Correct": values.get("correct"),
                          "Truncated": values.get("truncated"), "Backend": backend_label(row)})
        speed_report = row.get("speed") or {}
        for context, values in (speed_report.get("summary") or {}).items():
            speed.append({"Model": row["label"], "Backend": backend_label(row),
                          "Input tokens": int(context),
                          "Prefill tokens/s": measured_number(values.get("prefill_tokens_per_second")),
                          "Decode tokens/s": measured_number(values.get("decode_tokens_per_second")),
                          "Repetitions": values.get("measured_repetitions"),
                          "All contexts complete": speed_report.get("completed") is True})
    for comparison, benchmarks in (report.get("paired_tasks") or {}).items():
        for benchmark, values in benchmarks.items():
            paired.append({"Comparison": comparison, "Benchmark": benchmark, "Items": values.get("items"),
                           "Accuracy delta (fraction)": values.get("mean_accuracy_delta"),
                           "Bootstrap 95% interval": values.get("bootstrap_interval"),
                           "Hoeffding 95% interval": values.get("hoeffding_interval"),
                           "Wins": values.get("candidate_wins"), "Losses": values.get("candidate_losses")})
    return {"models": models, "tasks": tasks, "speed": speed, "paired": paired}


def quality_figure(models):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for axis, metric, title in zip(axes, ("Teacher KL", "Perplexity"),
                                   ("Difference from FP16 · lower is better", "Held-out perplexity · lower is better")):
        for row in models:
            x, y = row["Native GB"], row[metric]
            if x is None or y is None:
                continue
            color, marker = model_style(row["Model"])
            axis.scatter(x, y, color=color, marker=marker, s=48, label=row["Model"])
            interval = row.get(metric + " 95% interval")
            if interval is not None:
                axis.vlines(x, interval[0], interval[1], color=color, alpha=0.6)
        axis.set(xlabel="Measured complete native payload (GB)", ylabel=metric, title=title)
        axis.grid(alpha=0.18)
        axis.margins(x=0.22, y=0.25)
    handles, labels = {}, []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in handles:
                labels.append(label)
                handles[label] = handle
    if labels:
        figure.legend([handles[label] for label in labels], labels, loc="outside lower center",
                      ncol=min(3, len(labels)), frameon=False)
    figure.suptitle("Model size and quality · document-bootstrap 95% intervals", fontsize=15)
    return figure


def task_figure(models, rows):
    import matplotlib.pyplot as plt

    sizes = {row["Model"]: row["Native GB"] for row in models}
    benchmarks = list(dict.fromkeys(row["Benchmark"] for row in rows))
    figure, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 5.4),
                                squeeze=False, layout="constrained")
    for axis, benchmark in zip(axes[0], benchmarks):
        for row in rows:
            x, y = sizes.get(row["Model"]), row["Accuracy (%)"]
            if row["Benchmark"] != benchmark or x is None or y is None:
                continue
            color, marker = model_style(row["Model"])
            axis.scatter(x, y, color=color, marker=marker, s=48, label=row["Model"])
        axis.set(xlabel="Measured complete native payload (GB)", ylabel="Accuracy (%)",
                 title=benchmark, ylim=(-2, 102))
        axis.grid(alpha=0.18)
        axis.margins(x=0.22)
    handles, labels = {}, []
    for axis in axes[0]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in handles:
                labels.append(label)
                handles[label] = handle
    if labels:
        figure.legend([handles[label] for label in labels], labels, loc="outside lower center",
                      ncol=min(3, len(labels)), frameon=False)
    figure.suptitle("Public task quality and size\nCounts and paired intervals in tables", fontsize=15)
    return figure


def speed_figure(rows):
    import matplotlib.pyplot as plt

    backends = list(dict.fromkeys(row["Backend"] for row in rows))
    figure, axes = plt.subplots(len(backends), 2, figsize=(13, 5.4 * len(backends)),
                                squeeze=False, layout="constrained")
    for pair, backend in zip(axes, backends):
        subset = [row for row in rows if row["Backend"] == backend]
        labels = list(dict.fromkeys(row["Model"] for row in subset))
        for axis, metric in zip(pair, ("Prefill tokens/s", "Decode tokens/s")):
            for label in labels:
                values = sorted((row for row in subset if row["Model"] == label and row[metric] is not None),
                                key=lambda row: row["Input tokens"])
                if values:
                    color, marker = model_style(label)
                    axis.plot([row["Input tokens"] for row in values], [row[metric] for row in values],
                              label=label, marker=marker, color=color)
            maximum = max((row[metric] for row in subset if row[metric] is not None), default=0)
            axis.set(xlabel="Input tokens", ylabel=metric, title=f"{backend}\n{metric}",
                     ylim=(0, max(1, maximum * 1.15)))
            axis.set_xticks(sorted({row["Input tokens"] for row in subset}))
            axis.grid(alpha=0.18)
            if axis.lines:
                axis.legend(loc="best", fontsize=9, frameon=False)
            else:
                axis.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=axis.transAxes)
                axis.set_xticks([])
                axis.set_yticks([])
    figure.suptitle("Measured runtime speed · medians, warmup excluded", fontsize=15)
    return figure


def render_comparison(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = comparison_tables(report)
    figures = []
    if any(row["Native GB"] is not None and (row["Teacher KL"] is not None or row["Perplexity"] is not None)
           for row in tables["models"]):
        figures.append(("quality-size", quality_figure(tables["models"])))
    sizes = {row["Model"]: row["Native GB"] for row in tables["models"]}
    tasks = [row for row in tables["tasks"] if row["Accuracy (%)"] is not None
             and sizes.get(row["Model"]) is not None]
    if tasks:
        figures.append(("tasks-size", task_figure(tables["models"], tasks)))
    speed = [row for row in tables["speed"]
             if row["Prefill tokens/s"] is not None or row["Decode tokens/s"] is not None]
    if speed:
        figures.append(("speed", speed_figure(speed)))
    paths = []
    for name, figure in figures:
        path = output / f"{name}.png"
        figure.savefig(path, dpi=160, facecolor="white")
        plt.close(figure)
        paths.append(str(path))
    result = {"complete": report.get("complete") is True, "tables": tables, "figures": paths,
              "selection": report.get("selection"), "error": report.get("error"),
              "scope": report.get("scope")}
    (output / "readout.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def port_cells(cells):
    """Retain the existing W4 setup and execution conventions in fresh cells."""
    replacements = {
        "scripts/run_qwen35_w4_research.py": "scripts/run_qwen35_ternary_research.py",
        "configs/qwen35_4b_w4_research_cuda.yaml": "configs/qwen35_4b_ternary_research_cuda.yaml",
        "rotquant/w4_research": "rotquant/ternary_research_v1",
        "rotquant-w4-research": "rotquant-ternary-research-v1",
        "Source is missing W4 research support": "Source is missing ternary research support",
        '"rotquant/activation_refit.py",': ('"rotquant/ternary.py", "rotquant/ternary_train.py", '
            '"scripts/ternary_research_bonsai.py", "scripts/ternary_bonsai_bridge.cpp",'),
        "render-w4-results.py": "render-ternary-results.py",
        "w4-reports-": "ternary-v1-reports-",
    }
    for cell in cells:
        for old, new in replacements.items():
            cell.source = cell.source.replace(old, new)
    return cells


def setup_cells():
    cells = port_cells(w4.setup_cells())
    cells[0] = md('''
        # RotQuant ternary study · version 1

        Compare **FP16**, fresh **W6/W6, W5/W6 and W4/W6** controls, and
        **ternary weights** before and after quantization-aware training (QAT).
        Include the official **Bonsai2-4B** model as a measured comparison.
        The study records held-out quality, public task scores, complete model
        bytes, and prefill/decode speed. This is an open research recipe;
        PrismML's private training method is not reproduced here.

        Select an **A100 40 GB-class GPU** and high host RAM. Choose **Run all**
        and upload the supplied source ZIP. The separate W4 notebook stays
        available. This notebook has no saved model results; the default
        16 examples per public benchmark are a pilot.
    ''')
    cells[2].source = cells[2].source.replace('SOURCE_MODE = "upload"',
        'STUDY_VERSION = "ternary-v1"\nSOURCE_MODE = "upload"')
    return cells


def experiment_cells():
    cells = port_cells(w4.experiment_cells())
    cells[0] = md('''
        ## Review the plan
        The driver checks native operators and whole-model graphs, prepares
        fresh controls, compares ternary initialization and block QAT, and
        tests separate recovery arms. Training, validation, selection and final
        tests use separate data. The official Bonsai comparison uses this run's
        inputs and measurements, never published scores copied into the table.

        The default pilot uses ternary MLP matrices, W4 for the remaining
        eligible backbone, W6 vocabulary and source-precision protected tensors.
        The config also supports an all-backbone ternary experiment. Every row
        records its precision scope. Canonical labels: `fp16`, `w6_v6`, `w5_v6`, `w4_v6`,
        `w4_v6_recovered`, `ternary_ptq`, `ternary_blockqat`,
        `ternary_recovered`, `bonsai2_4b`.
    ''')
    cells[1].source = cells[1].source.replace('identity = {"source": SOURCE_ID,',
        'identity = {"study": STUDY_VERSION, "source": SOURCE_ID,')
    cells[1].source = cells[1].source.replace('config["seed"] = seed',
        'config["seed"] = seed\n    config["ternary"]["quant"]["seed"] = seed')
    return cells


def result_cells():
    cells = port_cells(w4.result_cells())
    cells[0] = md('''
        ## Results
        These tables and plots use saved measurements from this run. Missing
        values stay empty; status and error fields identify unfinished arms.
        The size plots use **complete native payload bytes**, including retained
        auxiliary tensors. Language-only and checkpoint sizes are separate.
        Backend and precision scope stay visible; speed plots separate backends.

        Task accuracy is shown as a percentage; paired deltas and intervals use
        fractions. Counts and truncations matter for this small pilot. Timings
        use frozen decode tokens and describe runtime speed, while public task
        scores measure generated answers. Selection uses its dedicated data.
        No row establishes unchanged capability by itself.
        Completed quality/task measurements remain visible if timing fails;
        that arm keeps its failed status. Resuming a fully completed run checks
        its saved receipts without starting GPU work.
    ''')
    functions = [measured_number, backend_label, model_style, model_comparison_row,
                 comparison_tables, quality_figure, task_figure, speed_figure, render_comparison]
    script = "import json, math, sys\nfrom pathlib import Path\n"
    script += "\n\n".join(inspect.getsource(function) for function in functions)
    script += '\nif __name__ == "__main__":\n    render_comparison(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])\n'
    cells[1] = code("PRESENTATION_SCRIPT = " + repr(script))
    cells[2].source = cells[2].source.replace('("Native speed", "speed")', '("Measured speed by backend", "speed")')
    cells[2].source = cells[2].source.replace('print("Selection record:",',
        'print("Scope:", readout.get("scope"))\n    print("Selection record:",')
    return cells


def build_notebook():
    cells = setup_cells() + w4.environment_cells() + experiment_cells() + result_cells()
    for index, cell in enumerate(cells):
        cell["id"] = f"ternary-v1-{index:02d}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "rotquant": {"study": "ternary-v1"},
    })


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)
