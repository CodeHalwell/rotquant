"""Build the uploaded-source, resumable W4 research Colab companion."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_qwen35_packed_validation_notebook import code, md

OUTPUT = ROOT / "notebooks/qwen35_4b_w4_research_colab.ipynb"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or any(part in {".git", ".venv"} for part in path.parts)):
        raise ValueError(f"Unsafe source archive member: {name!r}")
    return path


def verify_source_bundle(repository):
    repository = Path(repository)
    if any(path.is_symlink() for path in repository.rglob("*")):
        raise ValueError("Source tree must not contain symbolic links")
    manifest_path = repository / "SOURCE_BUNDLE.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "rotquant-source-bundle-v1":
        raise ValueError("Unrecognized source bundle manifest")
    entries = manifest.get("files_sha256")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("The source bundle must list its file checksums")
    for name, expected in entries.items():
        path = repository / safe_member_name(name)
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    expected_files = set(entries) | {"SOURCE_BUNDLE.json"}
    actual_files = {str(path.relative_to(repository)) for path in repository.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                    and not any(part.endswith(".egg-info") for part in path.parts)}
    if actual_files != expected_files:
        raise ValueError("Source tree has unlisted files; use a fresh source directory")
    return file_sha256(manifest_path)


def unpack_source_bundle(archive_path, cache_root):
    archive_path, cache_root = Path(archive_path), Path(cache_root)
    archive_hash = file_sha256(archive_path)
    repository = cache_root / archive_hash
    if repository.exists():
        return repository, verify_source_bundle(repository)
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=cache_root) as temporary:
        target = Path(temporary) / "repository"
        target.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > 50000 or sum(item.file_size for item in members) > 512 * 2**20:
                raise ValueError("Source archive exceeds the 512 MiB extraction limit")
            seen = set()
            for item in members:
                name = safe_member_name(item.filename)
                mode = stat.S_IFMT(item.external_attr >> 16)
                if str(name) in seen or mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Duplicate or non-regular source archive entry")
                seen.add(str(name))
                destination = target / name
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
        identity = verify_source_bundle(target)
        target.rename(repository)
    return repository, identity


def measured_number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def model_comparison_row(row):
    quality = row.get("quality") or {}
    short = (quality.get("test") or {}).get("summary") or {}
    long = (quality.get("long_test") or {}).get("summary") or {}
    checkpoint = measured_number(row.get("checkpoint_bytes"))
    native = measured_number(row.get("native_payload_bytes"))
    return {"Model": row["label"], "Checkpoint GB": None if checkpoint is None else checkpoint / 1e9,
            "Native GB": None if native is None else native / 1e9,
            "Teacher KL": measured_number(short.get("mean_teacher_kl")),
            "Teacher KL 95% interval": (short.get("confidence_intervals") or {}).get("mean_teacher_kl"),
            "Perplexity": measured_number(short.get("ppl")), "Perplexity 95% interval": short.get("ppl_interval"),
            "Top-1 agreement": measured_number(short.get("top1_agreement")),
            "Long-context KL": measured_number(long.get("mean_teacher_kl")),
            "Test documents": short.get("documents"), "Test tokens": short.get("tokens")}


def comparison_tables(report):
    models, tasks, speed, paired = [], [], [], []
    for row in report.get("rows", []):
        models.append(model_comparison_row(row))
        for benchmark, values in ((row.get("tasks") or {}).get("summary") or {}).items():
            tasks.append({"Model": row["label"], "Benchmark": benchmark,
                          "Accuracy": measured_number(values.get("accuracy")),
                          "Items": values.get("items"), "Correct": values.get("correct"),
                          "Truncated": values.get("truncated")})
        speed_report = row.get("speed") or {}
        for context, values in (speed_report.get("summary") or {}).items():
            speed.append({"Model": row["label"], "Input tokens": int(context),
                          "Prefill tokens/s": measured_number(values.get("prefill_tokens_per_second")),
                          "Decode tokens/s": measured_number(values.get("decode_tokens_per_second")),
                          "Repetitions": values.get("measured_repetitions"),
                          "All contexts complete": speed_report.get("completed") is True})
    for comparison, benchmarks in (report.get("paired_tasks") or {}).items():
        for benchmark, values in benchmarks.items():
            paired.append({"Comparison": comparison, "Benchmark": benchmark, "Items": values.get("items"),
                           "Accuracy delta": values.get("mean_accuracy_delta"),
                           "Bootstrap 95% interval": values.get("bootstrap_interval"),
                           "Hoeffding 95% interval": values.get("hoeffding_interval"),
                           "Wins": values.get("candidate_wins"), "Losses": values.get("candidate_losses")})
    return {"models": models, "tasks": tasks, "speed": speed, "paired": paired}


def model_style(label):
    labels = ["fp16", "w5_v6", "w4_g128_mse", "w4_g128_h", "w4_g64_h", "w4_g32_h",
              "w4_codes", "w4_recovered", "w4_codes_recovered"]
    index = labels.index(label) if label in labels else sum(map(ord, label))
    colors = ["#494949", "#245ca6", "#a96919", "#526c38", "#9c4f77"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    return colors[index % len(colors)], markers[index % len(markers)]


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
        axis.set(xlabel="Measured native model payload (GB)", ylabel=metric, title=title)
        axis.grid(alpha=0.18)
        axis.margins(x=0.22, y=0.25)
    handles, legend = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend, loc="outside lower center", ncol=min(3, len(handles)), frameon=False)
    figure.suptitle("Model size and quality · document-bootstrap 95% intervals", fontsize=15)
    return figure


def speed_series(rows, label):
    return sorted((row for row in rows if row["Model"] == label), key=lambda row: row["Input tokens"])


def speed_axis(axis, rows, metric, labels):
    measured = [row for row in rows if row[metric] is not None]
    for label in labels:
        values = speed_series(measured, label)
        if values:
            color, marker = model_style(label)
            axis.plot([row["Input tokens"] for row in values], [row[metric] for row in values],
                      label=label, marker=marker, color=color)
    maximum = max((row[metric] for row in measured), default=0)
    axis.set(xlabel="Input tokens", ylabel=metric, title=metric + " · higher is better",
             ylim=(0, max(1, maximum * 1.15)))
    axis.set_xticks(sorted({row["Input tokens"] for row in rows}))
    axis.grid(alpha=0.18)


def speed_figure(rows):
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(row["Model"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for axis, metric in zip(axes, ("Prefill tokens/s", "Decode tokens/s")):
        speed_axis(axis, rows, metric, labels)
    handles, legend = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend, loc="outside lower center", ncol=min(3, len(handles)), frameon=False)
    figure.suptitle("Fresh native CUDA timings · medians, warmup excluded", fontsize=15)
    return figure


def render_comparison(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = comparison_tables(report)
    paths = []
    for name, values, function in (("quality-size", tables["models"], quality_figure),
                                   ("speed", tables["speed"], speed_figure)):
        if values:
            figure = function(values)
            path = output / f"{name}.png"
            figure.savefig(path, dpi=160, facecolor="white")
            plt.close(figure)
            paths.append(str(path))
    result = {"complete": report.get("complete") is True, "tables": tables, "figures": paths,
              "selection": report.get("selection"), "error": report.get("error")}
    (output / "readout.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def embedded(functions):
    return "\n\n".join(inspect.getsource(function) for function in functions)


def setup_cells():
    return [
        md('''
        # RotQuant W4 research

        Compare **FP16**, a freshly prepared **W5/W6 baseline**, and smaller
        **W4/W6 candidates** using the same native CUDA runtime. Test group sizes
        128, 64 and 32, fit scales using calibration activations, refine selected
        codes, and compare short scale-only recovery runs with and without code
        refinement from the selected initial candidate.

        Use a fresh **A100 40 GB-class GPU** with high host RAM. Choose **Run all**
        and upload the supplied source ZIP when asked. The ZIP includes the new
        implementation; no commit or push is required. This notebook has no saved
        model results. Its default 16 examples per public benchmark are a pilot,
        not evidence that capability is unchanged.
        '''),
        md('''
        ## Settings
        Keep the same source ZIP and settings to resume. Logs and completed
        checkpoints stay on Drive. The eight-hour maximum is divided across
        seeds; setup and idle GPU time are additional. The run stops at its
        budget, so a cold build or interrupted session may leave partial results.
        '''),
        code('''
        from pathlib import Path
        import hashlib, inspect, json, math, os, re, shutil, stat, subprocess, sys, tempfile, zipfile
        from pathlib import PurePosixPath

        SOURCE_MODE = "upload"  # "git" requires an explicitly published full commit below.
        SOURCE_ARCHIVE = ""  # Optional: paste the saved Drive ZIP path to reuse it.
        GIT_COMMIT = ""  # Exactly 40 hexadecimal characters; branches such as main are rejected.
        SEEDS = [0]
        PUBLIC_SAMPLES = 16  # Raise before the run for a larger independent task comparison.
        BUDGET_MINUTES = 480
        RUN_EXPERIMENT = True
        REQUIRE_FAST_HADAMARD = True
        USE_HF_SECRET = False
        RESULT_BASE = Path("/content/drive/MyDrive/rotquant/w4_research")
        LOCAL_BASE = Path("/content/rotquant-w4-research")
        UV_VERSION = "0.9.5"

        assert SOURCE_MODE in {"upload", "git"}
        assert SEEDS and len(set(SEEDS)) == len(SEEDS) and all(type(seed) is int and seed >= 0 for seed in SEEDS)
        assert type(BUDGET_MINUTES) is int and len(SEEDS) <= BUDGET_MINUTES <= 480
        assert type(PUBLIC_SAMPLES) is int and 1 <= PUBLIC_SAMPLES <= 541
        '''),
        md('''
        ## Load the code
        Upload the supplied ZIP, or enter its saved Drive path above. Every source
        file is checked against the bundle manifest, including on resume. A changed
        source or configuration gets a separate result directory.
        '''),
        code(embedded([file_sha256, safe_member_name, verify_source_bundle, unpack_source_bundle])),
        code('''
        from google.colab import drive, files
        import torch
        from importlib import metadata
        assert torch.cuda.is_available(), "Choose a GPU runtime."
        assert torch.cuda.get_device_properties(0).total_memory >= 35 * 2**30, "Choose an A100 40 GB-class GPU."
        assert shutil.which("nvcc"), "This runtime needs the CUDA toolkit."
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        RESULT_BASE.mkdir(parents=True, exist_ok=True)
        LOCAL_BASE.mkdir(parents=True, exist_ok=True)
        TORCH_VERSION = metadata.version("torch")
        TORCH_FILE = str(Path(torch.__file__).resolve())
        if SOURCE_MODE == "upload":
            if not SOURCE_ARCHIVE:
                uploaded = files.upload()
                assert len(uploaded) == 1, "Upload exactly one source ZIP."
                name, content = next(iter(uploaded.items()))
                assert name.lower().endswith(".zip"), "Select the source ZIP."
                archive_hash = hashlib.sha256(content).hexdigest()
                archive = RESULT_BASE / "source_uploads" / (archive_hash + ".zip")
                archive.parent.mkdir(parents=True, exist_ok=True)
                if archive.exists():
                    assert file_sha256(archive) == archive_hash, "Saved source ZIP checksum differs."
                else:
                    archive.write_bytes(content)
                SOURCE_ARCHIVE = str(archive)
                del uploaded, content
            REPO_DIR, SOURCE_ID = unpack_source_bundle(SOURCE_ARCHIVE, LOCAL_BASE / "sources")
            print("Saved ZIP for reconnect:", SOURCE_ARCHIVE)
        else:
            assert re.fullmatch(r"[0-9a-fA-F]{40}", GIT_COMMIT), "Set a published full commit SHA."
            REPO_DIR = LOCAL_BASE / "sources" / GIT_COMMIT.lower()
            if not REPO_DIR.exists():
                subprocess.run(["git", "clone", "--no-checkout", "https://github.com/CodeHalwell/rotquant.git", str(REPO_DIR)], check=True, timeout=300)
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", GIT_COMMIT], check=True, timeout=180)
                subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", GIT_COMMIT], check=True, timeout=60)
            assert subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip() == GIT_COMMIT.lower()
            assert not subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip(), "Source has changed; use a fresh directory."
            SOURCE_ID = GIT_COMMIT.lower()
        for name in ("scripts/run_qwen35_w4_research.py", "scripts/w4_research_measure.py", "rotquant/activation_refit.py", "configs/qwen35_4b_w4_research_cuda.yaml"):
            assert (REPO_DIR / name).is_file(), f"Source is missing W4 research support: {name}"
        sys.path.insert(0, str(REPO_DIR))
        from scripts.colab_runtime import run_live as _run_live
        SETUP_LOGS = RESULT_BASE / "setup" / SOURCE_ID[:16]
        def run_live(command, label, timeout_seconds=None):
            return _run_live(command, label, repo_dir=REPO_DIR, log_root=SETUP_LOGS, timeout_seconds=timeout_seconds)
        print({"source": SOURCE_ID, "gpu": torch.cuda.get_device_name(0), "results": str(RESULT_BASE)})
        '''),
    ]


def environment_cells():
    return [
        md('''
        ## Prepare the runtime
        Dependencies live in a separate environment. The notebook checks that
        Colab's exact CUDA PyTorch version and location remain unchanged. Setup
        and fast-kernel build logs are saved to Drive.
        '''),
        code('''
        from urllib.request import urlopen
        uv = shutil.which("uv")
        if uv is None:
            tool_dir = LOCAL_BASE / "tools" / ("uv-" + UV_VERSION)
            tool_dir.mkdir(parents=True, exist_ok=True)
            installer = tool_dir / "install.sh"
            with urlopen(f"https://astral.sh/uv/{UV_VERSION}/install.sh", timeout=60) as response:
                installer.write_bytes(response.read())
            install_env = {**os.environ, "UV_INSTALL_DIR": str(tool_dir), "UV_NO_MODIFY_PATH": "1"}
            subprocess.run(["sh", str(installer)], env=install_env, check=True, timeout=180)
            uv = str(tool_dir / "uv")
        print(subprocess.check_output([uv, "--version"], text=True).strip())
        VENV = LOCAL_BASE / "environments" / SOURCE_ID[:16]
        PYTHON = VENV / "bin/python"
        if not PYTHON.exists():
            run_live([uv, "venv", "--python", sys.executable, "--system-site-packages", str(VENV)], "create-environment", 180)
        check = [str(PYTHON), str(REPO_DIR / "scripts/native_gpu_environment.py"),
                 "--prefix", str(VENV), "--expected-torch", TORCH_VERSION, "--expected-torch-file", TORCH_FILE]
        run_live([*check, "--output", str(SETUP_LOGS / "torch-before.json")], "torch-before", 60)
        constraints = VENV / "torch-constraint.txt"
        constraints.write_text(f"torch=={TORCH_VERSION}\\n")
        packages = ["datasets==4.8.5", "accelerate==1.13.0", "pandas==2.3.3",
                    "matplotlib==3.10.9", "seaborn==0.13.2", "huggingface_hub==1.17.0",
                    "nltk==3.9.2", "langdetect==1.0.9", "immutabledict==4.2.1", "absl-py==2.3.1",
                    "setuptools==80.9.0", "wheel==0.45.1"]
        install = [uv, "pip", "install", "--python", str(PYTHON), "--constraint", str(constraints)]
        run_live([*install, "-r", str(REPO_DIR / "requirements/native-gpu.txt"), *packages], "dependencies", 1200)
        run_live([*install, "--no-deps", "--no-build-isolation", "-e", str(REPO_DIR)], "install-project", 300)
        run_live([*check, "--output", str(SETUP_LOGS / "torch-after.json")], "torch-after", 60)
        os.environ.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
        smoke = ("import torch; from fast_hadamard_transform import hadamard_transform; "
                 "x=hadamard_transform(torch.ones(1,128,device='cuda',dtype=torch.float16),scale=128**-0.5); "
                 "assert torch.isfinite(x).all(); print('Fast Hadamard CUDA smoke passed')")
        try:
            run_live([*install, "--no-deps", "--no-build-isolation",
                      "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0.post2"], "fast-hadamard-build", 1800)
            run_live([str(PYTHON), "-c", smoke], "fast-hadamard-smoke", 60)
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
        except (subprocess.CalledProcessError, TimeoutError):
            if REQUIRE_FAST_HADAMARD:
                raise
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
        run_live([*check, "--output", str(SETUP_LOGS / "torch-final.json")], "torch-final", 60)
        if USE_HF_SECRET:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
            del token
        '''),
    ]


def experiment_cells():
    return [
        md('''
        ## Review the plan
        Calibration, candidate selection, recovery and final tests use separate
        data. The run first checks native operators and tiny whole-model graphs.
        It then prepares new checkpoints, selects a group size, compares code
        refinement, recovery alone and their combination, and measures held-out quality,
        public tasks, actual model bytes and matched prefill/decode speed.
        '''),
        code('''
        config_reader = "import json,yaml,sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"
        base_config = json.loads(subprocess.check_output([str(PYTHON), "-c", config_reader,
            str(REPO_DIR / "configs/qwen35_4b_w4_research_cuda.yaml")], text=True))
        RUNS = []
        minutes_per_seed = BUDGET_MINUTES // len(SEEDS)
        for seed in SEEDS:
            config = json.loads(json.dumps(base_config))
            config["seed"] = seed
            config["research"]["public_samples_per_benchmark"] = PUBLIC_SAMPLES
            identity = {"source": SOURCE_ID, "config": config, "budget_minutes": minutes_per_seed,
                        "fast_hadamard_required": REQUIRE_FAST_HADAMARD}
            run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            root = RESULT_BASE / f"{run_id}-s{seed}"
            root.mkdir(parents=True, exist_ok=True)
            config_path = root / "config.yaml"
            text = json.dumps(config, indent=2) + "\\n"  # JSON is also valid YAML.
            if config_path.exists():
                assert config_path.read_text() == text, "Stored run settings differ."
            else:
                config_path.write_text(text)
            (root / "notebook-identity.json").write_text(json.dumps(identity, indent=2) + "\\n")
            command = [str(PYTHON), "-u", str(REPO_DIR / "scripts/run_qwen35_w4_research.py"),
                       "--config", str(config_path), "--output-dir", str(root),
                       "--work-dir", str(LOCAL_BASE / "work" / run_id),
                       "--budget-minutes", str(minutes_per_seed)]
            RUNS.append({"seed": seed, "root": root, "command": command})
            _run_live([*command, "--dry-run"], "plan", repo_dir=REPO_DIR, log_root=root / "launch-logs", timeout_seconds=120)
        print("Total execution allowance:", minutes_per_seed * len(SEEDS), "minutes")
        '''),
        md('''
        ## Run the comparison
        Progress and logs stay on Drive. A failed gate stops that seed and leaves
        its reports available below. Rerun with unchanged settings to reuse
        verified completed stages; a lost Colab runtime may require local files
        to be rebuilt. Avoid running two copies against the same result folder.
        '''),
        code('''
        if RUN_EXPERIMENT:
            for run in RUNS:
                try:
                    _run_live(run["command"], f"research-s{run['seed']}", repo_dir=REPO_DIR,
                              log_root=run["root"] / "launch-logs")
                except (subprocess.CalledProcessError, TimeoutError) as error:
                    print("This seed stopped:", run["seed"], str(error))
                    print("Inspect its saved reports below:", run["root"])
        else:
            print("Plan only; no model experiment was launched.")
        '''),
    ]


def result_cells():
    presentation = embedded([measured_number, model_comparison_row, comparison_tables, model_style,
                             quality_figure, speed_series, speed_axis, speed_figure, render_comparison])
    script = "import json, math, sys\nfrom pathlib import Path\n" + presentation
    script += '\nif __name__ == "__main__":\n    render_comparison(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])\n'
    return [
        md('''
        ## Results
        Charts use only this run's saved measurements. Empty values mean missing
        measurements; they are never filled with zero. Checkpoint and native
        payload sizes stay separate. Timings exclude warmup and use frozen decode
        tokens, so they describe runtime speed rather than generated-answer quality.

        The pilot task scores and paired intervals are exploratory. Read the
        counts and both interval estimates before interpreting a difference.
        Candidate selection uses its dedicated selection data, not these tests.
        '''),
        code("PRESENTATION_SCRIPT = " + repr(script)),
        code('''
        from html import escape
        from IPython.display import HTML, Image, display
        def display_table(title, rows):
            print(title)
            if not rows:
                print("Not measured yet.")
                return
            keys = list(rows[0])
            def value(item):
                if item is None:
                    return "—"
                return f"{item:.6g}" if isinstance(item, float) else str(item)
            heading = "".join(f"<th>{escape(key)}</th>" for key in keys)
            body = "".join("<tr>" + "".join(f"<td>{escape(value(row.get(key)))}</td>" for key in keys) + "</tr>" for row in rows)
            display(HTML(f"<div style='overflow-x:auto'><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"))
        for run in RUNS:
            report_path = run["root"] / "comparison.json"
            print("Seed", run["seed"], "·", run["root"])
            if not report_path.exists():
                print("Comparison not available yet. Read launch-logs and stage reports.")
                continue
            script = LOCAL_BASE / "render-w4-results.py"
            script.write_text(PRESENTATION_SCRIPT)
            figures = run["root"] / "figures"
            _run_live([str(PYTHON), str(script), str(report_path), str(figures)], "render-results",
                      repo_dir=REPO_DIR, log_root=run["root"] / "launch-logs", timeout_seconds=120)
            readout = json.loads((figures / "readout.json").read_text())
            print("Complete" if readout["complete"] else "Partial — some measurements are missing")
            if readout.get("error"):
                print("Stopped:", readout["error"])
            for title, key in (("Quality and measured bytes", "models"), ("Public tasks", "tasks"),
                               ("Native speed", "speed"), ("Paired task differences", "paired")):
                display_table(title, readout["tables"][key])
            for path in readout["figures"]:
                display(Image(filename=path))
            print("Selection record:", json.dumps(readout["selection"], indent=2))
        '''),
        md('''
        ## Save the reports
        The compact download contains reports, task outputs and figures. Full
        checkpoints remain in Drive. When finished, disconnect
        and delete the Colab runtime; the execution budget does not stop billing.
        '''),
        code('''
        archive = RESULT_BASE / ("w4-reports-" + SOURCE_ID[:16] + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for run in RUNS:
                for path in sorted(run["root"].rglob("*")):
                    if (path.is_file() and path.suffix in {".json", ".log", ".csv", ".png", ".yaml"}
                            and path.stat().st_size <= 20 * 2**20
                            and not any(part in {"checkpoint", "teacher", "hessians"} for part in path.parts)):
                        output.write(path, str(path.relative_to(RESULT_BASE)))
        print("Reports:", archive)
        files.download(str(archive))
        '''),
    ]


def build_notebook():
    cells = setup_cells() + environment_cells() + experiment_cells() + result_cells()
    for index, cell in enumerate(cells):
        cell["id"] = f"w4-research-{index:02d}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
    })


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)
