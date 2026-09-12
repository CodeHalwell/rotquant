"""Build the clean end-to-end native GPU Colab notebook and its legacy alias."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_qwen35_packed_validation_notebook import code, md


def build_notebook():
    return new_notebook(cells=[
        md('''
        # RotQuant Qwen3.5-4B — native GPU end-to-end validation

        ## Goal
        Open a **fresh GPU runtime**, check the saved-evidence path, then choose
        **Runtime → Run all**. No repair cells or mid-run Git updates.

        This runs the saved **W5/scale8 backbone + W6 vocabulary** in our native
        llama.cpp integration. No retraining or requantization. Original Drive
        evidence is read-only. The previous A100 40GB is a sensible test machine,
        not a measured minimum-memory requirement.

        This is correctness validation, **not a demonstrated quality gain or
        speedup**. CUDA/full-4B success must come from this run, not compilation.
        The first CUDA build took about 26 minutes. The default allowance is
        **90 minutes of active stage execution**. Notebook idle time does not
        consume that allowance, but Colab can still charge while idle.
        '''),
        md('''
        ## Setup — settings
        `SOURCE_ROOT` must contain the original `b5_v6_s0/checkpoint/`,
        `prepared.json`, `preparation.json` and `packed_probes.safetensors`.
        Result-only ZIPs are insufficient. Leave timings off for this first run.

        Reusing unchanged settings resumes verified completed stages and creates
        fresh attempts for failures. Choose a new `RUN_NAME` for changed settings
        or a separate W8 run; previous reports are never overwritten.
        '''),
        code('''
        from pathlib import Path
        import json, re, shutil, subprocess, sys

        REPO_REF = "main"  # resolved once to a full commit before execution
        SOURCE_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
        ARMS = ("b5_v6_s0",)
        RUN_NAME = "run1"
        RUN_TIMING = False
        ACTIVE_BUDGET_MINUTES = 90
        BUILD_JOBS = 2

        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", RUN_NAME)
        assert 1 <= ACTIVE_BUDGET_MINUTES <= 180
        assert ARMS and len(set(ARMS)) == len(ARMS) and set(ARMS) <= {"b5_v6_s0", "b5_v8_s0"}
        assert 1 <= BUILD_JOBS <= 32
        '''),
        md('''
        ### Connect Drive and pin code
        This checks paths and fetches code; it does not start the execution
        budget. Dependencies go into a dedicated virtual environment, preserving
        Colab's installed CUDA PyTorch and notebook packages. Setup does not
        require `ensurepip`: the notebook's pip installs into the explicitly
        selected venv. Its identity and inherited Torch are checked before and
        after installation, including when retrying an incomplete setup.
        '''),
        code('''
        from google.colab import drive
        drive.mount("/content/drive")
        assert shutil.which("nvidia-smi") and shutil.which("nvcc"), "Select a GPU runtime with the CUDA toolkit."
        subprocess.run(["nvidia-smi"], check=True, timeout=30)
        for arm in ARMS:
            for item in ("checkpoint", "prepared.json", "preparation.json", "packed_probes.safetensors"):
                assert (SOURCE_ROOT / arm / item).exists(), f"Missing original evidence: {SOURCE_ROOT / arm / item}"
        REPO_DIR = Path("/content/rotquant-native-gpu-e2e/repository")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", "https://github.com/CodeHalwell/rotquant.git", str(REPO_DIR)], check=True, timeout=300)
        assert not subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip(), "Repository has edits; do not reset them. Use a fresh runtime."
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_REF], check=True, timeout=120)
        COMMIT = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "FETCH_HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", COMMIT], check=True, timeout=30)
        RUNNER = REPO_DIR / "scripts/run_native_gpu_validation.py"
        assert RUNNER.exists(), "This revision predates the end-to-end notebook. Stop here."
        WORK_DIR = Path("/content/rotquant-native-gpu-e2e/cache") / COMMIT[:12]
        RESULT_ROOT = Path("/content/drive/MyDrive/rotquant/native_gpu_e2e") / COMMIT[:12] / RUN_NAME
        print({"commit": COMMIT, "results": str(RESULT_ROOT), "arms": ARMS,
               "active_budget_minutes": ACTIVE_BUDGET_MINUTES, "timing": RUN_TIMING})
        '''),
        md('''
        ## Steps and checks — complete workflow
        1. Isolated dependencies, import/CUDA checks, original checkpoint and
           saved-probe verification **before** the long build.
        2. Pinned native build including the loader fix, strict Linux linking,
           and a fresh-process binding-load check.
        3. CPU/CUDA packed operators, W6/W8 tiny whole-model graphs, and an
           offline Transformers-to-native conversion check.
        4. Exact retained-model export, then GPU parity against frozen logits
           and generation. Optional timing runs only after parity passes.

        Progress, 30-second heartbeats, attempt logs and a stage ledger are
        saved to Drive. No failed gate is bypassed. Rerun **this cell** after an
        interruption in the same session; verified completed stages are reused.
        After losing a runtime, Run all rebuilds missing local files while
        retaining earlier attempt reports. Active time is cumulative on resume.

        Text inference only: non-text tensors remain in a counted sidecar, not
        a working vision tower. No public-task sweep, competitor downloads,
        external Hadamard build or `llama-cpp-python` wheel.
        '''),
        code('''
        sys.path.insert(0, str(REPO_DIR))
        from scripts.colab_runtime import run_live
        command = [sys.executable, "-u", str(RUNNER),
                   "--output-dir", str(RESULT_ROOT), "--work-dir", str(WORK_DIR),
                   "--source-root", str(SOURCE_ROOT),
                   "--active-minutes", str(ACTIVE_BUDGET_MINUTES), "--jobs", str(BUILD_JOBS)]
        for arm in ARMS:
            command.extend(["--arm", arm])
        if RUN_TIMING:
            command.append("--timing")
        # The driver owns active/per-phase budgets; no timer starts in setup.
        run_live(command, "end-to-end", repo_dir=REPO_DIR,
                 log_root=RESULT_ROOT / "launch-logs")
        '''),
        md('''
        ## Results and next steps
        A pass means these bounded checks passed on this runtime, not full-suite
        quality or production speed. Ordinary failures also write a summary and
        reports-only ZIP. Hard VM loss may interrupt archiving, but previously
        written stage receipts/logs remain on Drive.

        Share the ZIP. Do not loosen thresholds if a numerical gate fails.
        **Disconnect and delete the Colab runtime when finished or blocked**:
        neither the execution cap nor notebook completion stops GPU billing.
        '''),
        code('''
        summary_path = RESULT_ROOT / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            print("Workflow:", summary["status"], "| active minutes:", round(summary["active_minutes"], 1))
            for stage in summary["stages"]:
                print(stage["name"], stage["status"], f"{stage.get('elapsed_seconds', 0):.1f}s")
            if summary.get("error"):
                print("Stopped:", summary["error"])
        for archive in sorted(RESULT_ROOT.parent.glob(RUN_NAME + "-reports-*.zip"))[-3:]:
            print("Reports ZIP:", archive)
        print("Stop/disconnect the GPU runtime to avoid further charges.")
        '''),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"})


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    directory = Path(__file__).resolve().parents[1] / "notebooks"
    for name in ("qwen35_4b_native_gpu_e2e_colab.ipynb", "qwen35_4b_native_gpu_validation_colab.ipynb"):
        nbformat.write(notebook, directory / name)
        print(directory / name)
