"""Build the bounded native-GPU Colab validation notebook (no quality sweep)."""
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
        # RotQuant Qwen3.5-4B native GPU validation

        ## Goal
        Run the retained **W5/scale8 backbone + shared W6/W8 vocabulary** through
        our isolated llama.cpp integration, on CUDA. No re-quantization, training,
        Python weight expansion, Unsloth downloads, or public-task sweep.

        This is the first CUDA validation, not an already-validated release.
        Compilation alone is not success. Small operator and whole-model gates
        must pass before the real 4B model is loaded. Then saved-probe parity
        must pass before bounded timings. Any failure stops the notebook.

        Existing Drive checkpoints/results are read-only. New logs and reports
        have a separate run root. Choose a GPU runtime; the first checks require
        CUDA and nvcc. The prior A100 40GB is a sensible test machine, not a
        measured minimum requirement for this new runtime. Disconnect/delete the
        runtime when finished: the software time cap cannot stop Colab billing.
        '''),
        md('''
        ## Setup — controls
        Start from a **fresh Colab runtime**. The new notebook and code must first
        be published to GitHub. `main` is resolved once to a commit, never pulled
        mid-run. Prefer replacing `REPO_REF` with that full commit SHA.

        Set `SOURCE_ROOT` to the original packed-validation directory containing
        `b5_v6_s0/checkpoint`, `prepared.json`, `preparation.json`, and
        `packed_probes.safetensors`. Downloaded result-only bundles are insufficient.
        Start with W6/seed 0; add `b5_v8_s0` only for a second independent test.
        `RUN_TIMING=False` runs just the short real-model parity check.
        '''),
        code('''
        from pathlib import Path
        from datetime import datetime, timezone
        import json, os, shutil, subprocess, sys, time

        REPO_REF = "main"
        SOURCE_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
        ARMS = ("b5_v6_s0",)
        RUN_RETAINED_MODEL = True
        RUN_TIMING = False  # enable only if you want the bounded 128/512/2048-token timing phase
        SESSION_BUDGET_MINUTES = 60
        BUILD_JOBS = 2
        RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        REPO_DIR = Path("/content/rotquant-native-gpu")
        LLAMA_DIR = Path("/content/llama-rotquant-native-v3")
        BUILD_DIR = Path("/content/llama-rotquant-native-v3-build")
        EXPORT_ROOT = Path("/content/rotquant-native-exports") / RUN_TAG
        assert 1 <= SESSION_BUDGET_MINUTES <= 180
        assert ARMS and len(set(ARMS)) == len(ARMS) and set(ARMS) <= {"b5_v6_s0", "b5_v8_s0"}
        STARTED = time.monotonic()
        '''),
        code('''
        from google.colab import drive
        drive.mount("/content/drive")
        import torch
        assert torch.cuda.is_available(), "Select a CUDA GPU runtime; CPU fallback is forbidden."
        assert shutil.which("nvcc"), "CUDA toolkit/nvcc is missing. Stop here."
        subprocess.run(["nvidia-smi"], check=True)
        if RUN_RETAINED_MODEL:
            for arm in ARMS:
                for name in ("checkpoint", "prepared.json", "preparation.json", "packed_probes.safetensors"):
                    assert (SOURCE_ROOT / arm / name).exists(), f"Missing original evidence: {SOURCE_ROOT / arm / name}"
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", "https://github.com/CodeHalwell/rotquant.git", str(REPO_DIR)], check=True)
        assert not subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip(), "Checkout has edits; do not reset them."
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_REF], check=True)
        COMMIT = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "FETCH_HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", COMMIT], check=True)
        assert (REPO_DIR / "scripts/build_rq3_runtime.py").is_file(), "This revision lacks the new runtime. Stop; do not run an older notebook."
        RESULT_ROOT = Path("/content/drive/MyDrive/rotquant/native_gpu_validation") / COMMIT[:12] / RUN_TAG
        RESULT_ROOT.mkdir(parents=True, exist_ok=False)
        LOG_ROOT = RESULT_ROOT / "logs"
        sys.path.insert(0, str(REPO_DIR))
        from scripts.colab_runtime import run_live as _run_live
        def run_live(command, label, phase_limit=900):
            remaining = SESSION_BUDGET_MINUTES * 60 - (time.monotonic() - STARTED)
            if remaining <= 0:
                raise TimeoutError("Session budget exhausted. Stop the Colab runtime.")
            return _run_live(command, label, repo_dir=REPO_DIR, log_root=LOG_ROOT,
                             timeout_seconds=min(remaining, phase_limit))
        def script(name, *args, label, phase_limit=900):
            return run_live([sys.executable, "-u", str(REPO_DIR / "scripts" / name), *map(str, args)], label, phase_limit)
        controls = {"commit": COMMIT, "source_root": str(SOURCE_ROOT), "arms": ARMS,
                    "run_retained": RUN_RETAINED_MODEL, "run_timing": RUN_TIMING,
                    "budget_minutes": SESSION_BUDGET_MINUTES, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        (RESULT_ROOT / "controls.json").write_text(json.dumps(controls, indent=2))
        print(controls, "\\nReports:", RESULT_ROOT)
        '''),
        md('''
        ## Steps — dependencies and isolated build
        Preserve Colab's installed CUDA PyTorch. The experiment does not need
        `llama-cpp-python`, its wheels, or the external fast-Hadamard extension.
        The native kernels are compiled with the pinned llama.cpp sources.
        Build output and 30-second heartbeats are persisted to Drive.
        '''),
        code('''
        run_live([sys.executable, "-m", "pip", "install", "transformers==5.9.0",
                  "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
                  "pyyaml==6.0.3", "ninja==1.13.0", "cmake>=3.24,<5"], "dependencies", 600)
        run_live([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "--no-deps"], "install-rotquant", 120)
        run_live([sys.executable, "-m", "pip", "freeze"], "environment", 60)
        script("build_rq3_runtime.py", "--source-dir", LLAMA_DIR, "--build-dir", BUILD_DIR,
               "--backend", "CUDA", "--jobs", BUILD_JOBS, label="native-build", phase_limit=2400)
        receipt = json.loads((BUILD_DIR / "build-receipt.json").read_text())
        LIBRARY = Path(receipt["library"])
        shutil.copy2(BUILD_DIR / "build-receipt.json", RESULT_ROOT / "build-receipt.json")
        print("Compiled, not yet validated:", LIBRARY)
        '''),
        md('''
        ## Checks — synthetic operators, then a whole Qwen graph
        The tiny random model has both linear and full attention, GDN permutations,
        a packed backbone and one shared packed embedding/head. It tests prompts
        of 1/4/17/64 tokens plus cached decoding. It is not a quality benchmark.
        CPU is used only as the small-fixture numerical reference; the CUDA graph
        forbids CPU arithmetic fallback.
        '''),
        code('''
        for backend in ("CPU", "CUDA0"):
            script("check_rq3_gpu.py", "--library", LIBRARY, "--backend", backend,
                   "--output", RESULT_ROOT / f"operators-{backend}.json", label=f"operators-{backend}")
        for bits in (6, 8):
            fixture = BUILD_DIR / f"synthetic-w{bits}-{RUN_TAG}.gguf"
            script("make_rq3_model_fixture.py", "--output", fixture, "--llama-dir", LLAMA_DIR,
                   "--vocabulary-bits", bits, label=f"fixture-w{bits}", phase_limit=120)
            script("check_rq3_model.py", "--library", LIBRARY, "--model", fixture,
                   "--backend", "CUDA0", "--output", RESULT_ROOT / f"model-w{bits}.json", label=f"model-w{bits}")
        script("check_rq3_conversion.py", "--library", LIBRARY, "--llama-dir", LLAMA_DIR,
               "--output-dir", BUILD_DIR / f"hf-conversion-{RUN_TAG}", "--backend", "CUDA0",
               "--report", RESULT_ROOT / "hf-conversion.json", label="hf-conversion")
        '''),
        md('''
        ## Retained 4B — exact export, saved-probe parity, optional timing
        Original codes, scales, codebooks and saved signs are copied, never
        retrained or re-quantized. Non-text tensors remain in a counted sidecar;
        this runtime executes **text only**, not the vision tower.

        Local exports can occupy about 3.5–3.6GB per arm plus temporary files.
        They are disposable copies. The canonical Drive checkpoints are unchanged.
        Timing uses generated repeated-token inputs only to measure execution;
        no accuracy or competitor speed claim can be made from those numbers.
        '''),
        code('''
        # Recheck every prerequisite so jumping directly to this cell cannot bypass a failed gate.
        from scripts.check_rq3_model import runtime_identity
        current_runtime = runtime_identity(LIBRARY)
        for name in ("operators-CPU.json", "operators-CUDA0.json", "model-w6.json", "model-w8.json", "hf-conversion.json"):
            gate = json.loads((RESULT_ROOT / name).read_text())
            assert gate["passed"], f"Failed prerequisite: {name}"
            assert gate["runtime_files"] == current_runtime, f"Runtime changed since {name}; use a new run tag and repeat checks."
        if RUN_RETAINED_MODEL:
            for arm in ARMS:
                exported = EXPORT_ROOT / arm
                script("export_rotquant_gguf_v2.py", SOURCE_ROOT / arm / "checkpoint", exported,
                       "--llama-cpp-dir", LLAMA_DIR, label=f"export-{arm}", phase_limit=900)
                args = ["--library", LIBRARY, "--backend", "CUDA0", "--source-arm", SOURCE_ROOT / arm,
                        "--export", exported, "--output-dir", RESULT_ROOT / arm]
                if RUN_TIMING:
                    args.append("--timing")
                script("run_rq3_retained_gpu.py", *args, label=f"retained-{arm}", phase_limit=900)
                shutil.copy2(exported / "export.json", RESULT_ROOT / arm / "export.json")
        else:
            print("Synthetic validation only; retained 4B was not tested.")
        '''),
        md('''
        ## Next steps
        Share this new result directory and its logs. A pass establishes only
        bounded native parity on this hardware. A failure must be investigated;
        do not relax thresholds or restart the old public-task sweep.

        Before a longer run we still need measured model timings/memory, a matched
        native baseline, a runtime-bound evaluation identity and an agreed budget.
        Stop/disconnect the paid Colab runtime now. Keep the original checkpoints.
        '''),
        code('''
        for path in sorted(RESULT_ROOT.rglob("report.json")):
            row = json.loads(path.read_text())
            print(path.parent.name, {key: row.get(key) for key in ("passed", "parity", "memory", "error")})
        archive = shutil.make_archive(str(RESULT_ROOT) + "-reports", "zip", RESULT_ROOT)
        print("Download reports (no weights):", archive)
        print("Stop the GPU runtime to avoid further charges.")
        '''),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"})


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    output = Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_gpu_validation_colab.ipynb"
    nbformat.write(notebook, output)
    print(output)
