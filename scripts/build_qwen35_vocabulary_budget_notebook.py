#!/usr/bin/env python3
"""Generate the vocabulary-budget Colab directly (no inherited string rewrites)."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path("notebooks/qwen35_4b_vocabulary_budget_colab.ipynb")


def md(value):
    return new_markdown_cell(dedent(value).strip())


def code(value):
    return new_code_cell(dedent(value).strip())


def build_notebook():
    cells = [
        md("""
        # RotQuant Qwen3.5-4B vocabulary-budget experiment

        ## Goal
        Test whether compressing the tied vocabulary makes room for a better
        backbone at the Unsloth bundle's total size. Nine seed-0 arms cross
        FP16/W4/W5 backbones with FP16/W8/W6 vocabulary. The source teacher is
        captured before mutation. Vocabulary W6/W8 uses FP16 scale metadata.

        **This first run is a quality screen, not a compressed-model claim.**
        Dense vocabulary reconstruction stays live; projected artifact bytes
        cannot pass promotion. No recovery, KV, A8, allocator sweep or final
        300-prompt evaluation is launched automatically. Existing small suites
        are reused for historical continuity, not as an untouched benchmark.

        Run top to bottom on an A100 40 GB-class GPU. The notebook is generated
        and CPU-validated locally; full execution requires Colab/CUDA and model
        downloads. Use a fresh runtime. Keep Drive persistence enabled.
        """),
        md("## 1. Settings"),
        code("""
        from pathlib import Path

        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # Resolves to an immutable commit below; code must be published first.
        REPO_DIR = Path("/content/rotquant-vocabulary-budget")
        USE_GOOGLE_DRIVE = True
        RESULT_BASE = Path("/content/drive/MyDrive/rotquant/qwen35_vocabulary_budget")
        SEEDS = (0,)
        RUN_SCREEN = True
        FORCE_RERUN = False
        REQUIRE_FAST_HADAMARD = True
        USE_HF_SECRET = False  # Optional; public downloads work without authentication.
        RUN_PROVIDER_HEADER_AUDIT = True  # Header only; does not run Unsloth inference.
        DOWNLOAD_RESULTS = True
        RUNNER_HEARTBEAT_SECONDS = 60
        """),
        md("## 2. GPU, Drive and immutable checkout"),
        code("""
        import json
        import os
        import subprocess
        import sys
        import time
        import torch

        assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU."
        gpu = torch.cuda.get_device_properties(0)
        assert gpu.total_memory >= 35 * 2**30, "Use an A100 40 GB-class GPU for this screen."
        if USE_GOOGLE_DRIVE:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
        else:
            RESULT_BASE = Path("/content/rotquant-vocabulary-results")
            print("WARNING: local results and caches disappear when this runtime is lost.")

        def git(*args):
            result = subprocess.run(["git", *args], cwd=REPO_DIR if REPO_DIR.exists() else None,
                                    capture_output=True, text=True)
            if result.returncode:
                print(result.stdout, result.stderr)
                result.check_returncode()
            return result.stdout.strip()

        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        assert (REPO_DIR / ".git").is_dir(), "REPO_DIR is not a Git checkout."
        assert not git("status", "--porcelain"), "Use a clean or fresh REPO_DIR."
        git("fetch", "origin", REPO_REF)
        git("checkout", "--detach", "FETCH_HEAD")
        COMMIT = git("rev-parse", "HEAD")
        required = REPO_DIR / "scripts/run_qwen35_vocabulary_budget.py"
        assert required.exists(), "This revision does not contain the new experiment; publish/select it first."
        RESULT_ROOT = RESULT_BASE / COMMIT[:12]
        LOG_ROOT = RESULT_ROOT / "logs"
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        print({"commit": COMMIT, "gpu": gpu.name, "memory_gib": gpu.total_memory / 2**30,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "result_root": str(RESULT_ROOT), "seeds": SEEDS}, flush=True)
        """),
        md("## 3. Persistent live logs"),
        code("""
        import contextlib
        import signal

        def run_live(command, label, timeout_seconds=None):
            path = LOG_ROOT / f"{label}.log"
            print("Running:", " ".join(map(str, command)), flush=True)
            print("Persistent log:", path, flush=True)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Do not pipe the runner through the notebook: it writes directly to
            # Drive, so a lost browser connection cannot block a full stdout pipe.
            started = time.monotonic()
            with path.open("a", buffering=1) as sink, path.open("r") as reader:
                reader.seek(0, 2)
                process = subprocess.Popen(command, cwd=REPO_DIR, env=env,
                                           stdout=sink, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                print({"pid": process.pid, "tail_command": f'tail -f "{path}"'}, flush=True)
                heartbeat = started
                try:
                    while True:
                        text = reader.read()
                        if text:
                            print(text, end="", flush=True)
                        if process.poll() is not None:
                            print(reader.read(), end="", flush=True)
                            break
                        now = time.monotonic()
                        if timeout_seconds and now - started > timeout_seconds:
                            raise TimeoutError(f"{label} exceeded {timeout_seconds}s; see {path}")
                        if now - heartbeat >= 30:
                            print(f"notebook heartbeat {label}: {(now-started)/60:.1f}m; pid={process.pid}", flush=True)
                            heartbeat = now
                        time.sleep(1)
                except BaseException:
                    # Explicit cell interruption stops the whole subprocess group,
                    # including compilers; reconnecting the browser does not.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            return path
        """),
        md("## 4. Install the tested experiment dependencies"),
        code("""
        packages = [
            "transformers==5.9.0", "datasets==4.8.5", "accelerate==1.13.0",
            "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
            "pyyaml==6.0.3", "pandas==2.3.3", "huggingface_hub==1.17.0",
            "ninja==1.13.0", "nbformat==5.10.4",
        ]
        run_live([sys.executable, "-m", "pip", "install", *packages], "install")
        run_live([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "--no-deps"], "install-project")
        # Keep Colab's GPU-compatible torch; never downgrade it as a side effect.
        os.environ.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
        try:
            probe = subprocess.run([sys.executable, "-c",
                "import torch; from fast_hadamard_transform import hadamard_transform; "
                "x=hadamard_transform(torch.ones(1,128,device='cuda',dtype=torch.float16),scale=128**-0.5); "
                "assert torch.isfinite(x).all()"], capture_output=True, text=True)
            if probe.returncode:
                run_live([sys.executable, "-m", "pip", "install", "-v", "--no-deps", "--no-build-isolation",
                          "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0.post2"],
                         "fast-hadamard-build", timeout_seconds=1200)
            run_live([sys.executable, "-c",
                "import torch; from fast_hadamard_transform import hadamard_transform; "
                "x=hadamard_transform(torch.ones(1,128,device='cuda',dtype=torch.float16),scale=128**-0.5); "
                "assert torch.isfinite(x).all(); print('Fast Hadamard CUDA smoke passed')"], "kernel-smoke")
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
        except (subprocess.CalledProcessError, TimeoutError):
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
            if REQUIRE_FAST_HADAMARD:
                raise RuntimeError(f"Fast kernel failed; inspect {LOG_ROOT}. Set REQUIRE_FAST_HADAMARD=False only to accept a slow fallback.")
            print("WARNING: using the slower torch Hadamard fallback.")
        """),
        md("## 5. Optional Hugging Face authentication"),
        code("""
        # Public weights work without a token. Never print secrets or store them in results.
        # To authenticate, use a Colab secret named HF_TOKEN with notebook access enabled.
        if USE_HF_SECRET:
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
                if token:
                    os.environ["HF_TOKEN"] = token
                del token
            except Exception:
                print("No Colab HF_TOKEN available; public downloads will be unauthenticated.")
        else:
            print("Secret lookup skipped. Set USE_HF_SECRET=True to use a Colab HF_TOKEN secret.")
        """),
        md("## 6. Audit the pinned provider's tensor-byte layout"),
        code("""
        if RUN_PROVIDER_HEADER_AUDIT:
            provider_url = (
                "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
                "e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-UD-Q4_K_XL.gguf"
            )
            run_live([sys.executable, "-u", str(REPO_DIR / "scripts/inspect_gguf_types.py"),
                      "--url", provider_url, "--head-bytes", str(16 * 2**20), "--json",
                      "--output", str(RESULT_ROOT / "unsloth_tensor_ledger.json")], "provider-header")
            print("Saved actual GGUF tensor types and header hash; no new Unsloth quality result is implied.")
        """),
        md("## 7. Dry-run and correctness preflight"),
        code("""
        command = [sys.executable, "-u", str(REPO_DIR / "scripts/run_qwen35_vocabulary_budget.py"),
                   "--output-dir", str(RESULT_ROOT), "--heartbeat-seconds", str(RUNNER_HEARTBEAT_SECONDS)]
        for seed in SEEDS:
            command.extend(["--seed", str(seed)])
        run_live([*command, "--dry-run"], "plan")
        run_live([sys.executable, "-u", str(REPO_DIR / "scripts/preflight_vocabulary_budget.py"),
                  "--device", "cuda", "--output", str(RESULT_ROOT / "runtime_preflight.json")], "preflight")
        print("Preflight passed. The next cell is the GPU experiment.", flush=True)
        """),
        md("## 8. Run or resume the nine-arm screen"),
        code("""
        if RUN_SCREEN:
            active_command = [*command, *(["--force"] if FORCE_RERUN else [])]
            run_live(active_command, "vocabulary-screen")
        else:
            print("Screen disabled. Set RUN_SCREEN=True when ready.")
        """),
        md("""
        ## 9. Checks and interpretation
        Completed arms and vocabulary chunks resume by hashes. Source Hessians
        also resume by layer. An interrupted backbone patch is rebuilt from
        those Hessians; completed evaluations are not repeated. Only one runner
        may write this result root. In the Colab terminal use the printed
        `tail -f` command, or inspect `progress.json` and `nvidia-smi`.
        """),
        code("""
        import pandas as pd

        summary_path = RESULT_ROOT / "screen_summary.json"
        if summary_path.exists():
            check = [sys.executable, "-u", str(REPO_DIR / "scripts/assess_qwen35_vocabulary_budget.py"),
                     "--input-dir", str(RESULT_ROOT / "model_trials"), "--output", str(summary_path)]
            for seed in SEEDS:
                check.extend(["--seed", str(seed)])
            run_live(check, "validate-screen")
            summary = json.loads(summary_path.read_text())
            display(pd.DataFrame(summary["rows"])[[
                "arm", "seed", "mean_teacher_kl", "top1_agreement", "diverse_mean_teacher_kl",
                "diverse_trajectory_token_agreement", "ppl_wikitext2", "projected_artifact_bytes"]])
            print({key: summary[key] for key in ("complete", "finalists", "next_step", "interpretation")})
            print("Storage figures above are PROJECTED. No artifact or provider winner is promoted.")
        else:
            print("No summary yet; run the screen first.")
        """),
        md("## 10. Download the compact evidence"),
        code("""
        import zipfile

        if DOWNLOAD_RESULTS and (RESULT_ROOT / "screen_summary.json").exists():
            archive = Path("/content") / f"qwen35_vocabulary_budget_{COMMIT[:12]}.zip"
            # Deliberate allow-list: exclude weights, token caches and multi-GB Hessians.
            paths = [*RESULT_ROOT.glob("*.json"), *(RESULT_ROOT / "model_trials").glob("*.json"),
                     *(RESULT_ROOT / "model_trials").glob("*.sha256"), *LOG_ROOT.glob("*.log")]
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(paths):
                    bundle.write(path, path.relative_to(RESULT_ROOT))
            print("Evidence bundle:", archive)
            from google.colab import files
            files.download(str(archive))
        """),
        md("""
        ## Next steps
        Bring back the compact bundle. A promising W5/W8 or W5/W6 row advances
        to real shared packed export and independent confirmation, not straight
        to a provider claim. Fresh validation/final manifests and rerun Unsloth
        scores are required for the later comparison. If neither row passes,
        inspect vocabulary-only damage before spending on allocation or recovery.
        """),
    ]
    # Stable IDs keep generated notebooks reviewable and reproducible.
    for index, cell in enumerate(cells):
        cell.id = f"vocabulary-{index:02d}"
    return new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": OUTPUT.name}})


def main():
    notebook = build_notebook()
    nbformat.validate(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
