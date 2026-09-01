#!/usr/bin/env python3
"""Generate the focused Qwen3.5-4B optimization Colab notebook."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path("notebooks/qwen35_4b_optimization_stage_colab.ipynb")


def md(source: str):
    return new_markdown_cell(dedent(source).strip())


def code(source: str):
    return new_code_cell(dedent(source).strip())


def build_notebook():
    cells = [
        md("""
        # RotQuant Qwen3.5-4B optimization stage

        This is the focused follow-up to the completed Algorithm Lab. It does
        not repeat the broad screen. The default run compares the validated
        calibrated and Gaussian W4 controls with streamed GPTQ under identical
        held-out PPL, token-distribution, and 32-token trajectory checks.
        """),
        md("""
        ## What the notebook can run

        - **W4 (default):** source FP16, Gaussian W4, calibrated W4, and both
          matched streamed-GPTQ variants.
        - **W4A8/E8 (opt-in):** anchors on promoted Gaussian FWHT+GPTQ W4;
          separately tests optimized weight-only composition and A8, then adds
          finite-rate E8 KV to the matched A8 arm.
        - **Recovery (opt-in, very expensive):** a resumable million-token
          block-reconstruction and distillation protocol.
        - **Long KV (opt-in):** four disjoint 8k-prefill/64-token confirmations.

        Every completed arm is stored in Google Drive. Resumption requires the
        same Git commit and fully resolved configuration; stale results are not
        silently mixed with a changed implementation.
        """),
        md("## 1. Settings"),
        code("""
        from pathlib import Path

        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # branch, tag, or exact commit
        REPO_DIR = Path("/content/rotquant-next-stage")

        USE_GOOGLE_DRIVE = True
        DRIVE_RESULT_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_next_stage")
        LOCAL_RESULT_ROOT = Path("/content/rotquant_qwen35_next_stage")

        RUN_W4 = True
        RUN_W4A8 = False
        RUN_RECOVERY = False       # deliberately opt-in: >=1M unique train tokens
        RUN_LONG_CONTEXT_KV = False
        SEEDS = (0,)               # use (0, 1, 2) only after seed 0 promotes
        FORCE_RERUN = False
        REQUIRE_FAST_HADAMARD = True
        DOWNLOAD_RESULTS = True

        print({
            "repo_ref": REPO_REF,
            "run_w4": RUN_W4,
            "run_w4a8": RUN_W4A8,
            "run_recovery": RUN_RECOVERY,
            "run_long_context_kv": RUN_LONG_CONTEXT_KV,
            "seeds": SEEDS,
        })
        """),
        md("## 2. CUDA, Drive, and an immutable checkout"),
        code("""
        import subprocess
        import sys

        import torch

        assert torch.cuda.is_available(), "Select a CUDA GPU runtime first."
        gpu = torch.cuda.get_device_properties(0)
        print({
            "gpu": gpu.name,
            "memory_gib": round(gpu.total_memory / 2**30, 1),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version,
        })
        assert gpu.total_memory >= 35 * 2**30, (
            "Use an A100 40 GB (or larger) for the matched W4/GPTQ ladder."
        )

        if USE_GOOGLE_DRIVE:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
            else:
                print("Google Drive is already mounted.")
            RESULT_BASE = DRIVE_RESULT_ROOT
        else:
            RESULT_BASE = LOCAL_RESULT_ROOT
        RESULT_BASE.mkdir(parents=True, exist_ok=True)

        def run_git(arguments, *, cwd=None):
            result = subprocess.run(
                ["git", *arguments], cwd=cwd, capture_output=True, text=True,
                check=False,
            )
            if result.returncode:
                print(result.stdout)
                print(result.stderr)
                result.check_returncode()
            return result

        if not REPO_DIR.exists():
            run_git(["clone", REPO_URL, str(REPO_DIR)])
        elif not (REPO_DIR / ".git").is_dir():
            raise RuntimeError(
                f"{REPO_DIR} exists but is not a Git checkout; choose a fresh path."
            )

        # Fetching then detaching FETCH_HEAD works for a branch, tag, or commit
        # and avoids the deleted-feature-branch failure of clone --branch.
        run_git(["fetch", "--force", "origin", REPO_REF], cwd=REPO_DIR)
        run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        RESULT_ROOT = RESULT_BASE / commit[:12]
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Using commit {commit}; results: {RESULT_ROOT}")
        """),
        md("## 3. Install the pinned evaluation runtime and fast FWHT"),
        code("""
        import importlib
        import os

        runtime_packages = [
            "transformers==5.9.0", "datasets==4.8.5", "accelerate==1.13.0",
            "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
            "pyyaml==6.0.3", "pandas==2.3.3", "matplotlib==3.10.9",
            "huggingface_hub==1.17.0", "ninja==1.13.0", "nbformat==5.10.4",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", *runtime_packages],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR), "--no-deps"],
            check=True,
        )

        # The Python 3.13/CUDA combination used by current Colab has no matching
        # v1.1.0.post2 release wheel. The completed A100 run succeeded with this
        # bounded source build, so avoid a known 404 and build it directly.
        fht_release = "v1.1.0.post2"
        kernel_env = os.environ.copy()
        kernel_env.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
        kernel_build = subprocess.run([
            sys.executable, "-m", "pip", "install", "-v", "--no-deps",
            "--no-build-isolation",
            f"git+https://github.com/Dao-AILab/fast-hadamard-transform.git@{fht_release}",
        ], check=False, env=kernel_env)
        fast_hadamard_available = kernel_build.returncode == 0
        fast_hadamard_error = None
        if fast_hadamard_available:
            try:
                importlib.invalidate_caches()
                from fast_hadamard_transform import hadamard_transform
                with torch.no_grad():
                    smoke_input = torch.randn(2, 128, device="cuda", dtype=torch.float16)
                    smoke_output = hadamard_transform(smoke_input.contiguous())
                    torch.cuda.synchronize()
                    assert smoke_output.shape == smoke_input.shape
                    assert bool(torch.isfinite(smoke_output).all().item())
                del smoke_input, smoke_output
                print("Fast Hadamard CUDA smoke test passed via source build.")
            except Exception as exc:
                fast_hadamard_available = False
                fast_hadamard_error = f"{type(exc).__name__}: {exc}"
        else:
            fast_hadamard_error = f"source build exited with {kernel_build.returncode}"

        if fast_hadamard_available:
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
        else:
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
        if REQUIRE_FAST_HADAMARD:
            assert fast_hadamard_available, (
                "fast-hadamard-transform could not be built or launched: "
                f"{fast_hadamard_error}. Set REQUIRE_FAST_HADAMARD=False only "
                "for a much slower diagnostic run."
            )
        elif not fast_hadamard_available:
            print("WARNING: using the slow pure-torch FWHT fallback.")

        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("Runtime ready without replacing Colab's CUDA PyTorch wheel.")
        """),
        md("## 4. Validate and display the planned arms"),
        code("""
        required = [
            REPO_DIR / "scripts/run_qwen35_next_stage.py",
            REPO_DIR / "configs/qwen35_4b_gptq_cuda.yaml",
            REPO_DIR / "configs/qwen35_4b_w4a8_e8_trials_cuda.yaml",
            REPO_DIR / "configs/qwen35_4b_recovery_cuda.yaml",
            REPO_DIR / "configs/qwen35_4b_long_context_kv_cuda.yaml",
        ]
        missing = [str(path) for path in required if not path.exists()]
        assert not missing, "Missing required files: " + ", ".join(missing)

        selected_stages = []
        if RUN_W4:
            selected_stages.append("w4")
        if RUN_W4A8:
            selected_stages.append("w4a8")
        if RUN_RECOVERY:
            selected_stages.append("recovery")
        if RUN_LONG_CONTEXT_KV:
            selected_stages.append("long-kv")
        assert selected_stages, "Enable at least one stage."

        dry_command = [
            sys.executable, str(REPO_DIR / "scripts/run_qwen35_next_stage.py"),
            "--output-dir", str(RESULT_ROOT), "--dry-run",
        ]
        for stage in selected_stages:
            dry_command.extend(["--stage", stage])
        for seed in SEEDS:
            dry_command.extend(["--seed", str(seed)])
        subprocess.run(dry_command, cwd=REPO_DIR, check=True)
        """),
        md("## 5. Run or resume the selected stages"),
        code("""
        import os

        command = [
            sys.executable, "-u",
            str(REPO_DIR / "scripts/run_qwen35_next_stage.py"),
            "--output-dir", str(RESULT_ROOT),
        ]
        for stage in selected_stages:
            command.extend(["--stage", stage])
        for seed in SEEDS:
            command.extend(["--seed", str(seed)])
        if FORCE_RERUN:
            command.append("--force")
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        """),
        md("## 6. Inspect quality, divergence, storage, and memory together"),
        code("""
        import json

        import pandas as pd
        from IPython.display import display

        summary_path = RESULT_ROOT / "next_stage_summary.json"
        summary = json.loads(summary_path.read_text())
        frame = pd.DataFrame(summary["rows"])
        display(frame)

        quality_columns = [column for column in [
            "stage", "arm", "seed",
            "ppl_wikitext2_relative_to_source", "ppl_c4_relative_to_source",
            "mean_teacher_kl", "p95_teacher_kl", "top1_agreement",
            "trajectory_token_agreement", "exact_trajectory_rate",
            "mean_matching_prefix", "complete_persistent_model_bytes",
            "peak_vram_bytes_patch", "peak_vram_bytes_eval",
        ] if column in frame]
        display(frame[quality_columns])

        print("Paired candidate-minus-control bootstrap reports:")
        print(json.dumps(summary.get("paired_comparisons", []), indent=2))

        print(
            "Promotion is paired: compare each GPTQ arm with the control using "
            "the same codebook. Do not infer a win from PPL alone; require KL, "
            "top-1, and free-running trajectory evidence to agree."
        )
        """),
        md("## 7. Preserve a downloadable bundle"),
        code("""
        import shutil

        archive = shutil.make_archive(
            f"/content/qwen35_next_stage_{commit[:12]}", "zip",
            root_dir=RESULT_ROOT,
        )
        print(f"Created {archive}; Drive results remain at {RESULT_ROOT}")
        if DOWNLOAD_RESULTS:
            from google.colab import files
            files.download(archive)
        """),
        md("""
        ## Interpretation boundary

        These fixed WikiText/C4 and held-out C4 trajectory metrics decide which
        optimizer variants deserve the expensive frozen 300-prompt run. They do
        not yet establish parity with Unsloth Dynamic 3.0. That comparison needs
        the licensed five-domain manifest, 32-token teacher KL/trajectory records,
        task-outcome scorers, and deployed artifacts matched within 1% of size.
        """),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
