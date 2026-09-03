#!/usr/bin/env python3
"""Generate the resumable Qwen3.5-4B allocator-v2 Colab experiment."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path("notebooks/qwen35_4b_allocator_v2_colab.ipynb")


def md(value: str):
    return new_markdown_cell(dedent(value).strip())


def code(value: str):
    return new_code_cell(dedent(value).strip())


def build_notebook():
    cells = [
        md("""
        # RotQuant Qwen3.5-4B allocator-v2 experiment

        This run replaces the failed RMS/no-GPTQ mixed-precision proxy. It
        scores the exact deployed MSE-search + GPTQ candidates, measures
        single-projection teacher KL, solves the complete-byte rate problem
        with a Pareto dynamic program, and compares adjacent, broad, protected,
        and random policies. All policies reuse one expensive candidate table.

        Seed 0 is a screen. Only recipes that beat the matched random control
        proceed to seeds 1 and 2. Progress and persistent logs are printed
        throughout. Use an A100 40 GB or larger.
        """),
        md("## 1. Settings"),
        code("""
        from pathlib import Path

        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # Prefer an exact merged commit for a frozen run.
        REPO_DIR = Path("/content/rotquant-allocator-v2")

        USE_GOOGLE_DRIVE = True
        DRIVE_RESULT_ROOT = Path(
            "/content/drive/MyDrive/rotquant/qwen35_allocator_v2"
        )
        LOCAL_RESULT_ROOT = Path("/content/rotquant_qwen35_allocator_v2")

        RUN_SCREEN = True
        RUN_CONFIRMATION = True
        RUN_UNSLOTH_KL = True
        RUN_FINAL_COMPARISON = True
        SCREEN_SEED = 0
        CONFIRM_SEEDS = (0, 1, 2)
        FORCE_RERUN = False
        EXPORT_FINALIST_SEED0 = True
        PERSIST_GGUF_IN_DRIVE = False
        REQUIRE_FAST_HADAMARD = True
        DOWNLOAD_RESULTS = True
        NOTEBOOK_HEARTBEAT_SECONDS = 30
        RUNNER_HEARTBEAT_SECONDS = 60

        print({
            "repo_ref": REPO_REF,
            "screen": RUN_SCREEN,
            "confirmation": RUN_CONFIRMATION,
            "unsloth_kl": RUN_UNSLOTH_KL,
            "confirm_seeds": CONFIRM_SEEDS,
        })
        """),
        md("## 2. GPU, Drive, and immutable checkout"),
        code("""
        import json
        import os
        import subprocess
        import sys
        import time

        import torch

        assert torch.cuda.is_available(), "Select a CUDA GPU runtime first."
        gpu = torch.cuda.get_device_properties(0)
        assert gpu.total_memory >= 35 * 2**30, (
            "Use an A100 40 GB or larger for this experiment."
        )
        print({
            "gpu": gpu.name,
            "memory_gib": round(gpu.total_memory / 2**30, 1),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version,
        })

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
                ["git", *arguments], cwd=cwd, capture_output=True, text=True
            )
            if result.returncode:
                print(result.stdout)
                print(result.stderr)
                result.check_returncode()
            return result

        if not REPO_DIR.exists():
            run_git(["clone", REPO_URL, str(REPO_DIR)])
        elif not (REPO_DIR / ".git").is_dir():
            raise RuntimeError(f"{REPO_DIR} is not a Git checkout")
        if run_git(["status", "--porcelain"], cwd=REPO_DIR).stdout.strip():
            raise RuntimeError("Use a fresh REPO_DIR; checkout has local changes.")
        run_git(["fetch", "--force", "origin", REPO_REF], cwd=REPO_DIR)
        run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()

        RESULT_ROOT = RESULT_BASE / commit[:12]
        MODEL_RESULT_ROOT = RESULT_ROOT / "model_trials"
        SUMMARY_ROOT = RESULT_ROOT / "phase_summaries"
        LOG_ROOT = RESULT_ROOT / "logs"
        ARTIFACT_ROOT = RESULT_ROOT / "artifacts"
        UNSLOTH_ROOT = RESULT_ROOT / "unsloth_kl"
        TOKEN_CACHE_ROOT = RESULT_BASE / "token_cache"
        DYNAMIC_SCORE_CACHE_ROOT = RESULT_BASE / "dynamic_score_cache"
        for directory in (
            RESULT_ROOT, MODEL_RESULT_ROOT, SUMMARY_ROOT, LOG_ROOT,
            ARTIFACT_ROOT, UNSLOTH_ROOT, TOKEN_CACHE_ROOT,
            DYNAMIC_SCORE_CACHE_ROOT,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ["ROTQUANT_TOKEN_CACHE_DIR"] = str(TOKEN_CACHE_ROOT)
        os.environ["ROTQUANT_DYNAMIC_SCORE_CACHE_DIR"] = str(
            DYNAMIC_SCORE_CACHE_ROOT
        )
        print(f"Using commit {commit}")
        print(f"Results: {RESULT_ROOT}")
        """),
        md("## 3. Install and validate the CUDA runtime"),
        code("""
        import importlib

        packages = [
            "transformers==5.9.0", "datasets==4.8.5", "accelerate==1.13.0",
            "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
            "pyyaml==6.0.3", "pandas==2.3.3", "matplotlib==3.10.9",
            "huggingface_hub==1.17.0", "ninja==1.13.0", "nbformat==5.10.4",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", *packages],
            check=True,
        )
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-e",
            str(REPO_DIR), "--no-deps",
        ], check=True)

        kernel_env = os.environ.copy()
        kernel_env.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
        print("Building fast-hadamard-transform...", flush=True)
        build = subprocess.run([
            sys.executable, "-m", "pip", "install", "-v", "--no-deps",
            "--no-build-isolation",
            "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0.post2",
        ], env=kernel_env)
        fast_hadamard_available = build.returncode == 0
        error = None
        if fast_hadamard_available:
            try:
                importlib.invalidate_caches()
                from fast_hadamard_transform import hadamard_transform
                smoke = torch.randn(2, 128, device="cuda", dtype=torch.float16)
                transformed = hadamard_transform(smoke.contiguous())
                torch.cuda.synchronize()
                assert transformed.shape == smoke.shape
                assert torch.isfinite(transformed).all()
                del smoke, transformed
            except Exception as exc:
                fast_hadamard_available = False
                error = f"{type(exc).__name__}: {exc}"
        if fast_hadamard_available:
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
            print("Fast Hadamard CUDA smoke test passed.")
        else:
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
        if REQUIRE_FAST_HADAMARD:
            assert fast_hadamard_available, f"Fast Hadamard failed: {error}"

        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        environment = {
            "commit": commit,
            "gpu": gpu.name,
            "gpu_memory_bytes": gpu.total_memory,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fast_hadamard": fast_hadamard_available,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (RESULT_ROOT / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        """),
        md("## 4. Live progress and resumable commands"),
        code("""
        import queue
        import shutil
        import threading

        def gpu_snapshot():
            try:
                return subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=5).strip().splitlines()[0]
            except Exception:
                return "status unavailable"

        def run_live(command, *, label):
            log_path = LOG_ROOT / f"{label}.log"
            print("Running:", " ".join(map(str, command)), flush=True)
            print(f"Persistent log: {log_path}", flush=True)
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command, cwd=REPO_DIR, env=child_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            messages = queue.Queue()
            def reader():
                for line in process.stdout:
                    messages.put(line)
                messages.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            started = time.monotonic()
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                while True:
                    try:
                        line = messages.get(timeout=NOTEBOOK_HEARTBEAT_SECONDS)
                    except queue.Empty:
                        elapsed = (time.monotonic() - started) / 60
                        line = (
                            f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                            f"notebook heartbeat {label}: elapsed={elapsed:.1f}m "
                            f"gpu={gpu_snapshot()}\\n"
                        )
                        print(line, end="", flush=True)
                        log.write(line)
                        continue
                    if line is None:
                        break
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = process.wait()
            thread.join(timeout=2)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            return log_path

        def stage_command(seeds, *, arms=None, export_arms=None):
            command = [
                sys.executable, "-u",
                str(REPO_DIR / "scripts/run_qwen35_next_stage.py"),
                "--output-dir", str(MODEL_RESULT_ROOT),
                "--stage", "allocator-v2",
                "--heartbeat-seconds", str(RUNNER_HEARTBEAT_SECONDS),
            ]
            for seed in seeds:
                command.extend(["--seed", str(seed)])
            for arm in arms or ():
                command.extend(["--arm", arm])
            if export_arms:
                command.extend(["--artifact-dir", str(ARTIFACT_ROOT)])
                command.extend(["--export-seed", "0", "--export-processor"])
                for arm in export_arms:
                    command.extend(["--export-arm", arm])
            if FORCE_RERUN:
                command.append("--force")
            return command

        def snapshot_summary(name):
            source = MODEL_RESULT_ROOT / "next_stage_summary.json"
            assert source.exists(), f"Runner did not create {source}"
            destination = SUMMARY_ROOT / f"{name}.json"
            shutil.copy2(source, destination)
            csv_source = MODEL_RESULT_ROOT / "next_stage_summary.csv"
            if csv_source.exists():
                shutil.copy2(csv_source, SUMMARY_ROOT / f"{name}.csv")
            print(f"Saved {destination}")
            return destination
        """),
        md("## 5. Dry-run the registered plan"),
        code("""
        required = [
            "scripts/run_qwen35_next_stage.py",
            "scripts/select_qwen35_allocator_v2_finalists.py",
            "scripts/compare_qwen35_dynamic_to_unsloth.py",
            "scripts/run_unsloth_qwen35_4b_kl.py",
            "configs/qwen35_4b_allocator_v2_cuda.yaml",
            "research/eval_suites/qwen35_diverse_development_v1.json",
        ]
        missing = [name for name in required if not (REPO_DIR / name).exists()]
        assert not missing, "Missing required files: " + ", ".join(missing)
        subprocess.run(
            [*stage_command((SCREEN_SEED,)), "--dry-run"],
            cwd=REPO_DIR, check=True,
        )
        print("Plan validation passed; no model was loaded.")
        """),
        md("## 6. Phase A — faithful seed-0 candidate screen"),
        code("""
        SCREEN_SUMMARY = SUMMARY_ROOT / "allocator_v2_screen.json"
        if RUN_SCREEN:
            run_live(
                stage_command((SCREEN_SEED,)),
                label="phase-a-allocator-v2-screen",
            )
            SCREEN_SUMMARY = snapshot_summary("allocator_v2_screen")
        else:
            assert SCREEN_SUMMARY.exists(), "No persisted allocator-v2 screen"
        """),
        md("## 7. Select finalists"),
        code("""
        import pandas as pd
        from IPython.display import display

        selection_path = RESULT_ROOT / "allocator_v2_finalists.json"
        subprocess.run([
            sys.executable, "-u",
            str(REPO_DIR / "scripts/select_qwen35_allocator_v2_finalists.py"),
            "--summary", str(SCREEN_SUMMARY),
            "--output", str(selection_path),
            "--seed", str(SCREEN_SEED),
        ], cwd=REPO_DIR, check=True)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        FINALIST_ARMS = tuple(selection["finalists"])
        CONFIRMATION_ARMS = tuple(selection["confirmation_arms"])
        display(pd.DataFrame(selection["decisions"]))
        print({
            "finalists": FINALIST_ARMS,
            "confirmation_arms": CONFIRMATION_ARMS,
            "interpretation": selection["interpretation"],
        })
        """),
        md("## 8. Phase B — seeds 0/1/2 confirmation and packed export"),
        code("""
        CONFIRM_SUMMARY = SUMMARY_ROOT / "allocator_v2_confirm.json"
        if RUN_CONFIRMATION and FINALIST_ARMS:
            export_arms = FINALIST_ARMS if EXPORT_FINALIST_SEED0 else ()
            run_live(
                stage_command(
                    CONFIRM_SEEDS,
                    arms=CONFIRMATION_ARMS,
                    export_arms=export_arms,
                ),
                label="phase-b-allocator-v2-confirm",
            )
            CONFIRM_SUMMARY = snapshot_summary("allocator_v2_confirm")
        elif RUN_CONFIRMATION:
            print("No seed-0 recipe passed; confirmation is intentionally empty.")
        else:
            print("Confirmation disabled.")
        """),
        md("## 9. Phase C — prompt-matched Unsloth anchor"),
        code("""
        def ensure_llama_cpp():
            probe = subprocess.run([
                sys.executable, "-c",
                "import diskcache, llama_cpp; print(llama_cpp.__version__)",
            ], capture_output=True, text=True)
            if probe.returncode == 0:
                print({"llama_cpp_python": probe.stdout.strip(), "status": "resume"})
                return
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "-U",
                "diskcache==5.6.3", "jinja2==3.1.6",
                "typing-extensions==4.15.0",
            ], check=True)
            llama_env = os.environ.copy()
            llama_env.update({
                "CMAKE_ARGS": "-DGGML_CUDA=on",
                "CMAKE_BUILD_PARALLEL_LEVEL": "2",
                "FORCE_CMAKE": "1",
            })
            revision = "3691546f1c9e0c1bf93323dff02230bd959cf562"
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-v", "--no-deps",
                f"git+https://github.com/abetlen/llama-cpp-python.git@{revision}",
            ], check=True, env=llama_env)
            importlib.invalidate_caches()
            subprocess.run([
                sys.executable, "-c", "import diskcache, llama_cpp"
            ], check=True)

        if RUN_UNSLOTH_KL:
            ensure_llama_cpp()
            gguf_dir = (
                RESULT_BASE / "artifacts/unsloth-qwen35-4b-gguf"
                if PERSIST_GGUF_IN_DRIVE
                else Path("/content/unsloth-qwen35-4b-gguf")
            )
            command = [
                sys.executable, "-u",
                str(REPO_DIR / "scripts/run_unsloth_qwen35_4b_kl.py"),
                "--output-dir", str(UNSLOTH_ROOT),
                "--artifact-dir", str(gguf_dir),
                "--batches", "24", "--prompt-len", "512", "--skip", "4096",
            ]
            if FORCE_RERUN:
                command.append("--force")
            run_live(command, label="phase-c-unsloth-kl")
        """),
        md("## 10. Same-input, same-byte comparison"),
        code("""
        COMPARISON_PATH = RESULT_ROOT / "allocator_v2_vs_unsloth.json"
        if RUN_FINAL_COMPARISON and FINALIST_ARMS:
            command = [
                sys.executable, "-u",
                str(REPO_DIR / "scripts/compare_qwen35_dynamic_to_unsloth.py"),
                "--summary", str(CONFIRM_SUMMARY),
                "--unsloth", str(UNSLOTH_ROOT / "unsloth_ud_q4_kl.json"),
                "--stage", "allocator-v2",
                "--output", str(COMPARISON_PATH),
            ]
            for arm in FINALIST_ARMS:
                command.extend(["--arm", arm])
            subprocess.run(command, cwd=REPO_DIR, check=True)
        else:
            print("Provider comparison skipped because there are no finalists.")
        """),
        md("## 11. Validate and display the decision record"),
        code("""
        def load_json(path):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        screen_summary = load_json(SCREEN_SUMMARY)
        confirm_summary = load_json(CONFIRM_SUMMARY)
        issues = []
        rows = (screen_summary or {}).get("rows", [])
        if (screen_summary or {}).get("code_revision") != commit:
            issues.append("screen code revision mismatch")
        if not (screen_summary or {}).get("complete"):
            issues.append("screen summary incomplete")
        if len(rows) != 9:
            issues.append(f"expected 9 screen rows, found {len(rows)}")
        for row in rows:
            if row["arm"].startswith(("random_", "pareto_")):
                if row.get("dynamic_actual_target_match") is not True:
                    issues.append(f"{row['arm']}: byte target missed")
                if row.get("dynamic_scoring_matches_deployed") is not True:
                    issues.append(f"{row['arm']}: scoring was not faithful")
                if row.get("dynamic_proxy_rank_correlation") is None:
                    issues.append(f"{row['arm']}: proxy audit missing")
            if row.get("logit_fidelity_tokens", 0) < 10_000:
                issues.append(f"{row['arm']}: fewer than 10k KL tokens")

        if RUN_CONFIRMATION and FINALIST_ARMS:
            identities = {
                (row["arm"], int(row["seed"]))
                for row in (confirm_summary or {}).get("rows", [])
            }
            expected = {
                (arm, seed) for arm in CONFIRMATION_ARMS
                for seed in CONFIRM_SEEDS
            }
            if expected - identities:
                issues.append(f"confirmation missing {sorted(expected - identities)}")

        comparison = load_json(COMPARISON_PATH)
        validation = {
            "commit": commit,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "ready_for_internal_decision" if not issues else "needs_revision",
            "issues": issues,
            "boundary": (
                "Internal 24-prompt and 25-prompt development suites only. "
                "Public claims require the registered 300-prompt engine-neutral run."
            ),
        }
        (RESULT_ROOT / "allocator_v2_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        print(json.dumps(validation, indent=2))
        assert not issues, "Validation failed; inspect the issue list."

        for label, summary in (
            ("Allocator-v2 screen", screen_summary),
            ("Allocator-v2 confirmation", confirm_summary),
        ):
            if not summary:
                continue
            frame = pd.DataFrame(summary.get("rows", []))
            if frame.empty:
                continue
            print(label)
            columns = [name for name in (
                "arm", "seed", "mean_teacher_kl", "top1_agreement",
                "ppl_wikitext2", "ppl_c4", "diverse_mean_teacher_kl",
                "diverse_top1_agreement", "diverse_trajectory_token_agreement",
                "complete_persistent_model_bytes", "dynamic_counts_by_bits",
                "dynamic_proxy_rank_correlation", "dynamic_solver",
                "dynamic_protected_layers", "dynamic_score_cache_hit",
                "dynamic_score_cache_source",
            ) if name in frame]
            display(frame[columns].sort_values(["arm", "seed"]))
        if comparison:
            print("RotQuant minus Unsloth")
            display(pd.DataFrame(comparison["comparisons"]))
        print({"finalists": FINALIST_ARMS, "result_root": str(RESULT_ROOT)})
        """),
        md("## 12. Download the compact result bundle"),
        code("""
        if DOWNLOAD_RESULTS:
            from google.colab import files
            archive_base = Path("/content") / f"qwen35_allocator_v2_{commit[:12]}"
            archive = shutil.make_archive(
                str(archive_base), "zip", root_dir=RESULT_ROOT, base_dir="."
            )
            print(f"Download bundle: {archive}")
            files.download(archive)
        else:
            print(f"Results remain in {RESULT_ROOT}")
        """),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "colab": {"name": OUTPUT.name, "provenance": []},
        },
    )


def main() -> None:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
