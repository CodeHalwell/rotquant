#!/usr/bin/env python3
"""Generate the full Qwen3.5-4B optimization experiment Colab notebook."""
from __future__ import annotations

import hashlib
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
        # RotQuant Qwen3.5-4B full optimization experiment

        This notebook runs the next decision-grade development experiment:
        seed-0 factor screening, pre-registered finalist selection, three-seed
        confirmation of viable W4 factors, corrected long-context E8P cache
        validation, and the pinned released-Unsloth Q4 anchor.

        **Unexecuted notebook:** conclusions are intentionally produced only
        after the saved records pass the validation cell near the end.
        """),
        md("""
        ## Goal and methods

        The main decision is whether any individual W4 factor improves on the
        promoted Gaussian FWHT+GPTQ control without hiding regressions in PPL,
        teacher KL, top-1 agreement, trajectories, or exact bytes.

        1. Run every factor once at seed 0 with the global catastrophic gate.
        2. Select replication candidates using committed, visible guardrails.
        3. Re-run the control and finalists at seeds 0/1/2; seed 0 resumes.
        4. Re-run the four-prompt 8k/64-token E8P engineering check on the fixed
           cache simulator and require the uniform-K8 endpoint to pass.
        5. Reproduce the released Unsloth UD-Q4_K_XL same-engine KL anchor.

        ### Key assumptions and boundaries

        - An A100 40 GB or larger is required. The complete experiment is
          intentionally resumable across Colab sessions; the previous 8k cache
          leg alone took about seven hours.
        - The four-prompt cache leg is an engineering check, not promotable
          statistical evidence. Clustered ≥20-prompt/≥10k-token evaluation still
          comes later.
        - Million-token recovery, the licensed 300-prompt competitive suite,
          mixed-precision allocation, and Qwen3.8-27B are not mixed into this run.
        """),
        md("## 1. Experiment settings"),
        code("""
        from pathlib import Path

        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # Use an exact commit here for a frozen run if desired.
        REPO_DIR = Path("/content/rotquant-full-experiment")

        USE_GOOGLE_DRIVE = True
        DRIVE_RESULT_ROOT = Path(
            "/content/drive/MyDrive/rotquant/qwen35_full_experiment"
        )
        LOCAL_RESULT_ROOT = Path("/content/rotquant_qwen35_full_experiment")

        RUN_ABLATION_SCREEN = True
        RUN_ABLATION_CONFIRM = True
        RUN_LONG_CONTEXT_KV = True
        RUN_UNSLOTH_KL = True

        SCREEN_SEED = 0
        CONFIRM_SEEDS = (0, 1, 2)
        FORCE_RERUN = False
        REQUIRE_FAST_HADAMARD = True
        DOWNLOAD_RESULTS = True

        # The notebook prints a heartbeat if a subprocess emits nothing for this
        # long. The runner independently reports arm progress and GPU state.
        NOTEBOOK_HEARTBEAT_SECONDS = 30
        RUNNER_HEARTBEAT_SECONDS = 60

        # Persisting GGUFs avoids redownloading ~12 GB after a runtime reset but
        # consumes Drive space. The BF16 prompt logits always persist in results.
        PERSIST_GGUF_IN_DRIVE = False

        print({
            "repo_ref": REPO_REF,
            "ablation_screen": RUN_ABLATION_SCREEN,
            "ablation_confirm": RUN_ABLATION_CONFIRM,
            "long_context_kv": RUN_LONG_CONTEXT_KV,
            "unsloth_kl": RUN_UNSLOTH_KL,
            "confirm_seeds": CONFIRM_SEEDS,
            "force_rerun": FORCE_RERUN,
        })
        """),
        md("## 2. CUDA, Drive, and immutable checkout"),
        code("""
        import json
        import os
        import subprocess
        import sys
        import time

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
            "Use an A100 40 GB (or larger) for this experiment."
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
                f"{REPO_DIR} exists but is not a Git checkout; use a fresh path."
            )
        dirty = run_git(["status", "--porcelain"], cwd=REPO_DIR).stdout.strip()
        if dirty:
            raise RuntimeError(
                "The Colab checkout has local changes. Use a fresh REPO_DIR so "
                "the recorded commit describes the code that actually ran."
            )

        # Fetch + detached FETCH_HEAD supports a branch, tag, or exact commit and
        # avoids failures when an old feature branch has been deleted.
        run_git(["fetch", "--force", "origin", REPO_REF], cwd=REPO_DIR)
        run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        RESULT_ROOT = RESULT_BASE / commit[:12]
        MODEL_RESULT_ROOT = RESULT_ROOT / "model_trials"
        SUMMARY_ROOT = RESULT_ROOT / "phase_summaries"
        LOG_ROOT = RESULT_ROOT / "logs"
        UNSLOTH_ROOT = RESULT_ROOT / "unsloth_kl"
        for directory in (
            RESULT_ROOT, MODEL_RESULT_ROOT, SUMMARY_ROOT, LOG_ROOT, UNSLOTH_ROOT
        ):
            directory.mkdir(parents=True, exist_ok=True)
        print(f"Using commit {commit}")
        print(f"Persistent results: {RESULT_ROOT}")
        """),
        md("## 3. Install the pinned model runtime and fast Hadamard kernel"),
        code("""
        import importlib

        runtime_packages = [
            "transformers==5.9.0", "datasets==4.8.5", "accelerate==1.13.0",
            "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
            "pyyaml==6.0.3", "pandas==2.3.3", "matplotlib==3.10.9",
            "huggingface_hub==1.17.0", "ninja==1.13.0", "nbformat==5.10.4",
        ]
        print("Installing pinned Python runtime...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U",
             *runtime_packages],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e",
             str(REPO_DIR), "--no-deps"],
            check=True,
        )

        # Current Colab Python/CUDA combinations generally have no matching
        # upstream wheel. Use the bounded source build that completed the A100 run.
        fht_release = "v1.1.0.post2"
        kernel_env = os.environ.copy()
        kernel_env.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
        print("Building fast-hadamard-transform from the pinned release...", flush=True)
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
                    smoke_input = torch.randn(
                        2, 128, device="cuda", dtype=torch.float16)
                    smoke_output = hadamard_transform(smoke_input.contiguous())
                    torch.cuda.synchronize()
                    assert smoke_output.shape == smoke_input.shape
                    assert bool(torch.isfinite(smoke_output).all().item())
                del smoke_input, smoke_output
                print("Fast Hadamard CUDA smoke test passed.")
            except Exception as exc:
                fast_hadamard_available = False
                fast_hadamard_error = f"{type(exc).__name__}: {exc}"
        else:
            fast_hadamard_error = f"source build exited {kernel_build.returncode}"

        if fast_hadamard_available:
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
        else:
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
        if REQUIRE_FAST_HADAMARD:
            assert fast_hadamard_available, (
                "fast-hadamard-transform could not build/launch: "
                f"{fast_hadamard_error}. Set REQUIRE_FAST_HADAMARD=False only "
                "for a much slower diagnostic run."
            )
        elif not fast_hadamard_available:
            print("WARNING: using the slow pure-torch FWHT fallback.")

        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        environment_manifest = {
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
            json.dumps(environment_manifest, indent=2), encoding="utf-8")
        print("Pinned runtime ready.")
        """),
        md("## 4. Live subprocess and persistent-log helpers"),
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
            # Stream every child line to the cell and a Drive log. If the child
            # is silent, emit a notebook heartbeat with GPU state.
            log_path = LOG_ROOT / f"{label}.log"
            print("Running:", " ".join(map(str, command)), flush=True)
            print(f"Persistent log: {log_path}", flush=True)
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            started = time.monotonic()
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
            lines = queue.Queue()

            def read_output():
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(
                    f"\\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                    f"START {' '.join(map(str, command))}\\n"
                )
                while True:
                    try:
                        line = lines.get(timeout=NOTEBOOK_HEARTBEAT_SECONDS)
                    except queue.Empty:
                        elapsed = (time.monotonic() - started) / 60
                        heartbeat = (
                            f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                            f"notebook heartbeat {label}: elapsed={elapsed:.1f}m "
                            f"gpu={gpu_snapshot()}\\n"
                        )
                        print(heartbeat, end="", flush=True)
                        log.write(heartbeat)
                        continue
                    if line is None:
                        break
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = process.wait()
                elapsed = time.monotonic() - started
                log.write(
                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                    f"END returncode={returncode} elapsed_seconds={elapsed:.1f}\\n"
                )
            reader.join(timeout=2)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            return log_path

        def stage_command(stage, seeds, *, arms=None):
            command = [
                sys.executable, "-u",
                str(REPO_DIR / "scripts/run_qwen35_next_stage.py"),
                "--output-dir", str(MODEL_RESULT_ROOT),
                "--stage", stage,
                "--heartbeat-seconds", str(RUNNER_HEARTBEAT_SECONDS),
            ]
            for seed in seeds:
                command.extend(["--seed", str(seed)])
            for arm in arms or ():
                command.extend(["--arm", arm])
            if FORCE_RERUN:
                command.append("--force")
            return command

        def snapshot_summary(label):
            source = MODEL_RESULT_ROOT / "next_stage_summary.json"
            assert source.exists(), f"Runner did not create {source}"
            destination = SUMMARY_ROOT / f"{label}.json"
            shutil.copy2(source, destination)
            csv_source = MODEL_RESULT_ROOT / "next_stage_summary.csv"
            if csv_source.exists():
                shutil.copy2(csv_source, SUMMARY_ROOT / f"{label}.csv")
            print(f"Saved phase summary: {destination}")
            return destination

        print(
            "During a running cell, output is persisted under "
            f"{LOG_ROOT}. Runner state is in "
            f"{MODEL_RESULT_ROOT / 'next_stage_progress.json'}."
        )
        """),
        md("## 5. Validate the plan before spending GPU hours"),
        code("""
        required = [
            REPO_DIR / "scripts/run_qwen35_next_stage.py",
            REPO_DIR / "scripts/select_qwen35_ablation_finalists.py",
            REPO_DIR / "scripts/run_unsloth_qwen35_4b_kl.py",
            REPO_DIR / "configs/qwen35_4b_w4_factor_ablation_cuda.yaml",
            REPO_DIR / "configs/qwen35_4b_long_context_kv_cuda.yaml",
        ]
        missing = [str(path) for path in required if not path.exists()]
        assert not missing, "Missing required files: " + ", ".join(missing)

        for label, command in (
            ("ablation-screen", stage_command("ablation", (SCREEN_SEED,))),
            ("long-kv", stage_command("long-kv", (0,))),
        ):
            dry = [*command, "--dry-run"]
            print(f"\\nDry run: {label}")
            subprocess.run(dry, cwd=REPO_DIR, check=True)
        print("Plan validation passed. No model has been loaded yet.")
        """),
        md("## 6. Phase A — all-factor seed-0 screen"),
        code("""
        SCREEN_SUMMARY = SUMMARY_ROOT / "ablation_screen.json"
        if RUN_ABLATION_SCREEN:
            run_live(
                stage_command("ablation", (SCREEN_SEED,)),
                label="phase-a-ablation-screen",
            )
            SCREEN_SUMMARY = snapshot_summary("ablation_screen")
        else:
            assert SCREEN_SUMMARY.exists(), (
                "Screen disabled but no persisted ablation_screen.json exists."
            )
            print(f"Using existing screen: {SCREEN_SUMMARY}")
        """),
        md("## 7. Select replication candidates using frozen guardrails"),
        code("""
        import pandas as pd
        from IPython.display import display

        selection_path = RESULT_ROOT / "ablation_finalists.json"
        selection_command = [
            sys.executable, "-u",
            str(REPO_DIR / "scripts/select_qwen35_ablation_finalists.py"),
            "--summary", str(SCREEN_SUMMARY),
            "--output", str(selection_path),
            "--seed", str(SCREEN_SEED),
        ]
        subprocess.run(selection_command, cwd=REPO_DIR, check=True)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        FINALIST_ARMS = tuple(selection["finalists"])
        CONFIRMATION_ARMS = tuple(selection["confirmation_arms"])
        decision_rows = []
        for decision in selection["decisions"]:
            decision_rows.append({
                "arm": decision["arm"],
                "selected": decision["selected"],
                "halted": decision["evaluation_halted"],
                "failed_guards": ", ".join(
                    name for name, passed in decision["guards"].items()
                    if not passed
                ),
                "positive_signals": ", ".join(
                    name for name, passed in decision["signals"].items()
                    if passed
                ),
            })
        display(pd.DataFrame(decision_rows))
        print({
            "finalists": FINALIST_ARMS,
            "confirmation_arms": CONFIRMATION_ARMS,
            "interpretation": selection["interpretation"],
        })
        """),
        md("## 8. Phase B — three-seed confirmation of viable arms"),
        code("""
        CONFIRM_SUMMARY = SUMMARY_ROOT / "ablation_confirm.json"
        if RUN_ABLATION_CONFIRM and FINALIST_ARMS:
            run_live(
                stage_command(
                    "ablation", CONFIRM_SEEDS, arms=CONFIRMATION_ARMS),
                label="phase-b-ablation-confirm",
            )
            CONFIRM_SUMMARY = snapshot_summary("ablation_confirm")
        elif RUN_ABLATION_CONFIRM:
            print(
                "No seed-0 candidate cleared the pre-registered selector; "
                "there is nothing to confirm at seeds 1/2."
            )
        else:
            print("Three-seed confirmation disabled by configuration.")
        """),
        md("## 9. Phase C — corrected four-prompt 8k E8P engineering run"),
        code("""
        LONG_KV_SUMMARY = SUMMARY_ROOT / "long_kv.json"
        if RUN_LONG_CONTEXT_KV:
            run_live(
                stage_command("long-kv", (0,)),
                label="phase-c-long-kv",
            )
            LONG_KV_SUMMARY = snapshot_summary("long_kv")
        else:
            print("Long-context KV engineering run disabled by configuration.")
        """),
        md("## 10. Phase D — pinned Unsloth UD-Q4_K_XL anchor"),
        code("""
        def ensure_llama_cpp():
            probe = subprocess.run([
                sys.executable, "-c",
                "import diskcache, llama_cpp; print(llama_cpp.__version__)",
            ], capture_output=True, text=True)
            if probe.returncode == 0:
                print({"llama_cpp_python": probe.stdout.strip(), "status": "resume"})
                return
            print("Building the pinned CUDA llama-cpp-python comparator...", flush=True)
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
            llama_revision = "3691546f1c9e0c1bf93323dff02230bd959cf562"
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-v", "--no-deps",
                f"git+https://github.com/abetlen/llama-cpp-python.git@{llama_revision}",
            ], check=True, env=llama_env)
            importlib.invalidate_caches()
            version = subprocess.check_output([
                sys.executable, "-c",
                "import diskcache, llama_cpp; print(llama_cpp.__version__)",
            ], text=True).strip()
            print({"llama_cpp_python": version, "revision": llama_revision})

        if RUN_UNSLOTH_KL:
            ensure_llama_cpp()
            artifact_dir = (
                RESULT_BASE / "artifacts/unsloth-qwen35-4b-gguf"
                if PERSIST_GGUF_IN_DRIVE
                else Path("/content/unsloth-qwen35-4b-gguf")
            )
            unsloth_command = [
                sys.executable, "-u",
                str(REPO_DIR / "scripts/run_unsloth_qwen35_4b_kl.py"),
                "--output-dir", str(UNSLOTH_ROOT),
                "--artifact-dir", str(artifact_dir),
            ]
            if FORCE_RERUN:
                unsloth_command.append("--force")
            run_live(unsloth_command, label="phase-d-unsloth-kl")
        else:
            print("Unsloth anchor disabled by configuration.")
        """),
        md("## 11. Validate completeness, provenance, and comparable inputs"),
        code("""
        def load_summary(path):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        validation_issues = []
        phase_summaries = {
            "screen": load_summary(SCREEN_SUMMARY),
            "confirm": load_summary(CONFIRM_SUMMARY),
            "long_kv": load_summary(LONG_KV_SUMMARY),
        }
        for name, summary in phase_summaries.items():
            if summary is None:
                continue
            if summary.get("code_revision") != commit:
                validation_issues.append(
                    f"{name}: revision {summary.get('code_revision')} != {commit}")
            if not summary.get("complete"):
                validation_issues.append(f"{name}: summary is not marked complete")

        if RUN_ABLATION_SCREEN:
            screen = phase_summaries["screen"] or {}
            screen_rows = screen.get("rows", [])
            if len(screen_rows) != 8:
                validation_issues.append(
                    f"screen: expected 8 arms, found {len(screen_rows)}")

        if RUN_ABLATION_CONFIRM and FINALIST_ARMS:
            confirm = phase_summaries["confirm"] or {}
            identities = {
                (row.get("arm"), int(row.get("seed", -1)))
                for row in confirm.get("rows", [])
            }
            expected = {
                (arm, seed) for arm in CONFIRMATION_ARMS for seed in CONFIRM_SEEDS
            }
            missing = expected - identities
            if missing:
                validation_issues.append(
                    "confirm: missing " + ", ".join(map(str, sorted(missing))))

        if RUN_LONG_CONTEXT_KV:
            long_summary = phase_summaries["long_kv"] or {}
            long_rows = long_summary.get("rows", [])
            if len(long_rows) != 3:
                validation_issues.append(
                    f"long_kv: expected 3 arms, found {len(long_rows)}")
            for row in long_rows:
                if row.get("kv_endpoint_passed") is not True:
                    validation_issues.append(
                        f"long_kv/{row.get('arm')}: K8 endpoint did not pass")

        # Verify paired prompt/window identities directly from the latest raw
        # records selected by the runner summary.
        raw_records = {}
        for path in MODEL_RESULT_ROOT.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            config = payload.get("config") or {}
            if config.get("stage_protocol") != "qwen35-next-stage-v2":
                continue
            identity = (
                config.get("stage_name"), config.get("stage_arm"),
                int(config.get("seed", -1)),
            )
            raw_records.setdefault(identity, []).append((path.stat().st_mtime_ns, payload))
        latest_records = {
            identity: max(records, key=lambda item: item[0])[1]
            for identity, records in raw_records.items()
        }

        calibration_names = {
            "hessian_calibration", "activation_calibration",
            "allocation_teacher", "block_calibration",
        }
        diagnostic_names = {"layer_drift", "trajectory", "logit_fidelity"}
        for identity, payload in latest_records.items():
            manifest = (payload.get("metrics") or {}).get("data_manifest") or {}

            def source_rows(name):
                return {
                    row for row in (manifest.get(name) or {}).get("source_rows", [])
                    if row is not None
                }

            calibration_rows = set().union(*(
                source_rows(name) for name in calibration_names
            ))
            diagnostics = [
                (name, source_rows(name)) for name in diagnostic_names
                if source_rows(name)
            ]
            for name, rows in diagnostics:
                if calibration_rows.intersection(rows):
                    validation_issues.append(
                        f"{identity}: calibration overlaps {name}")
            for index, (left_name, left_rows) in enumerate(diagnostics):
                for right_name, right_rows in diagnostics[index + 1:]:
                    if left_rows.intersection(right_rows):
                        validation_issues.append(
                            f"{identity}: {left_name} overlaps {right_name}")

        for stage_seed in sorted({(key[0], key[2]) for key in latest_records}):
            records = [
                payload for (stage, _arm, seed), payload in latest_records.items()
                if (stage, seed) == stage_seed
            ]
            for metric_block, identity_key in (
                ("logit_fidelity", "input_hash"),
                ("trajectory", "input_hash"),
            ):
                identities = []
                for payload in records:
                    rows = ((payload.get("metrics") or {}).get(metric_block) or {}).get(
                        "prompt_metrics", [])
                    if rows:
                        identities.append(tuple(row[identity_key] for row in rows))
                if identities and len(set(identities)) != 1:
                    validation_issues.append(
                        f"{stage_seed}: {metric_block} prompts differ across arms")
            for dataset in ("wikitext2", "c4"):
                hashes = []
                for payload in records:
                    details = (payload.get("metrics") or {}).get(
                        f"ppl_{dataset}_details") or {}
                    if details.get("window_hashes"):
                        hashes.append(tuple(details["window_hashes"]))
                if hashes and len(set(hashes)) != 1:
                    validation_issues.append(
                        f"{stage_seed}: {dataset} windows differ across arms")

        unsloth_path = UNSLOTH_ROOT / "unsloth_ud_q4_kl.json"
        if RUN_UNSLOTH_KL:
            if not unsloth_path.exists():
                validation_issues.append("Unsloth result is missing")
            else:
                unsloth = json.loads(unsloth_path.read_text(encoding="utf-8"))
                if not unsloth.get("completed_rotquant_w4a8_input_hashes_match"):
                    validation_issues.append(
                        "Unsloth prompts do not match the registered RotQuant inputs")

        validation = {
            "commit": commit,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "ready_for_internal_decision" if not validation_issues else "needs_revision",
            "issues": validation_issues,
            "boundary": (
                "Internal development evidence only. Four-prompt cache intervals "
                "are not reliable promotion intervals; no same-byte Unsloth claim."
            ),
        }
        (RESULT_ROOT / "full_experiment_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps(validation, indent=2))
        assert not validation_issues, "Experiment validation failed; inspect issues above."
        """),
        md("## 12. Results and decision view"),
        code("""
        import matplotlib.pyplot as plt

        screen = phase_summaries.get("screen") or {}
        screen_frame = pd.DataFrame(screen.get("rows", []))
        if not screen_frame.empty:
            baseline = screen_frame.loc[
                screen_frame["arm"] == "promoted_w4"
            ].iloc[0]
            view = screen_frame.copy()
            view["kl_vs_control"] = (
                view["mean_teacher_kl"] / baseline["mean_teacher_kl"] - 1)
            view["top1_delta"] = (
                view["top1_agreement"] - baseline["top1_agreement"])
            view["c4_ppl_vs_control"] = (
                view["ppl_c4"] / baseline["ppl_c4"] - 1)
            view["bytes_vs_control"] = (
                view["complete_persistent_model_bytes"]
                / baseline["complete_persistent_model_bytes"] - 1)
            columns = [
                "arm", "evaluation_halted", "mean_teacher_kl",
                "kl_vs_control", "top1_delta", "c4_ppl_vs_control",
                "trajectory_token_agreement", "bytes_vs_control",
            ]
            display(view[columns].sort_values("mean_teacher_kl"))

            plot = view.set_index("arm")[[
                "kl_vs_control", "top1_delta", "c4_ppl_vs_control",
                "bytes_vs_control",
            ]]
            axes = plot.plot(
                kind="bar", subplots=True, layout=(2, 2), figsize=(14, 8),
                legend=False, sharex=True,
                title=[
                    "Teacher KL change vs W4 (lower better)",
                    "Top-1 change vs W4 (higher better)",
                    "C4 PPL change vs W4 (lower better)",
                    "Persistent bytes change vs W4 (lower better)",
                ],
            )
            for axis in axes.flat:
                axis.axhline(0, color="black", linewidth=0.8)
                axis.tick_params(axis="x", labelrotation=70)
            plt.tight_layout()
            plt.show()

        confirm = phase_summaries.get("confirm") or {}
        confirm_frame = pd.DataFrame(confirm.get("rows", []))
        if not confirm_frame.empty:
            print("Three-seed confirmation records")
            display(confirm_frame[[
                "arm", "seed", "mean_teacher_kl", "top1_agreement",
                "ppl_wikitext2", "ppl_c4", "trajectory_token_agreement",
                "complete_persistent_model_bytes",
            ]].sort_values(["arm", "seed"]))
            paired_comparisons = [
                report for report in confirm.get("paired_comparisons", [])
                if report.get("baseline_arm") == "promoted_w4"
            ]
            if paired_comparisons:
                print("Paired candidate-minus-promoted-W4 reports")
                display(pd.DataFrame([{
                    "arm": report["candidate_arm"],
                    "seed": report["seed"],
                    **report.get("aggregate_deltas", {}),
                } for report in paired_comparisons]))

        long_summary = phase_summaries.get("long_kv") or {}
        long_frame = pd.DataFrame(long_summary.get("rows", []))
        if not long_frame.empty:
            print("Corrected 8k E8P engineering records")
            display(long_frame[[
                "arm", "kv_endpoint_passed", "kv_endpoint_mean_teacher_kl",
                "kv_mean_teacher_kl", "kv_top1_agreement", "kv_effective_bpv",
                "kv_total_cache_compression_ratio",
            ]])

        if RUN_UNSLOTH_KL and unsloth_path.exists():
            unsloth = json.loads(unsloth_path.read_text(encoding="utf-8"))
            print("Pinned Unsloth UD-Q4_K_XL anchor")
            print(json.dumps({
                "complete_artifact_bytes": unsloth["candidate"][
                    "complete_artifact_bytes"],
                **unsloth["metrics"],
                "warning": unsloth["comparison_warning"],
            }, indent=2))

        print(
            "Interpretation: finalists are replication candidates, not wins. "
            "A production recipe still requires consistent three-seed paired "
            "evidence, matched-byte allocation, and the frozen 300-prompt run."
        )
        """),
        md("## 13. Preserve a downloadable audit bundle"),
        code("""
        bundle_root = Path(f"/content/qwen35_full_experiment_{commit[:12]}_bundle")
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        # Full BF16 logit arrays can be ~1 GB and already persist in Drive. Keep
        # them out of the browser download while retaining every manifest,
        # prompt record, result, validation file, phase summary, and live log.
        shutil.copytree(
            RESULT_ROOT,
            bundle_root,
            ignore=shutil.ignore_patterns("*.npz", "*.gguf"),
        )
        archive = shutil.make_archive(
            f"/content/qwen35_full_experiment_{commit[:12]}", "zip",
            root_dir=bundle_root,
        )
        print(f"Created {archive}")
        print(f"Canonical Drive records remain at {RESULT_ROOT}")
        if DOWNLOAD_RESULTS:
            from google.colab import files
            files.download(archive)
        """),
        md("""
        ## Takeaways and next decision

        Use the executed tables and `full_experiment_validation.json`, not
        scrollback, as the decision record. If one or more W4 factors reproduce
        across seeds without a tail/trajectory regression, the next build is the
        exact-byte mixed-precision allocator around those factors. If none do,
        retain promoted uniform W4 and move directly to allocation rather than
        combining individually negative tweaks.

        The cache result only clears engineering correctness. Before promotion,
        implement the fixed-FP16-teacher collector and prompt/document-cluster
        resampling, then run the registered ≥20-prompt/≥10k-continuation-token
        protocol. Qwen3.8-27B remains gated on the Qwen3.5-4B recipe.
        """),
    ]
    for index, cell in enumerate(cells):
        digest = hashlib.sha256(
            f"{cell.cell_type}\0{cell.source}".encode()
        ).hexdigest()[:10]
        cell["id"] = f"rotquant-{index:02d}-{digest}"

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
