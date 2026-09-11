"""Build the post-correctness native performance pilot; keep the old notebook intact."""
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
        # RotQuant Qwen3.5-4B — native GPU performance pilot

        ## Goal
        Measure our **unchanged W5/scale8 + W6 vocabulary** native llama.cpp
        path after fresh correctness checks. Use a fresh **A100 40GB** runtime
        for continuity with the successful run, then **Runtime → Run all**.

        The previous run passed saved-model parity but did **not** measure
        throughput. This run measures synchronous prefill, cached decoding and
        sampled process VRAM at **128, 512 and 2,048 input tokens**.
        It is not a new quantization, task-accuracy test or Unsloth comparison.

        Built libraries and lossless exports are saved to a **private Drive
        cache**. The first run still needs a build (previously about 27 minutes)
        because the old reports ZIP does not contain binaries. Later compatible
        runs restore checked files locally and repeat numerical gates.
        '''),
        md('''
        ## Setup — explicit controls
        Keep W6 alone for the first pilot. W8 is supported as a separate follow-up:
        set `ARMS = ("b5_v8_s0",)` and use a new run name when ready.

        `SOURCE_ROOT` must contain the original checkpoint, `prepared.json`,
        `preparation.json`, and `packed_probes.safetensors` for each arm.
        A reports-only ZIP is insufficient. The source is never modified.

        The default **90 active-minute** allowance includes dependency setup,
        hashing/copying, build and tests. Each performance context has a **3-minute
        ceiling**. The 2 tok/s and 16 GiB limits below are conservative spending
        guards, not pass marks for quality or production readiness.
        '''),
        code('''
        from pathlib import Path
        import json, re, shutil, subprocess, sys

        REPO_REF = "main"  # resolved once; use the published full commit for repeat runs
        SOURCE_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
        CACHE_ROOT = Path("/content/drive/MyDrive/rotquant/native_artifact_cache/v1")
        ARMS = ("b5_v6_s0",)
        RUN_NAME = "pilot1"
        ACTIVE_BUDGET_MINUTES = 90
        BUILD_JOBS = 2
        DECODE_STEPS = 32
        MEASURED_REPETITIONS = 3  # plus one excluded warmup per context
        MIN_DECODE_TOKENS_PER_SECOND = 2.0
        MAX_PROCESS_VRAM_MIB = 16384

        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", RUN_NAME)
        assert 1 <= ACTIVE_BUDGET_MINUTES <= 180 and 1 <= BUILD_JOBS <= 32
        assert ARMS and len(set(ARMS)) == len(ARMS) and set(ARMS) <= {"b5_v6_s0", "b5_v8_s0"}
        '''),
        md('''
        ### Connect Drive and pin the repository
        This uses Colab's normal Drive mount; local `gws` authentication is not
        required. The managed venv preserves Colab CUDA Torch and does not need
        `ensurepip`, an external Hadamard kernel or `llama-cpp-python`.

        Keep `CACHE_ROOT` private. Hashes detect accidental corruption, **not**
        a malicious replacement of both cached binaries and their manifest.
        Do not use downloaded/shared caches from an untrusted party.
        '''),
        code('''
        from google.colab import drive
        drive.mount("/content/drive")
        assert shutil.which("nvidia-smi") and shutil.which("nvcc"), "Select a CUDA GPU runtime."
        subprocess.run(["nvidia-smi"], check=True, timeout=30)
        for arm in ARMS:
            for item in ("checkpoint", "prepared.json", "preparation.json", "packed_probes.safetensors"):
                assert (SOURCE_ROOT / arm / item).exists(), f"Missing original evidence: {SOURCE_ROOT / arm / item}"

        REPO_DIR = Path("/content/rotquant-native-pilot/repository")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", "https://github.com/CodeHalwell/rotquant.git", str(REPO_DIR)], check=True, timeout=300)
        assert not subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip(), "Repository has edits; use a fresh runtime. No automatic reset."
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_REF], check=True, timeout=120)
        COMMIT = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "FETCH_HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", COMMIT], check=True, timeout=30)
        for name in ("run_native_gpu_validation.py", "native_gpu_cache.py", "run_rq3_performance_pilot.py", "native_pilot_controls.py"):
            assert (REPO_DIR / "scripts" / name).exists(), "Pilot code is not present at REPO_REF. Stop before spending on the build."
        sys.path.insert(0, str(REPO_DIR))
        from scripts.native_pilot_controls import validate_controls
        validate_controls(128, DECODE_STEPS, MEASURED_REPETITIONS,
                          MIN_DECODE_TOKENS_PER_SECOND, MAX_PROCESS_VRAM_MIB)
        WORK_DIR = Path("/content/rotquant-native-pilot/work") / COMMIT[:12]
        RESULT_ROOT = Path("/content/drive/MyDrive/rotquant/native_gpu_pilot") / COMMIT[:12] / RUN_NAME
        print({"commit": COMMIT, "results": str(RESULT_ROOT), "cache": str(CACHE_ROOT),
               "arms": ARMS, "active_minutes": ACTIVE_BUDGET_MINUTES,
               "contexts": [128, 512, 2048], "decode_steps": DECODE_STEPS})
        '''),
        md('''
        ## Steps and checks — run the complete pilot
        1. Verify environment, original checkpoint, tokenizer and saved probes.
        2. Restore an exact-compatible runtime or build it, then persist checked
           libraries. GPU/driver/toolchain changes cause a cache miss.
        3. Fresh binding, CPU/CUDA operator, W6/W8 tiny-model and conversion gates.
        4. Restore or export the saved model losslessly, including non-text sidecar.
        5. Fresh native parity against saved **quantized-model** logits/traces.
        6. Three independently capped timing processes, only after parity passes.

        Expect `CACHE MISS` on the first run, `CACHE PERSISTED` after saving, and
        `CACHE HIT` on a compatible restore. Build/export passes are reusable;
        all numerical gates and timing are fresh. Corrupt caches fail closed
        and are preserved, not silently overwritten or executed.

        Every phase prints its cap, PID, persistent log and 30-second heartbeat.
        Each repetition prints warmup/measured status and decode progress every
        eight tokens. Raw measurements are checkpointed outside timed regions.
        On failure, **run the Results cell below**; do not add repair cells.

        For an interruption, rerun this launch cell without changing settings.
        After VM loss, Run all with the same pinned commit/run name. Active time
        is cumulative; successful timings are deliberately rerun. Use a new
        `RUN_NAME` for changed settings. Never update Git while the driver runs.
        '''),
        code('''
        from scripts.colab_runtime import run_live
        command = [sys.executable, "-u", str(REPO_DIR / "scripts/run_native_gpu_validation.py"),
                   "--output-dir", str(RESULT_ROOT), "--work-dir", str(WORK_DIR),
                   "--source-root", str(SOURCE_ROOT), "--persistent-cache-dir", str(CACHE_ROOT),
                   "--active-minutes", str(ACTIVE_BUDGET_MINUTES), "--jobs", str(BUILD_JOBS),
                   "--performance-pilot", "--decode-steps", str(DECODE_STEPS),
                   "--repetitions", str(MEASURED_REPETITIONS),
                   "--min-decode-tps", str(MIN_DECODE_TOKENS_PER_SECOND),
                   "--max-vram-mib", str(MAX_PROCESS_VRAM_MIB)]
        for arm in ARMS:
            command.extend(["--arm", arm])
        run_live(command, "native-performance-pilot", repo_dir=REPO_DIR,
                 log_root=RESULT_ROOT / "launch-logs")
        '''),
        md('''
        ## Results — also run this cell after a stop
        The table reports the latest attempt per context, including partial
        results. Warmup is excluded. Rate = total measured tokens / total measured
        seconds. Each context has its own process/model reservation and sampled
        VRAM; the sidecar remains on disk, not in the text runtime.

        These are **synchronous bridge call rates**, including full-vocabulary
        logit copies, validation, synchronization and Python argmax. Checkpoint
        writes and progress logging are outside timed regions. Repeated prompt
        IDs and fixed decoding through EOS test shapes, not meaningful output.
        This is neither long-context accuracy nor a pure-kernel/bandwidth profile.

        No Unsloth/FP16 speed ratio is claimed without a matched baseline run.
        Share the reports ZIP; it excludes weights/binaries, which remain in the
        private cache. Failed/partial attempts never become successful benchmarks.
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
        def number(value):
            return "—" if value is None else f"{value:.2f}"
        from IPython.display import Markdown, display
        rows = ["| Arm | Input tokens | Status | Measured reps | Prefill tok/s | Decode tok/s | Peak MiB* |",
                "| :-- | --: | :-- | --: | --: | --: | --: |"]
        for arm in ARMS:
            for context in (128, 512, 2048):
                attempts = sorted((RESULT_ROOT / "stages" / f"pilot-{arm}-ctx{context}").glob("attempt-*"))
                report_path = attempts[-1] / "pilot/report.json" if attempts else None
                report = json.loads(report_path.read_text()) if report_path and report_path.exists() else {}
                metrics, memory = report.get("summary") or {}, report.get("memory") or {}
                status = report.get("status", "not completed" if attempts else "not run")
                rows.append(f"| {arm} | {context} | {status} | {metrics.get('measured_repetitions', 0)} | "
                            f"{number(metrics.get('prefill_tokens_per_second'))} | {number(metrics.get('decode_tokens_per_second'))} | "
                            f"{number(memory.get('sampled_peak_process_vram_mib'))} |")
        display(Markdown("\\n".join(rows)))
        print("* 0.5-second process samples may miss transient peaks; not Torch allocator memory.")
        for archive in sorted(RESULT_ROOT.parent.glob(RUN_NAME + "-reports-*.zip"))[-3:]:
            print("Reports ZIP:", archive)
        print("DISCONNECT AND DELETE the GPU runtime now to avoid further charges.")
        '''),
        md('''
        ## Next steps
        Inspect the pilot before extending it. If speed/VRAM are unsuitable,
        profile native copies and kernel hot spots before paying for a task sweep.
        If usable, validate retained W5/W8 and longer cache behavior, then add
        hardware/token/cache-matched FP16 and Unsloth baselines.

        **Completion, a timeout, or the 90-minute cap does not stop Colab billing.**
        Disconnect/delete the runtime yourself. No new GPU job is started merely
        by opening this notebook.
        '''),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"})


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    destination = Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_gpu_pilot_colab.ipynb"
    nbformat.write(notebook, destination)
    print(destination)
