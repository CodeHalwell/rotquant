"""Generate a separate opt-in overnight notebook; preserve all existing notebooks."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_qwen35_packed_validation_notebook import code, md


def build_notebook():
    return new_notebook(cells=[md('''
        # RotQuant Qwen3.5-4B — overnight native runtime experiment

        Use **A100 40GB**, check the controls and original checkpoint path, then
        **Runtime → Run all**. This tests new execution kernels, **not new quants**.
        The unchanged W5/scale8 backbone + W6 vocabulary checkpoint is retained.

        **Maximum: eight active hours AND eight wall hours from driver launch.**
        Setup/builds, failed trials and fresh controls count. The suite finishes
        early when no useful candidate remains. Resuming the same run preserves
        consumed time and the original deadline; idle/disconnected time still
        consumes the wall allowance. Colab availability is not guaranteed.

        This notebook is a GPU-validation handoff, not evidence that the new
        kernels work or are faster. No GPU has been launched by preparing it.
        Existing notebooks and runtime defaults remain unchanged.
        '''), md('''
        ## Controls — review before Run all

        `AUTO_RELEASE_RUNTIME=True` asks Colab to disconnect/delete **this runtime**
        after reports are saved and verified on Drive, on success or a handled
        failure. Set False if you want manual recovery. The API can fail and
        is **not a provider-side billing cap**; check the runtime afterwards.
        No keepalive tricks, automatic provisioning, quantization or training.

        Keep the original source and private binary/export cache untouched.
        A reports-only ZIP is insufficient: the checkpoint/probes are required.
        Initial builds may take roughly half an hour; later compatible builds
        are hash-checked and reused. GPU correctness gates are always rerun.
        '''), code('''
        from pathlib import Path
        import json, re, shutil, subprocess, sys

        REPO_REF = "main"  # publish first; pin the printed full commit for resuming
        RUN_NAME = "overnight1"
        ACTIVE_BUDGET_MINUTES = 480
        BUILD_JOBS = 2
        AUTO_RELEASE_RUNTIME = True
        RUN_CONVENTIONAL_CONTROLS = True  # fresh BF16 + Unsloth UD-Q4, not every quant
        SOURCE_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
        CACHE_ROOT = Path("/content/drive/MyDrive/rotquant/native_artifact_cache/v1")
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", RUN_NAME)
        assert 10 <= ACTIVE_BUDGET_MINUTES <= 480 and 1 <= BUILD_JOBS <= 32
        assert type(AUTO_RELEASE_RUNTIME) is bool and type(RUN_CONVENTIONAL_CONTROLS) is bool
        '''), md('''
        ## Mount Drive and freeze the code revision

        Do this while present. Keep the cache private; hashes do not authenticate
        a malicious replacement of both binary and manifest. Do not update any
        checkout used by a running experiment. An existing overnight checkout
        is reused without pulling main again; a changed explicit revision is
        rejected. Use a fresh runtime to switch code.
        '''), code('''
        from google.colab import drive
        drive.mount("/content/drive")
        assert shutil.which("nvcc") and shutil.which("nvidia-smi"), "Select a CUDA GPU runtime."
        subprocess.run(["nvidia-smi"], check=True, timeout=30)
        for name in ("checkpoint", "prepared.json", "preparation.json", "packed_probes.safetensors"):
            assert (SOURCE_ROOT / "b5_v6_s0" / name).exists(), f"Missing original evidence: {name}"
        REPO_DIR = Path("/content/rotquant-native-overnight/repository")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", "https://github.com/CodeHalwell/rotquant.git", str(REPO_DIR)], check=True, timeout=300)
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_REF], check=True, timeout=120)
            subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", "FETCH_HEAD"], check=True, timeout=30)
        assert not subprocess.check_output(["git", "-C", str(REPO_DIR), "status", "--porcelain"], text=True).strip(), "Edited checkout: stop; no automatic reset."
        COMMIT = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
        assert REPO_REF == "main" or COMMIT == REPO_REF, "Use the original full commit or a fresh runtime."
        for name in ("run_native_overnight.py", "native_overnight_safety.py", "run_rq3_overnight_screen.py", "run_rq3_overnight_fidelity.py"):
            assert (REPO_DIR / "scripts" / name).is_file(), "Overnight code has not been published at this revision. Stop."
        WORK_DIR = Path("/content/rotquant-native-overnight/work") / COMMIT[:12] / RUN_NAME
        RESULT_ROOT = Path("/content/drive/MyDrive/rotquant/native_overnight") / COMMIT[:12] / RUN_NAME
        sys.path.insert(0, str(REPO_DIR))
        print({"commit": COMMIT, "results": str(RESULT_ROOT), "source": str(SOURCE_ROOT),
               "active_and_wall_minutes": ACTIVE_BUDGET_MINUTES, "auto_release": AUTO_RELEASE_RUNTIME})
        '''), md('''
        ## Frozen experiment and stop rules

        Six independent candidates: two FP32 GEMM chunk sizes, three FP16 GEMM
        chunk sizes, and an exact warp-based vocabulary head. A seventh combined
        candidate runs only if its two components independently pass. **Tile8**
        is the performance control; the original kernel is the operator oracle.

        Both GEMM paths reconstruct a bounded chunk, round operands to the same
        FP16 boundary already used by native execution, accumulate FP32 and
        half-round outputs. They change reduction order. The FP32 control
        disables TF32; the FP16 path permits Tensor Cores but does not claim
        actual hardware utilization without profiling. No persistent dense model.
        Requested scratch is capped at **256 MiB**; pool reservations, cuBLAS
        workspace and process VRAM are additional and reported separately.

        1. Build/restore, original numerical gates and unchanged retained parity.
        2. W1–W8, scales, tails, permutations and long reductions; failures are
           rejected before model benchmarking. A bad shared reference, missing
           dispatch, timeout or invalid provenance stops the whole workflow.
        3. Warm resident synthetic AB/BA screens; actual model matrix inventory
           is saved separately. These are not frequency-weighted/cold-cache rates.
        4. At most two backbone finalists plus the head: tiny W6/W8 model gates,
           retained parity, extended common-input KL/top1/logit/greedy checks.
        5. Full-model 128/512/2048-token fixed replay, counterbalanced candidate
           order and bracketing controls. Fresh conventional controls follow
           only if a candidate survives. One separate diagnostic profile pair.

        Extended probes repeat saved prefixes; they test numerical stability,
        **not semantic long-context accuracy**. The teacher is the unchanged
        quantized tile8 implementation, **not the original FP16 model**.
        No task-accuracy improvement, serving readiness or promotion is assumed.
        W5/W8 retained runs, all Unsloth sizes and task sweeps are deferred.
        '''), code('''
        command = [sys.executable, "-u", str(REPO_DIR / "scripts/run_native_overnight.py"),
                   "--output-dir", str(RESULT_ROOT), "--work-dir", str(WORK_DIR),
                   "--source-root", str(SOURCE_ROOT), "--persistent-cache-dir", str(CACHE_ROOT),
                   "--active-minutes", str(ACTIVE_BUDGET_MINUTES), "--jobs", str(BUILD_JOBS)]
        if not RUN_CONVENTIONAL_CONTROLS:
            command.append("--skip-conventional")
        subprocess.run([*command, "--dry-run"], cwd=REPO_DIR, check=True, timeout=30)
        '''), md('''
        ## Run end to end

        The driver owns persisted budgets. A parent watchdog separately bounds
        the process. Each phase prints its cap, PID, log path and heartbeat;
        numerical cases and each timing repetition are checkpointed to Drive.
        No endless retries. A rejected candidate is not a GPU pass.

        If Colab interrupts or deletes the VM, neither code nor saved logs can
        guarantee clean shutdown. If final archive verification fails, automatic
        release is withheld so you can recover evidence. Watch the first setup
        and numerical-screen output before leaving it unattended.
        '''), code('''
        from scripts.native_overnight_safety import supervise
        supervise(command, RESULT_ROOT, REPO_DIR, ACTIVE_BUDGET_MINUTES,
                  auto_release=AUTO_RELEASE_RUNTIME)
        '''), md('''
        ## Results and recovery (manual if automatic release disconnected Colab)

        Results remain under the printed Drive path, with per-attempt JSON/logs,
        candidate rejections, common-input fidelity, raw timings and comparisons.
        The ZIP excludes weights, binaries and the local large reference logits.
        Those logits are reproducible and hash-bound; they are not task scores.
        Download the reports from Drive after an automatic release. No browser
        download popup is required for unattended operation.
        '''), code('''
        from IPython.display import Markdown, display
        from scripts.native_performance_summary import tables
        since_epoch = float("inf")
        if (RESULT_ROOT / "experiment.json").exists():
            experiment = json.loads((RESULT_ROOT / "experiment.json").read_text())
            since_epoch = experiment["started_utc_epoch"]
            print("Completed:", experiment["completed"], "Outcome:", experiment.get("outcome", "incomplete"))
            print("Screen finalists:", experiment["finalists"])
            print("Numerically confirmed:", experiment.get("numerically_confirmed", []))
            for candidate in experiment["candidates"]:
                print(candidate["kernel"], "eligible:", candidate["eligible"], "rejected:", candidate.get("rejected_at"))
        throughput, diagnostics = tables(RESULT_ROOT, since_epoch=since_epoch)
        display(Markdown("### Uninstrumented throughput\\n\\n" + throughput))
        display(Markdown("### Diagnostic profiles — not throughput\\n\\n" + diagnostics))
        print("Reports:", *sorted(RESULT_ROOT.parent.glob(RUN_NAME + "-reports-*.zip")), sep="\\n")
        print("Verify this GPU runtime is disconnected/deleted to avoid further charges.")
        ''')])


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    path = ROOT / "notebooks/qwen35_4b_native_overnight_colab.ipynb"
    nbformat.write(notebook, path)
    print(path)
