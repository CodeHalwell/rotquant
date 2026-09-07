#!/usr/bin/env python3
"""Generate the focused packed-artifact Colab; no inherited string rewrites."""
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path("notebooks/qwen35_4b_packed_validation_colab.ipynb")
REVALIDATION_OUTPUT = Path("notebooks/qwen35_4b_packed_revalidation_colab.ipynb")


def md(text):
    return new_markdown_cell(dedent(text).strip())


def code(text):
    return new_code_cell(dedent(text).strip())


def build_notebook(*, revalidate=False):
    revalidation_setting = ('Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")'
                            if revalidate else 'None')
    cells = [
        md('''
        # RotQuant Qwen3.5-4B saved-checkpoint revalidation

        ## Goal
        Reuse the W5/W6 and W5/W8 artifacts from `8f10ee60fc7f` with the repaired
        loader. **No Hessian collection, quantization or checkpoint export.**
        Preserve all original files, including the failed W6 validation. New
        records bind the original preparation and the current validator identities.

        The old run stopped on reload numerical parity; full packed quality was
        skipped. The loader fix preserves framework FP32 rotary-position buffers.
        A tiny local reproduction reached exact reload parity after that fix;
        **the actual 4B CUDA result is still unverified**. All thresholds stay unchanged.

        Use the same A100 40 GB-class GPU and pinned software as the original run,
        with high host RAM. Keep the checkpoints AND probe safetensors on Drive:
        the compact evidence ZIP is insufficient. This still loads the source
        teacher and runs full quality evaluation, so allow local download/cache
        space and time; it is not a fused-kernel benchmark or a provider comparison.
        ''' if revalidate else '''
        # RotQuant Qwen3.5-4B packed-vocabulary validation

        ## Goal
        Turn the two promising W5 backbone / W6 or W8 vocabulary prototypes
        into measured, reloadable artifacts. Test numerical parity, real file
        bytes, shared vocabulary ownership and packed-runtime quality. This is
        a **reference runtime**, not a fused-kernel speed benchmark or a provider win.

        The prior seed-0 screen recorded KL 0.004741 (W6 vocabulary) and 0.004037
        (W8), versus W4/FP16 vocabulary 0.016497. Those are historical dense
        prototype measurements, **not results of this notebook**.

        Use a fresh A100 40 GB-class Colab runtime. Reserve at least 45 GB of
        persistent free space for checkpoints, Hessians, caches and staging, plus
        local source-download space. CPU/format tests run locally; this complete
        GPU experiment has not been executed locally.
        '''),
        md('''
        ## Setup
        ### 1. Freeze settings
        Defaults run both seed-0 candidates, W6 first. No new allocation, LoRA,
        A8/KV, independent confirmation or Unsloth inference is launched.
        Keep the printed commit when resuming. `REVALIDATE_FROM` opts into a new,
        separately recorded validation of old artifacts; it never resumes old
        results under a different identity. Leave it `None` for a new full run.
        '''),
        code(f'''
        from pathlib import Path
        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # Publish this implementation before loading from main.
        REPO_DIR = Path("/content/rotquant-packed-validation")
        RESULT_BASE = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation")
        SEEDS = (0,)
        RUN_EXPERIMENT = True
        PREPARE_ONLY = False  # True exports/probes only; full validation remains incomplete.
        REVALIDATE_FROM = {revalidation_setting}
        REQUIRE_FAST_HADAMARD = True
        USE_HF_SECRET = False
        DOWNLOAD_RESULTS = True
        HEARTBEAT_SECONDS = 60
        '''),
        md("### 2. GPU, Drive and immutable checkout"),
        code('''
        import json, os, shutil, subprocess, sys
        import torch
        from google.colab import drive
        assert torch.cuda.is_available(), "Select a GPU runtime."
        gpu = torch.cuda.get_device_properties(0)
        assert gpu.total_memory >= 35 * 2**30, "Use an A100 40 GB-class GPU."
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")

        def git(*args):
            result = subprocess.run(["git", *args], cwd=REPO_DIR,
                                    capture_output=True, text=True)
            if result.returncode:
                print(result.stdout, result.stderr)
                result.check_returncode()
            return result.stdout.strip()

        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        assert (REPO_DIR / ".git").is_dir(), "REPO_DIR is not a checkout."
        assert not git("status", "--porcelain"), "Keep the checkout clean; use a new path for edits."
        git("fetch", "origin", REPO_REF)
        git("checkout", "--detach", "FETCH_HEAD")
        COMMIT = git("rev-parse", "HEAD")
        assert (REPO_DIR / "scripts/run_qwen35_packed_validation.py").exists(), "Publish/select the new code first."
        if REVALIDATE_FROM is not None:
            REVALIDATE_FROM = Path(REVALIDATE_FROM).resolve()
            assert not PREPARE_ONLY, "Recovery is validation-only, not preparation."
            assert REVALIDATE_FROM.is_dir(), "Original Drive run is missing."
            RESULT_ROOT = RESULT_BASE / f"{COMMIT[:12]}-revalidate-{REVALIDATE_FROM.name}"
            for seed in SEEDS:
                for arm in ("b5_v6", "b5_v8"):
                    source_arm = REVALIDATE_FROM / f"{arm}_s{seed}"
                    for name in ("prepared.json", "preparation.json", "dense_probes.safetensors",
                                 "packed_probes.safetensors", "checkpoint/rotquant_config.json"):
                        assert (source_arm / name).is_file(), f"Missing original evidence: {source_arm / name}"
        else:
            RESULT_ROOT = RESULT_BASE / COMMIT[:12]
        if REVALIDATE_FROM is not None:
            assert not RESULT_ROOT.resolve().is_relative_to(REVALIDATE_FROM), "Output overlaps original run."
            assert not REVALIDATE_FROM.is_relative_to(RESULT_ROOT.resolve()), "Output overlaps original run."
        LOG_ROOT = RESULT_ROOT / "logs"
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        print({"commit": COMMIT, "result_root": str(RESULT_ROOT), "gpu": gpu.name,
               "seeds": SEEDS, "filesystem_free_gb": shutil.disk_usage(RESULT_ROOT).free/1e9})
        print("Filesystem free space is NOT your Google Drive quota.")
        if REVALIDATE_FROM is None:
            print("Check Drive has >=45 GB free for a full preparation run.")
        else:
            print({"checkpoint_only": True, "original_run_unchanged": str(REVALIDATE_FROM)})
        sys.path.insert(0, str(REPO_DIR))
        from scripts.colab_runtime import run_live as _run_live
        def run_live(command, label, timeout_seconds=None):
            return _run_live(command, label, repo_dir=REPO_DIR, log_root=LOG_ROOT,
                             timeout_seconds=timeout_seconds)
        '''),
        md('''
        ### 3. Dependencies and fast Hadamard smoke
        Keep Colab's GPU-compatible PyTorch. Installation/build output goes to
        persistent log files. An explicit cell interruption stops the process
        group; closing/reconnecting the browser does not depend on a stdout pipe.
        '''),
        code('''
        packages = ["transformers==5.9.0", "datasets==4.8.5", "accelerate==1.13.0",
                    "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
                    "pyyaml==6.0.3", "pandas==2.3.3", "huggingface_hub==1.17.0",
                    "ninja==1.13.0", "nbformat==5.10.4"]
        run_live([sys.executable, "-m", "pip", "install", *packages], "install")
        run_live([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "--no-deps"], "install-project")
        os.environ.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
        smoke = ("import torch; from fast_hadamard_transform import hadamard_transform; "
                 "x=hadamard_transform(torch.ones(1,128,device='cuda',dtype=torch.float16),scale=128**-0.5); "
                 "assert torch.isfinite(x).all(); print('Fast Hadamard CUDA smoke passed')")
        try:
            probe = subprocess.run([sys.executable, "-c", smoke], capture_output=True, text=True)
            if probe.returncode:
                run_live([sys.executable, "-m", "pip", "install", "-v", "--no-deps", "--no-build-isolation",
                          "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0.post2"],
                         "fast-hadamard-build", timeout_seconds=1200)
            run_live([sys.executable, "-c", smoke], "kernel-smoke")
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
        except (subprocess.CalledProcessError, TimeoutError):
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
            if REQUIRE_FAST_HADAMARD:
                raise RuntimeError(f"Fast kernel failed: inspect {LOG_ROOT}. Set REQUIRE_FAST_HADAMARD=False only to accept a slow fallback.")
            print("WARNING: slower pure-torch rotation selected.")
        '''),
        md("### 4. Optional authentication"),
        code('''
        if USE_HF_SECRET:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
            del token
        else:
            print("Skipping Colab secret lookup; public downloads work unauthenticated.")
        '''),
        md('''
        ## Steps
        ### 5. Plan and synthetic CUDA export/reload preflight
        This tests the shared owner, tiny Qwen hybrid generation, save/reload,
        stored metadata and numerical gates before spending time on full-model calibration.
        '''),
        code('''
        command = [sys.executable, "-u", str(REPO_DIR / "scripts/run_qwen35_packed_validation.py"),
                   "--output-dir", str(RESULT_ROOT), "--heartbeat-seconds", str(HEARTBEAT_SECONDS)]
        for seed in SEEDS:
            command.extend(["--seed", str(seed)])
        if REVALIDATE_FROM is not None:
            command.extend(["--revalidate-from", str(REVALIDATE_FROM)])
        run_live([*command, "--dry-run"], "plan")
        run_live([sys.executable, "-u", str(REPO_DIR / "scripts/preflight_packed_validation.py"),
                  "--device", "cuda", "--output", str(RESULT_ROOT / "runtime_preflight.json")], "preflight")
        '''),
        md('''
        ### 6. Export and validate
        One W5 backbone quantization per seed is shared by both vocabulary
        candidates. Each fresh worker reloads from disk with `fallback=False`.
        The vocabulary head reconstructs only one tile at a time using the
        prototype's FP16 rounding. Full evaluation can be slow: there are no
        fused packed kernels. Stage markers and 60-second GPU heartbeats remain visible.

        **With `REVALIDATE_FROM` set**, skip preparation entirely and read the
        existing exports/probes. The plan must report zero backbone quantizations.
        Current runtime, model/config, prompt contents and thresholds must match
        the original preparation; only the validator code and checkout path may
        change. Any mismatch stops before model work. Numerical results print
        immediately; failed probes still skip the expensive full quality pass.
        '''),
        code('''
        if RUN_EXPERIMENT:
            run_live([*command, *(["--prepare-only"] if PREPARE_ONLY else [])], "packed-validation")
        else:
            print("Experiment disabled; preflight/plan only.")
        '''),
        md('''
        ## Checks
        ### 7. Verify saved results and inspect the decision
        No estimated storage can pass. The size report distinguishes an under-
        budget artifact from a two-sided 1%-matched artifact. Both still need
        independent quality confirmation and a fresh provider comparison.

        On reconnection, first check for the printed runner PID before restarting.
        Tail `logs/packed-validation.log`; inspect root and arm `progress.json`.
        Complete checkpoints and validation records resume by hashes. An interrupted
        patch reuses source Hessians; incomplete preparation is rebuilt. A completed
        export recovers from its hash-bound preparation record after interruption.
        Unverifiable/orphan artifacts are preserved, never destructively overwritten.
        '''),
        code('''
        import pandas as pd
        if not PREPARE_ONLY and (RESULT_ROOT / "summary.json").exists():
            run_live([*command, "--assess-only"], "assess")
            summary = json.loads((RESULT_ROOT / "summary.json").read_text())
            display(pd.DataFrame(summary["rows"]))
            print({key: summary[key] for key in ("complete", "missing", "artifact_validated_arms", "interpretation")})
        else:
            print("No complete validation summary yet. Check the logs and per-arm records.")
        print("Checkpoints stay on Drive; the compact download below excludes all tensor binaries.")
        '''),
        md("### 8. Download compact evidence"),
        code('''
        import zipfile
        if DOWNLOAD_RESULTS:
            archive = Path("/content") / f"qwen35_packed_validation_{RESULT_ROOT.name}.zip"
            paths = [*RESULT_ROOT.glob("*.json"), *RESULT_ROOT.glob("*.sha256"), *LOG_ROOT.glob("*.log")]
            for arm_root in RESULT_ROOT.glob("b5_v*_s*"):
                paths.extend(arm_root.glob("*.json"))
                paths.extend(arm_root.glob("*.sha256"))
                manifest = arm_root / "checkpoint/rotquant_config.json"
                if manifest.exists():
                    paths.append(manifest)
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(set(paths)):
                    bundle.write(path, path.relative_to(RESULT_ROOT))
                if REVALIDATE_FROM is not None:
                    # Include small, original records under a separate namespace
                    # without editing them or copying any checkpoint/probe tensors.
                    for seed in SEEDS:
                        for arm in ("b5_v6", "b5_v8"):
                            source_arm = REVALIDATE_FROM / f"{arm}_s{seed}"
                            originals = [*source_arm.glob("*.json"), *source_arm.glob("*.sha256"),
                                         source_arm / "checkpoint/rotquant_config.json"]
                            for path in sorted(set(originals)):
                                bundle.write(path, Path("original") / path.relative_to(REVALIDATE_FROM))
            from google.colab import files
            files.download(str(archive))
        '''),
        md('''
        ## Next Steps
        Bring back the compact bundle. Retain the actual checkpoints on Drive
        for inference and independent verification. A passed artifact gate is
        not a claim of Unsloth superiority: next freeze fresh validation inputs,
        replicate recipes across seeds, and rerun both providers with explicit
        reference/engine parity. Do not launch recovery or another allocation
        sweep automatically if an export/conformance gate fails.
        '''),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"packed-validation-{index:02d}"
    return new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": (REVALIDATION_OUTPUT if revalidate else OUTPUT).name}})


def main():
    for output, revalidate in ((OUTPUT, False), (REVALIDATION_OUTPUT, True)):
        notebook = build_notebook(revalidate=revalidate)
        nbformat.validate(notebook)
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        print(output)


if __name__ == "__main__":
    main()
