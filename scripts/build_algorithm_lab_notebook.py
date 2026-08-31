"""Generate the reader-facing Google Colab algorithm laboratory notebook."""
from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path("notebooks/rotquant_algorithm_lab_colab.ipynb")


def md(source: str):
    return new_markdown_cell(dedent(source).strip())


def code(source: str):
    return new_code_cell(dedent(source).strip())


def build_notebook():
    cells = [
        md("""
        # RotQuant algorithm laboratory

        A resumable Google Colab experiment notebook for low-bit vector
        quantization, calibrated scalar grids, mixed-bit allocation, W3/W4
        ablations, and selective K/V value retrieval.
        """),
        md("""
        ## Goal

        Screen every proposed algorithmic improvement under a matched protocol,
        promote only exact-rate Pareto candidates, and validate promoted recipes
        across seeds and a second model family. The notebook writes a compact,
        content-addressed audit trail to Google Drive so interrupted runs resume
        without repeating completed trials.
        """),
        md("""
        ### Key assumptions and boundaries

        - CUDA fallback is used for **quality only**. It materializes reconstructed
          weights and cannot support packed-memory or throughput claims.
        - The finite-rate vector codebook is research-only. It has exact packed
          rate and model-quality coverage, but checkpoint/native export remains
          deliberately blocked until it wins.
        - Every algorithmic arm uses the same model revision, evaluation subset,
          sequence length, rotation, exclusions, and seed.
        - `packed_weight_bytes` is the existing code/scale metric. Promotion uses
          `accounted_weight_bytes`, which additionally charges one fp32 centroid
          table per projection, including the larger vector tables.
        - Seed-0 screening is developmental. Promotion requires the full protocol,
          three seeds on the primary model, and optional cross-family confirmation.
        - Promoted profiles also receive a small held-out C4 free-running
          trajectory gate. It catches compounding token drift, but it is not a
          substitute for the planned diverse 300-prompt comparison suite.
        - Screening stores paired per-window NLL, confidence intervals, exact
          token identities, logit fidelity, complete persistent bytes, and wall
          time. Aggregate PPL alone never establishes a win.
        - The seeded random allocator and scalar 2.75-bpw allocator are negative
          controls, not novel candidates. Allocation/vector claims require a
          matched-format, matched-rate advantage.
        - Dense-attention top-k is an upper-bound oracle for selective V reads. It
          measures value sparsity; packed-key candidate recall is a separate gate.
        """),
        md("""
        ## Trial ladder

        1. Synthetic exact-rate scalar-versus-vector preflight.
        2. Four-window source/W4 sentinel with historical sanity gates.
        3. Seed-0, 16-window screen with a four-window catastrophic-loss stop.
        4. Exact-byte, paired-confidence, and matched-control promotion gates.
        5. Three-seed, 64-window WikiText/C4 validation with 32-token
           free-running trajectory agreement.
        6. Cross-family PPL, logit, layer-drift, and trajectory confirmation.
        7. Real-attention selective-V oracle with quantized-value and dense
           confidence-fallback curves.
        """),
        md("## Setup"),
        code("""
        from pathlib import Path

        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"
        REPO_DIR = Path("/content/rotquant-algorithm-lab")
        CONFIG_RELATIVE_PATH = Path("configs/algorithm_lab_cuda.yaml")

        MODEL_CASES = [
            {
                "name": "qwen35_4b",
                "model": "unsloth/Qwen3.5-4B",
                "revision": "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636",
                "include": ["model.language_model.layers."],
                "exclude": [
                    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
                    "lm_head", "embed_out",
                ],
            },
            {
                "name": "qwen25_3b",
                "model": "Qwen/Qwen2.5-3B",
                "revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
                "include": ["model.layers."],
                "exclude": ["lm_head", "embed_out"],
            },
        ]
        PRIMARY_MODEL = MODEL_CASES[0]
        DATASET_REVISIONS = {
            "c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
            "wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
        }

        USE_GOOGLE_DRIVE = True
        DRIVE_RESULT_ROOT = Path("/content/drive/MyDrive/rotquant/algorithm_lab")
        LOCAL_RESULT_ROOT = Path("/content/rotquant_algorithm_lab")

        CONFIRM_EXPENSIVE_RUN = False
        FORCE_RERUN = False
        RUN_SYNTHETIC_PREFLIGHT = True
        RUN_SENTINEL = True
        RUN_PRIMARY_SCREEN = True
        RUN_FULL_SEED_VALIDATION = True
        RUN_CROSS_FAMILY = True
        RUN_TRAJECTORY_VALIDATION = True
        RUN_LOGIT_FIDELITY = True
        RUN_SCREEN_LAYER_DRIFT = True
        RUN_LAYER_DRIFT = True
        RUN_SELECTIVE_KV_ORACLE = True
        DOWNLOAD_RESULTS = True
        REQUIRE_FAST_HADAMARD = True

        SENTINEL_MAX_SAMPLES = 4
        SENTINEL_W4_MAX_RELATIVE_PPL = 0.20
        SCREEN_MAX_SAMPLES = 16
        CONFIRM_MAX_SAMPLES = 64
        EVAL_SEQUENCE_LENGTH = 256
        VALIDATION_SEEDS = (0, 1, 2)
        CROSS_FAMILY_SEEDS = (0,)
        MAX_PROMOTED_PER_TRACK = 2
        MAX_SCREEN_RELATIVE_PPL = 1.0
        MAX_CONFIRM_RELATIVE_PPL = 0.25
        MIN_ALLOCATION_BYTE_SAVING = 0.01
        MATCHED_RATE_TOLERANCE = 0.0025
        EARLY_STOP_AFTER = 4
        EARLY_STOP_RELATIVE_PPL = 1.0
        BOOTSTRAP_DRAWS = 2000
        MATCHED_CONTROL_MAP = {
            **{
                f"vector_d2_w{bits}_rms": f"scalar_w{bits}_rms"
                for bits in (1, 2, 3)
            },
            "dynamic_local_3p625": "dynamic_random_3p625",
            "dynamic_teacher_3p625": "dynamic_random_3p625",
            "dynamic_guarded_teacher_3p625": "dynamic_random_3p625",
            "dynamic_vector_2p75": "dynamic_scalar_teacher_2p75",
        }
        PROMOTION_OUTCOME_PREFERENCE = (
            "dynamic_teacher_3p625",
            "dynamic_guarded_teacher_3p625",
        )

        # The in-notebook trajectory check is a cheap developmental sentinel.
        # A competitive claim requires the separate, diverse 300-prompt suite.
        COMPETITIVE_PROMPT_COUNT = 300
        COMPETITIVE_GENERATION_TOKENS = 32
        COMPETITIVE_DOMAINS = (
            "agentic", "code", "math", "multilingual", "long_document",
        )

        DYNAMIC_TEACHER_SKIP = 2048
        SCREEN_LAYER_DRIFT_SEQUENCE_LENGTH = 16
        SCREEN_LAYER_DRIFT_SKIP = 4096
        LAYER_DRIFT_SEQUENCE_LENGTH = 64
        LAYER_DRIFT_SKIP = 6144

        TRAJECTORY_BATCHES = 4
        TRAJECTORY_PROMPT_LENGTH = 64
        TRAJECTORY_NEW_TOKENS = 32
        TRAJECTORY_SKIP = 8192

        LOGIT_FIDELITY_BATCHES = 2
        LOGIT_FIDELITY_PROMPT_LENGTH = 64
        LOGIT_FIDELITY_SKIP = 12288

        RETRIEVAL_PROMPT_LENGTH = 512
        RETRIEVAL_SKIP = 16384
        RETRIEVAL_MASS_THRESHOLDS = (0.90, 0.95)
        RETRIEVAL_FRACTIONS = (1/16, 1/8, 1/4, 1/2, 1.0)
        RETRIEVAL_RECENT_WINDOW = 16
        RETRIEVAL_SINK_TOKENS = 4
        RETRIEVAL_VALUE_BITS = 3
        MAX_RETRIEVAL_LAYERS = None  # set an integer for a bounded smoke run

        print({
            "repo_ref": REPO_REF,
            "primary_model": PRIMARY_MODEL["model"],
            "screen_samples": SCREEN_MAX_SAMPLES,
            "confirm_samples": CONFIRM_MAX_SAMPLES,
            "confirm_expensive_run": CONFIRM_EXPENSIVE_RUN,
        })
        """),
        md("### 1. Verify the CUDA runtime"),
        code("""
        import os
        import subprocess
        import sys

        import torch

        assert torch.cuda.is_available(), "Select a CUDA GPU runtime before continuing."
        gpu = torch.cuda.get_device_properties(0)
        vram_gib = gpu.total_memory / 2**30
        print(f"GPU: {gpu.name} | VRAM: {vram_gib:.1f} GiB")
        print(f"torch={torch.__version__} | CUDA={torch.version.cuda}")
        if vram_gib < 24:
            print("WARNING: the 4B fallback matrix may OOM below 24 GiB; use the 3B case first.")
        subprocess.run(["nvidia-smi"], check=True)
        """),
        md("### 2. Mount Drive and fetch the exact repository revision"),
        code("""
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
            run_git([
                "clone", "--branch", REPO_REF, "--single-branch",
                REPO_URL, str(REPO_DIR),
            ])
        else:
            if not (REPO_DIR / ".git").is_dir():
                raise RuntimeError(
                    f"{REPO_DIR} exists but is not a Git checkout. "
                    "Choose a fresh REPO_DIR or remove the incomplete directory."
                )
            run_git(["fetch", "origin", REPO_REF], cwd=REPO_DIR)
            run_git(["checkout", REPO_REF], cwd=REPO_DIR)
            run_git(["pull", "--ff-only", "origin", REPO_REF], cwd=REPO_DIR)

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        COMMIT_ROOT = RESULT_BASE / commit[:12]
        COMMIT_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Using commit {commit}; result root: {COMMIT_ROOT}")
        """),
        md("### 3. Install the evaluation runtime without replacing CUDA PyTorch"),
        code("""
        import hashlib
        import importlib
        import importlib.metadata
        import json
        import platform

        runtime_packages = [
            "transformers==5.9.0", "datasets==4.8.5", "accelerate==1.13.0",
            "safetensors==0.7.0", "sentencepiece==0.2.1", "scipy==1.15.3",
            "pyyaml==6.0.3", "pandas==2.3.3", "matplotlib==3.10.9",
            "huggingface_hub==1.17.0", "ninja==1.13.0",
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", *runtime_packages],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR), "--no-deps"],
            check=True,
        )
        # The upstream v1.1.0 PyPI sdist points at a GitHub release with no
        # wheel assets, so pip falls through to a fragile Colab source build.
        # v1.1.0.post2 publishes wheels for the current CUDA/PyTorch matrix,
        # while retaining 1.1.0 as the distribution version inside the wheel.
        fht_release = "v1.1.0.post2"
        fht_version = "1.1.0"
        torch_major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        cuda_major = str(torch.version.cuda).split(".")[0]
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        machine = platform.machine().lower()
        wheel_platform = {
            "x86_64": "linux_x86_64",
            "amd64": "linux_x86_64",
            "aarch64": "linux_aarch64",
            "arm64": "linux_aarch64",
        }.get(machine, f"linux_{machine}")
        cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
        fht_wheel_name = (
            f"fast_hadamard_transform-{fht_version}+cu{cuda_major}"
            f"torch{torch_major_minor}cxx11abi{cxx11_abi}-"
            f"{python_tag}-{python_tag}-{wheel_platform}.whl"
        )
        fht_wheel_url = (
            "https://github.com/Dao-AILab/fast-hadamard-transform/"
            f"releases/download/{fht_release}/{fht_wheel_name}"
        )

        def install_kernel(command, label, *, env=None, stream_output=False):
            print(f"Installing fast Hadamard kernel via {label}...")
            run_options = {"check": False, "env": env}
            if not stream_output:
                run_options.update({"capture_output": True, "text": True})
            result = subprocess.run(command, **run_options)
            if result.returncode:
                if stream_output:
                    print(
                        f"{label} failed (exit {result.returncode}); "
                        "full output was streamed above."
                    )
                else:
                    output = "\\n".join(
                        part for part in (result.stdout, result.stderr) if part
                    )
                    print(f"{label} failed (exit {result.returncode}); output tail:")
                    print("\\n".join(output.splitlines()[-60:]))
            return result

        kernel_build = install_kernel([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            fht_wheel_url,
        ], "the upstream prebuilt wheel")
        fast_hadamard_install_method = "prebuilt_wheel"
        if kernel_build.returncode:
            kernel_env = os.environ.copy()
            kernel_env.update({"MAX_JOBS": "2", "NVCC_THREADS": "2"})
            kernel_build = install_kernel([
                sys.executable, "-m", "pip", "install", "-v", "--no-deps",
                "--no-build-isolation",
                f"git+https://github.com/Dao-AILab/fast-hadamard-transform.git@{fht_release}",
            ], "a bounded source build", env=kernel_env, stream_output=True)
            fast_hadamard_install_method = "source_build"

        fast_hadamard_available = kernel_build.returncode == 0
        fast_hadamard_error = (
            None if fast_hadamard_available else
            f"{fast_hadamard_install_method} exited with status "
            f"{kernel_build.returncode}; see installer output above"
        )
        if fast_hadamard_available:
            try:
                importlib.invalidate_caches()
                from fast_hadamard_transform import hadamard_transform

                with torch.no_grad():
                    smoke_input = torch.randn(
                        2, 128, device="cuda", dtype=torch.float16
                    )
                    smoke_output = hadamard_transform(smoke_input.contiguous())
                    torch.cuda.synchronize()
                    if smoke_output.shape != smoke_input.shape:
                        raise RuntimeError(
                            "CUDA smoke test returned an unexpected output shape"
                        )
                    if not bool(torch.isfinite(smoke_output).all().item()):
                        raise RuntimeError("CUDA smoke test returned non-finite values")
                del smoke_input, smoke_output
                print(
                    "Fast Hadamard CUDA smoke test passed via "
                    f"{fast_hadamard_install_method}."
                )
            except Exception as exc:
                fast_hadamard_available = False
                fast_hadamard_error = f"{type(exc).__name__}: {exc}"
                print(f"Fast Hadamard CUDA smoke test failed: {fast_hadamard_error}")
        if fast_hadamard_available:
            os.environ.pop("ROTQUANT_DISABLE_FAST_HADAMARD", None)
        else:
            # Fail closed even if the extension imported but its CUDA launch failed.
            # rotquant.rotate checks this flag at call time before selecting the kernel.
            os.environ["ROTQUANT_DISABLE_FAST_HADAMARD"] = "1"
        if REQUIRE_FAST_HADAMARD:
            assert fast_hadamard_available, (
                "No compatible fast-hadamard-transform kernel could be loaded. "
                "Installer diagnostics are printed above. For a slower diagnostic "
                "run only, set REQUIRE_FAST_HADAMARD=False and rerun this cell. "
                f"Attempted wheel: {fht_wheel_name}. "
                f"Smoke-test error: {fast_hadamard_error}"
            )
        elif not fast_hadamard_available:
            print("WARNING: using the much slower pure-torch FWHT fallback.")
        repo_path = str(REPO_DIR)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        driver_version = subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version",
            "--format=csv,noheader",
        ], text=True).strip().splitlines()[0]
        tracked_distributions = [
            "transformers", "datasets", "accelerate", "safetensors",
            "sentencepiece", "scipy", "pyyaml", "pandas", "matplotlib",
            "huggingface-hub",
        ]
        RUNTIME_MANIFEST = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver": driver_version,
            "fast_hadamard_transform": fast_hadamard_available,
            "fast_hadamard_disabled": not fast_hadamard_available,
            "fast_hadamard_install_method": (
                fast_hadamard_install_method if fast_hadamard_available else None
            ),
            "fast_hadamard_attempted_wheel": fht_wheel_name,
            "fast_hadamard_version": (
                importlib.metadata.version("fast-hadamard-transform")
                if fast_hadamard_available else None
            ),
            "packages": {
                name: importlib.metadata.version(name)
                for name in tracked_distributions
            },
        }
        RUNTIME_FINGERPRINT = hashlib.sha256(
            json.dumps(RUNTIME_MANIFEST, sort_keys=True).encode()
        ).hexdigest()[:12]
        RESULT_ROOT = COMMIT_ROOT / RUNTIME_FINGERPRINT
        RECORD_ROOT = RESULT_ROOT / "records"
        RAW_RUN_ROOT = RESULT_ROOT / "raw_runs"
        RECORD_ROOT.mkdir(parents=True, exist_ok=True)
        RAW_RUN_ROOT.mkdir(parents=True, exist_ok=True)
        print("Runtime ready without replacing the preinstalled CUDA torch wheel.")
        print(json.dumps(RUNTIME_MANIFEST, indent=2))
        print(f"Runtime-specific results: {RESULT_ROOT}")
        """),
        md("### 4. Validate the repository and predeclared matrix"),
        code("""
        import json

        import yaml

        CONFIG_PATH = REPO_DIR / CONFIG_RELATIVE_PATH
        required_paths = [
            CONFIG_PATH,
            REPO_DIR / "scripts/algorithmic_trials.py",
            REPO_DIR / "scripts/algorithmic_selection.py",
            REPO_DIR / "rotquant/codebooks.py",
            REPO_DIR / "rotquant/kv_cache.py",
            REPO_DIR / "scripts/run_experiment.py",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        assert not missing, "Missing required files: " + ", ".join(missing)

        from scripts.algorithmic_selection import (
            select_promoted_profiles,
            validation_status,
        )
        from scripts.algorithmic_trials import algorithmic_trial_matrix, trial_by_name

        TRIALS = algorithmic_trial_matrix()
        assert len(TRIALS) == len({trial.name for trial in TRIALS})
        assert {trial.track for trial in TRIALS} == {
            "control", "low_bit_vector", "codebook_scale", "allocation"
        }
        with CONFIG_PATH.open() as handle:
            base_config = yaml.safe_load(handle)
        assert base_config["patch"]["fallback"] is True
        print(f"Validated {len(TRIALS)} profiles across four research tracks.")
        print([trial.name for trial in TRIALS])
        """),
        md("## Context & Methods"),
        md("""
        ### 5. Run the synthetic exact-rate preflight

        This is a cheap implementation check, not a model-quality claim. It
        verifies that scalar and dimension-2 vector arms retain identical packed
        code and scale bytes before the expensive model sweep begins.
        """),
        code("""
        import pandas as pd

        synthetic_rows = []
        if RUN_SYNTHETIC_PREFLIGHT:
            from rotquant.eval.quantization import quantized_weight_metrics
            from rotquant.quantize import QuantConfig, Quantizer
            from rotquant.rotate import RandomizedHadamard

            generator = torch.Generator(device="cpu").manual_seed(2026)
            source_weight = torch.randn(256, 128, generator=generator)
            rotation = RandomizedHadamard(128, block=128, seed=17)
            rotated_weight = rotation.rotate_weight(source_weight)
            probes = torch.randn(128, 128, generator=generator)
            for bits in (1, 2, 3):
                for kind in ("gaussian", "vector"):
                    config = QuantConfig(
                        bits=bits, codebook=kind, scale="rms", group_size=128,
                        vector_dim=2, vector_samples=16384, vector_iters=25,
                    )
                    quantized = Quantizer(config).quantize_weight(rotated_weight)
                    metrics = quantized_weight_metrics(
                        rotated_weight, quantized, probes=probes
                    )
                    synthetic_rows.append({
                        "bits": bits, "codebook": kind,
                        "packed_words": quantized.packed.data.numel(), **metrics,
                        "codebook_bytes": (
                            quantized.codebook.centroids.numel()
                            * quantized.codebook.centroids.element_size()
                        ),
                        "matrix_code_scale_bytes": (
                            quantized.packed.data.numel()
                            * quantized.packed.data.element_size()
                            + quantized.scales.numel()
                            * quantized.scales.element_size()
                        ),
                    })
            synthetic_table = pd.DataFrame(synthetic_rows)
            for bits in (1, 2, 3):
                matched = synthetic_table[synthetic_table["bits"] == bits]
                assert matched["effective_bpw"].nunique() == 1
                assert matched["packed_words"].nunique() == 1
                assert matched["matrix_code_scale_bytes"].nunique() == 1
            display(synthetic_table)
        else:
            synthetic_table = pd.DataFrame()
            print("Synthetic preflight disabled.")
        """),
        md("### 6. Define the content-addressed, resumable model runner"),
        code("""
        import gc
        import hashlib
        import os
        import time
        import traceback
        from dataclasses import asdict

        import numpy as np

        from scripts.run_experiment import run as run_experiment
        from rotquant.utils import write_result

        trial_records = {}
        FAILURE_ROOT = RESULT_ROOT / "failures"
        FAILURE_ROOT.mkdir(parents=True, exist_ok=True)

        def persist_frame(name, frame):
            # Atomically checkpoint a derived table beside per-trial records.
            if frame.empty:
                return None
            path = RESULT_ROOT / f"{name}.csv"
            temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)
            return path

        def ppl_datasets(stage):
            return (
                ["wikitext2", "c4"]
                if stage in {"validation", "cross_family"}
                else ["wikitext2"]
            )

        def trajectory_settings(stage):
            if not RUN_TRAJECTORY_VALIDATION or stage not in {
                "validation", "cross_family"
            }:
                return False
            return {
                "batches": TRAJECTORY_BATCHES,
                "prompt_len": TRAJECTORY_PROMPT_LENGTH,
                "new_tokens": TRAJECTORY_NEW_TOKENS,
                "skip": TRAJECTORY_SKIP,
                "use_cache": True,
            }

        def logit_fidelity_settings(stage):
            if not RUN_LOGIT_FIDELITY or stage not in {
                "validation", "cross_family"
            }:
                return False
            return {
                "batches": LOGIT_FIDELITY_BATCHES,
                "prompt_len": LOGIT_FIDELITY_PROMPT_LENGTH,
                "skip": LOGIT_FIDELITY_SKIP,
                "temperature": 1.0,
            }

        def layer_drift_settings(stage):
            if stage == "screen" and RUN_SCREEN_LAYER_DRIFT:
                return {
                    "enabled": True,
                    "seq_len": SCREEN_LAYER_DRIFT_SEQUENCE_LENGTH,
                    "skip": SCREEN_LAYER_DRIFT_SKIP,
                }
            if RUN_LAYER_DRIFT and stage in {"validation", "cross_family"}:
                return {
                    "enabled": True,
                    "seq_len": LAYER_DRIFT_SEQUENCE_LENGTH,
                    "skip": LAYER_DRIFT_SKIP,
                }
            return {"enabled": False, "seq_len": 0, "skip": 0}

        def record_path(stage, model_case, profile, seed, max_samples):
            specification = {
                "commit": commit,
                "stage": stage,
                "model": model_case,
                "profile": asdict(profile),
                "seed": seed,
                "max_samples": max_samples,
                "seq_len": EVAL_SEQUENCE_LENGTH,
                "ppl_datasets": ppl_datasets(stage),
                "trajectory": trajectory_settings(stage),
                "logit_fidelity": logit_fidelity_settings(stage),
                "layer_drift": layer_drift_settings(stage),
                "dynamic_teacher_skip": DYNAMIC_TEACHER_SKIP,
                "early_stop": {
                    "after": EARLY_STOP_AFTER,
                    "relative_ppl": EARLY_STOP_RELATIVE_PPL,
                },
                "runtime": RUNTIME_FINGERPRINT,
            }
            digest = hashlib.sha256(
                json.dumps(specification, sort_keys=True).encode()
            ).hexdigest()[:16]
            return RECORD_ROOT / f"{stage}_{model_case['name']}_{profile.name}_s{seed}_{digest}.json"

        def release_cuda_memory():
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as exc:
                    # A device-side assert poisons the CUDA context. Preserve the
                    # original trial exception instead of replacing it here.
                    print(
                        "WARNING: CUDA cleanup failed; restart the runtime before "
                        f"continuing. Cleanup error: {type(exc).__name__}: {exc}"
                    )

        def record_complete(record, stage):
            try:
                metrics = record["payload"]["metrics"]
                for dataset in ppl_datasets(stage):
                    assert f"ppl_{dataset}" in metrics
                    assert f"ppl_{dataset}_details" in metrics
                if trajectory_settings(stage):
                    assert "trajectory" in metrics
                if logit_fidelity_settings(stage):
                    assert "logit_fidelity" in metrics
                if layer_drift_settings(stage)["enabled"]:
                    assert "layer_mse" in metrics
                return True
            except (AssertionError, KeyError, TypeError):
                return False

        def assert_data_partitions(metrics):
            manifest = metrics.get("data_manifest", {})
            calibration_names = {
                "hessian_calibration", "activation_calibration",
                "allocation_teacher", "block_calibration",
            }
            diagnostic_names = {
                "layer_drift", "trajectory", "logit_fidelity",
            }

            def rows(name):
                return {
                    row for row in manifest.get(name, {}).get("source_rows", [])
                    if row is not None
                }

            calibration_rows = set().union(*(
                rows(name) for name in calibration_names
            ))
            for name in diagnostic_names:
                assert calibration_rows.isdisjoint(rows(name)), (
                    f"Calibration and {name} rows overlap."
                )
            active_diagnostics = [
                (name, rows(name)) for name in diagnostic_names if rows(name)
            ]
            for index, (left_name, left_rows) in enumerate(active_diagnostics):
                for right_name, right_rows in active_diagnostics[index + 1:]:
                    assert left_rows.isdisjoint(right_rows), (
                        f"Held-out partitions {left_name} and {right_name} overlap."
                    )

        def print_trial_progress(record):
            completed = len(trial_records)
            screen_completed = sum(
                key[0] == "screen" for key in trial_records
            )
            observed = [
                float(item.get("trial_wall_seconds", 0.0))
                for item in trial_records.values()
                if item.get("trial_wall_seconds")
            ]
            if observed:
                mean_seconds = sum(observed) / len(observed)
                print({
                    "completed_records": completed,
                    "latest_minutes": round(
                        record.get("trial_wall_seconds", 0.0) / 60, 2
                    ),
                    "mean_minutes": round(mean_seconds / 60, 2),
                    "estimated_screen_minutes_remaining": round(
                        max(0, len(TRIALS) - screen_completed)
                        * mean_seconds / 60, 1
                    ),
                })

        def run_trial(stage, model_case, profile, seed, max_samples):
            path = record_path(stage, model_case, profile, seed, max_samples)
            key = (stage, model_case["name"], profile.name, seed, max_samples)
            if path.exists() and not FORCE_RERUN:
                try:
                    record = json.loads(path.read_text())
                except json.JSONDecodeError:
                    record = None
                if record is not None and record_complete(record, stage):
                    assert_data_partitions(record["payload"]["metrics"])
                    trial_records[key] = record
                    print(f"resume {path.name}")
                    return record
                invalid_path = path.with_name(
                    f"{path.stem}.invalid-{time.time_ns()}{path.suffix}"
                )
                os.replace(path, invalid_path)
                print(f"quarantined incomplete record: {invalid_path.name}")
            sets = [
                *profile.sets(),
                ("patch.include", model_case["include"]),
                ("patch.exclude", model_case["exclude"]),
                ("dynamic_teacher_skip", DYNAMIC_TEACHER_SKIP),
                ("eval.ppl.seq_len", EVAL_SEQUENCE_LENGTH),
                ("eval.ppl.max_samples", max_samples),
                ("eval.ppl_datasets", ppl_datasets(stage)),
                ("eval.trajectory", trajectory_settings(stage)),
                ("eval.logit_fidelity", logit_fidelity_settings(stage)),
                ("eval.layer_mse", layer_drift_settings(stage)["enabled"]),
                ("eval.layer_mse_seq_len", layer_drift_settings(stage)["seq_len"]),
                ("eval.layer_mse_skip", layer_drift_settings(stage)["skip"]),
                ("eval.zeroshot", False),
            ]
            if stage == "screen" and profile.name != "source_fp16":
                source_key = (
                    "screen", model_case["name"], "source_fp16", 0,
                    SCREEN_MAX_SAMPLES,
                )
                source_metrics = trial_records[source_key]["payload"]["metrics"]
                source_details = source_metrics["ppl_wikitext2_details"]
                sets.extend([
                    ("eval.ppl.early_stop_after", EARLY_STOP_AFTER),
                    ("eval.ppl.early_stop_relative_ppl", EARLY_STOP_RELATIVE_PPL),
                    ("eval.ppl.reference_window_nll_sums",
                     source_details["window_nll_sums"]),
                    ("eval.ppl.reference_window_tokens",
                     source_details["window_tokens"]),
                ])
            started = time.perf_counter()
            try:
                payload = run_experiment(
                    str(CONFIG_PATH),
                    output_dir=str(RAW_RUN_ROOT),
                    overrides={
                        "model": model_case["model"], "device": "cuda",
                        "seed": seed, "model_revision": model_case["revision"],
                    },
                    sets=sets,
                )
                assert_data_partitions(payload["metrics"])
            except Exception as exc:
                failure = {
                    "stage": stage,
                    "model_case": model_case,
                    "profile": asdict(profile),
                    "seed": seed,
                    "max_samples": max_samples,
                    "wall_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failure_path = FAILURE_ROOT / (
                    f"{path.stem}.failure-{time.time_ns()}.json"
                )
                write_result(str(failure_path), failure)
                raise
            finally:
                release_cuda_memory()
            record = {
                "stage": stage,
                "model_case": model_case,
                "profile": asdict(profile),
                "seed": seed,
                "max_samples": max_samples,
                "payload": payload,
                "trial_wall_seconds": time.perf_counter() - started,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            assert record_complete(record, stage)
            write_result(str(path), record)
            trial_records[key] = record
            print_trial_progress(record)
            return record

        def paired_window_statistics(candidate, source):
            if not candidate or not source:
                return {}
            windows = min(
                len(candidate["window_mean_nll"]),
                len(source["window_mean_nll"]),
            )
            assert candidate["window_hashes"][:windows] == source[
                "window_hashes"
            ][:windows], "paired PPL windows do not match"
            differences = np.asarray(
                candidate["window_mean_nll"][:windows], dtype=float
            ) - np.asarray(source["window_mean_nll"][:windows], dtype=float)
            rng = np.random.default_rng(20260831)
            sampled = differences[rng.integers(
                0, windows, size=(BOOTSTRAP_DRAWS, windows)
            )].mean(axis=1)
            source_tokens = sum(source["window_tokens"][:windows])
            source_nll = sum(source["window_nll_sums"][:windows])
            source_ppl = float(np.exp(source_nll / source_tokens))
            return {
                "paired_windows": windows,
                "paired_source_ppl": source_ppl,
                "paired_nll_delta": float(differences.mean()),
                "paired_nll_ci_low": float(np.quantile(sampled, 0.025)),
                "paired_nll_ci_high": float(np.quantile(sampled, 0.975)),
                "paired_window_win_rate": float((differences < 0).mean()),
            }

        def report_row(record, source_record=None):
            payload = record["payload"]
            metrics = payload["metrics"]
            config = payload["config"]
            dynamic = metrics.get("dynamic_quantization", {})
            trajectory = metrics.get("trajectory") or {}
            logit_fidelity = metrics.get("logit_fidelity") or {}
            layer_mse = metrics.get("layer_mse") or {}
            layer_errors = layer_mse.get("mse") or {}
            ppl = float(metrics["ppl_wikitext2"])
            ppl_details = metrics.get("ppl_wikitext2_details") or {}
            source_metrics = (
                source_record["payload"]["metrics"]
                if source_record is not None else {}
            )
            paired = paired_window_statistics(
                ppl_details, source_metrics.get("ppl_wikitext2_details")
            )
            source_ppl = paired.get("paired_source_ppl")
            c4_ppl = metrics.get("ppl_c4")
            c4_paired = paired_window_statistics(
                metrics.get("ppl_c4_details"),
                source_metrics.get("ppl_c4_details"),
            )
            quant = config.get("quant") or {}
            bits = quant.get("bits")
            codebook = str(quant.get("codebook", "gaussian")).lower()
            vector_dim = int(quant.get("vector_dim", 2))
            quant_layers = int(metrics.get("n_quant_layers", 0))
            vector = codebook in {
                "vector", "vector_kmeans", "product_vector", "pq"
            }

            def centroid_values(candidate_bits):
                if candidate_bits is None:
                    return 0
                levels = 1 << (candidate_bits * vector_dim if vector else candidate_bits)
                return levels * vector_dim if vector else levels

            counts_by_bits = dynamic.get("counts_by_bits") or {}
            if counts_by_bits:
                centroid_values_total = sum(
                    int(count) * centroid_values(int(candidate_bits))
                    for candidate_bits, count in counts_by_bits.items()
                )
            else:
                centroid_values_total = quant_layers * centroid_values(bits)
            # Current scalar checkpoints retain one fp32 centroid table per
            # projection. Charge the same conservative per-layer scope to the
            # research vector arm even though a future format could share it.
            codebook_bytes = metrics.get(
                "codebook_bytes", centroid_values_total * 4
            )
            packed_weight_bytes = metrics.get("packed_weight_bytes")
            accounted_weight_bytes = (
                packed_weight_bytes + codebook_bytes
                if packed_weight_bytes is not None else None
            )
            quantized_weights = metrics.get("fp16_weight_bytes", 0) / 2
            return {
                "stage": record["stage"],
                "model": record["model_case"]["name"],
                "profile": record["profile"]["name"],
                "track": record["profile"]["track"],
                "research_only": record["profile"]["research_only"],
                "seed": record["seed"],
                "max_samples": record["max_samples"],
                "bits": bits,
                "codebook": quant.get("codebook"),
                "scale": quant.get("scale"),
                "ppl": ppl,
                "relative_ppl": (
                    ppl / source_ppl - 1 if source_ppl is not None else float("nan")
                ),
                **paired,
                "ppl_windows": ppl_details.get("windows", 0),
                "ppl_stopped_early": ppl_details.get("stopped_early", False),
                "ppl_input_digest": ppl_details.get("input_digest"),
                "ppl_c4": c4_ppl,
                "relative_ppl_c4": (
                    c4_ppl / c4_paired["paired_source_ppl"] - 1
                    if c4_ppl is not None and c4_paired else float("nan")
                ),
                "paired_c4_nll_delta": c4_paired.get(
                    "paired_nll_delta", float("nan")
                ),
                "packed_weight_bytes": packed_weight_bytes,
                "codebook_bytes": codebook_bytes,
                "accounted_weight_bytes": accounted_weight_bytes,
                "effective_bpw": metrics.get("bits_per_weight_mean"),
                "accounted_bpw": (
                    accounted_weight_bytes * 8 / quantized_weights
                    if quantized_weights else None
                ),
                "complete_persistent_model_bytes": metrics.get(
                    "complete_persistent_model_bytes"
                ),
                "quality_runtime_model_bytes": metrics.get(
                    "quality_runtime_model_bytes"
                ),
                "fallback_cache_bytes": metrics.get("fallback_cache_bytes", 0),
                "patch_seconds": metrics.get("patch_seconds", 0.0),
                "model_load_seconds": metrics.get("model_load_seconds", 0.0),
                "peak_vram_bytes_patch": metrics.get("peak_vram_bytes_patch", 0),
                "peak_vram_bytes_eval": metrics.get("peak_vram_bytes_eval", 0),
                "trial_wall_seconds": record.get("trial_wall_seconds", 0.0),
                "target_reached": dynamic.get("target_reached", True),
                "dynamic_achieved_bpw": dynamic.get("achieved_bpw"),
                "dynamic_counts": dynamic.get("counts_by_bits"),
                "trajectory_prompts": trajectory.get("prompts", 0),
                "trajectory_token_agreement": trajectory.get(
                    "token_agreement", float("nan")
                ),
                "trajectory_exact_rate": trajectory.get(
                    "exact_trajectory_rate", float("nan")
                ),
                "trajectory_mean_matching_prefix": trajectory.get(
                    "mean_matching_prefix", float("nan")
                ),
                "mean_teacher_kl": logit_fidelity.get(
                    "mean_teacher_kl", float("nan")
                ),
                "median_teacher_kl": logit_fidelity.get(
                    "median_teacher_kl", float("nan")
                ),
                "p95_teacher_kl": logit_fidelity.get(
                    "p95_teacher_kl", float("nan")
                ),
                "max_teacher_kl": logit_fidelity.get(
                    "max_teacher_kl", float("nan")
                ),
                "mean_logit_cosine": logit_fidelity.get(
                    "mean_logit_cosine", float("nan")
                ),
                "top1_agreement": logit_fidelity.get(
                    "top1_agreement", float("nan")
                ),
                "logit_nll_delta": logit_fidelity.get(
                    "nll_delta", float("nan")
                ),
                "mean_layer_nmse": (
                    float(np.mean(list(layer_errors.values())))
                    if layer_errors else float("nan")
                ),
                "worst_layer_nmse": (
                    max(layer_errors.values())
                    if layer_errors else float("nan")
                ),
                "worst_layer": (
                    max(layer_errors, key=layer_errors.get)
                    if layer_errors else None
                ),
                "data_manifest": metrics.get("data_manifest", {}),
            }
        """),
        md("### 7. Confirm the expensive matrix"),
        code("""
        requested = any([
            RUN_SENTINEL, RUN_PRIMARY_SCREEN, RUN_FULL_SEED_VALIDATION,
            RUN_CROSS_FAMILY, RUN_SELECTIVE_KV_ORACLE,
        ])
        if requested:
            assert CONFIRM_EXPENSIVE_RUN, (
                "Review the model IDs, gates, Drive path, and estimated GPU cost, "
                "then set CONFIRM_EXPENSIVE_RUN=True."
            )
        """),
        md("## Data & Trials"),
        md("### 8. Run the source/W4 fail-fast sentinel"),
        code("""
        sentinel_table = pd.DataFrame()
        if RUN_SENTINEL:
            sentinel_source = run_trial(
                "sentinel", PRIMARY_MODEL, trial_by_name("source_fp16"),
                0, SENTINEL_MAX_SAMPLES,
            )
            sentinel_w4 = run_trial(
                "sentinel", PRIMARY_MODEL, trial_by_name("gaussian_w4_mse"),
                0, SENTINEL_MAX_SAMPLES,
            )
            sentinel_table = pd.DataFrame([
                report_row(sentinel_source, sentinel_source),
                report_row(sentinel_w4, sentinel_source),
            ])
            w4_row = sentinel_table[
                sentinel_table["profile"] == "gaussian_w4_mse"
            ].iloc[0]
            assert np.isfinite(w4_row["ppl"])
            assert w4_row["relative_ppl"] <= SENTINEL_W4_MAX_RELATIVE_PPL, (
                "Known W4 control missed its sentinel quality range; stop before "
                "spending GPU time on the matrix."
            )
            assert w4_row["complete_persistent_model_bytes"] < sentinel_table[
                sentinel_table["profile"] == "source_fp16"
            ]["complete_persistent_model_bytes"].iloc[0]
            assert sentinel_w4["payload"]["metrics"]["patched_modules"] > 0
            persist_frame("sentinel", sentinel_table)
            display(sentinel_table)
        """),
        md("### 9. Run the prioritized primary-model seed-0 screen"),
        code("""
        screen_records = []
        if RUN_PRIMARY_SCREEN:
            for profile in TRIALS:
                screen_records.append(run_trial(
                    "screen", PRIMARY_MODEL, profile, 0, SCREEN_MAX_SAMPLES
                ))
        else:
            print("Primary screen disabled.")
        """),
        md("### 10. Check exact rates and build the screening table"),
        code("""
        if screen_records:
            source_record = next(
                record for record in screen_records
                if record["profile"]["name"] == "source_fp16"
            )
            screen_source_ppl = float(
                source_record["payload"]["metrics"]["ppl_wikitext2"]
            )
            screen_table = pd.DataFrame([
                report_row(record, source_record) for record in screen_records
            ])
            records_by_name = {
                record["profile"]["name"]: record for record in screen_records
            }
            screen_table["matched_control"] = screen_table["profile"].map(
                MATCHED_CONTROL_MAP
            )
            screen_table["matched_control_nll_delta"] = np.nan
            screen_table["matched_control_ci_low"] = np.nan
            screen_table["matched_control_ci_high"] = np.nan
            screen_table["matched_rate"] = False
            for candidate_name, control_name in MATCHED_CONTROL_MAP.items():
                comparison = paired_window_statistics(
                    records_by_name[candidate_name]["payload"]["metrics"][
                        "ppl_wikitext2_details"
                    ],
                    records_by_name[control_name]["payload"]["metrics"][
                        "ppl_wikitext2_details"
                    ],
                )
                candidate_bytes = float(screen_table.loc[
                    screen_table["profile"] == candidate_name,
                    "complete_persistent_model_bytes",
                ].iloc[0])
                control_bytes = float(screen_table.loc[
                    screen_table["profile"] == control_name,
                    "complete_persistent_model_bytes",
                ].iloc[0])
                selector = screen_table["profile"] == candidate_name
                screen_table.loc[
                    selector, "matched_control_nll_delta"
                ] = comparison["paired_nll_delta"]
                screen_table.loc[
                    selector, "matched_control_ci_low"
                ] = comparison["paired_nll_ci_low"]
                screen_table.loc[
                    selector, "matched_control_ci_high"
                ] = comparison["paired_nll_ci_high"]
                screen_table.loc[selector, "matched_rate"] = (
                    abs(candidate_bytes / control_bytes - 1)
                    <= MATCHED_RATE_TOLERANCE
                )
            for bits in (1, 2, 3):
                pair = screen_table[screen_table["profile"].isin([
                    f"scalar_w{bits}_rms", f"vector_d2_w{bits}_rms"
                ])]
                assert len(pair) == 2
                assert pair["packed_weight_bytes"].nunique() == 1, pair
                assert pair["effective_bpw"].nunique() == 1, pair
            assert screen_table.loc[
                screen_table["track"] == "allocation", "target_reached"
            ].all()
            persist_frame("screen", screen_table)
            display(screen_table.sort_values([
                "track", "complete_persistent_model_bytes", "ppl"
            ]).style.format({
                "ppl": "{:.4f}", "relative_ppl": "{:+.2%}",
                "paired_nll_delta": "{:+.5f}",
                "paired_nll_ci_low": "{:+.5f}",
                "paired_nll_ci_high": "{:+.5f}",
                "effective_bpw": "{:.4f}", "accounted_bpw": "{:.4f}",
                "packed_weight_bytes": "{:,.0f}",
                "accounted_weight_bytes": "{:,.0f}",
                "complete_persistent_model_bytes": "{:,.0f}",
            }))
        else:
            screen_source_ppl = float("nan")
            screen_table = pd.DataFrame()
        """),
        md("### 11. Promote only bounded, matched-control Pareto candidates"),
        code("""
        promotion_decisions = pd.DataFrame()
        promoted_names = []
        if not screen_table.empty:
            promoted_names, decision_rows = select_promoted_profiles(
                screen_table.to_dict("records"),
                matched_controls=MATCHED_CONTROL_MAP,
                always_include=("gaussian_w4_mse",),
                outcome_preference=PROMOTION_OUTCOME_PREFERENCE,
                max_per_track=MAX_PROMOTED_PER_TRACK,
                max_relative_ppl=MAX_SCREEN_RELATIVE_PPL,
                min_allocation_byte_saving=MIN_ALLOCATION_BYTE_SAVING,
            )
            promotion_decisions = pd.DataFrame(decision_rows)
            persist_frame("promotion_decisions", promotion_decisions)
            display(promotion_decisions)

        promoted_profiles = [trial_by_name(name) for name in promoted_names]
        validation_profile_names = set(promoted_names)
        validation_profile_names.update(
            MATCHED_CONTROL_MAP[name]
            for name in promoted_names if name in MATCHED_CONTROL_MAP
        )
        validation_profile_names = sorted(validation_profile_names)
        validation_profiles = [
            trial_by_name(name) for name in validation_profile_names
        ]
        print({
            "promoted_profiles": promoted_names,
            "validation_profiles_including_controls": validation_profile_names,
        })
        """),
        md("### 11. Validate promoted profiles across three full primary-model seeds"),
        code("""
        validation_records = []
        full_source_record = None
        if RUN_FULL_SEED_VALIDATION:
            full_source_record = run_trial(
                "validation", PRIMARY_MODEL, trial_by_name("source_fp16"),
                0, CONFIRM_MAX_SAMPLES,
            )
            for profile in validation_profiles:
                for seed in VALIDATION_SEEDS:
                    validation_records.append(run_trial(
                        "validation", PRIMARY_MODEL, profile,
                        seed, CONFIRM_MAX_SAMPLES,
                    ))
        else:
            print("Full seed validation disabled; seed-0 rankings remain developmental.")

        if full_source_record is not None:
            full_source_ppl = float(
                full_source_record["payload"]["metrics"]["ppl_wikitext2"]
            )
            validation_table = pd.DataFrame([
                report_row(record, full_source_record)
                for record in validation_records
            ])
            for record in validation_records:
                manifest = record["payload"]["metrics"]["data_manifest"]
                diagnostic_digests = [
                    manifest[name]["digest"]
                    for name in ("layer_drift", "trajectory", "logit_fidelity")
                    if name in manifest
                ]
                assert len(diagnostic_digests) == len(set(diagnostic_digests)), (
                    "Held-out diagnostic partitions overlap."
                )
            persist_frame("validation", validation_table)
            display(validation_table.sort_values(["profile", "seed"]))
        else:
            full_source_ppl = float("nan")
            validation_table = pd.DataFrame()

        def matched_control_table(records, candidate_names, seeds):
            rows = []
            by_key = {
                (
                    record["model_case"]["name"],
                    record["profile"]["name"],
                    record["seed"],
                ): record
                for record in records
            }
            models = sorted({record["model_case"]["name"] for record in records})
            for model_name in models:
                for candidate_name in candidate_names:
                    control_name = MATCHED_CONTROL_MAP.get(candidate_name)
                    if control_name is None:
                        continue
                    for seed in seeds:
                        candidate_record = by_key[
                            (model_name, candidate_name, seed)
                        ]
                        control_record = by_key[(model_name, control_name, seed)]
                        candidate_metrics = candidate_record["payload"]["metrics"]
                        control_metrics = control_record["payload"]["metrics"]
                        wiki = paired_window_statistics(
                            candidate_metrics["ppl_wikitext2_details"],
                            control_metrics["ppl_wikitext2_details"],
                        )
                        c4 = paired_window_statistics(
                            candidate_metrics["ppl_c4_details"],
                            control_metrics["ppl_c4_details"],
                        )
                        candidate_bytes = candidate_metrics[
                            "complete_persistent_model_bytes"
                        ]
                        control_bytes = control_metrics[
                            "complete_persistent_model_bytes"
                        ]
                        matched_rate = abs(candidate_bytes / control_bytes - 1) <= (
                            MATCHED_RATE_TOLERANCE
                        )
                        rows.append({
                            "model": model_name,
                            "candidate": candidate_name,
                            "control": control_name,
                            "seed": seed,
                            "complete_byte_ratio": candidate_bytes / control_bytes,
                            "matched_rate": matched_rate,
                            "wiki_nll_delta": wiki["paired_nll_delta"],
                            "wiki_ci_low": wiki["paired_nll_ci_low"],
                            "wiki_ci_high": wiki["paired_nll_ci_high"],
                            "c4_nll_delta": c4["paired_nll_delta"],
                            "c4_ci_low": c4["paired_nll_ci_low"],
                            "c4_ci_high": c4["paired_nll_ci_high"],
                            "confidence_confirmed": (
                                matched_rate
                                and wiki["paired_nll_ci_high"] < 0
                                and c4["paired_nll_ci_high"] < 0
                            ),
                        })
            return pd.DataFrame(rows)

        validation_control_table = matched_control_table(
            validation_records,
            promoted_names if validation_records else [],
            VALIDATION_SEEDS,
        )
        if not validation_control_table.empty:
            persist_frame("validation_matched_controls", validation_control_table)
            display(validation_control_table)
        """),
        md("### 12. Confirm promoted profiles on a second model family"),
        code("""
        cross_family_records = []
        if RUN_CROSS_FAMILY:
            for model_case in MODEL_CASES[1:]:
                source = run_trial(
                    "cross_family", model_case, trial_by_name("source_fp16"),
                    0, CONFIRM_MAX_SAMPLES,
                )
                for profile in validation_profiles:
                    for seed in CROSS_FAMILY_SEEDS:
                        cross_family_records.append(run_trial(
                            "cross_family", model_case, profile,
                            seed, CONFIRM_MAX_SAMPLES,
                        ))
        else:
            print("Cross-family validation disabled.")

        cross_rows = []
        for record in cross_family_records:
            model_case = record["model_case"]
            source_key = (
                "cross_family", model_case["name"], "source_fp16",
                0, CONFIRM_MAX_SAMPLES,
            )
            cross_rows.append(report_row(record, trial_records[source_key]))
        cross_family_table = pd.DataFrame(cross_rows)
        if not cross_family_table.empty:
            persist_frame("cross_family", cross_family_table)
            display(cross_family_table.sort_values(["model", "profile", "seed"]))

        cross_family_control_table = matched_control_table(
            cross_family_records,
            promoted_names if cross_family_records else [],
            CROSS_FAMILY_SEEDS,
        )
        if not cross_family_control_table.empty:
            persist_frame(
                "cross_family_matched_controls", cross_family_control_table
            )
            display(cross_family_control_table)
        """),
        md("## Selective K/V retrieval"),
        md("""
        ### 13. Collect real attention probabilities and cache values

        This stage loads the source primary model with eager attention, captures
        one held-out 512-token C4 prompt, and evaluates both source values and
        3-bit RotQuant-reconstructed values. Dense probabilities choose the
        candidates, so the result is a value-bandwidth upper bound.
        """),
        code("""
        retrieval_inputs = None
        retrieval_output = None
        retrieval_model = None
        retrieval_manifest = None
        if RUN_SELECTIVE_KV_ORACLE:
            from scripts.run_experiment import (
                build_calib_loader,
                load_hf_model,
                token_batch_manifest,
            )

            retrieval_model, retrieval_tokenizer, _ = load_hf_model(
                PRIMARY_MODEL["model"], torch.float16, "cuda", "auto",
                PRIMARY_MODEL["revision"],
            )
            if hasattr(retrieval_model, "set_attn_implementation"):
                retrieval_model.set_attn_implementation("eager")
            retrieval_model.config.output_attentions = True
            retrieval_inputs = build_calib_loader(
                retrieval_tokenizer, 1, RETRIEVAL_PROMPT_LENGTH,
                "cuda", skip=RETRIEVAL_SKIP,
                revision=DATASET_REVISIONS["c4"],
            )[0]
            retrieval_manifest = token_batch_manifest(
                [retrieval_inputs], dataset="allenai/c4", split="train",
                revision=DATASET_REVISIONS["c4"], skip=RETRIEVAL_SKIP,
                seq_len=RETRIEVAL_PROMPT_LENGTH,
            )
            with torch.inference_mode():
                retrieval_output = retrieval_model(
                    **retrieval_inputs,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
            print("Captured a real held-out attention/cache boundary.")
        """),
        md("### 14. Measure source and quantized-value retrieval curves"),
        code("""
        retrieval_rows = []

        def find_output_attribute(root, name, max_depth=3):
            frontier = [(root, 0)]
            seen = set()
            while frontier:
                current, depth = frontier.pop(0)
                if id(current) in seen:
                    continue
                seen.add(id(current))
                value = getattr(current, name, None)
                if value is not None:
                    return value
                if depth >= max_depth or torch.is_tensor(current):
                    continue
                for child in getattr(current, "__dict__", {}).values():
                    if child is not None and not isinstance(
                        child, (str, int, float, bool)
                    ):
                        frontier.append((child, depth + 1))
            return None

        if retrieval_output is not None:
            from rotquant.kv_cache import (
                KVQuantConfig,
                build_kv_rotation,
                oracle_value_retrieval_curve,
                quantize_kv,
            )

            attentions = find_output_attribute(retrieval_output, "attentions")
            cache = find_output_attribute(retrieval_output, "past_key_values")
            assert attentions is not None, (
                "Model returned no attentions; ensure eager attention is active."
            )
            assert cache is not None and hasattr(cache, "layers")
            used_layers = 0
            for layer_index, attention in enumerate(attentions):
                if attention is None or layer_index >= len(cache.layers):
                    continue
                values = getattr(cache.layers[layer_index], "values", None)
                if values is None or values.numel() == 0 or values.ndim != 4:
                    continue
                attention = attention[..., -1:, :values.shape[-2]]
                sequence = values.shape[-2]
                mandatory = RETRIEVAL_RECENT_WINDOW + RETRIEVAL_SINK_TOKENS
                counts = sorted({
                    min(sequence, max(mandatory, round(sequence * fraction)))
                    for fraction in RETRIEVAL_FRACTIONS
                })
                value_config = KVQuantConfig(
                    bits=RETRIEVAL_VALUE_BITS,
                    value_bits=RETRIEVAL_VALUE_BITS,
                    group_size=min(64, values.shape[-1]),
                    rotation_block=values.shape[-1],
                    seed=layer_index,
                )
                value_rotation = build_kv_rotation(
                    values.shape[-1], value_config, value=True, device=values.device
                )
                quantized_values = quantize_kv(
                    values, value_rotation, value_config, value=True
                ).dequantize(original_basis=True)
                for threshold in RETRIEVAL_MASS_THRESHOLDS:
                    for value_kind, candidate_values in (
                        ("source_v", values),
                        (f"rotquant_v{RETRIEVAL_VALUE_BITS}", quantized_values),
                    ):
                        curve = oracle_value_retrieval_curve(
                            attention,
                            candidate_values,
                            counts,
                            reference_values=(
                                values if value_kind != "source_v" else None
                            ),
                            recent_window=RETRIEVAL_RECENT_WINDOW,
                            sink_tokens=RETRIEVAL_SINK_TOKENS,
                            mass_threshold=threshold,
                        )
                        retrieval_rows.extend({
                            "layer": layer_index,
                            "value_kind": value_kind,
                            **row,
                        } for row in curve)
                used_layers += 1
                if MAX_RETRIEVAL_LAYERS is not None and used_layers >= MAX_RETRIEVAL_LAYERS:
                    break
            assert retrieval_rows, "No full-attention cache layers were available."
            retrieval_table = pd.DataFrame(retrieval_rows)
            persist_frame("retrieval", retrieval_table)
            display(retrieval_table.head(20))
            retrieval_model = retrieval_output = retrieval_inputs = None
            release_cuda_memory()
        else:
            retrieval_table = pd.DataFrame()
        """),
        md("## Checks & Results"),
        md("### 15. Summarize promotion evidence without overstating support"),
        code("""
        validation_summary = pd.DataFrame()
        if not validation_table.empty:
            validation_summary = validation_table.groupby(
                ["profile", "track", "research_only"], as_index=False
            ).agg(
                mean_ppl=("ppl", "mean"),
                worst_ppl=("ppl", "max"),
                mean_relative_ppl=("relative_ppl", "mean"),
                worst_relative_ppl=("relative_ppl", "max"),
                mean_ppl_c4=("ppl_c4", "mean"),
                worst_relative_ppl_c4=("relative_ppl_c4", "max"),
                mean_paired_nll_delta=("paired_nll_delta", "mean"),
                worst_paired_nll_ci_high=("paired_nll_ci_high", "max"),
                mean_packed_bytes=("packed_weight_bytes", "mean"),
                mean_accounted_bytes=("accounted_weight_bytes", "mean"),
                mean_complete_persistent_bytes=(
                    "complete_persistent_model_bytes", "mean"
                ),
                mean_quality_runtime_bytes=("quality_runtime_model_bytes", "mean"),
                mean_effective_bpw=("effective_bpw", "mean"),
                mean_accounted_bpw=("accounted_bpw", "mean"),
                mean_trajectory_token_agreement=(
                    "trajectory_token_agreement", "mean"
                ),
                worst_trajectory_token_agreement=(
                    "trajectory_token_agreement", "min"
                ),
                mean_trajectory_exact_rate=("trajectory_exact_rate", "mean"),
                mean_trajectory_matching_prefix=(
                    "trajectory_mean_matching_prefix", "mean"
                ),
                mean_teacher_kl=("mean_teacher_kl", "mean"),
                worst_teacher_kl=("mean_teacher_kl", "max"),
                mean_median_teacher_kl=("median_teacher_kl", "mean"),
                worst_p95_teacher_kl=("p95_teacher_kl", "max"),
                worst_max_teacher_kl=("max_teacher_kl", "max"),
                mean_top1_agreement=("top1_agreement", "mean"),
                mean_logit_nll_delta=("logit_nll_delta", "mean"),
                worst_layer_nmse=("worst_layer_nmse", "max"),
                mean_trial_wall_seconds=("trial_wall_seconds", "mean"),
                peak_patch_vram_bytes=("peak_vram_bytes_patch", "max"),
                peak_eval_vram_bytes=("peak_vram_bytes_eval", "max"),
                seeds=("seed", "nunique"),
            )
            statuses = []
            for _, summary_row in validation_summary.iterrows():
                profile = summary_row["profile"]
                cross = cross_family_table[
                    cross_family_table["profile"] == profile
                ] if not cross_family_table.empty else pd.DataFrame()
                primary_quality_passed = bool(
                    summary_row["worst_relative_ppl"] <= MAX_CONFIRM_RELATIVE_PPL
                    and summary_row["worst_relative_ppl_c4"]
                    <= MAX_CONFIRM_RELATIVE_PPL
                )
                cross_family_available = not cross.empty
                cross_family_quality_passed = bool(
                    cross_family_available
                    and (cross["relative_ppl"] <= MAX_CONFIRM_RELATIVE_PPL).all()
                    and (cross["relative_ppl_c4"] <= MAX_CONFIRM_RELATIVE_PPL).all()
                    and (~cross["ppl_stopped_early"]).all()
                )
                if profile in MATCHED_CONTROL_MAP:
                    primary_controls = validation_control_table[
                        validation_control_table["candidate"] == profile
                    ]
                    cross_controls = cross_family_control_table[
                        cross_family_control_table["candidate"] == profile
                    ]
                    primary_matched_control_passed = bool(
                        not primary_controls.empty
                        and primary_controls["confidence_confirmed"].all()
                    )
                    cross_family_matched_control_passed = bool(
                        not cross_controls.empty
                        and cross_controls["confidence_confirmed"].all()
                    )
                else:
                    primary_matched_control_passed = True
                    cross_family_matched_control_passed = True
                statuses.append(validation_status(
                    research_only=bool(summary_row["research_only"]),
                    control_only=profile not in promoted_names,
                    primary_quality_passed=primary_quality_passed,
                    cross_family_available=cross_family_available,
                    cross_family_quality_passed=cross_family_quality_passed,
                    primary_matched_control_passed=(
                        primary_matched_control_passed
                    ),
                    cross_family_matched_control_passed=(
                        cross_family_matched_control_passed
                    ),
                ))
            validation_summary["status"] = statuses
            display(validation_summary.sort_values([
                "mean_complete_persistent_bytes", "mean_ppl"
            ]))

        retrieval_summary = pd.DataFrame()
        if not retrieval_table.empty:
            retrieval_summary = retrieval_table.groupby([
                "value_kind", "mass_threshold", "retrieval_k",
                "sequence_length",
            ], as_index=False).agg(
                mean_attention_mass=("mean_attention_mass", "mean"),
                p05_attention_mass=("p05_attention_mass", "mean"),
                dense_value_relative_attention_mse=(
                    "dense_value_relative_attention_mse", "mean"
                ),
                dense_fallback_fraction=("dense_fallback_fraction", "mean"),
                gated_relative_attention_mse=(
                    "gated_relative_attention_mse", "mean"
                ),
                effective_value_read_fraction=(
                    "effective_value_read_fraction", "mean"
                ),
                layers=("layer", "nunique"),
            )
            display(retrieval_summary)
        """),
        md("### 16. Plot the rate-quality and retrieval frontiers"),
        code("""
        import matplotlib.pyplot as plt

        figure_paths = []
        if not screen_table.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            quantized = screen_table[
                (screen_table["profile"] != "source_fp16")
                & ~screen_table["ppl_stopped_early"]
                & (screen_table["relative_ppl"] <= MAX_SCREEN_RELATIVE_PPL)
            ]
            for track, group in quantized.groupby("track"):
                ax.scatter(
                    group["complete_persistent_model_bytes"] / 1e9,
                    group["relative_ppl"], label=track, alpha=0.8,
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(
                xlabel="Complete persistent model bytes (GB)",
                ylabel="Relative WikiText-2 PPL",
                title="Seed-0 algorithmic screen",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            path = RESULT_ROOT / "algorithm_screen.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            figure_paths.append(path)
            plt.show()

        if not retrieval_summary.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for (kind, threshold), group in retrieval_summary.groupby([
                "value_kind", "mass_threshold"
            ]):
                group = group.sort_values("effective_value_read_fraction")
                ax.plot(
                    group["effective_value_read_fraction"],
                    group["gated_relative_attention_mse"],
                    marker="o", label=f"{kind}, mass>={threshold:.2f}",
                )
            ax.set(
                xlabel="Effective V rows read (including dense fallback)",
                ylabel="Gated relative attention-output MSE",
                title="Real-attention selective-V oracle",
            )
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.legend()
            ax.grid(alpha=0.2)
            path = RESULT_ROOT / "selective_v_curve.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            figure_paths.append(path)
            plt.show()
        """),
        md("### 17. Persist the audit trail and compact archive"),
        code("""
        import shutil

        table_paths = {}
        for name, frame in (
            ("synthetic_preflight", synthetic_table),
            ("sentinel", sentinel_table),
            ("screen", screen_table),
            ("promotion_decisions", promotion_decisions),
            ("validation", validation_table),
            ("validation_summary", validation_summary),
            ("validation_matched_controls", validation_control_table),
            ("cross_family", cross_family_table),
            ("cross_family_matched_controls", cross_family_control_table),
            ("retrieval", retrieval_table),
            ("retrieval_summary", retrieval_summary),
        ):
            if not frame.empty:
                path = persist_frame(name, frame)
                table_paths[name] = str(path)

        summary = {
            "git_sha": commit,
            "runtime": RUNTIME_MANIFEST,
            "primary_model": PRIMARY_MODEL,
            "model_cases": MODEL_CASES,
            "dataset_revisions": DATASET_REVISIONS,
            "screen_profiles": [trial.name for trial in TRIALS],
            "promoted_profiles": promoted_names,
            "validation_seeds": list(VALIDATION_SEEDS),
            "screen_max_samples": SCREEN_MAX_SAMPLES,
            "confirm_max_samples": CONFIRM_MAX_SAMPLES,
            "screening": {
                "sentinel_max_samples": SENTINEL_MAX_SAMPLES,
                "sentinel_w4_max_relative_ppl": SENTINEL_W4_MAX_RELATIVE_PPL,
                "early_stop_after": EARLY_STOP_AFTER,
                "early_stop_relative_ppl": EARLY_STOP_RELATIVE_PPL,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "matched_rate_tolerance": MATCHED_RATE_TOLERANCE,
                "max_confirm_relative_ppl": MAX_CONFIRM_RELATIVE_PPL,
            },
            "competitive_claim_contract": {
                "status": "not_run",
                "prompt_count": COMPETITIVE_PROMPT_COUNT,
                "generation_tokens": COMPETITIVE_GENERATION_TOKENS,
                "decoding": "greedy",
                "domains": list(COMPETITIVE_DOMAINS),
                "required_metrics": [
                    "mean_median_p95_teacher_kl",
                    "top1_agreement",
                    "trajectory_token_agreement",
                    "exact_trajectory_rate",
                    "mean_matching_prefix",
                    "exact_deployed_artifact_bytes",
                ],
                "requires_size_matched_external_artifacts": True,
                "requires_disjoint_calibration_manifest": True,
                "requires_item_level_token_hash_disjointness": True,
            },
            "trajectory_validation": {
                "enabled": RUN_TRAJECTORY_VALIDATION,
                "batches": TRAJECTORY_BATCHES,
                "prompt_len": TRAJECTORY_PROMPT_LENGTH,
                "new_tokens": TRAJECTORY_NEW_TOKENS,
                "skip": TRAJECTORY_SKIP,
                "dataset": "C4",
            },
            "logit_fidelity": {
                "enabled": RUN_LOGIT_FIDELITY,
                "batches": LOGIT_FIDELITY_BATCHES,
                "prompt_len": LOGIT_FIDELITY_PROMPT_LENGTH,
                "skip": LOGIT_FIDELITY_SKIP,
            },
            "data_partitions": {
                "allocation_teacher_skip": DYNAMIC_TEACHER_SKIP,
                "screen_layer_drift_skip": SCREEN_LAYER_DRIFT_SKIP,
                "layer_drift_skip": LAYER_DRIFT_SKIP,
                "trajectory_skip": TRAJECTORY_SKIP,
                "logit_fidelity_skip": LOGIT_FIDELITY_SKIP,
                "retrieval_skip": RETRIEVAL_SKIP,
                "retrieval_manifest": retrieval_manifest,
            },
            "record_data_manifests": {
                "|".join(map(str, key)): record["payload"]["metrics"].get(
                    "data_manifest", {}
                )
                for key, record in trial_records.items()
            },
            "tables": table_paths,
            "figures": [str(path) for path in figure_paths],
            "boundaries": {
                "fallback_quality_only": True,
                "vector_checkpoint_supported": False,
                "retrieval_selection": "dense-attention oracle",
                "packed_key_recall_measured": False,
            },
        }
        summary_path = RESULT_ROOT / "algorithm_lab_summary.json"
        write_result(str(summary_path), summary)

        log_lines = [
            f"## RotQuant algorithm lab ({commit[:12]})",
            "",
            f"- Primary model: {PRIMARY_MODEL['model']}",
            f"- Screened profiles: {len(TRIALS)}",
            f"- Promoted profiles: {', '.join(promoted_names) or 'none'}",
            f"- Full validation seeds: {VALIDATION_SEEDS if RUN_FULL_SEED_VALIDATION else 'disabled'}",
            f"- Cross-family validation: {'enabled' if RUN_CROSS_FAMILY else 'disabled'}",
            f"- Held-out trajectory validation: {'enabled' if RUN_TRAJECTORY_VALIDATION else 'disabled'} ({TRAJECTORY_BATCHES} prompts x {TRAJECTORY_NEW_TOKENS} tokens)",
            f"- Held-out logit fidelity: {'enabled' if RUN_LOGIT_FIDELITY else 'disabled'}",
            "- Promoted validation reports paired WikiText-2 and C4 window confidence intervals.",
            "- Vector profiles are research-only and cannot be exported yet.",
            "- Selective-V results use dense attention for candidate selection; packed-key recall remains open.",
        ]
        log_path = RESULT_ROOT / "experiment_log_entry.md"
        log_path.write_text("\\n".join(log_lines) + "\\n")
        archive_base = RESULT_BASE / (
            f"algorithm_lab_{commit[:12]}_{RUNTIME_FINGERPRINT}"
        )
        archive_path = Path(shutil.make_archive(
            str(archive_base), "zip", RESULT_ROOT
        ))
        print(json.dumps(summary, indent=2))
        print({"summary": str(summary_path), "archive": str(archive_path)})
        """),
        md("### 18. Optionally download the compact result archive"),
        code("""
        if DOWNLOAD_RESULTS:
            from google.colab import files
            try:
                files.download(str(archive_path))
            except Exception as exc:
                print(
                    "Automatic browser download failed, but the archive is already "
                    f"persisted at {archive_path}. Error: {type(exc).__name__}: {exc}"
                )
        """),
        md("""
        ## Takeaways and next steps

        Interpret the generated tables only after all enabled gates complete:

        - Promote vector quantization only if it beats its scalar control within
          the declared complete-byte tolerance, then confirms on paired
          WikiText-2/C4 windows and cross-family validation.
        - Promote a mixed-bit allocator only if it saves at least the declared
          byte threshold, beats the seeded random allocation control, and does
          not merely trade a tiny saving for worse PPL.
        - Treat the small C4 trajectory gate as a compounding-drift check, not
          as a claim of equivalence to a diverse Divergence-300-style benchmark.
        - Treat calibrated, spherical, length-corrected, and TurboQuant-scale
          profiles as independent ablations; do not combine them until one wins.
        - Use the selective-V curve to decide whether an architecture-specific
          packed-query/key capture trial is justified. Dense oracle selection is
          not a deployable retrieval result.
        - Append the generated experiment-log entry and raw summary only after
          checking model revisions, sample counts, and all failed/interrupted rows.
        """),
    ]
    for index, cell in enumerate(cells):
        digest = hashlib.sha256(cell.source.encode()).hexdigest()[:8]
        cell["id"] = f"rq-{index:02d}-{digest}"
    notebook = new_notebook(
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
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
