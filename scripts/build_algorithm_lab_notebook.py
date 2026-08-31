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
        - Dense-attention top-k is an upper-bound oracle for selective V reads. It
          measures value sparsity; packed-key candidate recall is a separate gate.
        """),
        md("""
        ## Trial ladder

        1. Synthetic exact-rate scalar-versus-vector preflight.
        2. Seed-0, 16-sample screen of every predeclared profile.
        3. Exact-byte checks and Pareto promotion by research track.
        4. Three-seed, 64-sample primary-model validation.
        5. Cross-family confirmation of promoted profiles.
        6. Real-attention selective-V oracle with quantized-value and dense
           confidence-fallback curves.
        """),
        md("## Setup"),
        code("""
        from pathlib import Path

        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "codex/canonical-serving-stage"
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
        RUN_PRIMARY_SCREEN = True
        RUN_FULL_SEED_VALIDATION = True
        RUN_CROSS_FAMILY = True
        RUN_SELECTIVE_KV_ORACLE = True
        DOWNLOAD_RESULTS = True
        REQUIRE_FAST_HADAMARD = False

        SCREEN_MAX_SAMPLES = 16
        CONFIRM_MAX_SAMPLES = 64
        EVAL_SEQUENCE_LENGTH = 256
        VALIDATION_SEEDS = (0, 1, 2)
        CROSS_FAMILY_SEEDS = (0,)
        MAX_PROMOTED_PER_TRACK = 2
        MAX_SCREEN_RELATIVE_PPL = 1.0
        MIN_ALLOCATION_BYTE_SAVING = 0.01

        RETRIEVAL_PROMPT_LENGTH = 512
        RETRIEVAL_SKIP = 4096
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
            drive.mount("/content/drive")
            RESULT_BASE = DRIVE_RESULT_ROOT
        else:
            RESULT_BASE = LOCAL_RESULT_ROOT
        RESULT_BASE.mkdir(parents=True, exist_ok=True)

        if not REPO_DIR.exists():
            subprocess.run([
                "git", "clone", "--branch", REPO_REF, "--single-branch",
                REPO_URL, str(REPO_DIR),
            ], check=True)
        else:
            subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO_DIR, check=True)
            subprocess.run(["git", "checkout", REPO_REF], cwd=REPO_DIR, check=True)
            subprocess.run(["git", "pull", "--ff-only", "origin", REPO_REF], cwd=REPO_DIR, check=True)

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
        import importlib.metadata
        import json

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
        kernel_build = subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "fast-hadamard-transform==1.1.0", "--no-build-isolation",
        ], check=False)
        fast_hadamard_available = kernel_build.returncode == 0
        if REQUIRE_FAST_HADAMARD:
            assert fast_hadamard_available, (
                "fast-hadamard-transform failed to build; inspect the pip output "
                "or set REQUIRE_FAST_HADAMARD=False for the slower torch fallback."
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
        tracked_distributions = [
            "transformers", "datasets", "accelerate", "safetensors",
            "sentencepiece", "scipy", "pyyaml", "pandas", "matplotlib",
            "huggingface-hub",
        ]
        RUNTIME_MANIFEST = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "fast_hadamard_transform": fast_hadamard_available,
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
            REPO_DIR / "rotquant/codebooks.py",
            REPO_DIR / "rotquant/kv_cache.py",
            REPO_DIR / "scripts/run_experiment.py",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        assert not missing, "Missing required files: " + ", ".join(missing)

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
            from eval.quantization import quantized_weight_metrics
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
        from dataclasses import asdict

        from scripts.run_experiment import run as run_experiment

        trial_records = {}

        def record_path(stage, model_case, profile, seed, max_samples):
            specification = {
                "commit": commit,
                "stage": stage,
                "model": model_case,
                "profile": asdict(profile),
                "seed": seed,
                "max_samples": max_samples,
                "seq_len": EVAL_SEQUENCE_LENGTH,
                "runtime": RUNTIME_FINGERPRINT,
            }
            digest = hashlib.sha256(
                json.dumps(specification, sort_keys=True).encode()
            ).hexdigest()[:16]
            return RECORD_ROOT / f"{stage}_{model_case['name']}_{profile.name}_s{seed}_{digest}.json"

        def release_cuda_memory():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def run_trial(stage, model_case, profile, seed, max_samples):
            path = record_path(stage, model_case, profile, seed, max_samples)
            key = (stage, model_case["name"], profile.name, seed, max_samples)
            if path.exists() and not FORCE_RERUN:
                record = json.loads(path.read_text())
                trial_records[key] = record
                print(f"resume {path.name}")
                return record
            sets = [
                *profile.sets(),
                ("patch.include", model_case["include"]),
                ("patch.exclude", model_case["exclude"]),
                ("eval.ppl.seq_len", EVAL_SEQUENCE_LENGTH),
                ("eval.ppl.max_samples", max_samples),
                ("eval.ppl_datasets", ["wikitext2"]),
                ("eval.zeroshot", False),
            ]
            payload = run_experiment(
                str(CONFIG_PATH),
                output_dir=str(RAW_RUN_ROOT),
                overrides={
                    "model": model_case["model"], "device": "cuda", "seed": seed,
                    "model_revision": model_case["revision"],
                },
                sets=sets,
            )
            record = {
                "stage": stage,
                "model_case": model_case,
                "profile": asdict(profile),
                "seed": seed,
                "max_samples": max_samples,
                "payload": payload,
            }
            path.write_text(json.dumps(record, indent=2))
            trial_records[key] = record
            release_cuda_memory()
            return record

        def report_row(record, source_ppl=None):
            payload = record["payload"]
            metrics = payload["metrics"]
            config = payload["config"]
            dynamic = metrics.get("dynamic_quantization", {})
            ppl = float(metrics["ppl_wikitext2"])
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
            codebook_bytes = centroid_values_total * 4
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
                "packed_weight_bytes": packed_weight_bytes,
                "codebook_bytes": codebook_bytes,
                "accounted_weight_bytes": accounted_weight_bytes,
                "effective_bpw": metrics.get("bits_per_weight_mean"),
                "accounted_bpw": (
                    accounted_weight_bytes * 8 / quantized_weights
                    if quantized_weights else None
                ),
                "patch_seconds": metrics.get("patch_seconds", 0.0),
                "target_reached": dynamic.get("target_reached", True),
                "dynamic_achieved_bpw": dynamic.get("achieved_bpw"),
                "dynamic_counts": dynamic.get("counts_by_bits"),
            }
        """),
        md("### 7. Confirm the expensive matrix"),
        code("""
        requested = any([
            RUN_PRIMARY_SCREEN, RUN_FULL_SEED_VALIDATION,
            RUN_CROSS_FAMILY, RUN_SELECTIVE_KV_ORACLE,
        ])
        if requested:
            assert CONFIRM_EXPENSIVE_RUN, (
                "Review the model IDs, gates, Drive path, and estimated GPU cost, "
                "then set CONFIRM_EXPENSIVE_RUN=True."
            )
        """),
        md("## Data & Trials"),
        md("### 8. Run the complete primary-model seed-0 screen"),
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
        md("### 9. Check exact rates and build the screening table"),
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
                report_row(record, screen_source_ppl) for record in screen_records
            ])
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
            display(screen_table.sort_values([
                "track", "accounted_weight_bytes", "ppl"
            ]).style.format({
                "ppl": "{:.4f}", "relative_ppl": "{:+.2%}",
                "effective_bpw": "{:.4f}", "accounted_bpw": "{:.4f}",
                "packed_weight_bytes": "{:,.0f}",
                "accounted_weight_bytes": "{:,.0f}",
            }))
        else:
            screen_source_ppl = float("nan")
            screen_table = pd.DataFrame()
        """),
        md("### 10. Promote only bounded Pareto candidates"),
        code("""
        import numpy as np


        def pareto_rows(frame):
            rows = frame.reset_index(drop=True)
            keep = []
            for _, row in rows.iterrows():
                dominated = (
                    (rows["accounted_weight_bytes"] <= row["accounted_weight_bytes"])
                    & (rows["ppl"] <= row["ppl"])
                    & (
                        (rows["accounted_weight_bytes"] < row["accounted_weight_bytes"])
                        | (rows["ppl"] < row["ppl"])
                    )
                ).any()
                keep.append(not dominated)
            return rows[np.array(keep, dtype=bool)]

        promoted_names = {"gaussian_w4_mse"}
        if not screen_table.empty:
            eligible = screen_table[
                (screen_table["profile"] != "source_fp16")
                & screen_table["target_reached"]
                & (screen_table["relative_ppl"] <= MAX_SCREEN_RELATIVE_PPL)
            ].copy()
            for track, group in eligible.groupby("track"):
                frontier = pareto_rows(group).sort_values([
                    "ppl", "accounted_weight_bytes"
                ])
                promoted_names.update(
                    frontier["profile"].head(MAX_PROMOTED_PER_TRACK).tolist()
                )

            # A vector arm advances only when it beats its exact-rate scalar
            # control. This prevents novelty from consuming confirmation runs.
            for bits in (1, 2, 3):
                scalar = screen_table[
                    screen_table["profile"] == f"scalar_w{bits}_rms"
                ].iloc[0]
                vector = screen_table[
                    screen_table["profile"] == f"vector_d2_w{bits}_rms"
                ].iloc[0]
                if vector["ppl"] >= scalar["ppl"]:
                    promoted_names.discard(vector["profile"])

            w4_bytes = float(screen_table[
                screen_table["profile"] == "gaussian_w4_mse"
            ]["accounted_weight_bytes"].iloc[0])
            for name in promoted_names.copy():
                row = screen_table[screen_table["profile"] == name]
                if row.empty or row.iloc[0]["track"] != "allocation":
                    continue
                saving = 1.0 - float(row.iloc[0]["accounted_weight_bytes"]) / w4_bytes
                if saving < MIN_ALLOCATION_BYTE_SAVING:
                    promoted_names.discard(name)

        promoted_names = sorted(promoted_names)
        promoted_profiles = [trial_by_name(name) for name in promoted_names]
        print({"promoted_profiles": promoted_names})
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
            for profile in promoted_profiles:
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
                report_row(record, full_source_ppl) for record in validation_records
            ])
            display(validation_table.sort_values(["profile", "seed"]))
        else:
            full_source_ppl = float("nan")
            validation_table = pd.DataFrame()
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
                source_ppl = float(source["payload"]["metrics"]["ppl_wikitext2"])
                for profile in promoted_profiles:
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
            source_ppl = float(
                trial_records[source_key]["payload"]["metrics"]["ppl_wikitext2"]
            )
            cross_rows.append(report_row(record, source_ppl))
        cross_family_table = pd.DataFrame(cross_rows)
        if not cross_family_table.empty:
            display(cross_family_table.sort_values(["model", "profile", "seed"]))
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
        if RUN_SELECTIVE_KV_ORACLE:
            from scripts.run_experiment import build_calib_loader, load_hf_model

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
                mean_packed_bytes=("packed_weight_bytes", "mean"),
                mean_accounted_bytes=("accounted_weight_bytes", "mean"),
                mean_effective_bpw=("effective_bpw", "mean"),
                mean_accounted_bpw=("accounted_bpw", "mean"),
                seeds=("seed", "nunique"),
            )
            validation_summary["status"] = np.where(
                validation_summary["research_only"],
                "research winner; format/kernel work still required",
                "eligible for runtime prototype if cross-family gates pass",
            )
            display(validation_summary.sort_values([
                "mean_packed_bytes", "mean_ppl"
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
            quantized = screen_table[screen_table["profile"] != "source_fp16"]
            for track, group in quantized.groupby("track"):
                ax.scatter(
                    group["accounted_weight_bytes"] / 1e9,
                    group["relative_ppl"], label=track, alpha=0.8,
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(
                xlabel="Accounted projection bytes (code + scale + codebook, GB)",
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
            ("screen", screen_table),
            ("validation", validation_table),
            ("cross_family", cross_family_table),
            ("retrieval", retrieval_table),
            ("retrieval_summary", retrieval_summary),
        ):
            if not frame.empty:
                path = RESULT_ROOT / f"{name}.csv"
                frame.to_csv(path, index=False)
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
        summary_path.write_text(json.dumps(summary, indent=2))

        log_lines = [
            f"## RotQuant algorithm lab ({commit[:12]})",
            "",
            f"- Primary model: {PRIMARY_MODEL['model']}",
            f"- Screened profiles: {len(TRIALS)}",
            f"- Promoted profiles: {', '.join(promoted_names) or 'none'}",
            f"- Full validation seeds: {VALIDATION_SEEDS if RUN_FULL_SEED_VALIDATION else 'disabled'}",
            f"- Cross-family validation: {'enabled' if RUN_CROSS_FAMILY else 'disabled'}",
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
            files.download(str(archive_path))
        """),
        md("""
        ## Takeaways and next steps

        Interpret the generated tables only after all enabled gates complete:

        - Promote vector quantization only if it beats its scalar control at the
          same exact packed bytes and survives cross-family validation.
        - Promote a mixed-bit allocator only if it saves at least the declared
          byte threshold and does not merely trade a tiny saving for worse PPL.
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
