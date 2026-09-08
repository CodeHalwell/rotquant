#!/usr/bin/env python3
"""Build the next Colab experiment with separate quality and replication phases."""

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_qwen35_packed_validation_notebook import build_notebook as packed_notebook
from scripts.build_qwen35_packed_validation_notebook import code, md

OUTPUT = ROOT / "notebooks" / "qwen35_4b_fresh_quality_colab.ipynb"


def build_notebook():
    # Reuse the tested dependency setup, not a divergent set of Colab versions.
    dependencies = next(
        c.source
        for c in packed_notebook().cells
        if c.cell_type == "code" and "packages = [" in c.source
    )
    dependencies += '\nrun_live([sys.executable, "-m", "pip", "install", "jinja2==3.1.6"], "template-dependency")'
    cells = [
        md("""
        # RotQuant Qwen3.5-4B fresh quality and recipe replication

        ## Goal
        Evaluate the validated W5/W6 and W5/W8 artifacts on **new frozen inputs**,
        compare the pinned Unsloth UD-Q4_K_XL, then replicate both recipes at seeds
        1/2. No new allocator, LoRA, activation/KV quantization or kernel tuning.

        The September 7 seed-0 revalidation passed the saved reload probes and
        development quality guards. That is historical evidence, not a result of
        this notebook. This next protocol has **not yet run on the full CUDA model**.

        Use A100 40 GB-class hardware and high host RAM. Budget **140 GB additional
        Drive space** for the full run (two replication seeds, checkpoints/Hessians,
        two full-logit reference sets, logs and margin), plus about 35 GB local
        space for HF/GGUF downloads/builds. These are estimates, not fit guarantees.
        Existing seed-0 tensors must remain on Drive; the compact ZIP is insufficient.
        """),
        md("""
        ## Setup
        ### 1. Controls and original artifact location
        Defaults run the full experiment. Keep the printed commit when resuming.
        The old `8f10ee60fc7f` folder holds the tensors; the successful revalidation
        folder contains new evidence only. **Do not point ARTIFACT_SOURCE at that
        compact revalidation folder.** Existing seed 0 is never requantized.

        For a shorter first session set RUN_REPLICATION=False. This produces a
        useful seed-0 comparison but does not complete seeds 1/2. Re-enable it later
        with the same commit/root. No failed/missing arm is treated as a success.

        To recover the tokenizer-gate failure from 733bb3d, set REUSE_FRESH_ROOT
        to its `fresh` directory. The new commit gets a separate output root;
        checksummed completed HF/seed-0 results retain their original identities
        and reference tensors stay in the old directory. Both directories must
        remain available. Missing/incomplete results are not silently adopted.
        """),
        code("""
        from pathlib import Path
        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # Publish this implementation first; pin the printed SHA to resume.
        REPO_DIR = Path("/content/rotquant-fresh-quality")
        ARTIFACT_SOURCE = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
        RESULT_BASE = Path("/content/drive/MyDrive/rotquant/qwen35_fresh_quality")
        # Optional recovery from the completed 733bb3d HF/packed collections:
        REUSE_FRESH_ROOT = None  # Path("/content/drive/MyDrive/rotquant/qwen35_fresh_quality/733bb3d2e477/fresh")
        GGUF_INPUT_POLICY = "frozen-hf"  # Same frozen IDs for all arms; native tokenization audited separately.
        RUN_UNSLOTH = True
        RUN_REPLICATION = True
        REPLICATION_SEEDS = (1, 2)
        REQUIRE_FAST_HADAMARD = True
        USE_HF_SECRET = False
        HEARTBEAT_SECONDS = 60
        DOWNLOAD_RESULTS = True
        """),
        md("""
        ### 2. Mount, freeze checkout, and establish persistent logs
        A browser disconnect does not block worker output. Explicit cell interruption
        terminates its process group. Before rerunning after reconnect, check the PID
        printed by run_live and use its `tail -f` command.
        """),
        code("""
        import json, os, shutil, subprocess, sys
        from google.colab import drive
        drive.mount("/content/drive")
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=REPO_DIR, text=True).strip()
        assert not git("status", "--porcelain"), "Use a clean checkout."
        subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR, check=True)
        COMMIT = git("rev-parse", "HEAD")
        assert (REPO_DIR / "scripts/run_qwen35_fresh_eval.py").exists(), "Publish/select the new code first."
        for arm in ("b5_v6", "b5_v8"):
            for name in ("prepared.json", "preparation.json", "checkpoint/rotquant_config.json",
                         "dense_probes.safetensors", "packed_probes.safetensors"):
                assert (ARTIFACT_SOURCE / f"{arm}_s0" / name).is_file(), f"Missing original artifact evidence: {arm}/{name}"
        RESULT_ROOT = RESULT_BASE / COMMIT[:12]
        EVAL_ROOT = RESULT_ROOT / "fresh"
        REPLICA_ROOT = RESULT_ROOT / "replicas"
        LOG_ROOT = RESULT_ROOT / "logs"
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(REPO_DIR))
        from scripts.colab_runtime import run_live as _run_live
        def run_live(command, label, timeout_seconds=None):
            return _run_live(command, label, repo_dir=REPO_DIR, log_root=LOG_ROOT,
                             timeout_seconds=timeout_seconds)
        print({"commit": COMMIT, "output": str(RESULT_ROOT), "source": str(ARTIFACT_SOURCE)})
        print("Drive filesystem free space is not a reliable account quota; check Drive storage manually.")
        """),
        md("""
        ### 3. Install the tested dependencies and fast Hadamard kernel
        Keep Colab's PyTorch. Pinning may report conflicts with unrelated preinstalled
        packages; the isolated subprocess checks below validate this workflow.
        """),
        code(dependencies),
        md("""
        ### 4. Optional authentication and tiny CUDA preflight
        The preflight exercises tiny random-weight multimodal Qwen, packed export
        and fresh-process reload; it is not full-model quality or vision validation.
        """),
        code("""
        if USE_HF_SECRET:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
            del token
        run_live([sys.executable, "-u", str(REPO_DIR / "scripts/preflight_packed_validation.py"),
                  "--device", "cuda", "--output", str(RESULT_ROOT / "runtime_preflight.json")], "preflight")
        """),
        md("""
        ## Steps
        ### 5. Freeze fresh inputs before inspecting candidate results
        24 C4 documents × 512 tokens, excluding recorded calibration/development rows.
        96 new authored tasks: 24 each for multilingual arithmetic, code tracing,
        JSON structure and tool selection. Six languages are represented.

        ### Key assumptions
        These are **synthetic diagnostics**, not real-world coding/agent execution,
        an independent public benchmark or Unsloth's private Divergence-300 suite.
        Generated code and tool calls are never executed. Oracles require exact JSON,
        including keys and types. Truncated outputs fail task success. Task generation
        stops at the source EOS set, with a 128-token cap and thinking disabled.
        KL uses up to 32 teacher-generated continuation positions; this is not an
        exact 32-token trajectory benchmark for outputs that end earlier.

        Exact row/token checks do not establish semantic near-dedup or pretraining
        cleanliness. Do not tune quantization against these results and still call
        them held out. The authored variants are grouped by family for uncertainty.
        """),
        code("""
        common = [sys.executable, "-u", str(REPO_DIR / "scripts/run_qwen35_fresh_eval.py"),
                  "--output-dir", str(EVAL_ROOT), "--evidence-root", str(ARTIFACT_SOURCE),
                  "--heartbeat-seconds", str(HEARTBEAT_SECONDS),
                  "--gguf-input-policy", GGUF_INPUT_POLICY]
        def fresh(phase, label, extra=()):
            run_live([*common, "--phase", phase, *extra], label)
        run_live([*common, "--phase", "freeze", "--dry-run"], "plan")
        if REUSE_FRESH_ROOT is not None:
            fresh("reuse", "reuse-completed-inputs", ["--reuse-root", str(REUSE_FRESH_ROOT)])
        else:
            fresh("freeze", "freeze-inputs")
        """),
        md("""
        ### 6. Capture the common FP16 source once
        Full-vocabulary logits persist per prompt with SHA-256 checks. No top-k KL
        approximation. Each worker loads only one model at a time, and reference
        scoring reads one prompt at a time. Allow up to roughly 31 GB total for the
        FP16-source and BF16-GGUF reference files, stored as FP32 arrays.
        """),
        code("""
        fresh("source", "common-fp16-teacher")
        """),
        md("""
        ### 7. Evaluate the existing two seed-0 checkpoints
        Artifact files/hashes, recipe settings, vocabulary ownership and no-dense-cache
        residency are verified. This does not repeat the completed old development
        suite. Logs print each prompt start/completion, KL and task outcome.
        """),
        code("""
        expected = ["source_fp16"]
        for arm in ("b5_v6", "b5_v8"):
            fresh("packed", f"fresh-{arm}-s0", ["--artifact-root", str(ARTIFACT_SOURCE),
                  "--arm", arm, "--seed", "0"])
            expected.append(f"{arm}_s0")
        """),
        md("""
        ### 8. Install the pinned provider engine and measure its BF16 bridge
        We verify the Git installation metadata, shared-library hashes, all token-ID
        mappings and exact decoded prompt bytes. Native chat retokenization is
        audited separately: four Hindi prompts differed in the first Colab check.
        With GGUF_INPUT_POLICY="frozen-hf", every arm receives the original HF
        IDs directly. Different native splitting alone does not block this controlled
        comparison. Mapping, padding, frozen HF retokenization and decoded-byte
        failures still stop. This is **not native-tokenizer serving parity**.
        Use "strict" to require native tokenization equality as an additional gate.
        No vocabulary slicing, changed prompts or omitted Hindi tasks.

        The bridge measures BF16 GGUF vs the common HF FP16 teacher, exposing their
        combined conversion/engine/precision discrepancy. Unsloth then gets BOTH
        common-FP16 KL and same-engine BF16 KL on the same frozen contexts. Do not
        subtract the bridge KL: KL divergences are not additive.

        Keep PyTorch imported before llama.cpp in custom diagnostics: the Colab
        build's system NCCL can conflict with PyTorch when loaded first. The runner
        already uses PyTorch-first order. Reuse the verified build; do not reinstall
        the engine to diagnose a text-tokenizer mismatch.
        """),
        code("""
        if RUN_UNSLOTH:
            from scripts.run_unsloth_qwen35_4b_kl import LLAMA_CPP_PYTHON_REVISION
            run_live([sys.executable, "-m", "pip", "install", "diskcache==5.6.3", "jinja2==3.1.6",
                      "typing-extensions==4.15.0"], "llama-dependencies")
            check_build = "from scripts.run_qwen35_fresh_eval import llama_build_identity; print(llama_build_identity())"
            probe = subprocess.run([sys.executable, "-c", check_build], cwd=REPO_DIR,
                                   capture_output=True, text=True)
            if probe.returncode:
                os.environ.update({"CMAKE_ARGS": "-DGGML_CUDA=on", "CMAKE_BUILD_PARALLEL_LEVEL": "2", "FORCE_CMAKE": "1"})
                run_live([sys.executable, "-m", "pip", "install", "-v", "--force-reinstall", "--no-deps",
                          f"git+https://github.com/abetlen/llama-cpp-python.git@{LLAMA_CPP_PYTHON_REVISION}"],
                         "llama-cuda-build", timeout_seconds=3600)
            run_live([sys.executable, "-c", check_build], "llama-build-check")
            fresh("bridge", "gguf-bf16-bridge")
            fresh("unsloth", "unsloth-common-and-bf16")
            expected += ["gguf_bf16_bridge", "unsloth_ud_q4"]
        else:
            print("Provider comparison disabled; not part of this run's completed scope.")
        """),
        md("""
        ### 9. Replicate recipes at seeds 1 and 2
        This is the expensive phase: one W5 backbone quantization per seed, shared
        by W6/W8 vocabulary. The existing preparation runner exports each checkpoint
        and performs its tiny prototype probes. We skip its old full reload evaluation;
        fresh-process loading/ownership and NEW quality measurement happen below.
        Preparation still uses the fixed development suite for its existing export
        evidence. It never sees the new task or C4 inputs when choosing quantization.

        Seeds vary rotation/quantization RNG, **not calibration document selection**.
        Keep seed results separate. Do not pool their shared prompts as independent
        observations or label them independent calibration-corpus replication.
        """),
        code("""
        if RUN_REPLICATION:
            assert REPLICATION_SEEDS == (1, 2), "This protocol registers seeds 1 and 2."
            preparation = [sys.executable, "-u", str(REPO_DIR / "scripts/run_qwen35_packed_validation.py"),
                           "--output-dir", str(REPLICA_ROOT), "--prepare-only",
                           "--heartbeat-seconds", str(HEARTBEAT_SECONDS)]
            for seed in REPLICATION_SEEDS:
                preparation.extend(["--seed", str(seed)])
            run_live([*preparation, "--dry-run"], "replication-plan")
            run_live(preparation, "replication-prepare")
            for seed in REPLICATION_SEEDS:
                for arm in ("b5_v6", "b5_v8"):
                    fresh("packed", f"fresh-{arm}-s{seed}", ["--artifact-root", str(REPLICA_ROOT),
                          "--arm", arm, "--seed", str(seed)])
                    expected.append(f"{arm}_s{seed}")
        else:
            print("Seeds 1/2 disabled. This is a seed-0-only experiment.")
        """),
        md("""
        ## Checks
        ### 10. Reconcile all expected arms and inspect domain/task outcomes
        A completed measurement does not automatically promote an artifact. Inspect
        strict task success, source-correct → candidate-wrong flips, malformed JSON,
        truncation, teacher KL and the cross-engine BF16 control side by side.
        Family-bootstrap contrasts use right-minus-left deltas; negative KL is better,
        positive task success is better. Single-family multilingual diagnostics have
        no defensible family-bootstrap CI and report null instead of false precision.
        """),
        code("""
        import pandas as pd
        # Derive expected scope from controls, not from successes in previous cells.
        expected = ["source_fp16", "b5_v6_s0", "b5_v8_s0"]
        if RUN_UNSLOTH:
            expected += ["gguf_bf16_bridge", "unsloth_ud_q4"]
        if RUN_REPLICATION:
            expected += [f"{arm}_s{seed}" for seed in REPLICATION_SEEDS for arm in ("b5_v6", "b5_v8")]
        summary_args = [arg for label in expected for arg in ("--expect", label)]
        fresh("summary", "summary", summary_args)
        summary = json.loads((EVAL_ROOT / "summary.json").read_text())
        display(pd.DataFrame(summary["rows"]))
        print({key:summary[key] for key in ("complete", "missing", "interpretation")})
        """),
        md("""
        ### 11. Download compact results; keep tensors on Drive
        No model weights, NumPy teacher arrays or probe safetensors are included.
        The full checkpoints/references are required to verify or reuse inference.
        """),
        code("""
        import zipfile
        if DOWNLOAD_RESULTS:
            archive = Path("/content") / f"qwen35_fresh_quality_{COMMIT[:12]}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(RESULT_ROOT.rglob("*")):
                    if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".sha256", ".log"):
                        if "hessians" not in path.parts and "vocabulary_cache" not in path.parts:
                            bundle.write(path, path.relative_to(RESULT_ROOT))
                # Keep imported per-prompt evidence in the compact handoff too.
                if (EVAL_ROOT / "reuse.json").exists():
                    import hashlib
                    reuse = json.loads((EVAL_ROOT / "reuse.json").read_text())
                    for name, digest in reuse["files"].items():
                        path = Path(reuse["root"]) / name
                        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Reused evidence changed: {name}"
                        bundle.write(path, "reused/" + name)
                        bundle.write(path.with_suffix(".sha256"), "reused/" + str(Path(name).with_suffix(".sha256")))
            from google.colab import files
            files.download(str(archive))
        """),
        md("""
        ## Next Steps
        Bring back the compact bundle. Preserve both old and new artifacts. W5/W6
        remains the under-budget candidate; W5/W8 remains the lower-development-KL
        comparator until this new evidence is assessed. Do not launch a new allocator
        or recovery fit based only on one diagnostic number.

        This protocol can expose task regressions and seed sensitivity. A competitive
        release still needs a larger independently sourced real-task benchmark,
        additional calibration-corpus replications, and separate runtime profiling.
        The notebook does not claim a Dynamic-v3.0 win or fused inference performance.
        """),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"fresh-quality-{index:02d}"
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": OUTPUT.name},
        },
    )


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)
