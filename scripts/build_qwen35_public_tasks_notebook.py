#!/usr/bin/env python3
"""Build the generation-only public-task Colab without changing old protocols."""

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_qwen35_fresh_eval_notebook import build_notebook as fresh_notebook
from scripts.build_qwen35_packed_validation_notebook import code, md

OUTPUT = ROOT / "notebooks/qwen35_4b_public_tasks_colab.ipynb"


def build_notebook():
    dependencies = next(c.source for c in fresh_notebook().cells
                        if c.cell_type == "code" and "packages = [" in c.source)
    cells = [
        md("""
        # RotQuant Qwen3.5-4B public-task release gate

        ## Goal
        Test whether the completed fresh-C4 fidelity gains translate into useful
        answers. **No quantization, training, allocator changes, or LoRA.** Reuse
        the six saved **W5 backbone / W6 or W8 vocabulary** checkpoints.
        These are not W4 models. Source FP16, a BF16-GGUF bridge and the pinned
        Unsloth UD-Q4_K_XL are mandatory controls on identical frozen HF input IDs.

        **New protocol; full CUDA model execution is not yet validated.** Start
        with the smoke option in a separate root. It checks plumbing, not quality.
        The old authored-task scores and their oracles remain unchanged.

        Public sources: [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
        [CRUXEval](https://huggingface.co/datasets/cruxeval-org/cruxeval), and
        [IFEval](https://huggingface.co/datasets/google/IFEval). Sources, splits,
        immutable revisions, row hashes, prompts, token IDs, caps and checker
        hashes are recorded before inference. No model output/program is executed.
        """),
        md("""
        ## Setup
        ### 1. Select scope before seeing answers
        Standard: 128 examples each of GSM8K, CRUXEval-O and IFEval = 384 prompts
        per arm; all seeds means **3,456 generations** across nine arms.
        SAMPLE_PER_BENCHMARK=0 uses all registered split rows (2,660 per arm).
        Subsets use a fixed SHA ordering, not the easiest or shortest examples.

        Use an A100 40 GB-class GPU and high host RAM. The existing Python packed
        reference path can make this a long, multi-session run: there is no measured
        runtime guarantee. The source phase gives an early timing estimate, not a
        prediction of packed speed. Stop/reconnect safely at any point; complete
        prompt files resume. More than one worker must not use the same root.

        Keep ~35 GB local space for downloads/builds and the original Drive
        checkpoints. This run does not create new weight/Hessian/full-logit copies;
        allow ~1 GB additional Drive headroom for standard text results/logs.
        Hardware/storage estimates are not fit guarantees. The compact ZIP alone
        cannot replace the saved original checkpoint/probe tensors.
        """),
        code("""
        from pathlib import Path
        REPO_URL = "https://github.com/CodeHalwell/rotquant.git"
        REPO_REF = "main"  # Publish this code first; pin the printed SHA when resuming.
        REPO_DIR = Path("/content/rotquant-public-tasks")
        SEED0_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
        REPLICA_ROOT = Path("/content/drive/MyDrive/rotquant/qwen35_fresh_quality/d4292d6fdec6/replicas")
        RESULT_BASE = Path("/content/drive/MyDrive/rotquant/qwen35_public_tasks")
        SCORER_CACHE = Path("/content/rotquant-public-scorers")
        GGUF_DIR = Path("/content/unsloth-qwen35-4b-gguf")
        SAMPLE_PER_BENCHMARK = 128
        SMOKE_RUN = True  # First run: eight per benchmark, seed 0, separate root. Then set False.
        SEED0_ONLY = False  # Full non-smoke run includes both recipes at seeds 0/1/2.
        REQUIRE_FAST_HADAMARD = True
        USE_HF_SECRET = False
        HEARTBEAT_SECONDS = 60
        DOWNLOAD_RESULTS = True
        """),
        md("""
        ### 2. Mount Drive and freeze the checkout
        The two artifact roots must point to original full checkpoints, not the
        compact result/revalidation bundles. Resume with the same commit, scope,
        dependency versions and GPU/runtime settings. A changed runtime fails
        closed instead of mixing evidence; a different code SHA gets a new root.
        """),
        code("""
        import json, os, shutil, subprocess, sys
        import torch
        from google.colab import drive
        assert torch.cuda.is_available(), "Choose a GPU runtime."
        assert torch.cuda.get_device_properties(0).total_memory >= 35 * 2**30, "Use A100 40 GB-class hardware."
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=REPO_DIR, text=True).strip()
        assert not git("status", "--porcelain"), "Use a clean checkout."
        subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR, check=True)
        COMMIT = git("rev-parse", "HEAD")
        assert (REPO_DIR / "scripts/run_qwen35_public_tasks.py").exists(), "Publish/select the new code first."
        samples = 8 if SMOKE_RUN else SAMPLE_PER_BENCHMARK
        seed0_only = SMOKE_RUN or SEED0_ONLY
        assert type(samples) is int and 0 <= samples <= 541
        scope = f"{'smoke' if SMOKE_RUN else 'public'}-n{samples}-{'s0' if seed0_only else 's012'}"
        RESULT_ROOT = RESULT_BASE / f"{COMMIT[:12]}-{scope}"
        EVAL_ROOT, LOG_ROOT = RESULT_ROOT / "tasks", RESULT_ROOT / "logs"
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(REPO_DIR))
        from scripts.colab_runtime import run_live as _run_live
        def run_live(command, label, timeout_seconds=None):
            return _run_live(command, label, repo_dir=REPO_DIR, log_root=LOG_ROOT,
                             timeout_seconds=timeout_seconds)
        seeds = (0,) if seed0_only else (0, 1, 2)
        for seed in seeds:
            for recipe in ("b5_v6", "b5_v8"):
                directory = (SEED0_ROOT if seed == 0 else REPLICA_ROOT) / f"{recipe}_s{seed}"
                for name in ("prepared.json", "preparation.json", "packed_probes.safetensors",
                             "dense_probes.safetensors", "checkpoint/rotquant_config.json"):
                    assert (directory / name).is_file(), f"Missing original artifact: {directory / name}"
        print({"commit": COMMIT, "scope": scope, "output": str(RESULT_ROOT), "seeds": seeds})
        print({"local_free_GB": shutil.disk_usage('/content').free / 1e9})
        """),
        md("### 3. Install tested model dependencies and the fast Hadamard kernel"),
        code(dependencies),
        md("""
        ### 4. Pin and preflight the public-task checkers
        Google IFEval's four upstream modules and license are downloaded by commit
        and verified by SHA-256. Sentence tables are pinned plain text, not pickle.
        No LLM judge or generated code execution. Checker dependency failure stops
        before model loading. Language detection and checker randomness are seeded.
        """),
        code("""
        run_live([sys.executable, "-m", "pip", "install", "nltk==3.9.2", "langdetect==1.0.9",
                  "immutabledict==4.2.1", "absl-py==2.3.1"], "checker-dependencies")
        if USE_HF_SECRET:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
            del token
        common = [sys.executable, "-u", str(REPO_DIR / "scripts/run_qwen35_public_tasks.py"),
                  "--output-dir", str(EVAL_ROOT), "--seed0-root", str(SEED0_ROOT),
                  "--replica-root", str(REPLICA_ROOT), "--scorer-cache", str(SCORER_CACHE),
                  "--gguf-dir", str(GGUF_DIR), "--samples", str(samples),
                  "--heartbeat-seconds", str(HEARTBEAT_SECONDS)]
        if seed0_only:
            common.append("--seed0-only")
        def public(phase, label, extra=()):
            run_live([*common, "--phase", phase, *extra], label)
        public("setup", "scorer-setup")
        run_live([sys.executable, "-u", str(REPO_DIR / "scripts/preflight_packed_validation.py"),
                  "--device", "cuda", "--output", str(RESULT_ROOT / "runtime_preflight.json")], "runtime-preflight")
        """),
        md("""
        ## Steps
        ### 5. Freeze public inputs and inspect scope
        GSM8K: numeric answer after a terminal `####` line. CRUXEval-O: exact
        typed Python literal between `[ANSWER]` and `[/ANSWER]`; this is code
        understanding, not code generation/pass@1. IFEval: original user prompts
        and Google's strict/loose checkers. Its split is named `train`, but no
        training happens here. Custom zero-shot chat prompts are not official
        leaderboard protocols or evidence of pretraining-contamination freedom.

        Thinking disabled; greedy generation. Caps: math 1,024, code 512,
        instructions 2,048 new tokens. Prompts over 2,048 tokens stop preparation,
        never silently disappear. Truncated outputs fail primary task success;
        raw IFEval checker scores are retained separately. Do not tune a recipe
        on this test set and continue calling it held out.
        """),
        code("""
        public("freeze", "freeze-public-inputs")
        manifest = json.loads((EVAL_ROOT / "manifest.json").read_text())
        print({"manifest": manifest["fingerprint"], "datasets": manifest["datasets"],
               "labels": manifest["identity"]["labels"], "prompts_per_arm": len(manifest["items"])})
        """),
        md("""
        ### 6. Source FP16 and existing seed-0 candidates
        Logs print prompt starts/completions, token counts, success and truncation;
        heartbeats continue during long model sections. Each result is checksummed
        on Drive. The log prints the PID and a `tail -f` command. After disconnect,
        check that PID before restarting. Interrupting run_live terminates its worker
        group; a new session resumes complete prompts, not the interrupted generation.
        Packed loads must pass original saved probes and residency checks first.
        """),
        code("""
        for label in ("source_fp16", "b5_v6_s0", "b5_v8_s0"):
            public("run", label, ["--label", label])
        """),
        md("""
        ### 7. Pinned BF16-GGUF bridge and Unsloth UD-Q4_K_XL
        Controls share the full token-axis mapping, exact prompt bytes, frozen HF
        IDs, stopping rules and caps. Native tokenizer differences are recorded,
        not discarded or asserted away. Keep PyTorch imported before llama.cpp.
        Build failure is not a benchmark result. No automatic CPU fallback.
        Multimodal sidecar bytes are included in size accounting; vision is untested.
        """),
        code("""
        from scripts.run_unsloth_qwen35_4b_kl import LLAMA_CPP_PYTHON_REVISION
        run_live([sys.executable, "-m", "pip", "install", "diskcache==5.6.3", "jinja2==3.1.6",
                  "typing-extensions==4.15.0"], "llama-dependencies")
        check_build = ("import torch; from scripts.run_qwen35_fresh_eval import llama_build_identity; "
                       "print(llama_build_identity()); import llama_cpp; "
                       "assert llama_cpp.llama_supports_gpu_offload(), 'CUDA offload unavailable'")
        probe = subprocess.run([sys.executable, "-c", check_build], cwd=REPO_DIR, capture_output=True, text=True)
        if probe.returncode:
            print(probe.stdout, probe.stderr)
            os.environ.update({"CMAKE_ARGS": "-DGGML_CUDA=on", "CMAKE_BUILD_PARALLEL_LEVEL": "2", "FORCE_CMAKE": "1"})
            run_live([sys.executable, "-m", "pip", "install", "-v", "--force-reinstall", "--no-deps",
                      f"git+https://github.com/abetlen/llama-cpp-python.git@{LLAMA_CPP_PYTHON_REVISION}"],
                     "llama-cuda-build", timeout_seconds=3600)
        run_live([sys.executable, "-c", check_build], "llama-build-check")
        for label in ("gguf_bf16_bridge", "unsloth_ud_q4"):
            public("run", label, ["--label", label])
        """),
        md("""
        ### 8. Existing seed-1/2 checkpoints—never requantize
        Both recipes stay registered regardless of seed-0 scores. Seeds vary
        quantization/rotation RNG on the same calibration selection, not independent
        training corpora. Do not select the luckiest seed or pool repeated prompts
        as independent samples. Seed-0-only mode is a separately labelled scope.
        """),
        code("""
        if not seed0_only:
            for seed in (1, 2):
                for recipe in ("b5_v6", "b5_v8"):
                    label = f"{recipe}_s{seed}"
                    public("run", label, ["--label", label])
        else:
            print("Registered seed-0-only scope; seeds 1/2 are not claimed complete.")
        """),
        md("""
        ## Checks
        ### 9. Reconcile every registered prompt and inspect each benchmark
        Missing arms cause a nonzero exit and appear in summary.json. No average
        across unrelated benchmarks. The primary score requires strict checker
        success and completed generation. Inspect format failures, truncation,
        FP16-correct → candidate-wrong flips, the BF16 bridge, and paired confidence
        intervals (right minus left, positive is better). Intervals are descriptive,
        unadjusted for multiple comparisons; there is no automatic promotion.
        The 128-example subset is a screen: nonsignificance does not prove
        non-inferiority, and identical observed outcomes do not prove equivalence.
        """),
        code("""
        import pandas as pd
        public("summary", "public-summary")
        summary = json.loads((EVAL_ROOT / "summary.json").read_text())
        display(pd.DataFrame(summary["rows"]))
        display(pd.DataFrame(summary["paired_contrasts"]))
        print({"complete": summary["complete"], "missing": summary["missing"],
               "interpretation": summary["interpretation"], "smoke_only": SMOKE_RUN})
        """),
        md("""
        ### 10. Download the compact handoff
        Keep original checkpoints on Drive. Results include public prompt/answer
        text under the source dataset licenses; generation may reproduce source
        text. No weights, teacher logits, credentials or checker caches are included.
        """),
        code("""
        import zipfile
        if DOWNLOAD_RESULTS:
            archive = Path("/content") / f"qwen35_public_tasks_{COMMIT[:12]}-{scope}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(RESULT_ROOT.rglob("*")):
                    if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".sha256", ".log"):
                        bundle.write(path, path.relative_to(RESULT_ROOT))
            from google.colab import files
            files.download(str(archive))
        """),
        md("""
        ## Next steps
        A successful smoke run only establishes plumbing. Set SMOKE_RUN=False for
        the registered public comparison; its output root is different. Bring back
        the full-run bundle. Keep W5/W6 as the under-budget candidate and W5/W8 as
        the fidelity alternative until per-benchmark evidence is assessed.

        Next engineering milestone: one measured packed serving path, starting
        with native/GGUF/llama.cpp interoperability. A Dynamic-v3.0-level product
        claim still needs genuine coding/tool-use workloads, broader models and
        measured speed/memory—not just this task subset or the existing C4 win.
        """),
    ]
    for i, cell in enumerate(cells):
        cell.id = f"public-task-{i:02d}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": OUTPUT.name}})


if __name__ == "__main__":
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(OUTPUT)
