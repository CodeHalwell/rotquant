# Qwen3.5-4B packed-vocabulary validation

Use [the new Colab notebook](../notebooks/qwen35_4b_packed_validation_colab.ipynb),
not another nine-arm screen. This implements the artifact milestone following
the [successful vocabulary screen](vocabulary_results_2026-09-07.md).

Status: code, generated notebook, CPU fixture tests and offline tiny multimodal
Qwen fresh-process reload checks are available. **The full pretrained CUDA run
has not been executed locally.** These changes must be published before the
notebook's default `REPO_REF = "main"` can load them. No new provider result,
independent confirmation, fused-kernel speedup or full-Qwen artifact pass is claimed.

## Recover the `8f10ee60fc7f` reload failure without requantizing

Use [the saved-checkpoint revalidation notebook](../notebooks/qwen35_4b_packed_revalidation_colab.ipynb).
Its default `REVALIDATE_FROM` points to the original Drive run. Run it top to
bottom **after the fixed code has been published**. Keep `PREPARE_ONLY = False`.
The plan must print `backbone_quantizations_per_seed: 0`.

The original W5/W6 run passed the prototype probes but failed the stricter reload
probes (mean KL 1.1783e-5); full packed quality was skipped. Investigation found
that the loader's blanket FP16 conversion rounded framework-created FP32 RoPE
buffers. Those nonpersistent tensors are absent from the checkpoint state; they
must be reconstructed and kept at their framework precision, not cast down and
back up. The local tiny-Qwen regression reaches exactly equal outputs after
preserving them. The actual 4B CUDA rerun is still required to confirm the cause.

The fix does not change codes, scales, artifact files, or numerical tolerances.
The preflight now constructs low-precision parameters while retaining FP32 RoPE
buffers, asserts those buffer dtypes after reload, and uses 64 input tokens and
eight generated tokens. The previous whole-model cast in the tiny fixture had
hidden this difference between the real source loader and the packed loader.

Recovery reads the two original `checkpoint/` directories, preparation records,
and dense/packed probe safetensors. It writes only to a **separate, nonoverlapping
output directory**, including a new token cache, logs and validation records.
The original failed validation remains unchanged. Each new result records the
original preparation identity, prepared-file checksum and manifest hash, plus
the current validator identity. No original record is relabelled as a new run.

Model/config, runtime versions/GPU, prompt contents, seeds and thresholds must
match. Only implementation provenance and hash-equivalent prompt-file locations
may change. Missing/corrupt artifacts or incompatible settings stop before
expensive model work. Closing the browser is safe; explicit cell interruption
stops the process group. Resume with the same fixed commit and output folder.

For an existing Colab session, once the checkout contains the fixed code, this
cell uses the saved artifacts directly (no installation or quantization):

```python
import subprocess, sys
from pathlib import Path

repo = Path("/content/rotquant-packed-validation")
sys.path.insert(0, str(repo))
from scripts.colab_runtime import run_live
source = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
assert "--revalidate-from" in (repo / "scripts/run_qwen35_packed_validation.py").read_text(), "Load the fixed code first."
output = source.parent / f"{revision[:12]}-revalidate-{source.name}"
logs = output / "logs"
command = [sys.executable, "-u", str(repo / "scripts/run_qwen35_packed_validation.py"),
           "--output-dir", str(output), "--revalidate-from", str(source),
           "--seed", "0", "--heartbeat-seconds", "60"]
run_live([*command, "--dry-run"], "plan", repo_dir=repo, log_root=logs)
run_live([sys.executable, "-u", str(repo / "scripts/preflight_packed_validation.py"),
          "--device", "cuda", "--output", str(output / "runtime_preflight.json")],
         "preflight", repo_dir=repo, log_root=logs)
run_live(command, "packed-validation", repo_dir=repo, log_root=logs)
```

Keep the original runtime's fast-Hadamard selection and package versions. The
revalidation notebook provides setup for a fresh Colab session. Both recovery
routes still load the pinned source teacher and repeat full development quality
after numerical parity passes. They skip Hessians, vocabulary quantization,
backbone GPTQ and re-export, not the required quality checks. The compact ZIP
keeps original small records under `original/`; tensors remain in the source
Drive directory. A failed new parity check prints its measurements immediately
and does not silently relax thresholds or proceed to a full quality run.

## What to run

Run the notebook top to bottom on an A100 40 GB-class runtime. Keep:

```python
SEEDS = (0,)
RUN_EXPERIMENT = True
PREPARE_ONLY = False
REQUIRE_FAST_HADAMARD = True
```

The two recipes are W5 backbone / W6 vocabulary first, then W5 / W8. Both use
the screened Gaussian/FWHT quantization: backbone act-order GPTQ, group128,
scale8; vocabulary group128 and FP16 scales without GPTQ. One source-calibrated
backbone quantization is shared across them. Activations remain FP16.

The 24-prompt primary KL, 25-snippet diverse KL, 25-prompt/32-token trajectories,
and 32-window WikiText-2/C4 perplexities are reused **for conformance**, not
relabeled as independent validation. The old W4 control and Unsloth aggregate
are historical context; neither is rerun in this notebook. There is no automatic
mixed-allocation, recovery, A8/KV or final benchmark stage.

## Execution sequence

1. Resolve the selected Git ref to a commit; print it and create a persistent,
   revision-specific Drive directory. Preserve this commit when resuming.
2. Install dependencies without replacing Colab's PyTorch; verify the fast
   Hadamard kernel. Source-build output is persistent and bounded by a timeout.
3. Run an offline, random-weight tiny **multimodal Qwen** fixture through both
   packed recipes and subprocess reload. This exercises the real loader, retained
   vision tensors and hybrid recurrent/attention text path, not vision quality.
4. Load the pinned source, prepare resumable source Hessians, and capture its
   unchanged reference logits/trajectories. Build both vocabulary owners from
   the original shared matrix and patch the W5 backbone once.
5. For each recipe, evaluate its dense vocabulary prototype and capture bounded
   numerical probes. Swap in the shared packed owner, clear backbone dense
   caches, test prototype parity and export a transactional checkpoint v3.
6. Launch a **fresh Python process per artifact**. Capture the unchanged teacher
   again, release the teacher model, then load only the exported student with
   `fallback=False`. References remain on CPU; no source teacher remains on GPU.
7. Check reload probes and packed ownership, then repeat full development
   quality evaluation. Verify pairing by prompt/token counts and PPL window
   hashes. Record actual complete file bytes and live cache state.

## Finite-precision projection semantics

An FP16 prototype stores `W_hat = inverse_rotation(dequantized_codes).half()`.
Computing `linear(rotate(x).half(), rotated_weights.half())` is equivalent in
exact arithmetic, but need not produce the same FP16 logits. The new explicit
`dense_equivalent` head reconstructs and rounds **one vocabulary row tile** to
the execution dtype before multiplying. It does not keep a full dense head.
GEMM tiling can still change reduction rounding; numerical tolerances and the
complete quality checks remain necessary.

[Checkpoint v3](packed_format_v3.md) records this execution mode separately from the quantized payload.
Old v3 files without the field keep their original `rotated` behaviour. It is
not a new quantizer or a stock GGUF format, and it is not a fused kernel.

## Preregistered gates

The gate values live in `rotquant.validation` and `rotquant.eval.promotion` and
are included in each experiment identity. Do not loosen them mid-run.

| Check | Prototype → packed | Pre-export packed → fresh reload |
|---|---:|---:|
| Maximum absolute logit error | ≤ 0.125 | ≤ 0.002 |
| Mean absolute logit error | ≤ 0.005 | ≤ 0.0002 |
| Mean KL on sampled logit positions | ≤ 1e-4 | ≤ 1e-6 |
| Top-1 agreement | ≥ 99% | 100% |
| Exact short greedy output | Reported, not an isolated hard gate | Required |

Default probes use the first four primary evaluation prompts, capped at 64
input tokens, four logit positions and eight generated tokens. These are small
numerical sentinels, not a substitute for the full suites. The thresholds are
engineering acceptance limits, not estimated statistical uncertainty.

The complete reloaded model must pass the existing no-regression guards against
its **same-run dense prototype**: primary/diverse KL and each PPL ratio ≤1.02,
primary/diverse top-1 loss ≤1 percentage point, trajectory agreement loss ≤2
percentage points. All metrics and pairing information must be present/finite.
The numerical probes run first; failed parity skips the expensive quality pass.

The size ceiling is the historical complete Unsloth bundle's 3,584,533,344 bytes
plus at most 1%. Actual artifacts materially below this budget are allowed on
the frontier. A separate **two-sided** 1% pairing flag distinguishes near-equal
size from merely under budget; the latter is not described as an exact-size pair.

An artifact passes this milestone only with valid file hashes, successful probes,
full quality guards, no dense model-owned vocabulary/backbone cache, and the size
ceiling. It is still **not** independently confirmed or provider-competitive.

## Bytes, resources and performance

`checkpoint/` includes the retained vision parameters, model/config/generation
files, one shared packed vocabulary, packed backbone and tokenizer/processor
files. Every file contributes to measured size. The verifier rejects unrecorded
files/symlinks, altered hashes, reused vocabulary code/scale keys and a retained
dense vocabulary matrix. The payload/container byte totals reconcile.

Keep at least **45 GB free in Drive** for two roughly 3.5 GB checkpoints, about
17 GB of source Hessians, vocabulary caches, token data and staging margin. More
space is needed for more seeds or retained interrupted staging directories.
Source downloads also need local Colab disk. The logged filesystem free-space
number is not a reliable Google Drive quota measurement. Source logit captures
can occupy roughly 12 GB of host RAM at the declared upper bound; use a high-RAM
runtime. These are planning estimates, not a memory-fit guarantee.

Packed state does not imply fast inference. The current backbone transiently
dequantizes each layer, and the vocabulary reconstructs a tile per projection.
Full trajectories may run substantially slower than the prior dense screen.
We record load/evaluation peak allocated CUDA bytes separately; neither is a
steady-state VRAM or throughput benchmark. Kernel performance is the later gate.

## Persistence, monitoring and failure handling

The notebook writes stdout/stderr **directly to Drive**, while displaying it
live. A browser reconnect cannot fill a pipe and block the worker. It prints a
PID and usable `tail -f` command. Runner/worker heartbeats include elapsed time
and GPU state; `progress.json` records source, Hessian, prototype, export,
reload, evaluation, completion or failure phases.

Before restarting after a disconnect, check the printed PID or:

```bash
ps -eo pid,etime,pcpu,pmem,args | grep '[r]un_qwen35_packed_validation'
```

An explicit cell interruption terminates the subprocess group. Rerun the same
notebook with the same commit/settings/output root to resume. Completed artifacts
and probe/result checksums are verified before skipping work. Source Hessians and
vocabulary chunks resume; an interrupted backbone patch is rebuilt from them.
There is one writer per root. Code, configuration, runtime or prompt-file identity
changes require a fresh output directory.

Preparation evidence and probe hashes are saved **before** export, and their
digest is embedded in the checkpoint manifest. If interruption occurs after
checkpoint publication but before `prepared.json`, the runner verifies those
bindings and safely finalizes the record without requantizing. An incomplete or
unverifiable orphan is preserved, never automatically overwritten; retain it
and choose a new root. Do not delete large caches/checkpoints to make a failed
gate pass. A recorded failed validation remains failed on resume; a code fix
requires a new revision/run. Use `--revalidate-from OLD_RUN --output-dir NEW_RUN`
for the explicit checkpoint-only path described above, not a normal run against
the old output directory.

The compact ZIP contains JSON records, checksums, logs and checkpoint manifests,
**not tensor binaries or probe safetensors**. Keep actual checkpoints and probes
on Drive; the compact bundle alone cannot independently reload the model.

## Local validation and exact GPU gap

After the reload fix, the complete local suite reports **643 passed, 17 skipped**
(uncompiled AVX2 native tests). FP16/BF16 tiny Qwen regressions preserve the FP32
rotary values and match saved/reloaded outputs exactly. Recovery tests verify
unchanged original files, no preparation calls, resume, lineage and rejection of
incompatible settings. Both generated notebooks are schema-validated and their
code cells compile; neither full notebook has been executed in Colab locally.

```bash
.venv/bin/python scripts/build_qwen35_packed_validation_notebook.py
.venv/bin/pytest -q tests/test_packed_validation.py
.venv/bin/python scripts/preflight_packed_validation.py --device cpu --output /tmp/packed-preflight.json
.venv/bin/python scripts/run_qwen35_packed_validation.py --output-dir /tmp/packed-plan --dry-run
```

Notebook cells are structurally validated and compiled locally, not executed
top-to-bottom: the local machine lacks CUDA and Colab Drive. To close that gap,
run the notebook in Colab, or on an equivalent configured CUDA host run:

```bash
python scripts/preflight_packed_validation.py --device cuda --output results/packed/runtime_preflight.json
python -u scripts/run_qwen35_packed_validation.py --output-dir results/packed --seed 0
```

After this gate, freeze fresh validation inputs, run seed 1/2 confirmation with
appropriate controls, then a common-input provider comparison with explicit
source/engine parity. Allocation and LoRA remain conditional on that evidence.
