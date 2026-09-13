# Native GPU follow-up: baseline completion and decode optimisation

## Run next

Use [qwen35_4b_native_followup_colab.ipynb](../notebooks/qwen35_4b_native_followup_colab.ipynb)
on an **A100 40GB**, verify `SOURCE_ROOT`, then **Run all**. The notebook has
persistent logs, per-phase caps, progress for every repetition, a results cell
that also works after a stop, and a reports download cell. Original checkpoints
and past reports are never edited. There is no calibration or training.

The [completed study evidence](../research/results/native_study_2026_09_11/README.md)
supports tiled4's 3.30–3.35× prefill gain, not a decode gain. BF16/Unsloth speed
comparisons remain unfinished. The new run prioritizes those missing results.

## What changed

- The native bridge explicitly assigns ordinary `token_embd.weight` to the
  selected device's buffer. The override lives with the model. CPU mode is
  unchanged, and `ROTQUANT_REQUIRE_GPU` is not weakened. All-layer offload alone
  does not guarantee GPU input-embedding placement in this pinned llama.cpp.
- Tiny ordinary BF16 and Q4_0 Qwen graphs test GPU prefill and cached decode
  against CPU before any retained export/timing or 4B baseline download. They
  must use **zero** RQ3 custom operators. These plumbing tests are not an
  approximation of Unsloth's dynamic quantization quality.
- Fresh same-runtime reference timings precede the pinned BF16 and UD-Q4
  controls. Controls run and checkpoint **before** the speculative candidate.
- `decode4` is opt-in. Four warps process four output rows, with one warp per
  row. The small codebook is shared in the block, group scales are broadcast
  within each warp, and the dot-product loop avoids block-wide barriers.
  Each lane retains four virtual lanes from the original 128-thread kernel.
  Combining offsets 64 then 32 before the warp reduction preserves the intended
  original FP32 association; FP16 rounding and non-fused arithmetic remain.
  It does not expand dense weights. Tiled4 still handles prefill; vocabulary
  decoding and rotations retain their prior implementations.
- CUDA tests demand exact candidate/reference operator outputs as well as
  canonical tolerances: 1–8-bit codebooks, both scale formats, row tails,
  permutations, widths through 11008 and token boundaries. Fresh tiny-model
  and retained parity follow. Dispatch counters must prove the requested
  decode and tiled paths were used. No successful CPU test authorizes CUDA
  execution, and no speed improvement is claimed before the Colab measurements.

## Bounded controls

| Control | Default | Alternative |
|---|---|---|
| `ARMS` | `("b5_v6_s0",)` | Keep the tested recipe for this follow-up |
| `CONTEXTS` | `(128, 512)` | Explicitly add 2048 with a new run name |
| `CANDIDATE` | `"decode4"` | `"none"`: fresh reference + baselines only |
| `BASELINES` | `("bf16", "ud_q4")` | Explicit subset; empty only with a candidate |
| `RUN_PROFILE` | `False` | Optional diagnostic runs with a candidate |
| Measurements | 3 repetitions + 1 warmup | 32 cached decode steps each |
| Active allowance | 90 minutes | Includes builds/downloads/checks, not idle setup |

Baseline-only mode deliberately still runs fresh reference timing and all
applicable correctness checks. It skips candidate gates/timings and profiles.
It does **not** import old timings from an incompatible native binary. The CLI
equivalent is `--performance-study --study-candidate none --skip-profile`.
The normal decode follow-up uses `--study-candidate decode4 --skip-profile`.
Selected phases have their own context-aware caps; failures preserve partial
measurements and prevent ratios from incomplete or instrumented reports.

Changed native code means one compatible new build (previously ~27 minutes).
Existing old-runtime caches cannot validate it; later compatible invocations
can restore verified binaries. The private Drive export cache may also be
conservatively invalidated, but exports reuse the original weights without
quantization. BF16/UD-Q4 downloads total about 11.34 GB on local VM disk.

Do not update a running checkout. New controls require a new `RUN_NAME`; the
source and previous evidence remain intact. **Timeouts do not stop billing.**
Download reports and disconnect/delete the GPU runtime yourself.

## Interpretation and local validation

The model, prompt IDs, FP16 cache, decode replay and native bridge are matched.
These are synchronous bridge rates including logit copying/validation, not a
pure-kernel rate or serving benchmark. BF16/UD-Q4 are not equal-size or
equal-quality controls. This experiment does not collect new quality results.

Local checks include a compiled/loaded CPU bridge, both ordinary fixtures
executing prefill and cached decode without RQ3 operations, 96 canonical CPU
operator/corner checks, reduction-order arithmetic, fail-closed workflow tests
and notebook execution from top to bottom under explicit Colab/GPU mocks.
The CPU backend does not execute the candidate CUDA code. A local Metal build
could not complete because the Metal compiler/toolchain is unavailable.
CUDA compilation, real GPU placement/parity, performance and rendered Colab
outputs remain unvalidated locally. Run the notebook on A100/Run all to close
those gaps. No GPU session was provisioned or billed by this development task.
