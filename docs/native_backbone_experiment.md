# W5 backbone kernel experiment

Use [qwen35_4b_native_backbone_colab.ipynb](../notebooks/qwen35_4b_native_backbone_colab.ipynb)
on a fresh **A100 40GB** runtime. Check the saved checkpoint path, then **Run all**.
This is the next experiment after the
[completed followup4 profile](../research/results/native_decode4_profile_2026_09_13/README.md).
Keep the previous notebook/reports for reproducibility; do not rerun an unchanged
profile or update a checkout while it is executing.

## Hypotheses, not results

| Opt-in kernel | Decode | Prefill |
|---|---|---|
| `decode4` | Validated four-row/warp implementation | Validated four-token tiles |
| `w5s8` | Compile-time W5 extraction and scale8/block256 metadata arithmetic | Shared codebook, warp-broadcast scale, four-token tiles |
| `w5s8-tile8` | Same W5 specialization | Eight-token tiles |
| `w5s8-tile16` | Same W5 specialization | Sixteen-token tiles |

The specialized device branch requires W5, scale8, metadata block256 and the
existing g128/aligned-input operator contract. Other formats use generic
arithmetic. One-token decode preserves four virtual accumulators per lane and
the original 128-lane reduction association. Prefill retains each token's
accumulation and tree order, including partial tiles. The existing strict
floating-point compiler flags are unchanged. No tensor-core accumulation change,
full dense weight expansion, vocabulary rewrite or rotation rewrite is made.

The default remains `reference`; all optimized variants are opt-in. `decode4`
is the **comparison baseline**, not a silent change to production defaults.
The retained W5/scale8/W6 model is unchanged, including codebooks and scales.
W6/W8 vocabulary and other scalar formats retain their arithmetic contracts.

## Bounded execution funnel

1. Verify the Colab environment and original source receipts. Build/restore a
   source-hash-compatible runtime and perform a fresh binding-load check.
2. Before loading the model, run decode4 and candidate dispatch/operator gates.
   Candidate tests cover row tails, column/row permutations, long reductions,
   W1–W8 formats, scale8/scale16, alternate metadata blocks and 4/8/16 tile tails.
   Exact operator agreement with the original native reference is required in
   addition to canonical Torch checks. A numerical/dispatch failure stops the run.
3. Screen each candidate against decode4 on three synthetic projection shapes:
   2560×2560, 9216×2560 and 2560×9216, at 1 and 128 tokens. Each shape/length
   uses AB then BA order, one excluded warmup per native call, three measured
   repetitions, ten resident graph executions per repetition. Persist every pair.
4. Shortlist at most two candidates. Every shape/order must reach **0.95×**
   decode4, and at least one phase's geometric-mean ratio must reach **1.05×**.
   Rank eligible candidates by their best phase ratio. Preserve rejected rows.
   If none qualifies, complete the screen and skip model export/inference.
5. For finalists, run fresh CPU/CUDA, tiny W6/W8 model, conversion and retained
   saved-checkpoint gates, then normal decode4 model timings. Repeat the
   candidate's full-model gates before its own timings. Keep original thresholds.
6. Reuse the exact decode4 prompt/decode IDs for full-model comparisons. Each
   candidate gets a distinct comparison file, bound to the current screen hash.
   Historical results from an earlier screen are excluded from the new readout.

The screen measures resident **rotation + packed matrix graph** execution,
including launches and completion synchronization. Allocation, static uploads,
warmup and output readback are outside timing. CUDA graphs are disabled; event
profiling is off. No dense source weight tensor is needed to generate the
synthetic fixture. This working set is smaller than a full model and may remain
in cache. Shapes/distributions are not a layer-frequency-weighted Qwen replay.
Consequently, screen ratios are shortlist heuristics, not serving speedups,
bandwidth measurements, confidence intervals or significance tests.

Full-model timings restore normal graph settings and existing synchronous-bridge
rules: three repetitions plus excluded warmup, 32 cached decode steps, FP16 KV,
fixed replay and sampled process VRAM. Only fresh comparable uninstrumented
reports yield ratios. The reference in these ratios is **decode4**, not the old
~20 tok/s implementation. Saved-model parity is not full-precision/task accuracy.

## Defaults and recovery

| Control | Default |
|---|---|
| `REPO_REF` | `main`, resolved once to a full commit |
| `RUN_NAME` | `backbone1` |
| `ARMS` | `("b5_v6_s0",)` |
| `CONTEXTS` | `(128,)` |
| `KERNEL_CANDIDATES` | All three W5 variants |
| `MAX_FINALISTS` | 2; choose 1 for less model work |
| `ACTIVE_BUDGET_MINUTES` | 90 |
| `BUILD_JOBS` | 2 |
| Original evidence | `MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f` |
| Private cache | `MyDrive/rotquant/native_artifact_cache/v1` |

The source must include the actual checkpoint and saved probes/receipts, not
just a reports ZIP. It is never modified. No model downloads, training,
requantization, public tasks or conventional GGUF benchmarks are requested.
The new CUDA/bridge hashes require a new binary cache entry. Compatible later
runs reuse it; the old binary cannot satisfy this source contract. Keep caches
private and do not clear them as a repair step. A previous cold build took
roughly 27 minutes; that is historical, not a promised duration for this build.

Every stage has a persistent log/PID/heartbeat and an active cap. Build cap:
45 minutes; operator/screen caps: six minutes each. All are bounded by the
remaining 90-minute allowance, which excludes notebook idle time. A failed
process is terminated and its partial reports retained; no guard is relaxed.
Read Results after any stop and download the reports ZIP. **Disconnect/delete
the GPU runtime yourself: script completion/timeouts do not stop billing.**

Rerun the launch cell with unchanged controls to recover an interruption. With
a new VM, select the same full commit and run name. Build/export reuse is hash
checked, while numerical screens and timing remain fresh. Change run name for
changed controls or renewed budget. Old screen comparisons are preserved but
excluded from the current screen's readout. The ZIP includes JSON/logs, not
weights or binaries.

## Local validation boundary

The native patch applies to the pinned llama.cpp base and all 28 entries in its
file hash contract match. A real CPU build/load, all 18 CPU operator cases and
repeated benchmark-output comparison passed locally. The full regression suite
passed **1,027 tests**, with **18 platform-dependent skips** (Linux ELF loading
and unavailable AVX2 kernels on this host); lint and whitespace checks passed.
Tests exercise screening
rejections, dispatch failures, baseline identity, preserved environment settings,
stage order and notebook execution using explicit mocks, never fake GPU data.

Compile-only validation used NVIDIA's CUDA 12.8.1 development container, without
a GPU or driver, targeting A100 `sm_80`. CUDA compilation succeeded and reported
no spills. Register counts were 48 for both decode instantiations and 64/91/128
for the W5 4/8/16-token tiles; static shared memory was 2176/4224/8320 bytes for
those tiles. These figures do not establish occupancy, correctness or speed.
Larger tiles may lose performance despite better weight reuse.

Reproduce the compile-only check after applying the verified patch in a fresh
source directory, using the [NVIDIA CUDA compiler/toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/):

```sh
nvcc -std=c++17 -arch=sm_80 --fmad=false --ftz=false \
  --prec-div=true --prec-sqrt=true -Xptxas=-v \
  -I/source/ggml/src/ggml-cuda -I/source/ggml/src \
  -I/source/ggml/include -I/source/include \
  -c /source/ggml/src/ggml-cuda/rq3.cu -o /out/rq3.o
```

The fresh patched source is `/source` in this example. Real CUDA operator/model
execution and the Colab-rendered result tables are still required. Open the
notebook on A100 and run all cells to perform those checks; local/mock execution
must not be substituted for that result or shipped as measured output.

## Following this run

A confirmed win earns 512/2048-token and retained W5/W8 checks, fresh conventional
controls and cost-bounded native task-quality evaluation. Vocabulary-head work
is the next independent kernel change, preserving inverse-rotation/FP16
reconstruction semantics. Do not enable kernels globally, restart allocator or
recovery sweeps, or scale to 27B on a synthetic-screen win alone.
