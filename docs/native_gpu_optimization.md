# Native GPU optimisation study — 11 September 2026

## Run next

Open [the end-to-end notebook](https://colab.research.google.com/github/CodeHalwell/rotquant/blob/main/notebooks/qwen35_4b_native_optimization_colab.ipynb),
select **A100 40GB**, verify the original saved-artifact path, then **Run all**.
It keeps the validated `b5_v6_s0` recipe unchanged. There is no calibration,
requantization, recovery training or task sweep. Local Google CLI/MCP login is
not required; Colab mounts your Drive normally.

Your [first native pilot](../research/results/native_pilot_2026_09_11/README.md)
measured approximately 36.4 prefill and 19.8 decode tok/s. The 2048 stage timed
out because four ~58-second repetitions could not fit in three minutes.
That scheduling error is corrected; it is not evidence of GPU instability.

## Experiment and gates

1. Recheck the saved source, isolated Python environment and native library.
   Build/cache integrity never substitutes for GPU conformance.
2. Repeat CPU/CUDA operator, tiny W6/W8, conversion and retained-model parity.
   Measure the existing reference kernel at each selected context.
3. Test experimental `tiled4` against canonical operator arithmetic and exact
   reference outputs, including 3/4/5-token tile boundaries and permutations.
   Repeat tiny whole-model and retained W5/W6 parity before timing it.
4. Profile both variants separately at the smallest selected context. Record
   CUDA-event milliseconds for rotation, matrix, vocabulary-head and embedding
   operations, split into prefill and decode.
5. Run pinned BF16 and Unsloth UD-Q4 controls through the **same compiled
   native bridge**, without a second llama-cpp-python engine/build.

The candidate reuses a decoded weight across four prompt tokens. It does not
materialize a dense weight matrix or change codebooks, scales, rotations,
FP16 rounding or reduction order. It applies only to backbone matrix operations
with at least four tokens. Single-token decode and the vocabulary head retain
the existing implementation. It is **opt-in, not promoted or CUDA-validated**
until the new run passes. A speedup is a hypothesis, not a result.

## Controls and cost

| Control | Default | Meaning |
|---|---|---|
| `ARMS` | `("b5_v6_s0",)` | Existing W5/scale8 + W6 vocabulary artifact |
| `CONTEXTS` | `(128, 512, 2048)` | One fresh process per selected shape |
| Decode / repetitions | 32 / 3 | One additional excluded warmup |
| Normal phase caps | 4 / 6 / 12 minutes | Load, verification and all repetitions included |
| `RUN_PROFILE` | `True` | Separate event-instrumented processes; doubled phase caps |
| `BASELINES` | `("bf16", "ud_q4")` | Pinned conventional GGUF controls |
| Active allowance | 90 minutes | Includes child stages; does not stop billing |
| Process VRAM guard | 16,384 MiB | Sampled memory; not a peak guarantee |
| Decode spending floor | 2 tok/s | Abort expensive slow runs; not scientific promotion |

The cap uses two minutes of load/verification allowance plus every warmup and
measured repetition budgeted at 16 prefill tok/s and 2 decode tok/s, rounded up.
Diagnostic caps double the normal cap, at most 30 minutes, and the cumulative
90-minute allowance still applies. No timeout or speed/memory guard is weakened
silently. Reports retain completed repetitions and the in-progress phase.

For a targeted rerun set `CONTEXTS = (2048,)` and a new `RUN_NAME`.
Set `RUN_PROFILE = False` and/or `BASELINES = ()` explicitly if those measurements
are already available. Correctness gates remain fresh, even on cache hits.
Changing controls requires a new result directory; do not update a running
checkout. The historical reports/weights are never overwritten.

New CUDA source means **one new cold build**. The previous build took ~27
minutes; that is an observation, not a promised duration. Incompatible cached
binaries cannot be reused. Compatible later runs can use the existing private,
hash-checked Drive cache. This patch also conservatively invalidates the export
cache contract; source weights are reused without quantization. The conventional
GGUF downloads total 11.34 GB and remain on local VM disk.

The notebook prints stage progress, heartbeat/log paths, prefill timings and
decode progress. Run Results even after a stop, then Download. **Disconnect and
delete the GPU runtime** after downloading: process timeouts do not stop Colab
billing, and no automated billing shutdown is performed.

## Measurement boundaries

Normal rates include synchronous native model calls, finite-logit checks,
full-vocabulary host copies and argmax. Report writing/logging are outside the
timers. This is a fixed-token private-bridge comparison, not llama-bench or a
production serving claim. All models use the same prompt ID hash, context
reservation, FP16 cache, batch-one sequence and warmup/repetition counts.

The reference produces greedy decode IDs; every candidate and control replays
the exact same IDs while still performing argmax/model calls. Determinism of
the reference trace and full token-ID vocabulary mapping are checked. No
multilingual retokenization is introduced. Controls are pinned to
`unsloth/Qwen3.5-4B-GGUF` revision `e87f176479d0855a907a41277aca2f8ee7a09523`,
with full file-size/SHA validation in `run_native_gguf_baseline.py`.
These are **not equal-size or equal-quality** controls, and this run measures
no new KL or task accuracy. Text GGUF bytes exclude the unloaded auxiliary
sidecar; the complete RotQuant payload remains reported separately.

CUDA profiles use [CUDA events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
on the execution stream, synchronizing each custom operation with CUDA graphs
disabled in a fresh child process. Event instrumentation perturbs scheduling.
Its wall rates must not enter speed comparisons. Operator counts are host
dispatches, not CUDA-graph replay/instruction counts. Profiles attribute custom
ops only: attention/SSM kernels, CPU overhead and transfers are not separately
measured, and event totals do not equal whole-model wall time. DRAM bytes and
static-weight transfer traces remain a later profiler task, not inferred from
file size or these events.

`comparison-*.json` emits ratios only for complete, matched, uninstrumented
runs. Partial/profile rows remain visibly separate. There is no automatic
kernel promotion. Review head-vs-backbone costs, repeated speed measurements
and unchanged parity before choosing the next kernel change; only then return
to retained W5/W8, longer-context quality and native public-task benchmarks.

## Local verification boundary

Notebook format/source consistency, top-to-bottom execution under explicit
Colab/GPU mocks, targeted workflow order, fail-closed gates, replay tokens,
budget arithmetic, partial reports, baseline identities and cache behavior are
tested locally. The patch is applied to a fresh pinned llama.cpp tree and a CPU
library is compiled/loaded with CPU operator checks. None of those constitutes
CUDA compilation, tiled-kernel parity or measured A100 improvement. Those are
the first gates of the supplied notebook; no paid GPU job was launched here.
Rendered Colab presentation and real GPU outputs remain uninspected locally;
open the notebook in Colab to inspect the saved sections, then use A100 / Run
all to validate real execution and the resulting tables. Mock outputs are not
saved into the distributable notebook as if they were experiment results.
