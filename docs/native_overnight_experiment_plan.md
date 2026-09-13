# Overnight native GPU experiment — implementation and runbook

User-selected ceiling: **eight active GPU-hours** on 13 September 2026.
The user subsequently approved local implementation. No GPU provisioning,
launch, purchase, scheduling or GitHub push has occurred as part of this work.
Use the separate [overnight notebook](../notebooks/qwen35_4b_native_overnight_colab.ipynb);
the existing notebooks retain their original budgets and defaults.

## Implemented first iteration

The design below remains the broader research direction. The first runnable
iteration deliberately contains **six independent candidates and one conditional
combination**, not 12–24 superficial configurations:

| Registry name | Implementation | Numerical lane |
|---|---|---|
| `head-warp` | Four vocabulary rows per block; warp-shuffle 128-point FWHT and original reduction order | Exact |
| `sgemm-128`, `sgemm-512` | Bounded reconstructed FP32 chunks; FP32 cuBLAS GEMM, TF32 disabled | Reordered |
| `hgemm-128`, `hgemm-512`, `hgemm-2048` | Bounded FP16 chunks; FP32 GEMM accumulation, Tensor Cores permitted | Reordered |
| `hgemm-512-head` | One fixed combination, only after both components independently pass | Reordered |

All preserve the checkpoint. Backbone decode uses the existing W5 path; GEMM
dispatch starts at four input tokens. The original implementation already
half-rounds decoded weights and activations: these staging paths **do not add
a new operand-quantization boundary**. They change accumulation order/FMA
behavior and therefore require separate numerical evidence. The same `.002`
relative / `.001` absolute canonical operator limits apply to the reordered
lane; unchanged paths/head exact lane additionally require bitwise reference
outputs. No thresholds are tuned after GPU failures.

The cuBLAS handle's math/pointer modes are restored after each operation.
SGEMM does not inherit llama.cpp's TF32 mode, and reduced-precision intermediate
reductions are disabled. Modern cuBLAS can choose Tensor Cores with the default
algorithm; **permitted is not measured utilization**. See NVIDIA's
[cuBLAS precision and GEMM documentation](https://docs.nvidia.com/cuda/cublas/index.html).

Requested input/weight/output scratch is bounded at 256 MiB **per custom op**.
Temporary decoded chunks are not retained as a dense weight cache. The pool
may retain reserved capacity; cuBLAS/context workspace and allocator rounding
are additional. Reports distinguish requested scratch from sampled process
VRAM. The preserved packed payload size is not the total serving-memory claim.

The first iteration records the actual export matrix inventory but screens
three synthetic backbone shapes and a bounded synthetic head. It does **not**
claim frequency-weighted or cold-cache microbenchmark coverage. Full-model
timings include staging and bridge costs. At most two backbone finalists plus
an independently eligible head proceed to tiny W6/W8, retained-model and
extended numerical gates, followed by fixed-token 128/512/2048 timing groups.
Candidate order reverses at alternate contexts; fresh controls bracket groups.
The conditional combination is limited to the one implemented pairing.

Extended fidelity uses 16 logit positions and up to 16 greedy steps for each
saved prefix repeated to 128/512/2048 tokens. All per-prompt guards must pass;
it is a synthetic numerical stress test against **quantized tile8**, not a
fresh FP16-teacher or task-quality evaluation. Original retained probes remain
a separate mandatory gate. A candidate's numerical rejection is saved and
followed by a fresh baseline health check; infrastructure errors/timeouts stop.

Fresh BF16/UD-Q4 controls use the established separately preflighted bridge,
after a candidate survives. Replay/settings/runtime hashes must reconcile
before computing a ratio. A final diagnostic profile pair is never mixed into
throughput. No winner is promoted automatically.

**Deferred deliberately:** additional two-dimensional SIMT/fused-unpack kernels,
learned shape dispatch, cold-working-set microbenchmarks, retained W5/W8,
the fresh FP16 teacher/task suite and the all-Unsloth-quant size curve. These
require subsequent evidence-driven work; this notebook does not claim them.

## Running and resuming

Select A100 40GB and review the notebook controls while present. `RUN_NAME`
defaults to `overnight1`; `AUTO_RELEASE_RUNTIME=True` requests release of that
Colab runtime after a verified Drive archive. Set False for manual recovery.
The driver never provisions resources. Budget starts at driver launch, after
interactive Drive mount/repository checkout; environment/build time is included.
Existing cached binaries/exports can be reused but current numerical gates and
measurements are rerun on a resume. All previous attempts remain preserved.
This is **not** a promise to skip every completed numerical/timing phase.
The same run cannot renew its consumed active time or original wall deadline.

The runner reserves finalization time and has both child-stage timeouts and an
independent notebook supervisor. On stop, inspect `experiment.json`,
`workflow.json`, per-attempt logs and the reports ZIP. Optional release only
follows CRC/content verification of the durable archive. Missing/failed archive
verification withholds release for manual recovery. Unassignment itself may
fail or terminate Python before a confirmation is saved; no billing-stop claim
is inferred. Confirm the runtime is disconnected/deleted afterwards.

## Decision to make

Can a different matrix implementation materially close the prefill gap while
preserving the compressed checkpoint, bounded runtime memory and model quality?
Can vocabulary-head work independently improve decode, and do the two changes
combine successfully?

Use Qwen3.5-4B, initially `b5_v6_s0`, on A100 40GB. Keep the original checkpoint
and probes immutable. Use `w5s8-tile8` as the principal performance control,
`decode4` as a regression anchor, and the original native implementation for
operator correctness. No 27B, allocator, LoRA, learned-rotation or quantization
recipe sweep belongs in this runtime experiment.

The supplied `backbone1-reports-1789332956558625649` bundle selected tile8 for
further testing. Its offline audit is separate from this design. Before
implementation is finalized, preserve that evidence and update the native
validation status; do not describe the new proposed candidates as validated.

## Broader candidate-family design (partly deferred above)

1. **Packed SIMT matrix improvements.** Two-dimensional reuse of weights and
   activations, vectorized loads and bounded shared-memory staging. Keep
   reduction/rounding semantics where the exact-reference contract requires it.
   Tile8 is the control; larger token tiles alone are not the experiment.
2. **Bounded unpack plus FP32 GEMM.** Reconstruct one bounded matrix chunk on the
   GPU, multiply it with cuBLAS, then reuse the workspace. Include reconstruction
   and synchronization in timing. This diagnoses whether matrix execution can
   overcome the cost of temporary unpacking without retaining a dense model.
   A changed reduction order is a numerical-contract change even with FP32.
3. **Tensor Core prefill.** Prototype an A100-compatible tiled implementation
   using a bounded FP16 staging path or fused unpacking with CUDA/CUTLASS.
   Weight reconstruction/operand rounding and accumulation order must be
   explicit. This is a separate numerical experiment, not an exact replacement
   merely because it computes the same algebraic expression.
4. **Vocabulary-head optimization.** Improve row tiling, reuse and reconstruction
   scheduling while preserving the existing inverse-rotation and FP16
   reconstruction semantics. Benchmark the head independently before combining
   it with a backbone finalist.
5. **Shape-dependent selection.** Consider dispatch thresholds only after the
   individual implementations pass. Tune on a declared shape subset and confirm
   on held-out shapes, lengths and full-model executions. Retain a safe fallback.

Use a finite, versioned candidate registry, not an unattended agent rewriting
code or a combinatorial search over arbitrary knobs. Target roughly 12–24
predefined configurations across the implemented families, only when each tests
a concrete hypothesis. Build compatible variants together and reuse binaries
by source hash. Do not include a family as a placeholder that silently executes
the control implementation.

NVIDIA's [efficient GEMM documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
describes hierarchical tiling, pipelining and reuse. Those are implementation
directions, not evidence of a RotQuant speedup. Use A100-compatible mechanisms;
Hopper-specific warp-group/TMA designs are not drop-in A100 candidates.

## Adaptive queue and budget

These are proposed allocation ceilings, not duration predictions or a target
to consume all eight hours. Finish as soon as the useful eligible work is done.

| Phase | Allowance | Deliverable |
|---|---:|---|
| Environment, build/cache and shared controls | 60 min | Provenance, load and baseline numerical gates |
| Candidate operator checks and staged microbenchmarks | 120 min | Complete pass/rejection table, shape-level timings and memory |
| Full-model confirmation | 90 min | At most two finalists, plus controls, at 128/512/2048 tokens |
| Matched baselines and bounded quality checks | 120 min | Same-run conventional comparisons and numerical/quality evidence |
| Targeted profiling, selected combination and finalization reserve | 90 min | Bottleneck evidence, optional independently justified combination, durable reports |
| **Global maximum** | **480 min** | Includes setup, downloads, compilation and failed trials |

The implementation must enforce both **480 active minutes** and a conservative
**480-minute wall deadline from launch**. These controls limit the job, not the
provider's billing. Preserve consumed active time and the original wall deadline
across a resume of the same run; renewing the allowance requires an explicit new
run name and visible controls. Leave time inside the deadline for finalization.

The current `Workflow` accepts at most 180 minutes, while existing notebook
defaults are lower still. Add an explicit overnight mode and tests; do not
silently remove the limit for every existing notebook.

### Candidate elimination

- Predeclare implementation identity, numerical lane and expected dispatch.
  Verify the executing CUDA dependency and that each candidate actually runs.
- Exercise tails, permutations, metadata blocks and all supported scalar
  formats; report unsupported shapes as unsupported, not as a candidate win.
- Reject numerical failures before expensive model work. Continue independent
  candidates only when the shared baseline and runtime remain healthy.
- Use fresh subprocesses after candidate failures. CUDA context corruption,
  lost GPU access, disk exhaustion, shared-baseline failure, invalid provenance
  or exhausted global allowance stops the entire run.
- Benchmark actual checkpoint-derived shape/format families, recording layer
  frequencies. Retain the synthetic square/expanding/contracting tests as
  diagnostics, not substitutes for representative model evidence.
- Test representative cold/large-working-set behavior as well as warm resident
  kernels. Include packing/unpacking, scratch allocation where applicable,
  transfers and synchronization in the appropriate end-to-end measurement.
- Screen cheaply first, increase repetitions only for survivors, and shortlist
  at most two backbone candidates. Benchmark head candidates separately, then
  at most one justified backbone/head combination.
- Randomize or counterbalance model-level control/candidate ordering, not just
  microbenchmarks. Repeat the control near the end to expose temporal drift.
- Keep diagnostic event profiles separate from ordinary throughput. Capture
  profiling only for an identified bottleneck or surprising result.

## Numerical, memory and comparison contracts

**Exact lane:** existing bitwise operator requirements remain in force for
implementations claiming unchanged arithmetic. Keep original model thresholds.

**Reordered/reduced-precision lane:** define and freeze justified operator
tolerances before GPU execution, based on the intended arithmetic and independent
CPU/reference checks. Do not relax thresholds after seeing failed candidates.
Keep existing retained-model guards and separately evaluate common-input logits,
teacher KL, top-1 agreement and bounded generation checks against a frozen
reference. Reuse a trustworthy pinned teacher cache when available; otherwise
include its generation in the budget. Clearly separate numerical fidelity from
task accuracy. Only execute pinned, bounded task fixtures with an established
runner; generated code must not run unrestricted on the Colab host.

**Memory:** retain packed model weights. Proposed initial scratch ceiling:
256 MiB, reported separately from persistent weight bytes and sampled process
VRAM. Record allocator high-water marks where supported. No persistent full
FP16 model cache may be presented as compressed inference. Bounded staging still
adds memory traffic; measure its total latency and do not infer bandwidth from
unchanged VRAM allocation. Report any faster-but-larger alternative separately.

**Comparisons:** use matching prompts, token IDs, batch/context/KV settings,
warmup rules and pinned runtime/model artifacts. Check conventional bridge
equivalence before fresh BF16/Unsloth timings. Evaluate the existing matched
Unsloth quant first; all published quant sizes are a separate size/quality study,
not filler for this runtime experiment. Broaden to retained W5/W8 only after
the W5/W6 candidate clears gates. Missing evidence is missing, never a pass.

A **2x prefill or 25% decode improvement over tile8** is an aspirational
milestone for a substantial new implementation, not a promised outcome or a
reason to change accuracy requirements. Smaller honest gains remain reportable;
no qualifying candidate is also a useful completed result.

## Unattended operation and readiness checklist

- Mount/authorize Drive and preflight enough disk/compute availability while the
  user is present. No keep-alive tricks or bypass of Colab resource restrictions.
- Log trial name, phase, elapsed/remaining time, GPU memory, current candidate,
  completed/failed/rejected counts and next action. Bound each subprocess.
- Save per-trial JSON and compact progress snapshots. Checkpoint to Drive
  continuously; do not rely on a final ZIP or browser download dialog at 3am.
  Write large working intermediates locally and persist only required artifacts.
- Save and verify a final reports bundle on Drive before attempting runtime
  release. Keep original evidence and private caches untouched. Retain rejected
  and failed results alongside successes.
- Provide an explicit notebook `AUTO_RELEASE_RUNTIME` control. Test finalization
  and release ordering with mocks. Use Colab's
  [runtime unassignment API](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py)
  when enabled; it can fail and is not a guaranteed provider-side billing cap.
  A wall-deadline watchdog and partial-result preservation are required, but
  cannot survive deletion of the VM or guarantee provider control-plane access.
- Colab does not guarantee uninterrupted availability; runtime duration and
  background execution depend on plan/resources. See the
  [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
- Before release: CPU/unit tests, notebook mock run-all, independent CUDA
  compile checks, failure/resume/budget tests, durable-export/release tests and
  a reviewed frozen candidate registry. Document that real A100 execution and
  Colab rendering still need validation.

**Current status:** the separate notebook, frozen registry, CUDA kernels,
numerical gates, orchestration and release safeguards are implemented locally.
The new translation unit compiles for `sm_80`; native CPU build/load and
all 318 declared operator fixtures pass against the canonical oracle on CPU.
CPU arithmetic/mocked notebook/pipeline
tests exercise budgets, rejection, archive verification and release ordering.
The full local regression run passed 1,061 tests with 18 platform skips; the
subsequent targeted tests also cover JSON-stable resume controls and isolated
per-run work/logit paths. The notebook validates with nbformat, executes its
cells under explicit no-network/no-GPU mocks, and exports to HTML. The CUDA
translation unit compiles with CUDA 12.8.1 for A100 (`sm_80`) with no reported
register spills; the complete patch applies to the clean pinned base and all
29 patched file hashes reconcile. Original notebooks were not regenerated.
These are **not GPU results**. A100 execution, cuBLAS graph-capture behavior,
performance and real Colab Run-all/rendering still need this explicit run.
Use the separate overnight notebook only after the implementation is published.
