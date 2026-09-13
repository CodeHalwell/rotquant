# Native GPU follow-up: focused decode4 bottleneck profiling

## Run next

Use [qwen35_4b_native_followup_colab.ipynb](../notebooks/qwen35_4b_native_followup_colab.ipynb)
on an **A100 40GB**, verify `SOURCE_ROOT`, then **Run all**. The notebook has
persistent logs, per-phase caps, progress for every repetition, a results cell
that also works after a stop, and a reports download cell. Use the new default
`RUN_NAME = "followup4-profile"`, not an edit to a completed run. Original checkpoints
and past reports are never edited. There is no calibration or training.

The [completed study evidence](../research/results/native_study_2026_09_11/README.md)
supports tiled4's 3.30–3.35× prefill gain, not a decode gain. BF16/Unsloth speed
comparisons were subsequently completed in [followup2](../research/results/native_followup_2026_09_13/README.md).
The [followup3 evidence](../research/results/native_decode4_2026_09_13/README.md)
now confirms all 20 gates, ~60% faster decode and the retained prefill gain.
The next run profiles the validated candidate at **128 tokens only**, alongside
fresh reference/decode4 timing controls. It repeats neither conventional
downloads nor 512/2048-token timings by default. `RUN_PROFILE = True` produces
two separate diagnostic reports after correctness and throughput checks.

Profiles measure custom rotation, backbone matrix, vocabulary head and embedding
CUDA-event time. They do not separately attribute non-custom attention/SSM,
host transfers, CPU work or DRAM traffic. Profile mode disables CUDA graphs and
synchronizes events; its wall throughput must not enter speed comparisons.
Use absolute event milliseconds by phase to locate the next optimisation target,
not old reference-kernel percentages. There is no new task-accuracy test.

## Validated repair: cached-library diagnostic binding

The uploaded followup2 bundle confirms exact private/public agreement on both
CPU and CUDA for tiny BF16/Q4_0 models, reference retained-model parity, and all
four baseline timings. It stopped only at the decode4 dispatch-counter gate,
after numerical cases passed. Its cached runtime restored in 17.9 seconds.

The cache intentionally persists SONAME aliases as separate regular files.
The old diagnostic helper opened `libggml-cuda.so` directly, whereas execution
could use its versioned dependency. A real Linux CPU-only ELF regression now
reproduces this state split: old alias counter zero, execution counter nonzero.
The repair resolves diagnostic symbols through the execution-library handle,
verifies both symbols' actual provider, and records its path and hash. It
does not modify the native kernel, relax the dispatch guard, or clear caches.

An early tiny dispatch probe checks **new counter increments** for reference,
one-token decode and four-token prefill, as well as exact reference outputs,
immediately after binding load. Wrong/missing dispatch stops before export or
timing. Full operator checks repeat that probe; failure receipts now preserve
binding/counter evidence instead of ending without a report.

Default `BASELINES = ()` avoids repeating the completed ~11.34 GB downloads
and four timings. Fresh reference and decode4 measurements still run in pairs.
Historical baselines remain archived, not imported as same-session results.
To explicitly repeat baselines, select `("bf16", "ud_q4")`; this also restores
the conventional preflight. Followup3 verified the repaired dispatch and bounded
CUDA parity/timing. The Linux regression remains a loader test, not GPU evidence.

## What changed

### September 13: distinguish bridge correctness from backend drift

The uploaded `2c4037e76f71/followup1` log shows a successful native build and
ordinary BF16 GPU preflight, followed by an ordinary Q4_0 numerical failure:

| Tiny fixture | CPU/CUDA max abs | Mean abs | Mean KL | Top-1 | Short greedy traces |
|---|---:|---:|---:|---:|---|
| BF16 | 0.005206 | 0.0009484 | 7.155e-7 | 100% | Exact |
| Q4_0 | 0.053020 | 0.0094214 | 7.080e-5 | 100% | Exact |

Both used **zero RQ3 custom operators**. These are random tiny-model probes,
not 4B quality measurements. The failed run did not reach the retained model,
matched baselines or decode4 execution. It is not evidence of a RotQuant
regression, but matching tokens alone does not prove the drift harmless.
Pinned llama.cpp uses different CPU/CUDA quantized dot-product paths; that is
a plausible source of drift, not a demonstrated diagnosis from this log alone.

The replacement preflight compiles a separate **public llama C-API caller**.
It uses explicit positions and sequence IDs rather than the private bridge's
batch helper. Each caller performs the same four prompts (1/4/17/64 tokens)
and eight cached greedy steps on CPU and CUDA independently:

- Private/public agreement **on each backend** is required: max absolute error
  ≤1e-6, mean absolute error ≤1e-7, KL ≤1e-10, top-1 100%, exact greedy traces.
- CPU/GPU top-1 100% and exact greedy traces remain required for both formats.
- BF16 retains its cross-backend numerical limits: 0.02 max absolute error,
  0.002 mean absolute error and 1e-5 KL.
- Q4_0 cross-backend numerics retain those original thresholds and their own
  pass/fail flags, but are now **diagnostics**, not bridge-equivalence gates.
  This is an explicit change of the conventional preflight contract, not a
  wider numerical tolerance or a relabeling of the old failed experiment.
- Missing/mismatched controls, changed binaries and CPU compute fallback stop
  the run. Raw logits/traces, hashes and per-prompt metrics enter the reports ZIP.
  RotQuant operator, synthetic-model and retained-model gates are unchanged.

**Boundary:** the independent caller links the same verified llama/ggml
libraries. It isolates API/bridge disagreement; it does not independently
validate bugs shared by both callers, upstream kernel correctness, Unsloth
quality, or the unexecuted decode4 candidate. No new GPU success is claimed.

### Existing runtime follow-up

- The native bridge explicitly assigns ordinary `token_embd.weight` to the
  selected device's buffer. The override lives with the model. CPU mode is
  unchanged, and `ROTQUANT_REQUIRE_GPU` is not weakened. All-layer offload alone
  does not guarantee GPU input-embedding placement in this pinned llama.cpp.
- Tiny ordinary BF16 and Q4_0 Qwen graphs test GPU prefill and cached decode
  through both API callers before retained export/timing or 4B downloads. They
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
  execution. Followup3 subsequently established the measured short-context
  speed improvement; other shapes/devices still require their own evidence.

## Bounded controls

| Control | Default | Alternative |
|---|---|---|
| `ARMS` | `("b5_v6_s0",)` | Keep the tested recipe for this follow-up |
| `RUN_NAME` | `"followup4-profile"` | A new name for changed controls |
| `CONTEXTS` | `(128,)` | Explicitly add longer contexts with a new run name |
| `CANDIDATE` | `"decode4"` | `"none"`: fresh reference + baselines only |
| `BASELINES` | `()` | Opt in to BF16/UD-Q4 reruns; empty only with a candidate |
| `RUN_PROFILE` | `True` | `False` skips this run's profiling question |
| Measurements | 3 repetitions + 1 warmup | 32 cached decode steps each |
| Active allowance | 60 minutes | Maximum, not expected duration; includes builds/checks |

Baseline-only mode deliberately still runs fresh reference timing and all
applicable correctness checks. It skips candidate gates/timings and profiles.
It does **not** import old timings from an incompatible native binary. The CLI
equivalent is `--performance-study --study-candidate none --skip-profile`.
The focused profile uses `--study-candidate decode4` without `--skip-profile`.
Both reference and candidate profiles run at the smallest selected context.
With default controls, each uninstrumented process has a four-minute cap and
each diagnostic process an eight-minute cap, inside the overall allowance.
Selected phases have their own context-aware caps; failures preserve partial
measurements and prevent ratios from incomplete or instrumented reports.

This preflight repair changes **no native kernels, bridge code or runtime-cache
inputs** relative to `2c4037e76f71`. That failed run already saved its successful
CUDA build in the private Drive cache. A matching GPU/toolchain can restore it;
the diagnostic repair requires no new native compilation. The small public-API
caller is compiled only when conventional baselines are requested. A cache miss or incompatible
older kernel still requires a cold build (previously ~27 minutes). Restoration
does not skip fresh conformance gates. The private Drive export cache may be
conservatively invalidated, but exports reuse the original weights without
quantization. Optional BF16/UD-Q4 downloads total about 11.34 GB on local VM disk;
the focused defaults skip them.

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
The new public-API executable also compiles and loads locally; both tiny
formats have **zero private/public CPU logit difference**, zero KL and exact
traces. CPU-only reports explicitly mark cross-backend checks unexecuted.
Tests cover divergent callers, changed identities, missing controls, retained
BF16 limits, Q4 diagnostics, malformed probes and notebook orchestration.
The CPU backend does not execute the candidate CUDA code. A local Metal build
could not complete because the Metal compiler/toolchain is unavailable.
The uploaded followup2 run demonstrated the public/private CUDA checks and
reference retained-model parity. The repair's real Linux loader test runs
without CUDA: `python tests/check_cuda_diagnostic_loader.py --work-dir /tmp/rq3-loader-test`
(choose a new empty directory). It reproduces old-alias zero counters and
repaired execution counters after an actual cache round trip.
Followup3 then passed repaired dispatch, candidate CUDA model parity and speed.
This notebook configuration has local top-to-bottom mocked execution coverage,
not a new GPU profile result. The focused local suite passed 100 tests; the
saved notebook matches its generator and contains no fabricated outputs.
An HTML preview was generated, but browser policy blocked local visual inspection.
Open it on A100 40GB, verify `SOURCE_ROOT`, keep
`followup4-profile` defaults, choose **Runtime → Run all**, then inspect Results.
Expect fresh reference/decode4 throughput plus two completed diagnostic groups:
`profile-reference-b5_v6_s0-ctx128` and `profile-decode4-b5_v6_s0-ctx128`.
Missing/stopped profiles leave this question unanswered even if timing passed.
Download reports and disconnect/delete the runtime. No GPU session was
provisioned or billed by this development task.
