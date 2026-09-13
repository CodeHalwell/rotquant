# Decode4 bottleneck profile — 13 September 2026

All **20 stages passed** in **11.5882 active minutes**. Fresh uninstrumented
timings reproduce the decode4 gain; separate instrumented measurements identify
backbone matrix operations as the largest remaining custom-kernel cost, followed
by the vocabulary head. This evidence record changes no executable code,
notebook defaults, weights or recipes and does not promote a new kernel.

Producer: `53b8dac6cecf316c22070b2931fee800155072f8`.
Uploaded bundle: `followup4-profile-reports-1789320861931053283`.
`original-reports.tar.gz` preserves all 61 supplied files byte-for-byte
(2,539,866 uncompressed bytes). Archive: 343,972 bytes; SHA-256
`866e0d42ba68bfc532fec7a89c522aee8f178e9fba26790f71944c3de28a9efb`.
The upload contains logs/reports, not model tensors, binaries or raw parity logits.

## Fresh paired throughput

A100-SXM4-40GB; retained `b5_v6_s0` W5 backbone / scale8 / W6 vocabulary;
128-token prompt, 32 cached decode steps, three measured repetitions and one
excluded warmup; batch one and FP16 KV cache.

| Kernel | Prefill tok/s | Decode tok/s | Sampled peak MiB |
|---|---:|---:|---:|
| Reference | 36.34894 | 20.03992 | 3450 |
| Decode4 | 120.10121 | 32.16720 | 3450 |

Prefill is **3.30412×** faster; decode throughput is **1.60516×**, or **60.52%**
higher. Candidate decode repetition range: 32.10338–32.22515 tok/s; this is an
observed range, not a confidence interval. Rates are total measured tokens
divided by total measured time.

These paired runs use matching runtime hashes, export, prompt IDs and fixed
decode-token replay. Event profiling is off and CUDA graphs are not explicitly
disabled. The synchronous private bridge includes logit checks, full-vocabulary
host copies and Python argmax; logging/persistence are outside timing. These are
not production-server or pure-kernel rates. Memory is sampled every 0.5 seconds,
so equal samples do not establish equal transient peaks or DRAM traffic.

The previous [followup3](../native_decode4_2026_09_13/README.md) measured the same
gain at 128 and 512 tokens. [Followup2's conventional baselines](../native_followup_2026_09_13/README.md)
remain historical measurements: this run does not repeat BF16 or Unsloth.

## Separate diagnostic profiles

The following are mean CUDA-event milliseconds per measured repetition, with
one prefill and 32 decode steps. Warmup is excluded. Event profiling is enabled,
CUDA graphs are disabled and the diagnostic lookup binds the actual execution
dependency, `libggml-cuda.so.0`. Timing values are nonzero and cumulative event
receipts reconcile with the per-repetition records.

| Phase | Custom operator | Reference ms | Decode4 ms | Decode4 share of custom event time |
|---|---|---:|---:|---:|
| Prefill | Backbone matrix | 3494.110 | 1038.982 | 98.73% |
| Prefill | Vocabulary head | 10.476 | 10.476 | 1.00% |
| Prefill | Rotation | 2.879 | 2.865 | 0.27% |
| Prefill | Embedding | 0.019 | 0.019 | <0.01% |
| Decode | Backbone matrix | 1123.644 | 522.853 | 57.80% |
| Decode | Vocabulary head | 335.193 | 335.136 | 37.05% |
| Decode | Rotation | 45.544 | 46.192 | 5.11% |
| Decode | Embedding | 0.396 | 0.399 | 0.04% |

Backbone decode event time fell by about 53.5%. The head is essentially unchanged
at 10.473 instrumented ms per decode step: its share increased because matrix
time decreased, not because the head became slower. Backbone work still
dominates prefill even more strongly.

**These percentages are shares of the four measured custom operators, not shares
of whole-model wall time.** Event synchronization and disabled graphs change
execution. Do not compare profiled rates with uninstrumented rates as a
regression, subtract event sums from ordinary wall time, infer DRAM bandwidth,
or derive a production speed ceiling from these shares. Standard attention,
other graph operations and host work are not included in the denominator.

## Correctness and provenance checks

All dispatch, CPU/CUDA operator, tiny W6/W8 whole-model, conversion and retained
model gates pass their existing thresholds. Candidate operators include 54
single-token cases with exact native-reference outputs. Both retained reports
have identical aggregate parity: max absolute error 0.0322265625, mean absolute
error 0.00283652, KL 8.6674e-6, 16/16 argmax agreements and 4/4 exact short
generation probes, with CPU compute fallback forbidden.

Retained parity compares with the saved **quantized checkpoint**, not a
full-precision teacher. Identical aggregate summaries do not prove bitwise
full-model equality; raw logits are absent. There is no new task accuracy,
99% teacher alignment, multimodal quality or 27B evidence.

The offline audit verifies 22 included artifact references and 42 producer-source
hashes, independently recomputes rates, ratios, elapsed totals and profile
shares, and rechecks recorded numerical limits. It counts 48 distinct referenced
external artifacts as absent, not verified. Archive members were compared
byte-for-byte with the upload. Hash agreement is internal receipt consistency,
not an independent GPU replay or verification of missing model tensors.

From the repository root, after extracting the archive into an empty directory:

```sh
node research/results/native_decode4_profile_2026_09_13/audit.mjs EXTRACTED_REPORT_DIRECTORY
```

This reproduces `audit.json`; an optional third argument writes a new file and
refuses overwrite. The repository must contain the producer commit.

## Next optimization decision

1. **Backbone first.** Investigate a W5/scale8-specialized packed load/unpack path
   and better weight/activation reuse. The current prefill kernel reuses each
   weight across four tokens; larger tiles are a testable candidate, not an
   established improvement. Preserve reduction order where required and avoid
   full dense-weight materialization. This profile does not establish whether
   loads, bit unpacking, synchronization or occupancy is the limiting resource.
2. **Vocabulary head second.** It now consumes about 37% of measured custom
   decode time. Optimize its reconstruction/reduction path while preserving
   `dense_equivalent_fp16` reconstruction and FP16 rounding semantics. An
   algebraic change of basis is not automatically numerically equivalent.
3. **Keep rotation work lower priority for runtime speed.** It is about 5% of
   measured custom decode time; embeddings are negligible. Learned-rotation
   quality research is a separate experiment, not the immediate runtime fix.
4. For each candidate, keep fresh dispatch/operator/whole-model/retained parity
   gates unchanged, then run paired uninstrumented timings. Confirm 512-token
   context and retained W5/W8 after a short-context win. Repeat conventional
   controls before reporting a new same-run competitive result.

The current run answers the bottleneck-ranking question. Another unchanged
profile or quantization sweep is not needed to obtain that same answer. No new
paid GPU run, kernel implementation or GitHub push was performed for this audit.
