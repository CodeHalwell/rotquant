# Native CUDA prefill optimisation study — 11 September 2026

Producer: `ed920464932a6eab09bda263fb547efa3e364b74`, original bundle
`study1-reports-1789125388926541216`. `original-reports.tar.gz` preserves the
reports and logs, not weights/binaries. SHA-256:
`f6a1004a996a4bbccc03c614c827c3063b2ca7154f4284e5b0137f4fb48a24f2`.

From the repository root, run `node research/results/native_study_2026_09_11/audit.mjs EXTRACTED_REPORT_DIRECTORY`
to independently reproduce `audit.json`. The audit checks 26 included artifact
hashes, 37 historical producer-source hashes, timing arithmetic and paired
evidence. 39 referenced external artifacts are absent; this is receipt/aggregate
validation, not replay of CUDA or raw model logits.

## Confirmed same-run result

| Prompt tokens | Reference prefill tok/s | Tiled4 prefill tok/s | Ratio | Reference decode tok/s | Tiled4 decode tok/s | Sampled peak MiB, both |
|---|---:|---:|---:|---:|---:|---:|
|128|36.332|119.995|3.303|19.864|19.934|3450|
|512|36.456|121.932|3.345|19.885|19.869|3972|
|2048|36.450|122.256|3.354|19.827|19.785|5542|

Every timing run completed three measured repetitions plus an excluded warmup,
with 32 cached decode steps. Paired runtime, model/export/probe hashes, prompt
IDs, cache/context settings and replayed decode IDs match. The 2048-token mean
prefill fell from 56.186 to 16.752 seconds. No sampled peak-memory increase was
observed; 0.5-second samples can miss transient peaks. This is one-session,
synchronous private-bridge throughput, not production serving performance.

The tiled candidate passed all 36 operator cases and tiny W6/W8 models.
Reference and tiled retained-parity aggregates are identical: max absolute
error 0.0322265625, mean error 0.002836524, mean KL 8.6674e-6, top-1 16/16 and
short generation 4/4. These compare saved **quantized-model** probes, not FP16
accuracy or a new task benchmark. The unchanged recipe is W5/scale8 backbone
plus W6 vocabulary. Complete payload remains 3,435,359,712 bytes, including the
667,061,152-byte auxiliary sidecar that is not loaded for text inference.

## Decode bottleneck

In the tiled4 diagnostic profile, per decoded token: backbone matrix operators
35.08 ms, vocabulary head 10.46 ms, rotations 1.48 ms, embeddings 0.015 ms.
Their shares of **custom operator event time** are approximately 74.6%, 22.2%,
3.1% and 0.03%. These are not shares of total model wall time. Event profiling
synchronizes each custom operator and disables CUDA graphs; its wall rates
must not enter the throughput comparison. Non-custom attention/SSM work,
transfers and host overhead remain unattributed.

Prefill matrix event time fell from 3493.96 to 1039.22 ms at 128 tokens, which
is consistent with the uninstrumented gain. The next runtime target is the
single-token backbone matrix path, then the vocabulary head, not rotations.

## Incomplete comparison and decision

24 stages passed; stage 25, BF16 at 128 tokens, failed before a complete warmup:
`model.input_embed (GET_ROWS) scheduled on CPU`. No BF16 timing or Unsloth
result was collected. All-layer offload had left ordinary input embeddings on
CPU; only the RQ3 input had a GPU placement exception. Keep the strict gate and
fix ordinary embedding placement. Active workflow time was 47.965 minutes;
build/load consumed 27.223 minutes.

The [targeted follow-up](../../../docs/native_gpu_followup.md) adds ordinary
BF16/Q4_0 preflight before 4B work, finishes matched baselines first, and tests
an opt-in decode candidate. Fresh reference timings are required after changing
the native binary. No default promotion, new compression claim or quality
claim follows from this study. Preserve this evidence and reuse original
weights rather than repeating quantization/calibration.
