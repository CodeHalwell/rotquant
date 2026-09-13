# Decode4 CUDA validation and speedup — 13 September 2026

**Ready within the reviewed scope:** the uploaded followup3 run passed all 20
stages in 10.6219 active minutes. The same retained W5 backbone / W6 vocabulary
checkpoint now decodes about 60% faster than the reference kernel, with the
existing tiled prefill gain preserved. No weights, recipes or defaults are
promoted by this evidence record.

Producer: `445a436e9f0c4ddc4f8ba9c2e7955e38dd837b06`.
Uploaded bundle: `followup3-reports-1789314983411251883`.
`original-reports.tar.gz` preserves all 61 supplied files byte-for-byte.
Archive: 346,877 bytes; SHA-256
`a4dccb8d8f349cdad6e92df930e8d01334c6c105b3874e983282f3c4fa4f6d45`.
The upload contains logs/reports, not binaries, tensors or raw parity logits.

## Matched A100 40GB measurements

| Prompt tokens | Kernel | Prefill tok/s | Decode tok/s | Sampled peak MiB |
|---|---|---:|---:|---:|
|128|reference|36.3307|19.9383|3450|
|128|decode4|119.9981|31.9156|3450|
|512|reference|36.4276|19.9051|3972|
|512|decode4|121.8154|31.8861|3972|

- Prefill speed ratio: **3.30294× / 3.34404×** at 128 / 512 tokens.
- Decode speed ratio: **1.60072× / 1.60190×**, or +60.07% / +60.19% throughput.
- Three measured repetitions and one excluded warmup per cell; 32 cached decode
  steps each. Rates are total tokens divided by total measured time, not an
  average of rounded rates. Decode4 repetition ranges are 31.8729–31.9926 and
  31.8458–31.9347 tok/s; these are observed ranges, not confidence intervals.
- Same binary hashes, export, prompt IDs, replayed decode IDs, FP16 cache and
  non-kernel execution settings. CUDA event profiling is off; graphs are not
  explicitly disabled. These are batch-one synchronous private-bridge timings,
  including finite-logit checking, host logit copies and Python argmax. Logging
  and report writes are excluded. Not production-server or pure-kernel rates.
- Memory is process allocation sampled every 0.5 seconds. Equal samples do not
  prove equal transient peaks or equal DRAM bandwidth. Text execution does not
  load the non-text sidecar; model payload remains 3,435,359,712 bytes, including
  the 667,061,152-byte sidecar and 2,768,298,560-byte text GGUF.

The previous [followup2](../native_followup_2026_09_13/README.md) conventional
measurements remain historical evidence, not new same-run Unsloth comparisons.
Followup3 intentionally ran neither BF16 nor Unsloth again.

## What the gates establish

The early four-call dispatch probe now binds the actual `libggml-cuda.so.0`
execution dependency and observes fresh reference/decode/tiled counter deltas.
The same gate passes again inside the full candidate operator check. This is
GPU confirmation of the cached-library diagnostic repair, beyond its earlier
CPU-only Linux loader regression.

Candidate operators pass 42 normal cases plus 54 single-token shape/format/tail
cases. The 54 decode cases record exact native-reference outputs, including
permutations and widths up to 11,008; their separately reported `max_abs` is
error against canonical Torch arithmetic, not against the native reference.
Tiny W6/W8 Qwen graphs and both retained-model paths pass their original guards,
with CPU compute fallback forbidden.

Reference and decode4 retained reports have identical aggregate parity values:
max absolute error 0.0322265625, mean absolute error 0.00283652, KL 8.6674e-6,
16/16 argmax agreements and 4/4 exact short generation probes. This compares
native execution with the saved **quantized** model, not the full-precision
teacher. Identical summaries are not proof of bitwise equivalence at every
model output; raw logits are absent from this upload. This run establishes no
new task accuracy, 99% teacher alignment, multimodal quality or 27B result.

Uninstrumented host dispatch counts demonstrate candidate path selection, not
total GPU invocations: graph replay need not increment host counters per token.
Operator milliseconds are zero because event timing was disabled. They cannot
be interpreted as zero operator cost or used for a bottleneck breakdown.

## Reproduce the offline audit

From the repository root, after extracting the archive into an empty directory:

```sh
node research/results/native_decode4_2026_09_13/audit.mjs EXTRACTED_REPORT_DIRECTORY
```

The output reproduces `audit.json`. An optional third argument writes a new
audit file, refusing overwrite. The audit independently recomputes timings,
ratios, elapsed totals and reported numerical guards; checks dispatch deltas;
and verifies 22 included artifact references and 42 producer-source hashes.
Native sources also match the older cached build's producer. The 48 distinct
referenced external artifacts missing from the upload are counted as absent,
not silently verified. Hash agreement establishes internal receipt consistency,
not an independent GPU rerun or proof of the missing model tensors.

## Next decision

1. Profile reference and decode4 at the same short context, separately from
   uninstrumented timings, using this retained artifact and cached build. Keep
   fresh paired timing controls and the existing parity/dispatch gates.
2. Use the new measurements to choose between backbone packed-matrix work,
   vocabulary-head work and bridge/transfer overhead. Earlier ~75% backbone /
   ~22% head shares were custom-event shares under the old kernel; they cannot
   be assumed to hold now. DRAM traffic needs separate profiling evidence.
3. After the next candidate passes, confirm longer context and retained W5/W8,
   then restore fresh conventional controls and resume the native task-quality
   gate when its cost is acceptable. No allocator/training sweep is implied.

This record changes no executable code, notebook defaults or retained artifacts.
