# First native CUDA throughput pilot — 11 September 2026

Original user bundle: `pilot1-reports-1789113892245467172`, producer
`1bc31d43acf62f669fe7e5d8af8a766e94698232`, A100-SXM4-40GB.
The byte-preserving reports are in `original-reports.tar.gz`; no weights or
native binaries are included. `audit.py` reproduces `audit.json` against the
extracted original reports and the historical Git producer objects.

## Outcome

13 stages passed. The last performance stage timed out; **the workflow did
not fully pass**. No correctness gate failed. The native runtime reproduced
the unchanged W5/scale8 backbone + W6 vocabulary checkpoint on bounded probes:
16/16 top-1 positions, 4/4 short generation probes, KL 8.6674e-6. This is
quantized-checkpoint parity, not agreement with the full-precision source.

| Input tokens | Measured reps | Prefill tok/s | Decode tok/s | Sampled peak MiB | Status |
|---|---:|---:|---:|---:|---|
| 128 | 3 | 36.38 | 19.86 | 3450 | passed |
| 512 | 3 | 36.47 | 19.84 | 3972 | passed |
| 2048 | 1 | 36.45 | 19.73 | 5542 | stopped; partial |

One warmup is excluded. The 2048 measurements are preliminary, not a completed
three-repetition result. A repetition needed approximately 56.2 seconds of
prefill + 1.62 seconds of decode. Four repetitions already need about 231
seconds, excluding loading, hashes and report writes; the 180-second cap was
too small. Increasing the cap is a scheduling correction, not a kernel fix.

These are synchronous private-bridge rates, including logit copying, finite
checks and Python argmax; not pure GPU kernel throughput. VRAM is sampled at
0.5-second intervals and may miss transient peaks. The near-flat prefill rate
identifies a performance problem but does not attribute it to a particular
operator without profiling. No Unsloth speed result was collected in this run.

## Size, cost and persistence

- Text GGUF: 2,768,298,560 bytes; loaded for text inference.
- Auxiliary sidecar: 667,061,152 bytes; not loaded by this text bridge.
- Complete payload: 3,435,359,712 bytes. Payload is not total VRAM.
- Active workflow time: 39.08 minutes; native build/load: 27.14 minutes.
- Runtime and export logs record `CACHE PERSISTED`. This was a cold miss;
  restoration on a later Colab VM is not demonstrated by these reports.

Audit checks 15 included artifact identities, 34 producer-source hashes,
stage-time totals, evidence bindings and timing aggregates from step rows.
38 referenced external artifacts are absent, so their bytes and GPU execution
cannot be independently revalidated from this reports-only bundle.

## Decision

Keep the recipe and parity thresholds unchanged. Run the
[native optimisation study](../../../docs/native_gpu_optimization.md): corrected
context budgets, explicit targeting, separate operator profiles, an opt-in
four-token weight-reuse candidate, and same-bridge pinned BF16/UD-Q4 controls.
No default-kernel promotion or quality claim follows from this pilot.
