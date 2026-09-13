# W5 backbone experiment — 13 September 2026

Producer `e435080f74d49fef405249b1b2dd82fa12f05295`; supplied bundle
`backbone1-reports-1789332956558625649`. All **29 stages passed**, in **40.1228
active minutes**, including **26.9563 minutes** building the native runtime.

All 83 original supplied files are byte-preserved in `original-reports.tar.gz`
(without adding macOS resource-fork sidecars).
Archive SHA-256:
`f23a600af3c0dba6f4ac686aed5316b9cf6fa6ed6b25627796ee2ab4b69845e9`.
`audit.json` contains the independently recomputed receipt audit. The archive
contains reports/logs, not model tensors, executing libraries or raw parity logits.

## Fresh within-run model measurements

A100-SXM4-40GB, unchanged `b5_v6_s0`, 128 prompt tokens, 32 decode steps, batch
one, FP16 KV; three measured repetitions and one excluded warmup. Same runtime,
export, probes and replay IDs. No CUDA-event instrumentation in this table.

| Kernel | Prefill tok/s | Decode tok/s | Sampled peak MiB |
|---|---:|---:|---:|
| Decode4 | 119.4901 | 31.6630 | 3450 |
| W5/scale8 tile8 | 160.0811 | 33.6984 | 3450 |
| W5/scale8 tile16 | 152.0771 | 33.6757 | 3450 |

Tile8 is **33.97% faster prefill** and **6.43% faster decode** than decode4 in
this run. Tile16 is not faster than tile8 here. Rates include the synchronous
private bridge; they are not a production-server benchmark. Equal 0.5-second
memory samples do not establish equal transient peaks or memory bandwidth.
The earlier conventional BF16/Unsloth measurements were **not repeated here**.

The three synthetic candidates passed the declared screen; two proceeded to
model testing. Each W5 candidate's existing gates include 84 primary cases,
54 decode cases and 168 format/tail cases. Original numerical limits remain
unchanged. Retained aggregate parity: max absolute error 0.0322265625, mean
absolute error 0.00283652, KL 8.6674e-6, 16/16 argmax positions, 4/4 short exact
generation probes. This compares with the saved **quantized checkpoint**, not
the FP16 teacher. No new task accuracy, 99% teacher alignment or 27B evidence.

The audit verified 31 included artifact references and 28 producer source
hashes; 38 external references are unavailable in the upload, not independently
verified. It recomputes timing totals, ratios, shortlist decisions and gate flags.
Identical aggregate parity summaries do not prove bitwise full-model equality.

To reproduce the audit after extracting the archive into an empty directory:

```sh
node research/results/native_backbone_2026_09_13/audit.mjs EXTRACTED_DIRECTORY REPOSITORY_DIRECTORY
```

## Decision

Use tile8 as the next performance control, without changing the global default.
The next [overnight runtime experiment](../../../docs/native_overnight_experiment_plan.md)
tests bounded GEMM staging and the vocabulary head, then longer contexts and
fresh conventional controls. New kernels are **not** validated by this result.
