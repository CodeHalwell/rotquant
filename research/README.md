# Reproducible research records

Latest: [completed fresh-quality evidence](results/raw/qwen35_fresh_quality_d4292d6fdec6/evidence_index.json)
preserves 1,080 prompt records across nine arms and all 1,125 SHA-256 record
pairs, byte-for-byte. Summary/oracle/paired-contrast reconciliation is repeatable
with `scripts/archive_fresh_quality.py`. See the
[results and limitations](../docs/fresh_quality_results_2026-09-09.md) and
[public-task protocol and sources](../docs/public_tasks_run_2026-09-09.md).
The subsequent public-task run was stopped for reference-runtime cost; only
partial progress is recorded in the [ledger](../docs/experiment_log.md), not a
completed archive. [Native serving preparation](../docs/native_runtime_v3.md)
now precedes another paid run. The fresh-quality archive above is unchanged.

Future reading: [hybrid-attention quantisation and learned-rotation follow-up](../docs/hybrid_attention_quantization_research_2026-09-09.md)
distinguishes external evidence from proposed 4B experiments and records the
quality, byte-budget and bandwidth constraints. It contains no new run results.

Comparison coverage: the [September 9 Unsloth inventory](unsloth_gguf_inventory_2026-09-09.json)
pins all 21 current 4B and 24 current 27B language-model quant variants, their
exact bytes and Hub-reported checksums. The
[full-frontier plan](../docs/unsloth_full_frontier_plan_2026-09-09.md) defines
the pending multi-artifact benchmark; the inventory contains no new scores.

The earlier [packed revalidation evidence](results/raw/qwen35_packed_revalidation_89d25f3/evidence_index.json)
still preserves both the failed and repaired W6 records. No old scores were replaced.

This directory contains compact, versioned records for completed external
experiments. Recoverable Colab bundles are preserved under `results/raw/` as
text-normalised JSON/CSV, while compact research records name the exact
Git/model revision, preserve the decision metrics, and record the SHA-256 of
every originally delivered raw file. Large model weights and generated
full-vocabulary reference-logit arrays remain in Drive rather than Git; their
revisions, hashes, and persistence locations are recorded by the experiment
protocols that create them.

The narrative chronology and negative results remain in
[`docs/experiment_log.md`](../docs/experiment_log.md). Scientific assumptions,
maths, and claim boundaries are in [`docs/how_rotquant_works.md`](../docs/how_rotquant_works.md),
[`docs/scientific_review_2026-08-31.md`](../docs/scientific_review_2026-08-31.md),
and [`docs/competitive_eval.md`](../docs/competitive_eval.md).
