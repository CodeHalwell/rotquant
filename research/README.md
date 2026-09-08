# Reproducible research records

Latest: [packed revalidation evidence](results/raw/qwen35_packed_revalidation_89d25f3/evidence_index.json)
preserves original JSON bytes/checksums, including the failed and repaired W6
records. See [the next fresh-quality protocol](../docs/fresh_quality_run_2026-09-08.md).

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
