# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Research
results and experiment decisions are recorded separately in
[`docs/experiment_log.md`](docs/experiment_log.md); this file tracks the
software.

## [Unreleased]

### Added

- MIT `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.
- Python CI: pytest (including the cross-language native conformance suite)
  on Python 3.10/3.11/3.12 plus ruff lint, on every push and pull request.
- Native CI additions: an ASan/UBSan sanitizer job and a check that the
  pinned llama.cpp integration patch still applies.
- `rotquant.__version__` (single-sourced into package metadata) and a
  `py.typed` marker; result provenance now records the rotquant version.
- Smoke tests for `scripts/verify_rotquant_gguf.py`, `eval` throughput and
  zero-shot wrappers.
- A competitive-data pipeline with immutable source/license manifests,
  post-template token identities, exact and near-duplicate leakage checks,
  fixed 300-prompt domain quotas, measured multi-file artifact identities,
  structured run failures, and paired/domain bootstrap reports.

### Changed

- The `eval` package is now `rotquant.eval`; installing the wheel no longer
  claims the global top-level module name `eval`.
- Library logging follows library convention: a `NullHandler`, no forced
  level, and per-module logger names under the `rotquant.` namespace.
- `transformers` and `torch` dependencies now carry upper bounds, because the
  KV-cache evaluation boundary relies on Transformers cache internals.
- `scripts/run_experiment.py`'s `run()` is decomposed into documented stage
  functions.

### Added

- `rotquant.eval.kv_cache`: a mandatory validity endpoint
  (`KVCacheEvalConfig.endpoint_check_bits`, default 8, and
  `endpoint_max_kl`, default 0.01). A uniform Gaussian cache at that width is
  evaluated on the held-out calls before any candidate or allocator; a run
  whose endpoint KL exceeds the limit is rejected, because a floor that
  survives 8-bit codes is not quantization error. `evaluate_kv_cache` returns
  the report as `endpoint_check`; a run's result JSON therefore carries it at
  `metrics["kv_cache"]["endpoint_check"]`.
- Tiered cache simulation now tracks absolute positions: a decode write is no
  longer treated as its own sequence, sink rows are decided by absolute
  position, and rows are packed exactly once when they leave the recent
  window (previously every decode write stayed fp16 forever).
- `train_rotation.select_butterfly_checkpoint_hessian` and
  `hessian_reconstruction_error`: the Hessian rotation objective is now gated
  against seeded FWHT under the exact deployed quantizer (including GPTQ), as
  the activation objective already was.
- `ButterflyRotation.enable_sign_training(init_magnitude=...)` and
  `RotationTrainConfig.sign_init_magnitude` (default 0.1): the previous ±1
  logit initialisation could never cross zero under the shipped learning
  rates, so the learned-sign arm was inert.
- `rotquant._internal.rotate_hessian` and `encoded_storage_scales`.
- Tests: KV bit-monotonicity on a hybrid model, endpoint-check plumbing,
  tiered ageing, exact code/scale storage consistency, sign initialisation,
  and the Hessian gate.
- `docs/scientific_validity_review_2026-09-01.md` and
  `docs/change_report_2026-09-01.md` (every change on the review branch, its
  reason and evidence, what is withdrawn, and what future generations must
  implement).

### Changed

- Every Colab notebook now pins `transformers==5.9.0`; the unpinned
  `>=5.9,<6` range resolved to 5.16.x on 2026-08-29 and exposed the cache
  simulator defect below.
- `scripts/run_qwen35_next_stage.py` paired intervals carry
  `interval_reliable` (false below 20 paired samples; a percentile bootstrap
  of four prompts is not a 95 % interval).
- Publication manifest and paper: all cache-quality results are marked
  withdrawn; storage is reported like-for-like against the loaded source
  tensors (the source index includes a 241 MB MTP head that the model never
  loads), 58.26 % rather than 59.34 %.
- `docs/roadmap.md` reserves the name QRAT for a future
  quantization-and-rotation-aware training method.

### Fixed

- The 8-bit scale encoder divided by a divisor clamped to the smallest normal
  fp16 value while decoding multiplied by the true step. Blocks whose 256
  scales span less than 0.0156 have a subnormal step, so every scale in them
  was pulled toward the block minimum (measured −18 % mean and −35 % worst on
  down-projection-like scales, +73 % weight quantization error). The encoder
  now divides by the exact step; GPTQ reuses the retained scale codes instead
  of re-deriving them. Found by the Codex review of the follow-up PR.
- `scripts/audit_publication.py` derives the MTP head bytes and the loaded
  tensor total from the source safetensors shard headers and checks them
  against the manifest, and always checks the manifest's own MTP arithmetic.
- With 8-bit double-quantised scales, GPTQ's lazily refit group scales were
  encoded per group column while the stored scales were encoded row-major, so
  packed codes were assigned against values the artifact did not store
  (~1e-3 relative). Every path now encodes scales exactly once and retains
  that triple verbatim; GPTQ snaps refit scales onto the frozen grid.
- `rotquant.eval.kv_cache` cloned only tensor-valued cache attributes, so on
  Transformers releases that keep linear-attention conv/recurrent state in
  `dict` attributes (5.16.x) the simulated packed cache shared, and the two
  decode passes corrupted, that state. Every K/V code width then produced the
  same next-token KL (~0.5–0.9 on Qwen3.5-4B). The clone now covers tensors
  inside containers and fails closed when any storage remains shared;
  `non_kv_state_bytes` accounting sees the same tensors. A regression test
  (`tests/test_kv_cache_bit_monotone.py`) requires near-zero KL at 8 bits and
  monotone KL across bit widths on a hybrid model. See
  `docs/scientific_validity_review_2026-09-01.md` for the affected results.

- `set_seed` no longer sets `PYTHONHASHSEED` at runtime (a no-op that
  suggested determinism it could not provide).
- `git_sha` provenance now resolves the repository containing the package
  rather than the process working directory, and records a dirty-tree flag.
- The pure-torch FWHT fallback warns once when the CUDA
  `fast-hadamard-transform` kernel is unavailable on a CUDA device.
- `Quantizer` with `codebook="calibrated"` refuses to silently reuse a grid
  fitted to a different weight matrix.
- The block-calibration data manifest recorded a hardcoded sequence length
  that could disagree with the loader actually used.

## [0.1.0] - 2026-08-31

Initial development version: rotation + quantization core (`rotate`,
`codebooks`, `quantize`, `pack`, `linear`, `patch`, `calibrate`,
`train_rotation`, `block_train`), packed checkpoint v1, native runtime v2
(Python reference and portable C++17 implementation with NEON/AVX2 paths),
experimental llama.cpp GGUF integration, KV-cache quantization and selective
retrieval oracle, the E1-E9 experiment harness, and GPTQ/AWQ/AQLM baseline
wrappers.
