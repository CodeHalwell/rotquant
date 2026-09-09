# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Research
results and experiment decisions are recorded separately in
[`docs/experiment_log.md`](docs/experiment_log.md); this file tracks the
software.

## [Unreleased]

### Project review (2026-09-09)

- `docs/project_review_2026-09-09.md` records the state of the evidence after
  the fresh-quality archive, verifies the baseline at `85e1254` (Python CI
  green on 3.10–3.13, locked ruff clean, 729 tests passing, native Release
  build warning-free with both conformance suites passing, CPU smoke config
  end to end), confirms the 8 September library defect queue is unchanged,
  and orders the next steps. No code changed.
- Recorded here because commit `5a98b99` had no changelog entry: the
  public-task release gate (`scripts/run_qwen35_public_tasks.py`,
  `scripts/public_task_suite.py`, the generated
  `notebooks/qwen35_4b_public_tasks_colab.ipynb` and its tests) and
  `scripts/archive_fresh_quality.py`, which imports a fresh-quality Colab
  bundle byte-for-byte with a SHA-256 evidence index.

### Project deep dive (2026-09-08)

- Python CI on `main` had failed on every Python version since `a4ac776`: the
  reuse-compatibility test diffs the checkout against the reviewed producer
  revision, which a shallow `actions/checkout` clone cannot resolve. The tests
  job now fetches full history and sets `ROTQUANT_REQUIRE_GIT_HISTORY` so the
  guard fails rather than skips; a shallow developer clone skips with
  instructions.
- `scripts/build_qwen35_fresh_eval_notebook.py` gains the repository-root
  bootstrap the other builders have, so it runs as a script from any working
  directory; the generated notebook is unchanged.
- Fresh-quality summary rows carry token-weighted `source_nll`,
  `candidate_nll` and `nll_delta`; the Colab llama.cpp CUDA build timeout is
  3600 s (the recorded build took about 29 minutes against 1800 s).
- Native C ABI: out-of-range `rq_native_v2_kernel`/`rq_native_v2_status`
  integers from a foreign caller were undefined behaviour at the parameter
  load (UBSan-reported); a sentinel enumerator makes every int32 a defined,
  rejected value.
- README: CI runs Python 3.10–3.13; on Linux `uv sync` installs the locked
  CUDA torch build, not a CPU wheel; the next-run pointer now names the
  fresh-quality Colab. The fresh-quality runbook records the operational
  hazards found in review (runtime-bound reuse, the archive as part of the
  frozen protocol, the degenerate `condition` task family, descriptive
  task-domain intervals). `docs/project_deep_dive_2026-09-08.md` records the
  full review, including the library defects queued behind the reuse freeze.

### Packed reload precision and checkpoint-only recovery (2026-09-07)

- Preserve framework-reconstructed floating nonpersistent buffers at their
  original precision during packed load. In particular, Qwen RoPE frequencies
  remain FP32 for FP16/BF16 execution rather than being irreversibly rounded.
- Strengthen the tiny Qwen preflight with source-like low-precision parameter
  construction, FP32 rotary-buffer assertions and 64-token/eight-generation
  probes; add exact FP16/BF16 CPU reload regression tests.
- Add `--revalidate-from`, a dedicated recovery Colab, separate output/cache
  roots and original-preparation/current-validator provenance. Saved artifacts
  and probes are verified and reused without calibration, quantization or export.
  Print numerical failure details immediately; acceptance thresholds are unchanged.

### Packed vocabulary validation (2026-09-07)

- Focused W5/W6 and W5/W8 export/reload runner, generated Colab notebook,
  persistent direct-to-Drive logs, per-stage identities/checksums, source-cache
  resume, actual file-byte ledgers and fail-closed numerical/quality/size gates.
- Explicit tiled `dense_equivalent` vocabulary projection preserves the dense
  prototype's execution-dtype weight rounding without a resident dense head.
  Checkpoint v3 records this mode; old files retain rotated-head semantics.
- Offline tiny multimodal Qwen hybrid export/fresh-process reload preflight,
  live packed-ownership checks and probe safetensors. This is a reference runtime,
  not a fused kernel, provider win, or locally completed full-Qwen CUDA run.
- Nine original vocabulary-screen records archived unchanged with SHA-256
  provenance; results and next-run instructions added to the project docs.

### Vocabulary-budget experiment (2026-09-06)

- Nine-arm Qwen3.5-4B vocabulary/backbone screen, generated Colab notebook,
  immutable source/runtime/data identities, checksummed results, direct-to-Drive
  logs, phase status, and layer/chunk-level calibration resume. It reuses two
  quantized backbones per seed and never calibrates on a vocabulary student.
- Bounded Gaussian W6/W8 tied-vocabulary quantization with FP16 scales,
  exception-safe dense quality reconstruction, and experimental shared packed
  lookup/projection wrappers. Optional checkpoint v3 saves one vocabulary
  payload and restores its aliases; ordinary artifacts still write v2.
- FP32 source-weight rotations and pure-torch low-precision FWHT intermediates;
  fast normalized FWHT scales inside the kernel. Packed scale/offset/step
  metadata retains its values and dtype across execution-dtype conversion.
- Corrected uniform-codebook MSE bounds, a uniform absmax reference, opt-in
  scale-storage diagnostics, and candidate-cache invalidation for the numerical
  changes. Historical quality results are not relabelled as repaired results.
- Component-byte ledgers, pinned GGUF header/tensor audit output, exploratory
  paired contrasts and fail-closed screen selection. Dense/projected results
  cannot pass artifact promotion. Packed-vocabulary dynamic allocation is
  explicitly rejected until its conditioned scoring/budget protocol exists.
- Thirty original allocator-v4 JSON records/manifests archived unchanged with
  SHA-256 provenance. Tensor/tokenizer binaries are deliberately omitted.

### Results review (2026-09-06)

- `scripts/inspect_gguf_types.py` summarises a GGUF artifact's tensor types,
  per-layer recipe and nominal byte shares from its header alone (local file
  or HTTP range fetch), so provider artifacts can be decomposed before they are
  used as matched-size controls. `docs/results_review_2026-09-06.md` records
  the finding it was written for: the Unsloth comparison keeps RotQuant's tied
  vocabulary at fp16 while the provider stores it as Q6_K, forcing the RotQuant
  backbone 1.7 bits/weight below the provider's at the same total bytes.

### Project review fixes (2026-09-05)

- Shared rotations support inference tensors and no longer retain activation
  inputs/outputs after a projection site returns, including exceptional returns.
- Parent-aware adapter exclusions protect PyTorch attention/transformer fast
  paths that read child weights; support reports distinguish discovery from
  validated execution.
- Packed exports stage a new generation before publication, retain a recoverable
  prior directory during replacement, and record per-file hashes. Resume checks
  require the exact recorded manifest digest. Overwrite never removes unrelated
  user files or unverified legacy directories.
- Allocator confirmation reapplies all quality guards on every seed and rejects
  nonfinite/unreliable evidence. Estimated bytes cannot satisfy measured-export
  gates; export identity includes revision, seed, trial and allocation.
- Dynamic scoring rejects unsupported shared/A8/trained-rotation combinations
  before calibration/cache reuse. Exact-byte allocation no longer treats larger
  savings as unconditional dominance; missed intervals get bounded integer repair.
- Notebook dependencies are locked; CI includes Python 3.13. The v4 notebook
  runs a six-format scored/deployed correctness preflight before Qwen calibration.

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
- The Qwen3.5-4B exact-byte experiment: 2/3/4/5/6/8-bit model-specific
  allocation, a matched random-mixed control, fp16 deployed butterfly angles,
  persistent token caches, multi-suite/domain fidelity metrics, three-seed
  finalist selection, packed export, and same-input Unsloth comparison in a
  resumable Colab notebook.
- Dynamic allocator v2: model-adapter projection discovery, faithful
  MSE-search/GPTQ candidate scoring, activation-relative and marginal-logit
  distortion, robust score normalization, a bucketed multiple-choice Pareto
  solver with exact final byte checks, measured sensitive-layer protection,
  proxy-rank diagnostics, and a persistent partial candidate-score cache.
- A generated Qwen3.5-4B allocator-v2 Colab, registered seed-0 finalist
  selector, and compact content-addressed result record for the completed
  learned-sign/dynamic-mixed experiment.
- Dynamic allocator v3: complete exported-artifact byte targets, deterministic
  exact-byte random Pareto controls, allocation fingerprints, and bounded
  single/pair exchange refinement over the measured rate-distortion objective.
- A generated allocator-v3 Colab and fail-closed selection/confirmation tools
  for binding W6/W8 sensitivity islands, direct three-seed paired random
  comparisons, duplicate-recipe rejection, and compact result downloads.
- Format-aware dynamic allocation: named full-`QuantConfig` candidate palettes,
  same-width format identity across caches, controls, and refinement,
  allocation-only palette restrictions, per-format diagnostics, and compatible
  reuse of earlier bit-only score caches.
- A generated Qwen3.5-4B allocator-v4 Colab, registered six-format W3/W4/W5
  experiment, fail-closed finalist/confirmation tools, and a compact validated
  record of the completed allocator-v3 run.

### Changed

- The `eval` package is now `rotquant.eval`; installing the wheel no longer
  claims the global top-level module name `eval`.
- Library logging follows library convention: a `NullHandler`, no forced
  level, and per-module logger names under the `rotquant.` namespace.
- `transformers` and `torch` dependencies now carry upper bounds, because the
  KV-cache evaluation boundary relies on Transformers cache internals.
- `scripts/run_experiment.py`'s `run()` is decomposed into documented stage
  functions.
- Packed checkpoint loading preserves each butterfly rotation's declared
  storage dtype instead of coercing angle metadata to the model load dtype.

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
  window (previously every decode write stayed fp16 forever). A write longer
  than the window (chunked or speculative decode) is packed identically to
  the same rows written one token at a time; an earlier revision of this
  change packed such a write's leading rows twice, and a later one packed the
  whole aged slice as a single artifact, so 8-bit scale blocks and a
  calibrated codebook were fitted across whichever rows happened to age
  together and the cache depended on chunk size (both found by the Codex
  review). Aged positions are now packed one at a time, which is what a
  one-token decode already did.
- `train_rotation.select_butterfly_checkpoint_hessian` and
  `hessian_reconstruction_error`: the Hessian rotation objective is now gated
  against seeded FWHT under the exact deployed quantizer (including GPTQ), as
  the activation objective already was. Both take an `activation_mean`, which
  `patch_model` supplies whenever the layer deploys `mean`/`length_mean` bias
  correction: that correction folds the mean error into the output bias, so the
  deployed error is weighted by the centered second moment `H - mu^T mu` rather
  than by `H`, and scoring `H` ranks a component the deployment cancels. Found
  by the Codex review of the follow-up PR.
- `hessian_reconstruction_error(..., damp_frac=...)` and
  `patch_model(*, hessian_damp_frac=...)` (keyword-only, so `stats_out` keeps
  its sixth positional slot) remove the ridge that
  `HessianAccumulator.finalize` folds into `H`, which GPTQ wants for Cholesky
  stability but which scoring measures as an extra `lambda ||E||^2`.
  `CalibrationResult.damp_frac` records what was applied. The default 0.0
  matches `scripts/run_experiment.py`, which damps only inside the GPTQ solver,
  so no recorded run is affected; measured on synthetic layers the ridge shifts
  the score by at most 1.009x and reversed no decision in 120 trials, because
  it inflates numerator and denominator of the ratio together.
- Both rotation checkpoint gates score a shared-rotation site as the separate
  packed weights it deploys, rather than as the concatenation it trains on.
  `patch_model` packs each sibling projection on its own, so a concatenated
  score fits one calibrated codebook across all siblings and lets 8-bit scale
  blocks straddle their boundaries: measured 7.1e-2 and 6.9e-3 relative
  difference in the packed weight. `configs/qwen35_4b_w4a8_e8_trials_cuda.yaml`
  combines `share_rotations: true`, `scale_bits: 8` and the Hessian objective,
  so its bundled `optimized_w4` arm carried this confound. Found by the Codex
  review of the follow-up PR.
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

- Calibrated KV-cache scalar grids are now fitted once from each layer's
  prefill K/V distributions, persisted for the cache lifetime and reused by
  every decode and ageing write. The previous row-at-a-time ageing path fitted
  and discarded a new grid for every row, so the artifact had no stable
  decoder; deployed byte accounting now includes both fp32 centroid grids.
- fp16 scale storage now preserves positive subnormal magnitudes instead of
  clamping offsets and 16-bit scales to fp16's smallest *normal*. The old clamp
  amplified genuinely small layers (synthetic weight sigma 1e-6: 8-bit-scale
  NMSE >47); the 8-bit affine step is also rounded upward when nearest-fp16
  rounding would otherwise leave the block maximum unreachable.
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
  `non_kv_state_bytes` accounting sees the same tensors. Cloning, the
  shared-storage check and the byte accounting now share one traversal
  (`_iter_cache_tensors`), which also covers tensors held on the cache object
  rather than in a layer; those were cloned and checked but not counted, so a
  cache keeping state there reported a whole-cache ratio that was too
  favourable. A regression test
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
