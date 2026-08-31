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

### Changed

- The `eval` package is now `rotquant.eval`; installing the wheel no longer
  claims the global top-level module name `eval`.
- Library logging follows library convention: a `NullHandler`, no forced
  level, and per-module logger names under the `rotquant.` namespace.
- `transformers` and `torch` dependencies now carry upper bounds, because the
  KV-cache evaluation boundary relies on Transformers cache internals.
- `scripts/run_experiment.py`'s `run()` is decomposed into documented stage
  functions.

### Fixed

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
