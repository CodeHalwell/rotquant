# Contributing to RotQuant

## Setup

```bash
uv sync --extra dev            # core + pytest/ruff (CPU only)
uv sync --extra dev --extra eval   # adds transformers/datasets/lm-eval
```

## Before opening a pull request

1. **Run the test suite.** `uv run pytest tests/ -q` must pass. The
   cross-language native suite (`tests/test_native_cpp.py`) needs `cmake` on
   the PATH; install it rather than letting those tests skip.
2. **Lint.** `uv run ruff check .` must be clean.
3. **Native changes** additionally need the compiled conformance suite:

   ```bash
   cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
   cmake --build build/native --parallel
   ctest --test-dir build/native --output-on-failure
   ```

4. **Format or bitstream changes** require a new format version per
   [`docs/packed_format_v1.md`](docs/packed_format_v1.md) — never change bit
   order, word layout, or reconstruction semantics in place.

## Project conventions

- Fail closed. No silent fallbacks: an unsupported backend, bit width, or
  artifact must raise, not degrade quietly. If a degraded path is
  deliberate (e.g. the fp16 quality-only fallback), it must be reported in
  results.
- Equal-bits discipline: any quality comparison must hold true bits/weight
  fixed, including scales, residuals, and padding. `BitBudget.assert_matches`
  is there to enforce it.
- Experiment results, including negative ones, are recorded in
  [`docs/experiment_log.md`](docs/experiment_log.md) using its entry
  template. Raw result JSONs accompany every claim.
- Helpers shared across modules live in public or explicitly documented
  internal APIs — do not add new cross-module imports of `_`-prefixed names.
- Software changes are summarized in [`CHANGELOG.md`](CHANGELOG.md).

## License

By contributing you agree that your contributions are licensed under the MIT
License in [`LICENSE`](LICENSE).
