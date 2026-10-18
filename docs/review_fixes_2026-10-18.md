# Review fixes — 18 October 2026

Addresses the 13 findings from the review of `6d5f620`.
No retained checkpoint or raw experiment result was replaced.

| Review finding | Change | Verification |
|---|---|---|
| 1. Nested worker timeout | Freeze and enumerate descendants across process sessions, terminate them, escalate and wait before returning | Real nested subprocess timeout; surviving worker check |
| 2. Checkpoint precision | Rotation storage moves by device without dtype rounding; adapters and gradients retain their storage precision; dense/learned rotation storage dtype is saved | Exact FP16/BF16 tiny-Llama output and state round trips across butterfly/dense/Cayley; FP64 dense storage and gradient checks |
| 3. FP16 butterfly invariant | Compute trigonometry and stages at least in FP32; score final stored angles; block selection and deployment use the same stored candidate | FP16/BF16/FP32 invariant and inverse checks; selection observes committed FP16 candidate/reference; block-training regressions |
| 4. Missing GPTQ calibration | Reject missing Hessians before model mutation; refresh accepts a source-basis Hessian | Missing/partial calibration rejection; successful calibrated refresh; identity-Hessian and storage regressions |
| 5. Shared projections | Replace every alias with one packed module; record aliases for checkpoint reconstruction; reject incompatible filters and unsafe attention owners | Shared-object and save/reload equality; filter and attention-parent negative controls |
| 6. Root linear no-op | Reject root projections in the in-place API and direct callers to `QuantLinear.from_linear` | Rejection leaves no unused child module |
| 7. Uniform scale bounds | Resolve automatic bounds from the current codebook through `mse_search_bounds`; explicit overrides remain explicit | Direct replacement and dynamic-palette checks; replaced/fresh uniform configurations produce identical codes |
| 8. Budget allocation | Use a one-sided interval for `target_bpw`, including refinement; retain two-sided matched-artifact targets | Original 14-byte counterexample selects distortion 1; 20 small instances match exhaustive enumeration |
| 9. Non-finite storage | Reject non-finite inputs, scale-storage overflow and non-finite reconstructed weights | FP16 and affine-scale overflow rejection; small finite scales remain supported |
| 10. Score-cache identity | Include actual model bytes, dtype/device, implementation hashes, dependency versions, runtime/math settings and stage fingerprints; advance cache schema to v3 | Weight/dtype/stage changes invalidate identity; allocation-only settings do not; disk-cache resume regressions |
| 11. KV writes | Count decode calls/tokens per full-attention layer and require coverage on each step for candidates and endpoints | Deliberate bypass fails; positive coverage is retained for every batch; Qwen cache tests pass |
| 12. Withdrawn publication evidence | Default audit is ineligible and exits nonzero; current output excludes withdrawn derived values and renders withdrawn macros; explicit historical mode preserves arithmetic/status | Both CLI modes tested; default generated report rebuilt and blocked as expected |
| 13. Current native integration CI | Add pinned v2 CPU build/load/operators/W6-W8 graphs and an sm80 CUDA compile-only job | Clean v2 CPU build/load, 18 operator cases and both tiny graphs ran locally; workflow structure checked |

The FP16 butterfly's measured squared-norm gain changed from `1.00334431246`
to `1.00000017009`; maximum orthogonality error is `1.70e-7` in the same
FP64 matrix diagnostic. This is a numerical invariant check, not a new model
quality result.

## Validation

- **323 passed**, two existing SWIG warnings, in **49.90 seconds** across 20
  focused test modules. This includes the three new regression modules and the
  existing API, checkpoint, rotation/block training, allocator, quantiser, cache,
  publication, workflow, format and runner tests. History-dependent guards ran.
- A separate final score-cache identity check passed after runtime identity was
  expanded; Ruff and the new helper/test complexity checks passed at limit 15.
- A clean, pinned llama.cpp v2 CPU Release build compiled, linked and loaded its
  actual binding. All **18** CPU packed-operator cases passed.
- Synthetic W6 and W8 two-layer Qwen graphs each passed reset/replay and cached
  decode checks, executing **168 custom operations** per graph.
- All **three** standalone native C/C++ conformance suites passed.
- The default publication audit correctly exits 1 for withdrawn evidence;
  `arithmetic_passed=true`, `publication_eligible=false`, `passed=false`.

Local validation uses Apple Silicon, Python 3.12.15 and Torch 2.12.0. The local
SciPy package was brought to the lockfile's Python-3.12 version, 1.17.1, replacing
the old 1.15.3 wheel that prevented Transformers imports. No dependency spec or
lockfile change was needed.

CUDA compilation is configured in CI but was not run on this Mac. No GPU
numerical validation, retained-model quality evaluation, training or paid GPU
job was launched. These checks make no new speed or model-quality claim.

## Changed contracts

- GPTQ now requires explicit calibration; refresh Hessians use the source input
  basis and are rotated internally.
- Model `.half()`/`.bfloat16()` operations preserve rotation and adapter storage.
  Select angle storage explicitly through `PatchConfig.rotation_storage_dtype`.
- Unspecified MSE bounds remain `None` in configuration serialisation and resolve
  through `mse_search_bounds` after replacements.
- Aliased projections require compatible include/exclude rules and one quantiser
  configuration. Checkpoints retain their module aliases.
- Score-cache v2 entries are not adopted by the new v3 scoring path.
- Historical fresh-quality collection reuse is disabled under the changed core.
  Existing archives remain readable; new execution requires a fresh result root.
- Publication arithmetic for withdrawn results requires `--historical` and
  remains explicitly ineligible for publication.

Detailed local receipts and logs are at
`/tmp/rotquant-fixes-2026-10-18/`; the compact verification bundle is also saved
beside the original review under `Documents/Codex/reviews/rotquant-2026-10-18/`.
