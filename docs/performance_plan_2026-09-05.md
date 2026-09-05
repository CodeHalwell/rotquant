# RotQuant: review fixes and performance plan — 2026-09-05

## Immediate objective

Improve **held-out model fidelity at a fixed complete artifact size**, while
making runtime correctness, resident memory, and throughput separately testable.
These are different objectives: better KL does not establish faster inference,
and a smaller packed checkpoint does not establish lower peak GPU memory when
the quality harness caches dequantized weights.

The [September project review](project_review_2026-09-04.md) remains a record
of the defects found before these changes. Its nine findings are addressed as
follows; this is not a claim that every possible project defect has been eliminated.

| Review | Change | Regression coverage |
|---|---|---|
| R1, R9 | Invocation-scoped shared-rotation cache; recompute for unversioned inference tensors; release on success and exceptions | Tiny Llama forward/generate, weak-reference lifetime checks, ordinary-input mutation test |
| R2 | Locked notebook dependency, clean-install collection, Python 3.13 CI entry | Fresh dev+eval environment and generated notebook tests |
| R3 | One quality-guard definition used at screening and on every confirmation seed; finite metrics and reliable paired samples required | Later-seed regressions, missing/NaN/infinite metrics, unreliable/reversed intervals |
| R4 | Separate estimated bytes from measured export bytes; bind seed-0 export to revision, trial, allocation and manifest digest | Estimate-only, missing, stale and wrong-seed exports cannot promote |
| R5 | Fail before scoring/cache lookup for dynamic shared rotations, A8 or rotation training, which the isolated scorer cannot reproduce | Unsupported settings rejected even with a warm cache; six-format scored/deployed equality preflight |
| R6 | Only equal-byte states dominate under a two-sided target; bounded integer repair when bucket search misses the interval | Original counterexample and small brute-force comparisons |
| R7 | Exclude PyTorch attention/transformer parents that directly read child weights; report exclusions and discovery-only status | MHA, encoder and decoder fast paths with surrounding safe projections |
| R8 | Stage complete checkpoint generations; publish by directory rename with a recoverable previous generation; hash all files and match the result's manifest digest | Injected packed-write and publication failures, corruption, stale generation, interrupted rename, unrelated-file protection |

## Next experiment: format-aware allocation on Qwen3.5-4B

Run [allocator-v4](../notebooks/qwen35_4b_allocator_v4_colab.ipynb) after these
changes are published to the notebook's selected immutable revision. Keep the
registered six-format palette and exact-size controls. Do not expand the sweep
to additional model families, all 1–8-bit formats, or recovery training yet.

1. **Preflight on the actual Colab GPU.** The new synthetic check compares
   packed values and outputs through the scorer and patcher for all six formats,
   then checks shared-rotation inference/generation. It downloads no model.
   A failure stops before the expensive Qwen calibration. Its timings are smoke
   diagnostics, not a throughput benchmark.
2. **Screen on seed 0.** Compare the re-run bits-only allocator with format-aware
   allocation, a same-palette random control, and uniform W4. The corrected
   solver may change allocations; do not reuse old final recipes as if produced
   by the new solver. Unchanged raw candidate scores may still be reused.
3. **Confirm at seeds 0/1/2.** Require primary KL wins and reliable paired KL
   intervals against both allocator controls, all registered no-regression
   guards on every seed, and a consistent secondary win. Report KL tails,
   top-1, diverse-domain fidelity, trajectories, and both perplexity datasets.
4. **Measure the exported artifact.** Seed 0 is the registered required export;
   do not imply all three seeds were measured. Serialization, tokenizer and
   configuration overhead count. Disabling export permits exploratory reports
   but blocks promotion. Old summaries without export identities cannot pass
   the new gate solely on an approximate size.

Why this direction: the archived v3 Pareto arm beat its weak broad-random
control, but still had about **2.62× the prompt-matched Unsloth anchor's mean KL**.
Uniform W4 had better fidelity but a larger persistent footprint, so that is
not an equal-byte win. Forced W6/W8 islands worsened the fixed-budget result:
extra bytes spent on selected layers required harmful downgrades elsewhere.
The observed winner used W3/W4/W5. Those results justify testing better codebooks
and finer groups before assuming a larger training budget is necessary.
See the [archived v3 record](../research/results/qwen35_4b_allocator_v3_c9efd2d56774.json).

## Decision after v4

- **If format allocation wins:** freeze the full per-layer recipe and artifact,
  then enter the registered engine-neutral 300-prompt/32-token comparison with
  disjoint calibration and task/loop/tool-failure checks. Do not turn a small
  development-suite improvement into a Dynamic 3.0 product claim.
- **If the additive allocator still misses:** first measure interactions between
  a small set of proposed layer exchanges on a separate calibration-validation
  split. A marginal single-layer KL table is not an exact whole-model objective.
  Accept exchanges only after replaying the combined deployed recipe; account
  for validation reuse rather than tuning on the final held-out benchmark.
- **If a residual quality gap remains:** run a separately registered frozen-code
  low-rank/distillation recovery experiment, with rank/storage/latency counted,
  an unrecovered control, a genuine token budget and disjoint validation.
  Train recovery parameters first to isolate the effect; training original
  weights or rotations changes the quantizer/recipe and needs another controlled
  ablation plus re-export. Do not label the current PTQ allocator QRAT.

Only move the frozen method to 27B after the 4B evidence identifies a repeatable
improvement. Larger models may respond differently; extra parameters alone are
not a guarantee that this recipe will quantize better.

## Runtime track and remaining limits

Start with one measured W4 packed-linear backend, separating decode GEMV from
prefill GEMM, before pursuing several engine forks. Require conformance against
the Python/native format, no persistent dense fallback, peak resident memory,
and matched-shape latency/throughput measurements on the intended GPU. Add W3/W5
only if the quality winner actually needs them. This keeps engineering effort
tied to the winning recipe. vLLM integration and wider model coverage follow
the operator acceptance gate already described in the roadmap.

The shared activation cache no longer retains whole-prefill tensors after a
site returns. Ordinary no-grad inputs still reuse their rotation within a site;
inference tensors intentionally recompute because they lack mutation counters.
This is a safety/memory fix, not a claim of fused-kernel speedup. Dynamic scoring
with shared rotations, activation quantization or rotation training remains an
explicitly unsupported combination until the scorer models it faithfully.

Checkpoint overwrite temporarily needs space for old and new artifacts. A
normal failure restores the previous directory; an abrupt interruption between
renames leaves `.NAME.rotquant-previous`, which the Python loader can recover.
Use a new export path if a recovery copy remains, and retain it until verified.
Unknown/legacy directories and unrecorded user files are never erased by
transactional overwrite. Legacy artifacts remain loadable, but hashless artifacts
do not satisfy new experiment-resume integrity checks. Filesystem/power-loss
durability and concurrent writers are not guaranteed; do not run two writers
against the same artifact directory.

Verification on the final implementation: a fresh `uv sync --locked --extra dev
--extra eval` environment passed **592 tests**, with 17 unavailable AVX2 tests
skipped and two SWIG deprecation warnings. Ruff and `git diff --check` passed;
an offline locked sync reports no dependency changes. The generated v4 notebook
matches its builder, validates as nbformat v4, compiles, and passes its plan
dry-run and six-format synthetic CPU preflight.

Local verification is CPU/macOS. CUDA, Colab Drive behaviour, full-Qwen quality,
GPU memory/throughput, remote CI and public provider competitiveness still require
their own runs. Notebook schema, generated source, plan dry-run and synthetic
CPU preflight can be checked locally; the full Colab notebook cannot be executed
end-to-end here because no CUDA/Colab runtime is attached.
