# RotQuant project review — 2026-09-04

Follow-up, 2026-09-05: implementation fixes and regression tests for R1–R9 are
recorded in [the performance plan](performance_plan_2026-09-05.md). The findings
below describe the reviewed state before those fixes, not nine still-open bugs.

## Verdict and scope

RotQuant has a substantial, testable research foundation: structured rotations,
calibration-aware quantization, resumable experiments, packed artifacts, and a
standalone native CPU implementation. It is not yet a dependable general-purpose
optimization library or an established competitor to Unsloth's deployed artifacts.

This review found **nine actionable defects: four P1 and five P2**. The most urgent
are an ordinary inference-mode crash, missing CI dependencies, and two ways for the
new experiment assessor to approve evidence that does not satisfy its intended
quality/artifact gates. Existing passing tests do not cover these cases.

Reviewed state: branch `codex/dynamic-mixed-precision-experiment`, HEAD
`c9efd2d5677420d231d88ed80292f6fe0656a052`, including the existing uncommitted
allocator-v4 implementation. This is a review of the local checkout, not a claim
that the remote main branch was freshly synchronized. The v4 assessor and notebook
are uncommitted; findings against those files must be fixed before publishing them.

Scope included the public API and adapters, rotations, scalar/vector quantization,
calibration and recovery, allocation, checkpoint/native formats, native runtime,
KV-cache evaluation, fidelity/trajectory statistics, notebook orchestration,
experiment records, CI, packaging, and roadmap. This was repository-wide inspection
with targeted reproductions, not a formal proof of every algorithm or an exhaustive
security audit. No implementation was changed and nothing was pushed.

## Findings

### R1 — [P1] Shared rotations crash under `torch.inference_mode()`

Location: [rotquant/rotate.py](../rotquant/rotate.py), lines 150–163.

The activation cache reads `x._version`. Tensors created inside
`torch.inference_mode()` do not have a version counter, so the first cached
rotation raises `RuntimeError: Inference tensors do not track version counter.`
The public `RotQuantConfig` enables shared rotations by default, making this a
normal inference usage failure rather than an obscure research configuration.

Reproduced end-to-end using a locally constructed, two-layer tiny Llama:
`optimize_model` followed by a `torch.no_grad()` forward succeeds; the same model's
forward under `torch.inference_mode()` fails. No model download is required.

Required change: handle inference tensors explicitly, using an invocation-scoped
cache or disabling this cache when version tracking is unavailable. Do not merely
remove the mutation check for ordinary tensors. Add public-API forward and
generation regressions for both modes and shared/unshared rotations.

### R2 — [P1] Fresh CI environments cannot collect the notebook tests

Locations: [pyproject.toml](../pyproject.toml), lines 73–77;
[Python CI](../.github/workflows/python-ci.yml);
[allocator-v4 tests](../tests/test_qwen35_allocator_v4.py), line 10.

The v2, v3, v4 and dynamic-experiment tests import `nbformat` unconditionally, but
neither the declared extras nor `uv.lock` include it. CI installs with
`uv sync --extra dev --extra eval`, so a fresh environment cannot import those
tests. A working developer environment masks the missing dependency.

Read-only reproduction:

```bash
uv sync --frozen --extra dev --extra eval --dry-run --offline
```

This reports that it would uninstall `nbformat==5.11.1` and its notebook-related
dependencies from the current environment. The local full suite passes because
those packages are currently installed outside the lock.

Required change: declare notebook build/test dependencies in an appropriate extra,
update the lock, and run collection plus the full suite in a clean synchronized
environment. Include Python 3.13 in the compatibility plan because that is the
Colab interpreter used in the supplied runs.

### R3 — [P1] Confirmation can promote a recipe with severe secondary regressions

Location: [scripts/assess_qwen35_allocator_v4.py](../scripts/assess_qwen35_allocator_v4.py),
lines 135–167. New, uncommitted v4 code.

The seed-0 selector has no-regression guards for top-1, perplexity, diverse KL and
trajectory agreement. Confirmation does not reapply them. It only requires
`any(secondary.values())`, allowing one small improvement to outweigh arbitrarily
large deterioration elsewhere on seeds 1/2.

A synthetic run passed the actual seed-0 selector, then was promoted after
confirmation seeds 1/2 deteriorated: top-1 fell from 0.91 to 0.80, diverse KL
doubled from 0.08 to 0.16, diverse top-1 fell from 0.85 to 0.50, and trajectory
agreement fell from 0.37 to zero. Primary KL and its paired intervals still
passed. A WikiText perplexity improvement from 10.0 to 9.9 in every seed was
sufficient to pass the secondary gate. This failure does not require bypassing
the seed-0 selection step.

Required change: apply the registered no-regression guards on confirmation data,
separately from the requirement for a positive signal. Validate all required
metrics as finite, require usable paired sample counts/reliability metadata, and
add negative tests in which seed 0 passes but confirmation regresses. An internal
winner should not be announced merely because primary KL and one other metric win.

### R4 — [P1] The exported-byte gate accepts results with no export

Locations: [scripts/compare_qwen35_dynamic_to_unsloth.py](../scripts/compare_qwen35_dynamic_to_unsloth.py),
lines 51–76; [v4 assessor](../scripts/assess_qwen35_allocator_v4.py), lines 146–184.

When `packed_artifact_bytes` is absent, the comparator substitutes
`complete_persistent_model_bytes`. Those quantities have different meanings: the
latter excludes the artifact's serialization/configuration/tokenizer overhead.
The comparator still sets `within_byte_gate=True`, which the assessor reports as
`within_exported_byte_gate` and accepts for promotion.

Reproduced by removing every exported size from a three-seed synthetic result.
Matching tensor bytes to the provider's artifact bytes still produces a passing
export gate and a promoted recipe. The notebook permits export to be disabled,
so this is a reachable operational path.

Required change: distinguish estimated and measured byte fields explicitly; an
estimate must not satisfy an export requirement. Require a measured export tied
to the selected recipe, seed and revision. Report which seeds were exported and
avoid treating one exported seed as a measurement of every seed. Preserve
estimate-only comparisons as clearly labeled development diagnostics.

The supplied v3 Pareto/random comparisons do contain seed-0 export sizes, so this
finding does **not** invalidate those particular measured byte comparisons.

### R5 — [P2] “Scoring matches deployed” ignores deployment-changing settings

Locations: [rotquant/dynamic.py](../rotquant/dynamic.py), lines 669–709, 762–764,
1652–1655; [scripts/run_experiment.py](../scripts/run_experiment.py), score-context
construction in `_apply_quantization`.

Scoring constructs independent rotations with `seed + layer_index`. Deployment
can instead share the first sibling's rotation across q/k/v or gate/up projections.
The candidate constructor also omits `activation_bits`, and it does not execute
rotation training. Nevertheless, fidelity is reported as true whenever the scale
and error-compensation overrides say `inherit`. The cache identity omits these
deployment settings too.

Reproduced on three 16-by-16 q/k/v projections with shared rotations and A8:

```text
candidate_scoring_matches_deployed: True
scored/deployed activation bits: None / 8
scored/deployed k-projection rotation equal: False
different packed words: 32
```

Required change: share one resolved deployment specification between scoring and
patching, including rotation-site identity, activation precision and any trained
rotation state. Until supported, reject incompatible combinations or label them
as proxy scoring; include the relevant settings in score-cache identity.

The registered v4 sweep uses unshared FWHT and no activation quantization or
rotation training, so these specific mismatches are not evidence that its default
candidate scores, or the corresponding v3 defaults, are invalid.

### R6 — [P2] Pareto pruning can discard the only exact-byte-feasible state

Location: [rotquant/dynamic.py](../rotquant/dynamic.py), lines 1150–1175.

Within a coarse bucket, greater savings and lower distortion dominate a state.
That rule is appropriate for an upper-budget problem, but not always for the
two-sided byte interval used here: greater savings can put a recipe below the
minimum permitted size. The final exact-byte check cannot recover a state already
discarded by pruning.

Reproduction using one layer, a 1,000-byte bucket, and candidates with
`(bytes, score)` equal to `(100000, 1.0)`, `(99500, 0.5)` and `(99400, 0.4)`:
for a target of `99500 ± 50` bytes, the solver returns 99,400 bytes and reports
failure, despite the exact 99,500-byte candidate being available. The small-model
exact-granularity shortcut is deliberately not triggered in this example.

Required change: make pruning aware of the lower and upper bounds, or add an
exact feasibility/repair pass before reporting failure. Test non-monotonic
same-bit formats and compare small instances against exhaustive search. This
reproduction establishes a solver defect, not that an archived v3 run missed a
better feasible solution.

### R7 — [P2] Generic model support can report success and break the model

Location: [rotquant/adapters.py](../rotquant/adapters.py), lines 27–28 and 67–72.

Every `nn.Linear` subclass is considered replaceable, and finding one makes
`ModelSupport.supported` true. Some parents consume a child's `.weight` directly
instead of calling its forward method. `QuantLinear` does not expose that tensor.

Reproduced with a wrapper around `nn.MultiheadAttention(16, 2, batch_first=True)`:
inspection reports `supported=True`, the original forward succeeds, optimization
returns successfully, and the next forward raises
`AttributeError: 'QuantLinear' object has no attribute 'weight'` because the
attention implementation directly accesses `out_proj.weight`.

Required change: exclude known incompatible parents/projection types or implement
parent-level adapters. Expose discovery separately from validated execution
support, and fail before mutating an unsupported model. Do not solve this by
silently adding a persistent dense `.weight`, which would defeat the storage
contract. Add functional-forward tests for each advertised architecture family.

### R8 — [P2] Interrupted overwrite can leave a mixed checkpoint marked complete

Locations: [rotquant/checkpoint.py](../rotquant/checkpoint.py), lines 399–432;
[scripts/run_qwen35_next_stage.py](../scripts/run_qwen35_next_stage.py), lines 660–665.

Saving the manifest last protects a new directory, but not `overwrite=True`:
the old manifest remains while the tensor files are replaced. Resume checks only
whether the three filenames exist, so it can accept files from different saves.

Reproduced by exporting a tiny model, beginning an overwrite with another model,
and injecting failure at the packed-state write. The ordinary-state file's hash
changes; the packed-state file and manifest retain their old hashes. Despite that
mixed generation, `_complete_artifact` returns true.

Required change: write a complete validated generation into a staging location,
then publish it atomically or through a versioned manifest pointer. Preserve the
last good generation. Resume must check manifest/file integrity and recipe
identity, not filename presence. Add interrupted-first-write and interrupted-
overwrite tests, including a truncated safetensors file.

### R9 — [P2] Shared activation caches retain whole-prefill activations

Location: [rotquant/rotate.py](../rotquant/rotate.py), lines 159–162; cache enabling
in [rotquant/patch.py](../rotquant/patch.py), line 283.

Each shared rotation holds strong references to both its input and rotated output.
There is no forward-completion or last-consumer cleanup; `clear_activation_cache`
is only called when explicitly disabling the cache. Consequently, inference
retains activations for all visited shared sites after a model forward returns.

The tiny two-layer Llama reproduction retains eight tensors across four sites,
totaling 12,288 bytes for a 12-token FP32 forward. The same allocation pattern
scales with layers × sequence length × hidden width. Illustratively, 32 layers,
two sites per layer, two cached tensors per site, 32,768 tokens and width 4,096
would retain 32 GiB at two bytes per element. That is an extrapolation, not a
measured large-model allocation.

Required change: bound reuse to a shared site's consumers within one invocation,
then release it, including exceptional paths. Add live-memory/retained-tensor
checks after prefill and decode. This is distinct from R1: avoiding the version
counter exception alone does not fix the lifetime problem.

## Experimental evidence: what the results support

The data-validation pass independently recomputed the supplied v3 confirmation
means and checked its SHA-256 against the compact result record. The bundle has
15 confirmation rows, seeds 0/1/2, and 21 paired reports. The record matches.

| Recipe | Mean teacher KL | Top-1 agreement | Diverse trajectory agreement | Measured seed-0 artifact bytes |
|---|---:|---:|---:|---:|
| Uniform scale8 W4 | 0.016801 | 93.17% | 50.71% | Not exported in this bundle |
| Broad random exact-size | 0.114340 | 83.03% | 14.92% | 3,587,634,335 |
| Pareto global | 0.031129 | 90.78% | 36.83% | 3,587,632,807 |
| Unsloth Q4 anchor | 0.011888 | 94.09% | Not measured here | 3,584,533,344 |

RotQuant entries are three-seed means; the provider row is the supplied anchor
evaluation. Uniform W4 has 3,759,868,416 registered persistent tensor bytes, so its
quality cannot be treated as an equal-size comparison to the smaller Pareto model.
The primary suite contains 24 prompts/12,264 scored tokens; the diverse suite
contains 25 prompts. These are development suites, not independent replications
of the provider's entire benchmark.

Interpretation:

- Measured allocation is materially better than the broad random control at
  approximately the same size. The 72.8% KL reduction is reproducible arithmetic.
- It does not establish an optimal allocator. The broad random palette produces
  a weak control; the bits-only Pareto control and same-palette format control in
  v4 are important additions.
- The Pareto model remains approximately 2.62 times the Unsloth anchor's KL and
  3.31 percentage points lower on top-1 at a seed-0 artifact only 0.0865% larger.
  This is a cross-engine development comparison, not a clean attribution of the
  entire difference to quantization alone.
- The hard W6/W8 protection experiments did not establish an improvement. This
  does not mean sensitive layers never benefit from higher precision: at a fixed
  budget, that benefit must outweigh the compensating downgrades elsewhere.
- No v4 quality result exists yet. Its broader palette is an experiment, not an
  established improvement.

The old withdrawn KV-cache claims must remain withdrawn. The current cloning,
endpoint checks and tiny-hybrid bit-monotonicity tests are valuable safeguards;
passing them does not retroactively validate results from the state-sharing bug.

## Architecture, science and release assessment

| Area | Assessment | Main remaining requirement |
|---|---|---|
| Rotation/quantization primitives | Coherent implementation with substantial numerical tests; no new primitive-level counterexample found in this review | Keep stored-scale, rotation and deployed-weight equivalence tests across every format |
| Dynamic allocation | Useful research implementation; v3 shows a real improvement over its random control | Faithful candidate deployment, reliable exact-byte feasibility, and whole-model confirmation |
| Evaluation | Stronger than PPL-only: KL, top-1, trajectories, prompt identities and paired comparisons | One reusable, fail-closed promotion contract and an untouched final test set |
| Public model API | A useful initial surface, but discovery is broader than demonstrated execution support | Parent-aware adapters, preflight rejection, normal inference-mode coverage and model-specific presets |
| Checkpointing | Versioned, pickle-free, tested round trips | Transactional export/resume, integrity metadata, and lower-memory loading |
| Native CPU runtime | Real standalone C/C++ and SIMD work, with cross-language conformance coverage | Integration with model execution and deployment-format conformance |
| GPU/server inference | Still a major product gap | A packed operator integrated into a serving path, measured on named hardware |
| Packaging/operations | License, metadata, tests and CI scaffolding exist | Clean installability, GPU acceptance checks, release wheels and reproducible artifacts |

The main scientific limitation is the allocator's additive approximation:
single-projection reconstruction and marginal teacher KL do not capture all
interactions after many projections are quantized together. Format-aware search
is reasonable, but should be checked against joint/block-level perturbations and
whole-model validation, with a fixed compute budget. Merely widening the palette
or forcing precision islands does not guarantee a better quality/size frontier.

The current Transformers `QuantLinear.forward` still dequantizes a whole weight
matrix before `F.linear` unless a dense fallback is cached. The standalone native
registry is not automatically invoked by that path. Thus native conformance and
packed file size do not establish end-to-end GPU speed or resident-memory gains.
The native-v2 block format also uses FP16 scales, while the current winning
research recipe uses uint8 scale metadata: deployment must account for the
conversion's representation, size and numerical behavior explicitly.

Packed loading constructs a full Transformers model with `from_config` before
replacing its linears. Host-memory demand can therefore be substantially larger
than the eventual packed artifact. Meta initialization/sharded loading belongs on
the path to 27B support, alongside GPU kernel work.

The v4 notebook builder inherits v3, which inherits earlier notebook builders,
and modifies source strings. Compilation and generated-file equality catch syntax
drift but not all semantic omissions. A shared parameterized experiment/notebook
builder with a single protocol schema would be easier to maintain than continued
version-by-version replacement chains.

For reproducibility, the new v3 compact record is presently untracked, and the
raw supplied v3 evidence remains in the Downloads bundle. Compact means and file
hashes alone cannot reconstruct per-prompt intervals. Before a release, archive
the permitted raw metric records, manifests and checksums in Git or versioned
release storage; retain licensed text/token data separately where redistribution
is restricted. This review did not publish those files.

## Recommended order of work

1. **Before the next expensive Colab sweep:** fix R2–R6, unify selector/assessor
   no-regression rules, require measured exports for artifact gates, and add
   adversarial decision tests. Retain the existing v3 results as the baseline.
2. **Before calling the API generally usable:** fix R1 and R7–R9; run tiny-model
   forward/generation and checkpoint tests under both `no_grad` and
   `inference_mode`, with shared/unshared rotations and interruption injection.
3. **Run a bounded v4 smoke, then the registered 4B sweep:** verify candidate
   versus deployed codes/scales/rotations on representative layers, resume after
   interruption, export/reload the chosen artifact, and preserve per-prompt
   evidence. Promote only after confirmation and no-regression gates pass.
4. **Make the competition measurable:** freeze the final 300-prompt protocol,
   source/tokenizer/template identity and byte policy; measure source-model
   cross-engine differences and real task outcomes. Keep re-used development
   suites distinct from the final test set.
5. **Build one deployable runtime path:** choose one canonical format and target
   backend, implement packed prefill/decode integration, then measure latency,
   throughput, peak memory and reload fidelity. Expand to other engines after
   that contract is proven; provision a GPU acceptance mechanism first.
6. **Move to 27B after the pipeline is frozen:** use a small transfer canary and
   memory preflight before committing to a full factorial experiment. Do not
   assume model size alone will close the provider gap.

## Verification record and limits

- `.venv/bin/pytest -q`: **556 passed, 17 skipped, 2 warnings**, 244.61 seconds.
  The skips were unavailable AVX2 cases on this Mac. Available native compilation
  and conformance tests were included. Warnings were SWIG deprecations.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed; existing implementation changes were preserved.
- Clean synchronization dependency audit: reproduced missing `nbformat` through
  `uv sync --frozen --extra dev --extra eval --dry-run --offline`; no packages were
  installed or removed.
- Targeted reproductions: inference-mode crash, retained activation tensors,
  incompatible attention replacement, scoring/deployment mismatch, lost feasible
  byte state, false confirmation, estimate-only export gate, and interrupted
  checkpoint overwrite. Temporary checkpoint fixtures were isolated and removed.
- v3 evidence: raw confirmation hash matched; three-seed means independently
  recomputed from the supplied result bundle.
- Environment: local Python 3.12, PyTorch 2.12.0; CUDA unavailable. No paid GPU
  work, full-size model run, Colab execution, live vLLM/SGLang server validation,
  fresh provider benchmark, or remote GitHub CI run was performed.

These findings justify a reliability pass before the next long experiment. They
do not establish that the previous valid weight-only measurements are wrong, and
they do not replace the planned hardware/runtime acceptance work.
