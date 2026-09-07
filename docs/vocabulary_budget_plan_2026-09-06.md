# Vocabulary-aware quantization plan — 2026-09-06

**2026-09-07 follow-up:** the first screen has now completed; its original
planning/registration text below is retained. Both W5 candidates passed the
development quality guards. See [results](vocabulary_results_2026-09-07.md) and
the new [packed-validation runbook](packed_vocabulary_validation_run.md).
Full-Qwen export/reload is now implemented as the next runnable notebook,
but its pretrained CUDA acceptance and independent confirmation are unexecuted.

Status: first-screen implementation and generated Colab notebook are ready for
GPU execution; no new full-Qwen quality result is claimed. See the
[runbook](vocabulary_budget_run.md). Numerical prerequisites, chunked quality
prototype, experimental shared packed wrappers/checkpoint v3, byte ledgers,
source-Hessian resume and screen assessment are implemented and CPU-tested.
Real full-Qwen export parity, conditioned rebudgeting, fresh manifests and
independent/final evaluation below remain contingent next work, not enabled
notebook stages. Local publication to GitHub is a separate action.
Planning baseline: `e10214e` (merged results review). This document changes the
next priority from interaction-aware allocation to testing the vocabulary-byte
hypothesis. It does not alter the registered v4 experiment retrospectively.

## 1. Objective and interpretation

Determine whether compressing Qwen3.5-4B's shared input embedding/output head
allows enough extra backbone precision to improve fidelity at approximately
the pinned Unsloth bundle's **3,584,533,344 bytes**. Keep the vision tower and
auxiliary-tensor policy unchanged. Remain on the 4B model, FP16 activations and
the existing KV policy; do not introduce recovery, A8, new codebooks, or new
serving-engine forks in the same experiment.

The [results review](results_review_2026-09-06.md) reports a 635,699,200-weight
tied vocabulary matrix: FP16 costs 1,271,398,400 bytes, versus 521,472,000 bytes
for the audited Q6_K tensor. The 749,926,400-byte difference is equivalent to
1.683 bits per one of the 3,565,158,400 targeted backbone weights. This is a
substantial allocation opportunity, **not proof** that it explains all of the
quality gap. Compressing the vocabulary changes both input representations and
output predictions. The existing artifacts did lose the matched-total-size
comparison; that observation remains valid.

Questions, in order:

1. How much error does vocabulary compression introduce by itself?
2. Does a W5 backbone recover more quality than a W8/W6 vocabulary loses?
3. Does mixed W4/W5/W6/W8 allocation improve on a feasible uniform backbone
   once the vocabulary no longer consumes 1.27 GB?
4. Does the actual exported and reloaded model retain those gains?

The previous v4 result is a useful negative control: full format selection
averaged KL 0.03134 versus bits-only 0.03113, despite trajectory agreement
improving from 36.83% to 45.79%. It does not justify abandoning mixed precision
or promoting v4. Raw evidence is the supplied
`qwen35_allocator_v4_8ba3b751bb82` bundle; 30 original JSON records and export
manifests are now archived with an
[SHA-256 index](../research/results/raw/qwen35_4b_allocator_v4_8ba3b751bb82/evidence_index.json).
Tensor/tokenizer binaries are not part of that compact evidence archive.

## 2. Work package A — correctness and accounting prerequisites

Land regression-tested fixes before freezing the experiment revision:

- Preserve packed scales, offsets, steps and other format-defining floating
  metadata at their recorded storage dtypes across `.to()`, FP16/BF16 casts
  and checkpoint reload. Changing execution dtype must not silently requantize
  the artifact. Check residual metadata and codebook/rotation ownership too.
- Prevent FP16 intermediate overflow in the pure-torch normalized FWHT; test
  finite large inputs against an FP32 reference, including CUDA fallback when
  the optional kernel is absent. Do not silently change the unnormalized API.
- Rotate source weights in FP32 before quantization. Version/invalidate
  affected candidate caches and rerun the controls under the new code revision.
- Correct the scale-search range for the `uniform` codebook and add a
  properly scaled absmax reference. Gaussian-vs-uniform claims need these
  corrected controls, even though the initial screen uses Gaussian codebooks.
- Record scale-range and reconstructed-scale error distributions by projection
  and width. Use FP16 scale metadata initially for vocabulary W6/W8 and
  high-precision backbone W6/W8, avoiding a new scale8 error floor. Retain the
  existing scale8 W4/W5 backbone recipe as the controlled baseline. A scale8
  high-precision optimization is a later named ablation, not a silent switch.
- Produce a component ledger on both sides: shared vocabulary, targeted
  backbone, vision/projector, small retained tensors, codes, scale metadata,
  codebooks/rotations, container/tokenizer files and complete artifact bytes.
  Integrate the pinned GGUF type inspection into comparison metadata; archive
  its tensor table, revision and digest. Report effective backbone and
  whole-model rates, not just the `Q4`/`W5` label.

Acceptance: fresh locked CPU test suite plus synthetic CUDA tests of every
newly used format; storage counts reconcile; unsupported configurations fail
before model calibration. Previous numerical results remain historical rather
than being silently relabelled as results of the repaired quantizer.

## 3. Work package B — bounded vocabulary quality prototype

### One logical matrix, two uses

Use adapter-resolved input/output embedding accessors and verify actual shared
storage, shape, dtype and vocabulary size. For this Qwen model, modifying the
single tied matrix must affect both lookup and output projection. Reject an
untied or unsupported model in this initial path; never silently quantize only
one alias. General untied support is a later explicit extension.

Use the repository's rotation convention, not ambiguous handwritten transpose
notation:

```text
W_rot = rotation.rotate_weight(W_fp32)
packed = quantize(W_rot)
W_hat = rotation.inverse_activation(packed.dequantize())
```

For its row-vector convention, `W_rot = W R^T`, `W_hat = Q(W R^T) R`, and
`x W_hat^T = (x R^T) Q(W R^T)^T`. Test both embedding lookup and output logits
against this dense reconstruction. Do not put rotated rows directly into an
ordinary embedding. An identity rotation is the corresponding no-rotation case.

Quantize in row chunks with one shared deterministic codebook and rotation.
Chunk boundaries must preserve code packing and second-level scale-block
boundaries, including the final partial chunk. Check chunked vs unchunked
equivalence on small matrices and chunk-size invariance. Stream outputs to CPU
or disk; do not create a second full FP32 vocabulary matrix or accumulate GPU
chunks. Preserve source rows for restoration or reload the immutable source
between arms; test that interruptions cannot leave another arm's vocabulary.

### Teacher and calibration isolation

- Capture reference logits and greedy sequences from the **unchanged source**
  before any mutation, and hash the reference state. Reuse neither a student
  nor an aliased mutated parameter as the teacher. Snapshot/reload tests must
  demonstrate that arm order cannot change a reference or result.
- For the factorial screen, collect backbone Hessians/activation calibration
  on the same source model and token manifest. Freeze them across vocabulary
  treatments so the screen isolates the deployed vocabulary intervention.
  Use the original quantized backbone unchanged for each of its vocabulary arms.
- Condition later mixed-allocation scoring on the selected compressed
  vocabulary student, but compare with the unchanged FP16 teacher. Include
  the vocabulary configuration and reconstructed/packed fingerprint in every
  affected score, Hessian, activation and trial cache identity. If reusing
  source Hessians, explicitly record their source basis; vocabulary-conditioned
  recalibration would be a different experiment.

### Honest storage reporting

The dense reconstruction is a **quality prototype**, not a compressed model.
Report separately: actual resident/dense bytes, actual packed vocabulary payload
bytes (when produced), projected complete artifact bytes, actual exported bytes
(absent until measured), and peak workspace. Do not subtract a live FP16
parameter from `registered_model_bytes` and call the result measured storage.
Estimated bytes cannot pass an artifact or provider-promotion gate.

## 4. First Colab screen — nine controlled arms, seed 0

Use a 3-by-3 factorial screen:

| Backbone | Vocabulary FP16 | Vocabulary W8 | Vocabulary W6 |
|---|---|---|---|
| FP16 | Source/reference control | Vocabulary-only error | Vocabulary-only error |
| Uniform W4 | Refreshed existing control | Combined W4/W8 | Combined W4/W6 |
| Uniform W5 | Backbone-only improvement | Primary W5/W8 candidate | Primary W5/W6 candidate |

Default compressed-vocabulary formats: Gaussian codebook, MSE scale search,
group 128, FP16 scales, FWHT block 128, no GPTQ, same rotation seed across
widths. Using the same recipe at W6 and W8 avoids confounding precision with
rotation. An unrotated W8 vocabulary is a later cost/quality ablation, not
another default axis. Higher-precision scale metadata adds only roughly 5 MB
versus the review's vocabulary scale8 estimates; actual packing decides size.

Backbone W4/W5 use the frozen FWHT/Gaussian/MSE/act-order-GPTQ recipe and the
same seed/calibration samples. No forced precision islands, learned rotations,
LoRA or calibration-size changes. FP16-vocabulary W5 is deliberately an
over-budget diagnostic, not a provider competitor. W5/W6 may be materially
under budget; show it on the frontier without calling it an exact-size match.

Run the existing 24-prompt primary KL and 25-prompt diverse/32-token suites,
plus the existing WikiText-2/C4 PPL sentinels, for historical continuity. These
are development data already used for selection, not an untouched benchmark.
Report mean/median/p95/max KL, top-1 agreement, NLL change, token agreement,
exact trajectories, matching prefix, domain summaries, loops/empty outputs and
the component ledger. A 32-token match is not task accuracy.

Report paired contrasts for vocabulary-only damage, W4-to-W5 improvement,
and their combined effect. For each compressed-vocabulary treatment also report
the interaction contrast `(W5,Vq - W5,V16) - (W4,Vq - W4,V16)`; do not assume
the two sources of KL error add independently.

Only two backbone quantizations are needed for the six quantized-backbone
cells when source Hessians and backbone weights are fixed. Reuse each backbone
and the same vocabulary reconstruction safely; do not recalibrate nine times.
Show every cell, including failed/halted arms, rather than only a winner.

### Screen decisions (before seeing results)

- Keep existing catastrophe cutoffs (primary KL > 0.25 or top-1 < 0.75), and
  stop immediately for nonfinite outputs, teacher/tie mismatches or failed
  conformance. A halted arm is ineligible, not missing data.
- Retain at most the two W5/compressed-vocabulary candidates. A candidate must
  have lower primary KL than refreshed W4/FP16 and pass the existing v4
  no-regression guards on primary top-1, diverse KL/top-1, both PPL datasets
  and trajectories, relative to that same W4/FP16 control. Use the shared guard
  implementation and serialize its numeric thresholds.
  Do not interpret an oversized diagnostic as a matched-byte improvement.
- If neither qualifies, stop the expensive confirmation/allocator stages and
  inspect vocabulary-only error and scale diagnostics. Do not automatically
  launch a recovery sweep. First investigate vocabulary quantization itself
  (for example rotation, stored scales or output-aware fitting) on calibration
  data under a new registered ablation.
- If one or both qualify, proceed to real packed representation and use the
  measured free budget to decide whether mixed allocation is warranted.

Current shared guard values at the planning revision: primary/diverse KL and
each PPL ratio at most 1.02; primary/diverse top-1 loss at most 1 percentage
point; trajectory token-agreement loss at most 2 percentage points. The screen
adds the stricter primary-KL improvement requirement above. Finite complete
metrics are mandatory; these guard allowances are not claims of equivalence.

## 5. Work package C — real shared packed vocabulary

Implement one packed vocabulary owner referenced by an embedding-lookup wrapper
and an output-projection wrapper. Store codes, scales, codebook and rotation
once; serialize aliases explicitly and reconstruct them on load. The source
FP16 matrix must be absent from the resulting packed artifact.

Lookup gathers/dequantizes only requested rows and reverses their rotation.
Output projection rotates hidden activations and consumes the same packed rows;
start with a bounded tiled reference path, not a permanently materialized dense
head. Explicit quality-only dense caches may remain opt-in and separately
accounted, but cannot satisfy packed-residency acceptance.

Tests: repeated-token lookup, all vocabulary rows on a tiny fixture, tied
storage identity, logits/generation vs prototype, FP16/BF16 execution without
metadata recasting, device moves, save/reload in a fresh process, file/manifest
hashes, duplicate-tensor detection, corrupted checkpoint rejection, missing
aliases, and prevention of accidental HF `tie_weights()`/resize reintroduction
of a dense vocabulary. Either support those mutation APIs or fail explicitly.

The existing native GGUF v1 is W4/FP16-scale only. Do not relabel the W5/W6/W8
Python checkpoint as stock GGUF-compatible. Extending that format and its
consumer is a separate runtime milestone after a recipe is selected.

## 6. Work package D — spend the recovered bytes, if necessary

For each retained vocabulary format, recompute the fixed-cost ledger from its
actual packed representation. Screen a bounded Gaussian g128 backbone palette:
W4 and W5 with the existing scale8 recipe; W6 and W8 with FP16 scales. Include
every format/metadata choice in candidate identities and byte accounting.

Use source-calibrated GPTQ as frozen above and score marginal changes in the
student with that vocabulary installed. Compare the feasible uniform backbone,
same-palette exact-budget random allocation and Pareto allocation. No forced
W8 islands; inspect whether the allocator selects sensitive projections now
that upgrades have a different opportunity cost.

The budget target remains 3,584,533,344 bytes, with at most 1% **measured**
artifact mismatch for direct provider pairing. Account for actual per-recipe
container overhead; an internal solver tolerance is not a substitute for export.
An under-budget recipe is still useful and should not be forced to spend bytes
when it worsens quality; distinguish its frontier point from a matched pair.

Skip this search initially if a feasible uniform W5 candidate already meets the
quality goal: validate the simpler recipe first. If search is needed, cap it at
two vocabulary settings and four backbone widths, reuse compatible tables, and
retain no more than two complete recipes. This is not another unrestricted
format sweep. Interaction-aware exchanges are the next branch only if the
corrected whole-model budget still leaves the additive allocator wanting.

## 7. Confirmation and external evidence

Before implementation completes, freeze a new 100-prompt development-validation
manifest (20 per domain) distinct from calibration and the existing 24/25-prompt
sets, plus an untouched 300-prompt final manifest (60 per domain). Record source
licenses/IDs, exact token IDs, chat template, lengths/truncation, duplicate and
near-duplicate checks. Do not use final prompts for allocation or recovery.

Confirm frozen finalists at seeds 0/1/2 on the new development-validation set.
Seeds 1/2 are the independent recipe-seed confirmation; screen seed 0 is not a
second independent sample when reused. Include source, refreshed uniform
W4/FP16 as an over-budget quality control, and the refreshed v3 bits-only policy
as the previous matched-budget control. Mixed-allocator claims additionally
need a measured near-budget uniform comparator and same-palette random control.
If a comparator is not within the byte gate, report the size difference and
do not claim a fixed-size allocator win over it.

Use the existing strict promotion contract: finite complete evidence, primary
KL win over the relevant matched control in every seed, reliable paired KL
interval below zero in every seed, all per-seed no-regression guards, and a
consistent secondary improvement. Do not relax these after seeing the screen.
Use two-sided 95% paired intervals, at least 4,000 seeded bootstrap draws and
the existing minimum of 20 paired prompts/reliability check. Declare the primary
comparisons before scoring; unregistered domain/format analyses are exploratory.
Bootstrap paired prompts (and PPL windows), not independent tokens or duplicated
seed/prompt rows; show seed-specific intervals and any cross-seed aggregate
separately. Inconclusive results remain inconclusive. Export/reload all promoted
seed artifacts and report their individual measured sizes and identities.

The pinned Unsloth 24-prompt KL 0.0118883/top-1 94.0884% is a historical anchor
only. Rerun the released artifact on the new exact token manifests for any new
comparison; do not compare a 100/300-prompt result with its old aggregate.
Keep the same source revision and auxiliary-module policy. Measure source
agreement across engines/precisions and, where feasible, compare both students
to one common pinned teacher on aligned token IDs. Continue to report each
engine's source-normalized metrics; engine differences are a caveat, not assumed
to cancel. Unsloth trajectory/task results must actually be collected.

After recipe freeze, execute the untouched 300-prompt/32-token contract with
domain and task/failure outcomes. No public parity/win claim without matched
actual bytes, complete paired evidence, recorded engine/reference semantics
and published reproducibility records. A code correctness fix invalidates all
affected arms and controls equally. See [competitive_eval.md](competitive_eval.md).

## 8. Notebook and implementation deliverables

Implementation inventory (later gated stages are deliberately not runnable yet):

| Deliverable | Scope |
|---|---|
| `rotquant/vocabulary.py` and adapter hooks | Chunked quality prototype, shared packed owner/wrappers, explicit supported-model contract |
| Quantizer/rotation/linear/checkpoint changes | Numerical fixes, alias-preserving format/load, unique-byte accounting |
| `configs/qwen35_4b_vocabulary_budget_cuda.yaml` | Frozen vocabulary/backbone definitions, data, cache identities and gates |
| `scripts/run_qwen35_vocabulary_budget.py` | Implemented nine-arm factorial only; conditional rebudgeting, confirmation and full-Qwen export are subsequent work |
| `scripts/assess_qwen35_vocabulary_budget.py` | One fail-closed interpretation shared with the notebook |
| `scripts/build_qwen35_vocabulary_budget_notebook.py` | Generate the notebook directly using common helpers; avoid another chain of inherited text replacements |
| `notebooks/qwen35_4b_vocabulary_budget_colab.ipynb` | Readable controls, live output, resume, compact result bundle |
| Tests and archived manifests/results | CPU unit/round-trip/accounting/resume checks and CLI dry-run; synthetic CUDA preflight is provided for Colab, not claimed executed locally |

First-run defaults: preflight and nine-arm seed-0 screen enabled; expensive
rebudgeting, three-seed confirmation, final 300 prompts, recovery, A8 and KV
sweeps are not automatic stages of this notebook. Print a decision and a next
action. An eventual optional
end-to-end mode may advance only through predeclared gates and a printed
resource budget; it must not keep spending GPU time after a failed stage.

Use the existing A100 40 GB environment as the target, subject to measured
preflight headroom; no larger GPU is assumed necessary. The implemented preflight
times synthetic vocabulary and W4/W5/W6/W8 projections and exercises tiny Llama
and Qwen3.5 hybrid generation. After source load the runner also prints a
shape-based resource ledger (tensor/Hessian/reference bytes and reported free
space). It does not extrapolate those timings into an unvalidated full-model
runtime forecast. A full model/host/Drive peak-headroom and
runtime forecast is still required before the later export/allocator stages.
Do not promise a fixed number of hours from operation counts alone. Stream exact
full-vocabulary teacher comparisons by prompt/token chunk; bound reference disk
usage or recompute references rather than substituting top-k KL silently. The
screen currently reuses full-logit references in host memory within its process
and recomputes them after runtime loss; it does not persist a teacher-logit disk
cache or substitute top-k KL.

Print unbuffered per-arm/per-layer/per-vocabulary-chunk progress, GPU memory,
elapsed time and measured ETA, plus 30/60-second heartbeats. Persist atomic
progress records, checkpoints, logs and failures to Drive. Identify every
resumed arm/cache by immutable revision, source and data hashes, vocabulary
fingerprint, quantizer settings and random seed. On disconnect, print the exact
status/log paths and last committed work item. A checksum-validated compact
bundle is downloadable without the multi-GB weights; full exports stay in Drive.

## 9. Execution order and stop points

1. Numerical/ledger fixes and tests; archive v4 evidence without rewriting it.
2. Vocabulary prototype, teacher/tie isolation and nine-arm notebook.
3. Run the seed-0 screen; stop and inspect if compression does not recover quality.
4. If promising, shared packed export/reload; optional corrected-budget allocation.
5. Fresh-manifest, three-seed confirmation; freeze recipe and exact artifacts.
6. Untouched external evaluation, then one measured packed GPU operator path.
7. Only after repeatable 4B evidence, plan the 27B transfer and its resource budget.

Low-rank recovery remains a separate later experiment with fixed codes and a
real token/storage budget. This plan is PTQ, not QRAT, and introduces no training
of original weights or rotations. Quality, artifact size and serving speed have
separate acceptance gates throughout.
