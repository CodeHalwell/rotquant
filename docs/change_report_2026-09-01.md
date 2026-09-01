# Change report — 2026-09-01 validity review and follow-up

This report records every change made on branch
`claude/rotquant-validity-review-4rbsp1`, the reason for each change, the
evidence behind it, what was verified, what is now withdrawn, and what the next
generations of RotQuant need to improve or implement. It complements
`docs/scientific_validity_review_2026-09-01.md`, which contains the review
itself, and `docs/experiment_log.md`, which records the research decision.

Nothing in the packed bitstream, the checkpoint format, the native runtime or
the released artifacts was changed. The changes are to the evaluation code,
one quantizer storage path, three calibration-time training details, notebook
pins, tests, and documents.

---

## 1. Summary of changes

| Area | File(s) | Change | Reason |
|---|---|---|---|
| KV simulator | `rotquant/eval/kv_cache.py` | Clone every tensor reachable through dict/list/tuple attributes of the cache and its layers; fail closed if any storage stays shared; count the same tensors in `non_kv_state_bytes`. | Under Transformers 5.16.x the source and packed decode passes shared linear-attention state and corrupted each other, producing a KL floor independent of bit width. |
| KV simulator | `rotquant/eval/kv_cache.py` | Mandatory endpoint check: a uniform 8-bit Gaussian cache must give KL ≤ 0.01 on the held-out calls before any candidate or allocator runs. | A floor that survives 8-bit codes cannot be quantization error; this would have caught the defect on the first run. |
| KV simulator | `rotquant/eval/kv_cache.py` | Absolute-position tiers: decode writes are no longer their own sequence, sink rows are decided by absolute position, rows are packed once when they leave the recent window. | Previously every decode write stayed fp16 forever, so tiered results were optimistic beyond the recent window. |
| Quantizer | `rotquant/quantize.py`, `rotquant/_internal.py`, `rotquant/linear.py` | `_encoded_storage_scales` encodes scales exactly once and every path retains that triple; GPTQ snaps lazily refit scales onto the frozen 8-bit grid. | With 8-bit scales, GPTQ assigned codes against per-group-encoded scales but stored row-major-encoded ones (1.1×10⁻³ relative mismatch); other paths re-encoded decoded values, which is not guaranteed idempotent. |
| Rotation training | `rotquant/train_rotation.py`, `rotquant/patch.py` | `hessian_reconstruction_error` and `select_butterfly_checkpoint_hessian`; `patch_model` gates the Hessian objective against seeded FWHT under the deployed quantizer. | The Hessian objective trained under a proxy (RMS assignment, no GPTQ) and deployed the best proxy step with no packed-versus-FWHT comparison, unlike the activation objective. |
| Rotation training | `rotquant/rotate.py`, `rotquant/train_rotation.py` | Learned-sign logits start at ±`init_magnitude` (default 0.1) instead of ±1. | With lr 10⁻³ and 200 Adam steps a logit moves at most ≈0.2, so ±1 could never cross zero: the arm was inert (measured flip rate 0). |
| Notebooks | eight `notebooks/*.ipynb` | `transformers>=5.9,<6` → `transformers==5.9.0`. | The open range resolved to 5.16.x on 2026-08-29 and exposed the simulator defect; the pinned stages already used 5.9.0. |
| Statistics | `scripts/run_qwen35_next_stage.py` | Paired intervals carry `interval_reliable` (false below 20 paired samples). | A percentile bootstrap of four prompts is not a 95 % interval. |
| Storage accounting | `paper/data/publication_results.json`, `scripts/audit_publication.py`, `paper/generated/*` | Record the 241 MB MTP head, the loaded-tensor total and the like-for-like reduction (58.26 %); new TeX macros. | The 59.34 % figure counts a head the model never loads on the source side only. |
| Paper | `paper/main.tex` | Abstract, contributions, method, a new validity-notice section, both cache tables, limitations, conclusion and appendix mark every cache result as withdrawn; storage is quoted like-for-like. | The cache numbers do not measure quantization. |
| Documents | `README.md`, `docs/how_rotquant_works.md`, `docs/experiment_log.md`, `docs/unsloth_qwen35_4b_comparison.md`, `docs/roadmap.md`, `configs/e3b_turboquant_bias.yaml`, `CHANGELOG.md` | Validity notices, MTP policy, the QRAT name reservation, the e3b null-control label, changelog entries. | Keep the written record consistent with the evidence. |
| Tests | `tests/test_kv_cache_bit_monotone.py`, `tests/test_scale_storage_consistency.py`, additions to `tests/test_kv_cache_eval.py` and `tests/test_train_rotation.py` | Guards for every change above. | Each finding gets a test that fails on the previous code. |

---

## 2. Changes in detail

### 2.1 KV simulator: cloning all cache state

**What was wrong.** `simulate_packed_kv_cache` builds the "packed" cache by
copying the source cache after prefill. The old `_clone_cache` cloned only
attributes that were themselves tensors and shallow-copied everything else.
Transformers 5.9.0 stores each linear-attention layer's `conv_states` and
`recurrent_states` as tensor attributes, so the clone was complete. Transformers
5.16.x stores them as `dict[int, Tensor]` attributes and updates them in place
(`update_conv_state`, `update_recurrent_state` and `causal_conv1d_update` all
use `copy_`). A shallow dict copy shares its tensors, so after cloning the two
caches shared every conv and recurrent state of all 24 linear-attention layers.
At each decode step the source pass advanced the shared state, the candidate
pass advanced it again, and from the second token both trajectories were
corrupted. `_cache_tensor_bytes` had the same blind spot and reported
`non_kv_state_bytes = 0`, which is the fingerprint of the affected runs.

**Evidence.** Replication on the pinned `unsloth/Qwen3.5-4B` weights on CPU
(bf16, source weights, prompt 256, two prompts × 8 continuation tokens,
Gaussian codebook, group 64, fp16 scales, no tiers, the protocol of the
recorded cache matrix):

| Transformers | Simulator | K8/V8 KL | K4/V4 KL | K2/V2 KL | top-1 at 8 bits | `non_kv_state_bytes` |
|---|---|---:|---:|---:|---:|---:|
| 5.9.0 | previous | 5.0×10⁻⁴ | 6.3×10⁻³ | 6.5×10⁻² | 0.94 | 26,738,688 |
| 5.16.1 | previous | 0.877 | 0.891 | – | 0.69 | 0 |
| 5.16.1 | fixed | 3.6×10⁻⁴ | 6.9×10⁻³ | – | 1.00 | 51,904,512 |

The recorded source-weight control (K4/V4, KL 0.8618) matches the defective
run to within noise. The next-token NLL of the *source* pass itself rose from
2.71 to 3.23 nats under the defect, confirming both passes were corrupted. A
tiny random hybrid Qwen3.5 shows the same signature (KL 1.1×10⁻⁶ at 8 bits on
5.9.0; KL 0.1307 at every bit width with top-1 0.000 on 5.16.1).

**Fix.** `_clone_tree` clones tensors wherever they live (tensor, dict, list,
tuple), the cache object's own container attributes are cloned too, and
`_storage_keys` verifies that no untyped storage is shared between source and
clone, raising otherwise. `_iter_tensors` gives the byte accounting the same
reach.

**Test.** `tests/test_kv_cache_bit_monotone.py` builds a tiny hybrid Qwen3.5
(linear and full attention) and requires `non_kv_state_bytes > 0`, KL < 10⁻⁴
at 8 bits and strictly decreasing KL from 2 to 4 to 8 bits, with and without
fp16 tiers. It fails on the previous code under 5.16.1 and passes with the fix
under 5.9.0 and 5.16.1.

### 2.2 KV simulator: mandatory endpoint check

**Why.** The recorded matrices contained an endpoint-equivalence check
(dynamic-at-8-bit ≡ uniform-8-bit) but no *near-zero* check. Equivalence holds
trivially when both runs are equally corrupted. A near-lossless endpoint is the
one invariant that no downstream recipe comparison can fake.

**Semantics.** `KVCacheEvalConfig.endpoint_check_bits` (default 8) and
`endpoint_max_kl` (default 0.01). `evaluate_kv_cache` first runs
`_evaluate_kv_cache` with a uniform Gaussian scalar cache at that width on the
*evaluation* calls, with the same tiers, group size and scale precision as the
candidate, and stores the result under `metrics["endpoint_check"]`. If the
endpoint KL exceeds the limit it raises `RuntimeError` before any candidate or
allocator runs. The check is disabled only explicitly
(`endpoint_check_bits: null`). The limit gives a 20× margin over the measured
8-bit KL on the real model (5×10⁻⁴) and is 80× below the defect (0.88).

**Cost.** One extra evaluation pass per `evaluate_kv_cache` call; negligible
against the allocator's dozens of passes and against weight patching.

**Tests.** `test_endpoint_check_runs_first_and_passes_on_a_faithful_simulator`,
`test_endpoint_check_fails_closed_on_a_bit_independent_floor` (monkeypatched
floor raises), `test_endpoint_check_can_be_disabled_explicitly`; the position
plumbing test now expects the endpoint pass plus the candidate pass.

### 2.3 KV simulator: absolute-position tiers

**What was wrong.** `quantize_kv` decides fp16 tiers from positions *inside the
tensor it is given*. A single-token decode write has length one, so with
`sink_tokens ≥ 1` every decode write was a "sink" and stayed fp16 forever;
the last `recent_window` prefill rows were likewise never re-packed. The E8P
smoke result (32 continuation tokens ≤ 32-token window) was unaffected in
value, but the corrected 8k/64-token configuration would have been optimistic
for the second half of every continuation.

**Fix.** The simulator records, per layer, the sink/recent settings and
`pending_start`, the first position held in fp16 only because it is inside the
window. On each write it computes how many of the written rows fall inside the
absolute sink prefix (`max(0, sinks − start)`), keeps the newest `recent` rows
in fp16, and after the write packs every row from `pending_start` up to
`new_length − recent` exactly once, in place in the cache tensor. Rows in the
sink prefix are never packed.

**Test.** `test_tiered_cache_requantizes_rows_that_age_out_of_the_recent_window`
checks, position by position, that the sink stays exact, that rows are packed
when and only when they leave the window, and that new rows are exact while
inside it; `test_decode_writes_without_tiers_are_packed_immediately` covers the
untiered case. Exactness is up to the fp16 rounding of the rotation round trip
(fp16 tiers are stored in the rotated basis).

### 2.4 Exact code/scale storage (8-bit double quantization)

**What was wrong.** For 8-bit scales the affine grid of each 256-entry block is
determined by that block's minimum and maximum. In `_gptq` the lazily refit
scale of one group was encoded as a column block (256 rows of one group), while
`quantize_weight` re-encoded the whole `[out, groups]` matrix row-major. Codes
were therefore assigned against values that the artifact did not store; the
measured discrepancy between the assignment reconstruction and `dequantize()`
was 1.1×10⁻³ relative Frobenius norm. Every other path re-encoded already
decoded values, which is exact only when each block's extreme codes are 0 and
255.

**Fix.** `_encoded_storage_scales` returns `(decoded, stored, offsets, steps)`
computed once; `quantize_weight`, the length-correction path, `_add_residual`
and `QuantLinear.commit_scale_finetuning` retain that triple verbatim. For
GPTQ with 8-bit scales, `_gptq_with_scale_grid` receives the grid fixed from
the initial scale selection and snaps every refit scale onto it (`code =
round((refit − offset)/step)` clipped to `[0, 255]`), so the value each code is
assigned against is bit-identical to the stored decode. The public `_gptq`
signature and 3-tuple return are unchanged. No format or bitstream change: the
stored tensors, dtypes and block layout are the same as before.

**Tests.** `tests/test_scale_storage_consistency.py` checks, for 8- and 16-bit
scales, that plain rounding, GPTQ with act-order and lazy refit, and the
residual pass all reproduce their packed codes from the stored scales and that
`dequantize()` equals the GPTQ assignment reconstruction exactly.

### 2.5 Gate for the Hessian rotation objective

**What was wrong.** `train_layer_rotation` with `objective: hessian` optimises
`tr(E H Eᵀ)/tr(W H Wᵀ)` under a proxy quantizer (`assignment_scale: rms`, no
error compensation) and keeps the best proxy step. `patch_model` then deployed
that rotation with the real quantizer (`mse_search`, GPTQ, possibly 8-bit
scales) without comparing it to the seeded FWHT it started from. The activation
objective already had such a gate. A rotation that is worse than FWHT under the
deployed quantizer could therefore be deployed silently. This cannot on its own
explain the catastrophic "optimized W4" arm, but it removes one unguarded path
before the factor ablation runs.

**Fix.** `hessian_reconstruction_error` quantizes the rotated weight with the
exact deployed configuration (rotating the Hessian for GPTQ) and returns the
Hessian-weighted output error of the un-rotated error matrix;
`select_butterfly_checkpoint_hessian` keeps the trained angles and signs only
if that error beats the FWHT reference by `selection_min_improvement`,
otherwise restores FWHT. `patch_model` applies it for every butterfly trained
with the Hessian objective (shared sites use the concatenated member weights
and the shared Hessian). `rotate_hessian` moved to `rotquant._internal` so
both `patch` and `train_rotation` use one implementation.

**Tests.** `test_hessian_gate_restores_fwht_for_a_worse_candidate_and_keeps_an_equal_one`
(a non-mixing θ=0 butterfly is rejected and restored, an identical one is
accepted, and GPTQ does not increase the weighted error) and
`test_patch_model_gates_hessian_objective_against_fwht` (the runner statistics
report the gate).

### 2.6 Learned-sign initialisation

**What was wrong.** `enable_sign_training` initialised the sign logits at the
signs themselves (±1). The forward pass uses hard signs, the backward pass a
`tanh` surrogate, and a sign flips only when its logit crosses zero. Adam moves
a parameter by roughly the learning rate per step, so with the shipped
`lr: 0.001` and `steps: 200` no logit could move more than ≈0.2. The recorded
"optimized W4" arm reported a sign-flip rate of exactly 0.0; the arm tested
nothing.

**Fix.** `enable_sign_training(temperature, init_magnitude=0.1)` and
`RotationTrainConfig.sign_init_magnitude` (validated positive). The forward
pass is unchanged at initialisation.

**Test.** `test_sign_training_starts_at_configured_magnitude_and_commits_flips`.

### 2.7 Transformers pin in every notebook

The eight notebooks that install `transformers>=5.9,<6` now install
`transformers==5.9.0`, matching `uv.lock`, the two generated notebooks and the
version recorded in the only raw result set in the repository. Transformers
5.9.0 was released on 2026-05-20 and 5.16.1 on 2026-08-26; the cache notebooks
ran on 2026-08-29/30 with `pip install -U` and would have resolved to 5.16.x.
The archived raw JSONs on Drive record `library_versions.transformers` and
should be checked to close the loop.

### 2.8 Bootstrap reliability flag

`_bootstrap_delta` in the stage runner now reports
`interval_reliable = paired_samples ≥ 20`. The stage summary's prompt-level
intervals had `paired_samples: 4`; the token-level intervals treat tokens
within a document as independent. Both are documented in the review; the flag
stops the four-sample intervals being quoted as 95 % intervals.

### 2.9 Like-for-like storage accounting

The pinned source safetensors index totals 9,319,737,856 bytes and includes an
`mtp.*` multi-token-prediction head of 241,199,104 bytes.
`Qwen3_5ForConditionalGeneration` has no `mtp` submodule, so those tensors are
dropped at load and are absent from the export. That head is the previously
unexplained 237,955,440-byte gap between the 4.0277 GB logical estimate and the
3.7898 GB export. Against the 9,078,538,752 bytes of loaded source tensors the
reduction is 58.26 %, not 59.34 %. The manifest records `mtp_head_bytes`,
`tensor_bytes_excluding_mtp` and `reported_like_for_like_tensor_reduction_pct`;
`audit_publication.py` derives and checks the like-for-like figure and emits
`\LikeForLikeTensorReductionPct` and `\SourceTensorExclMtpGB`; the paper
quotes those and explains the whole-file figures. The Unsloth comparison note
states that both totals already exclude the head.

### 2.10 Paper and documents

`paper/main.tex`: the abstract no longer makes a cache claim; contributions 2
and 3 are qualified; the method section states the endpoint requirement; a new
subsection "Validity notice: cache results withdrawn" (`sec:validity`) gives
the mechanism and the replication numbers; the cache-transfer and matched
tables and their sections are marked withdrawn and retained as a record; the
storage paragraphs use the like-for-like figure; a "Cache quality is
unmeasured" limitation is added; the conclusion and the appendix map caption
are rewritten. `paper/data/publication_results.json` carries a `validity`
block naming the withdrawn blocks; `evidence_status` and `claim_boundary`
are updated. README, the science guide and the experiment log carry the
withdrawal; `docs/roadmap.md` reserves the name **QRAT** for the future
quantization-and-rotation-aware training method and states that none of the
current calibration stages may be called QRAT; `configs/e3b_turboquant_bias.yaml`
is labelled a null control with the measured ×πd/2k penalty.

---

## 3. Verification performed

- Lint: `ruff check .` clean.
- Test suite: 428 tests passed before this work; with the new and modified
  tests, 446 pass on the final branch state (C++ conformance suite not built
  in this environment).
- Regression tests run under both Transformers 5.9.0 (the pin) and 5.16.1
  (the version that exposed the defect).
- Real-model replication on CPU as tabulated in §2.1; synthetic checks of the
  QJL variance ratio (×104 at d=4096, ×270 at d=11008, matching πd/2k), the
  sign-flip impossibility, the tiered write semantics, the GPTQ/8-bit mismatch
  (1.1×10⁻³ before, 0 after), and the Lloyd–Max anchors.
- Source checkpoint audit from the safetensors headers: 3,565,158,400
  quantisable weights × 4.125 bpw / 8 = 1,838,284,800 bytes, equal to the
  recorded `packed_weight_bytes`; `mtp.*` = 241,199,104 bytes.
- `scripts/audit_publication.py` passes 15 checks including the new
  like-for-like check.

---

## 4. What is withdrawn and what stands

| Result | Status | Reason |
|---|---|---|
| Qwen3.5-4B weight-only PPL, teacher KL, top-1, trajectories; three-seed GPTQ promotion; W4A8 comparison | Stands | These paths never use the cache simulator. |
| OPT-125M/1.3B rotation and recovery ablations | Stand (single-seed, development) | No simulator involvement. |
| Seed-0 K/V matrix, three-seed K/V validation, 1,024-token confirmation | Withdrawn | Bit-independent KL floor; simulator defect. |
| Frozen-map transfer (24.0 % / 10.6 %) and the context-length sensitivity reversal | Withdrawn | Same. |
| Matched K4/V4 controls (16.5–23.2 %; ratios 0.802 / 0.835) | Withdrawn | Same. |
| Selection of `uniform_w4__frozen_mixed_3.25` as joint winner; the per-layer map embedded in the released checkpoint and GGUF | Weight half stands; cache half unvalidated | The joint score's cache term was invalid. |
| E8P 2-bit tiered cache smoke (KL 0.030 on 64 tokens) | Plausible, not a claim | Ran on pinned 5.9.0 with recurrent state visible; too few tokens. |
| 59.34 % tensor-file reduction | Replaced by 58.26 % like-for-like | MTP head counted on one side only. |
| Exported artifact bytes, native GGUF conformance (200 projections) | Stand | Verified independently of the simulator. |

---

## 5. What needs to be improved and implemented in future generations

### 5.1 Immediately, before any cache claim

1. **Re-run the cache study with the fixed simulator.** Seed-0 matrix,
   three-seed validation, cross-context transfer and matched controls, all with
   the 8-bit endpoint row present and passing, `transformers==5.9.0` (or a
   later version re-verified by the regression test), and the affected raw
   JSONs archived in `research/results/` rather than only on Drive.
2. **Size the evaluation for the real effect.** Under the fixed simulator the
   K4/V4 control costs about 6×10⁻³ nats on the source model, so differences
   between 3.25-bpv recipes will be at the 10⁻³ level. Budget at least 10⁴
   evaluated tokens per seed, at least 20 prompts, and report document-level
   (cluster) bootstrap intervals, not token-level ones.
3. **Re-select and re-embed the deployment map**, or drop the mixed map in
   favour of uniform K4/V4 if the corrected differences are not resolvable;
   update `rotquant_config.json` deployment metadata, the native GGUF and the
   publication manifest accordingly.
4. **Confirm the environment of the archived runs** from
   `environment.library_versions.transformers` in the Drive JSONs and record
   it in the experiment log.

### 5.2 Evaluation methodology

- Add an **environment guard** to the runners: assert that the installed
  `transformers`/`torch` versions match the lock (or an explicit allow-list)
  and record the check in every result JSON.
- Generalise the **endpoint principle** to the weight pipeline: an
  identity-quantizer arm (rotation on, codes at 16 bits or `patch.enabled`
  with a no-op quantizer) must reproduce source logits to round-off before any
  weight recipe is compared; a **hybrid-model assertion** should require
  `non_kv_state_bytes > 0` whenever the model has linear-attention layers.
- Run the hybrid KV monotonicity test in CI on **both the pinned and the
  latest Transformers** so a cache-layout change is caught before a Colab run.
- Replace token-level bootstraps with **cluster bootstraps over documents**,
  report sample standard deviations (not population) for three-seed results,
  and never quote an interval below 20 paired samples.
- Move the hand-written cache notebooks into **generated notebooks** (as the
  algorithm-lab and optimization-stage notebooks already are) so pins, gates
  and archival are enforced by the builder scripts.
- Make the **fail-fast logit gate** (`eval.fail_fast`) the default for every
  stage so a catastrophic arm stops before long evaluation.
- Complete the missing external validity work already listed in the roadmap:
  full WikiText-2/C4 perplexity, a fixed zero-shot bundle, the 300-prompt
  domain suite with task outcomes, long-context perplexity and retrieval at
  8k–32k, matched GPTQ/AWQ/QuaRot/SpinQuant/QuIP#/AQLM/standard-GGUF baselines
  at exact bytes, and resident-memory and throughput measurements on the
  native packed path only.

### 5.3 Algorithms

- **QRAT (reserved name).** The future quantization-and-rotation-aware
  training method should train rotations, scales, signs and recovery
  parameters under simulated quantization at a genuine budget: block-wise
  reconstruction over thousands of sequences, then end-to-end distillation over
  at least 10⁶ tokens with validation on ≥5×10⁴ held-out tokens, the
  straight-through estimator carried through the rotation as well as the codes
  (the 2026-08-31 review's §2.3 question), periodic code refresh, shared
  rotations per input site, and the same fail-closed rollback and
  deployed-quantizer gates as the calibration stages. Expected outcome from
  the literature: W4 near-lossless and W3 viable, which would move the
  size/quality frontier more than any codebook refinement. The properly
  budgeted `configs/qwen35_4b_recovery_cuda.yaml` profile is its first
  candidate; none of the current stages should be labelled QRAT.
- **Factor ablation of the catastrophic bundled arm**, now that its
  confounds are removed: 8-bit scales, mean-bias correction, shared rotations,
  butterfly control, Hessian objective with the gate, flippable signs, each
  against the promoted W4 recipe with the fail-fast gate.
- **Mixed-precision allocation.** Replace the greedy per-byte downgrade with
  an exact multiple-choice knapsack (dynamic programming over integer byte
  budgets is cheap at 200 projections), score candidates with a commensurable
  Hessian-weighted objective, and constrain choices to kernel-deployable
  formats. Retain the random-allocation control.
- **Cache.** Keep the sink/recent tiers now that ageing is simulated;
  evaluate E8P against Gaussian at 2–3 bits with the endpoint; test
  per-channel key handling against rotation on the outlier channels of
  post-RoPE keys; move the selective-V oracle to packed-key scoring; run the
  8k–32k confirmations where the cache actually dominates memory.
- **Weight codebooks at ≤3 bits.** The scalar ceiling question (E6) is
  decidable only against a real packed vector or trellis method (E8P as a
  packed format, QuIP#, QTIP); the d=2 vector arm is nearly powerless by
  construction.
- **Retire or reframe arms** that cannot win as designed: the spherical
  codebook at d=128 (identical to Gaussian), the QJL weight sketch (keep as
  the E3 null control), per-row scales under block-diagonal rotation (use a
  full-row Kronecker rotation if the arm is to be tested), and E7's
  single-layer mismatched control (test fused residual-stream rotation with
  folded norms against online rotation instead).
- **Activation quantization.** Measure a native A8 GEMM before any W4A8
  speed claim; add a W4A4 arm so the second half of hypothesis E1 (learned
  rotations pull ahead once activations are quantized) becomes testable.

### 5.4 Engineering and protocol

- Any change to the 8-bit scale block layout or to reconstruction semantics
  requires a new packed-format version and native conformance vectors; the
  exactness fix above deliberately kept the layout.
- Record every claim's raw result JSON in the repository under
  `research/results/` with content-addressed names; the three-seed GPTQ
  promotion and every cache run currently exist only on Drive.
- Keep one implementation of each mathematical primitive
  (`rotate_hessian`, `encoded_storage_scales`) in `rotquant._internal` and
  import it, rather than duplicating it across modules.
- Add resident-memory and peak transient-memory probes to the native runtime
  benchmark so the storage figures gain their runtime counterparts.
