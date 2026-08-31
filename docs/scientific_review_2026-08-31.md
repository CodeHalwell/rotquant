# Scientific and mathematical review — 2026-08-31

Scope: full-repository review of the algorithmic and statistical content, aimed at
the project goal of **maximum size reduction at minimum quality loss**. Every
core module was read line-by-line (`rotate`, `codebooks`, `quantize`, `pack`,
`linear`, `patch`, `calibrate`, `train_rotation`, `block_train`, `kv_cache`,
`dynamic`, the eval protocol, runner, configs, and the experiment log). The
review is organized as: verified foundations, findings ranked by expected
impact, and directions worth exploring.

This is a review document, not a change set: nothing in library code was
modified.

> **Implementation follow-up (2026-09-01).** The actionable findings were
> implemented on `codex/scientific-review-implementation`: bounded-memory
> randomized sampling and disjoint calibration/evaluation rows; streamed GPTQ
> Hessians with actorder and scale refits; million-token recovery profiles with
> sparse teacher calls and resumable checkpoints; calibrated mean-bias
> correction; shared/joint projection rotations; a trace-exact Hessian rotation
> objective; blockwise uint8 scale metadata; fp16 sink/recent KV storage;
> finite-rate E8P weight/KV codebooks; per-token W4A8 semantics; learned hard
> sign vectors; and paired bootstrap intervals. The analysis below is retained
> as the dated rationale, so statements such as “currently absent” describe the
> reviewed 2026-08-31 tree rather than the follow-up implementation.

---

## 1. What was checked and holds up

These were verified by direct derivation against the code, not assumed:

- **FWHT** (`rotate.py:82`): the iterative pairing reproduces the orthonormal
  Walsh–Hadamard transform; `H/√d` is its own inverse.
- **RandomizedHadamard inverse** (`rotate.py:204`): with `R^T = D_s H`,
  `inverse_activation(x) = FWHT(x)·s = x R` is exactly right.
- **ButterflyRotation** (`rotate.py:209`): the stage matrix
  `[[c, s], [s, -c]]` is orthogonal and self-inverse; at θ=π/4 each stage is the
  normalized Hadamard butterfly, so the initialization equals
  `RandomizedHadamard` exactly, and the reversed-stage inverse is correct. The
  parameter count `d/2·log2(block)` matches the implementation.
- **Cayley map** (`rotate.py:408`): `solve(I+A, I−A)` with skew `A` is
  orthogonal; caching/invalidations are handled correctly.
- **GPTQ** (`quantize.py:652`): the upper-Cholesky-of-`H⁻¹` recursion, the
  `err = (col − q)/Hinv[i,i]` update, and the lazy block propagation match the
  reference implementation; `H' = R H Rᵀ` (`patch.py:252`) is the correct
  Hessian basis change for rotated inputs.
- **QJL sketch estimator** (`quantize.py:758`, `linear.py:205`): the
  asymmetric estimator `√(π/2k)·‖r‖·(xG)·sign(rG)` is exactly unbiased for
  Gaussian G at any angle — the constant, the `1/√k` column normalization, and
  the sign-side-only design are all correct. (See §2.1 for the variance
  problem.)
- **Spherical marginal** (`codebooks.py:121`): `(1−u²)^{(d−3)/2}` is the exact
  coordinate density of a uniform point on `S^{d−1}`, and the `√d` scaling
  gives unit variance. The d=3 uniform-coordinate special case
  (`tests/test_source_coding.py:45`,
  `test_spherical_codebook_matches_uniform_sphere_in_three_dimensions`) is the
  right analytic anchor.
- **Lloyd–Max anchors**: 0.1175 (2-bit) and 0.0345 (3-bit) are the classic Max
  (1960) values; the Panter–Dite constant `√3·π/2` in `turboquant_mse_bound`
  is correct as a high-rate approximation (see §2.11 for framing).
- **D8/E8 nearest-point** (`codebooks.py:517`): the round-then-fix-worst
  coordinate decoder and the `D8 ∪ (D8+½)` union are the standard exact
  algorithms.
- **Bit packing** (`pack.py`): LSB-first packing via `scatter_add_` is correct
  because per-word bit ranges are disjoint (add ≡ or); spill handling for codes
  straddling word boundaries is right for bits ≤ 16.
- **Bit accounting**: stored-bytes-based `BitBudget`, fp16 scale rounding
  *before* index assignment (`_storage_scales`), partial-group handling in
  `_group_scales_rms` and the padded-slot masking in the MSE scale search are
  all careful and honest. `compare_quantizers` enforcing matched
  `effective_bpw` is exactly the right discipline.
- **Value-basis attention** (`kv_cache.py`): storing V rotated and applying one
  inverse rotation to the attention output is exact by linearity — a genuinely
  nice trick that avoids per-token inverse rotations.
- **KV rotation training** (`train_kv_rotations`, `kv_cache.py:446`): the STE
  fake-quant (`_fake_quant`, `kv_cache.py:412`) composed with rotations applied
  *differentiably* on both Q and K/V sides (`_fake_rotquant_attention`,
  `kv_cache.py:434`) is the right gradient structure — both sides carry
  gradient through the standard STE surrogate (contrast with §2.3).
- **Methodology**: the experiment log records negative results, protocol bugs
  (the seed-0 uniform/dynamic evaluation-batch mismatch), endpoint-equivalence
  checks (dynamic-at-8-bit ≡ uniform-8-bit), matched-control corrections, and
  predeclared gates. The GGUF export was verified byte-for-byte against the
  producer. This is unusually good hygiene for a research repo.

---

## 2. Findings (ranked by expected impact on the size/quality frontier)

### 2.1 The TurboQuant weight-sketch correction adds ~πd/2k more error than it removes

`error_comp="turboquant"` stores `k=64` sign bits + a norm per output row and
corrects the inner product at inference (`linear.py:205`). The estimator is
unbiased, but unbiasedness is not the objective — squared error is.

Per output unit, without correction the missing term is `x·r` with
`E[(x·r)²] ≈ ‖x‖²‖r‖²/d` for near-isotropic rotated activations
(d = in_features). With correction, the error becomes zero-mean noise with
variance `≈ (π/2k)·‖x‖²‖r‖²`. The ratio is

```
E[err²_with] / E[err²_without] ≈ π·d / (2k)
```

For d = 4096 and k = 64 that is ≈ **100×**. The sketch converts a small,
partly systematic error into a large zero-mean one. It can only win if the
activation–residual correlation is ~10× above the random-direction level, or
if `k` is scaled to O(d) — which defeats the storage savings.

This is not a coding bug; it is a scale mismatch when transplanting a KV-cache
technique (TurboQuant/QJL operate at `d = head_dim = 128`, where `k ≈ d/2` is
affordable and the estimate is accumulated across many positions) to weight
rows at `d = 4096`. Note also that the project's own E3 hypothesis —
*deterministic beats stochastic residuals at equal bits* — applies to this arm:
the sketch **is** a stochastic residual code.

Recommendations:
- Put this variance analysis next to the e3b config and pre-register the
  expectation that the arm loses for weights at k ≪ d; keep it as a null
  control rather than a candidate.
- The *systematic* part of the bias it targets can be removed exactly on the
  calibration distribution: fold the calibration-mean term into a bias vector,
  `b += μ_rot · (W_rot − Q)ᵀ` where `μ_rot = E[x_rot]` (one extra running
  mean in `HessianAccumulator`). This is classic bias correction (Banner et
  al., 2019). It adds no storage when a bias already exists; a biasless layer
  needs `out_features` extra values plus either a fused epilogue or an explicit
  addition. It dominates the sketch for the calibration-mean component, but
  the correction can become stale under distribution shift.
- If the sketch stays, the right home is the KV cache (d = 128), not weights.

### 2.2 Per-row "turboquant" scale is inconsistent with block-diagonal rotation

The rationale in `quantize.py:466` ("after Hadamard rotation the distribution
shape is universal, one scale per row suffices") assumes the rotation mixes the
entire row. But `RandomizedHadamard`/`ButterflyRotation` are **block-diagonal**
(block = 128): each 128-wide block's energy is exactly preserved, so a row's
inter-block energy variation — which is large in real LLM weights
(outlier input channels) — survives rotation untouched. A single per-row scale
then systematically mis-scales blocks with atypical energy, and the Gaussian
codebook premise fails at the row level even though it holds within a block.

This doesn't invalidate E4b as an experiment, but it means the arm as
implemented tests "per-row scale + 128-block rotation", which is strictly
weaker than TurboQuant's actual setting, and the expected loss vs per-group
scales is larger than the theory being cited predicts.

Recommendations:
- For the per-row-scale arm, use a full-row rotation when possible
  (4096 = 2¹²; for non-power-of-two dims like 11008 = 172·64, a Kronecker
  `H_172 ⊗ H_64` construction as in QuaRot restores full mixing).
- Alternatively state the arm as "per-row scale under block mixing" and expect
  a penalty scaling with the inter-block energy variance.
- Note the default config (`group_size = block = 128`, aligned) is exactly
  self-consistent — that alignment is worth calling out as a design invariant.

### 2.3 Block-training gradients are taken against a frozen-weight surrogate

In `FakeQuantButterflyLinear._assigned_weight` (`block_train.py:440`), the
entire weight side — rotation of W, scale selection, quantization — is computed
under `no_grad`; θ receives gradient only through
`rotate_activation(x)`. The layerwise activation objective in
`train_rotation.py:193` has the same structure. To be precise, nothing is
wrong in autograd terms: once indices and scales are detached, the dequantized
matrix is a constant, and the activation-only path **is** the exact gradient
of that frozen-assignment surrogate. With fixed scales it is also the a.e.
local derivative of the piecewise-constant quantized map. The potential gap is
at finite optimizer steps: the next forward re-quantizes from the new θ, so
assignments can jump, and detached scale changes are omitted.

Why this may matter: at the current θ, write the dequantized matrix as
`q̄ = W R(θ)ᵀ + E`. Against the full-precision target, the prediction residual is
`e = x R(θ)ᵀ Eᵀ = O(‖E‖)`. With `q̄` frozen, the *prediction Jacobian*
contains an O(‖W‖) component,
`x·dRᵀ/dθ·(W Rᵀ)ᵀ`, plus an O(‖E‖) component. The squared-loss gradient,
however, is `2 Jᵀ e`, so it is O(‖W‖‖E‖), not O(1), and it vanishes in the
unquantized limit. The potential mismatch is therefore not a non-vanishing
gradient at `E = 0`; it is that this local derivative cannot anticipate
assignment jumps after an optimizer step (or the omitted scale derivative),
while its O(‖W‖) Jacobian component can dominate the finite-error update.
Whether that behavior is harmful is an empirical question and cannot, by
itself, explain the 12 → 18 step result.

A standard alternative already used elsewhere in this repo
(`_fake_rotquant_attention`, `kv_cache.py:434`, via the `_fake_quant` STE at
`kv_cache.py:412`) is the straight-through estimator with the rotation
differentiable on the quantized side too:

```python
rotated = self.rotation.rotate_weight(self.weight)     # differentiable
with torch.no_grad():
    q = quantize(rotated.detach())
return rotated + (q - rotated).detach()                # STE
```

Under STE the identity-aligned component cancels in the prediction Jacobian and
the surviving term is `x·dRᵀ/dθ·Eᵀ = O(‖E‖)`. For a squared reconstruction
loss, its gradient is therefore typically O(‖E‖²) near the unquantized limit.
To be clear, STE is itself a surrogate (`dq/dv := I` is not the a.e. derivative
of a piecewise-constant quantizer), so neither scaling argument establishes
which surrogate optimizes deployed quality better; the question is empirical.
The likely reason for the current design is memory
(backprop through the butterfly stages of an [out, in] weight stores per-stage
intermediates); gradient checkpointing over stages, or chunking rows, makes
the STE version affordable. Run both on OPT-125M and quantify the gap — if STE
gives a materially better deployed PPL at the same steps, this is a cheap win
for every learned-rotation arm. (A second, smaller gap in the frozen
surrogate: with `assignment_scale: rms` the deployed scales are a continuous
function of θ, and freezing them drops that term as well.)

The **weight-MSE** objective in `train_rotation.py` is unaffected (its
gradient flows through `rotate_weight` and is a proper Lloyd-style
majorize-minimize step). One footnote there: RMS scales are re-computed and
then frozen each step; since RMS is not the argmin of the MSE objective, the
envelope argument doesn't cover the scale term and the gradient carries a small
bias. Harmless given best-checkpoint restore, but worth a comment.

### 2.4 The LoRA-QAT / distillation "negative results" are underpowered, not negative

The checked-in base recovery configs train on ~**256–512 tokens** total
(`distill_train_batches: 2` × 128–256-token sequences), for ≤ 12 steps, with
validation on 1–2 batches and early-stopping patience 3–4
(`configs/qwen35_4b_lora_qat_cuda.yaml`, `e1f`, `e1g`). The executed Colab
matrix did go further: its medium rank-4 profile used 8 training batches and
24 steps, while the large rank-4 and rank-8 profiles used 16 batches and 32
steps, with 2–4 validation batches. Those profiles are stronger than the base
configs but still train ~4M adapter parameters on only a few thousand unique
sequence tokens, repeatedly reused across steps, with a noisy validation gate.
"Best step 0" is therefore unsurprising, but not predetermined. The
literature that establishes QAT recovery at these bit-rates (EfficientQAT,
LLM-QAT, ApiQ) uses 10⁶–10⁷ tokens of block-wise then end-to-end training.

The experiment log correctly refuses to claim adapters help, but the framing
("rank 4, more data, more steps all failed") risks hardening into a conclusion
the data cannot support. Given the project goal, this is the single largest
untapped quality lever: published W4-g128 QAT results are near-lossless, versus
the current +4.7% mean PPL, and W3 typically becomes viable — which would beat
uniform W4 on the size/quality Pareto outright.

Recommendation: one properly-budgeted run — block-wise reconstruction with a
few thousand sequences, then end-to-end KD at ≥ 1M tokens, validation on ≥ 50k
tokens — before recording any "recovery doesn't help" decision. The
`propagate_quantized_inputs` machinery is already the right skeleton.

### 2.5 GPTQ never made it into any release recipe

E5 exists to show "GPTQ helps with real activations", the implementation is
correct, and yet every Qwen recipe (`qwen35_4b_*.yaml`) ships
`error_comp: none` — the released W4 artifact is round-to-nearest with
MSE-searched scales. GPTQ at W4/g128 reliably recovers a large fraction of the
RTN gap at zero inference cost and zero extra bits; combined with rotation it
is the QuaRot recipe. The blocker appears to be Hessian memory (~25 GB noted in
the README) — but Hessians can be accumulated per-layer sequentially (hook one
layer at a time, or stream by block) at the cost of extra forward passes, which
the roadmap's layer-streaming item already contemplates.

For the stated aim, this is the highest-confidence, lowest-novelty improvement
available: it composes with everything else in the repo and would likely
reclaim a substantial part of the +4.7%.

Two implementation notes for when it is used:
- No `actorder` (activation-order) support: quantizing columns in decreasing
  `diag(H)` order is a one-line permutation that measurably helps at ≤ 4 bits.
- Group scales are selected from the *original* rotated weight before error
  feedback (`quantize_weight` → `select_scales(w)`), whereas reference GPTQ
  re-derives scales per group from the *updated* W as it sweeps; and
  `mse_search` optimizes plain-rounding MSE, not the GPTQ objective. Both are
  small systematic disadvantages for the E5 arm as implemented.

### 2.6 Calibrated-codebook sampling can alias badly

`_calibration_sample_indices` (`quantize.py:56`) samples the flattened,
normalized weight at evenly spaced indices. For a 4096×4096 matrix with the
default 65,536 samples the stride is ≈ 256, so sampled columns cluster around
multiples of 256, and each column is only ever seen from a narrow band of rows
(the slow drift term couples column identity to row position). Under FWHT with
random signs, columns are approximately exchangeable and the damage is limited;
but for *trained* butterflies, non-square matrices, or any weight with
column-periodic structure, the Lloyd fit sees a biased sample.

Fix: use a seeded bounded-memory sample without replacement, such as Floyd's
algorithm or a keyed permutation over only the selected indices. A full
`randperm(total)[:count]` removes aliasing but allocates O(total) int64 storage
(~128 MiB for 4096² and ~344 MiB for a 4096×11008 projection) to select only
65,536 entries, and scales poorly to the planned 27B model.

Related inconsistency: `_fit_calibrated_codebook` normalizes by **RMS** scales,
but deployment may quantize with `mse_search` scales (0.5–1.5×RMS), so the
fitted grid is matched to a distribution the deployed quantizer never sees.
Either fit under the deployed scale rule, or alternate (fit grid → refit
scales → refit grid) for a few rounds.

### 2.7 Data-disjointness relies on `skip` counts that don't commute across loaders

`build_calib_loader` counts *eligible* documents (length ≥ seq_len) when
applying `skip`. Different stages use different seq_lens, so eligibility
differs: the KV evaluator (`skip: 384` at seq_len ≈ 137) can still draw
documents whose prefixes fed Hessian/block calibration (`skip: 0` at
seq_len 2048), because the first 128 ≥2048-token documents are scattered
arbitrarily deep into the ≥137-token ordering. The manifests already record
`source_rows`, so this is checkable — but it isn't checked.

Fix: a runner-level assertion that calibration-stage and evaluation-stage
`source_rows` sets are disjoint (and fail loudly otherwise). The competitive
data pipeline already does exact/near leakage checking; the internal harness
should get the cheap version of the same guarantee.

Also worth noting: requiring documents ≥ seq_len biases calibration toward long
C4 documents (a different content mix than average text). Concatenating short
documents (GPTQ-style) or sampling windows would remove the bias.

### 2.8 Activation capture is position- and document-biased

`collect_activations` (`calibrate.py:139`) keeps the **first** `max_tokens`
tokens seen. With `e1b`'s `rotation_n_calib: 1`, training tokens are positions
0–63 and "disjoint" selection tokens are positions 64–127 **of the same single
document**. Early positions are exactly where attention-sink / massive-
activation anomalies live, and same-document selection is highly correlated
with training. The layer-local gate is therefore much weaker than it looks.

Fix: reservoir-sample tokens across batches and positions (seeded), and draw
selection tokens from different documents than training tokens.

### 2.9 Dynamic weight allocation compares incommensurable scores across layers

`_local_error` (`dynamic.py:184`) is a *relative* output MSE, normalized per
layer. The greedy knapsack then compares `Δscore/Δbytes` across layers — but a
relative error of 1e-3 on `down_proj` and on a small projection are not the
same damage to the network. The global-KL term is commensurable; the local term
is not, and with `global_kl_batches: 0` the allocation runs on the local term
alone. This plausibly contributes to the observed result that dynamic weight
allocation lost to uniform W4 (+2.0% PPL for 0.098% bytes).

Fixes, cheapest first: compute squared error summed across output dimensions
and averaged across calibration examples, then apply at most one normalization
shared by all layers. Merely dropping the per-layer energy denominator is not
enough if the implementation still takes a mean over `out_features`; multiply
that mean by the output width. Alternatively, use a Hessian-weighted proxy with
the statistics the repo already knows how to collect. The exact per-layer
expected output-MSE contribution is `tr(ΔW·H·ΔWᵀ)` and needs the full `H`
(already computed on GPTQ runs); the cheap variant
`Σᵢ Hᵢᵢ·‖ΔW[:,i]‖²` uses only `diag(H)` and is the standard diagonal
approximation. Either is commensurable across layers, unlike per-layer
relative MSE. The additive-interaction caveat and the uniform-restore guard
are already handled well.

### 2.10 E7's "mismatched" mode is a straw man

`mode: "mismatched"` rotates the weight and feeds the raw activation
(`patch.py:94`) — the layer then computes `x R Wᵀ ≠ x Wᵀ`, which is wrong for a
*single* layer; no cross-layer mechanism is needed to predict catastrophic
output. The scientifically interesting consistency question in the
QuaRot/SpinQuant lineage — whether rotations can be *fused* through RMSNorm and
residual streams without runtime cost, and what breaks when γ-folding is
approximate — is designed away by this repo's per-layer online rotation.
That design choice is legitimate (and the llama.cpp integration pays the
rotation honestly), but E7 as implemented confirms a triviality. Either reframe
E7 as a sanity check, or test the interesting version: fused residual-stream
rotation with folded norms vs online rotation, at matched bits.

### 2.11 Smaller technical points

- **`normal_float` is not bitsandbytes NF4** (`codebooks.py:236`): the real NF4
  is asymmetric with an exact zero (8 negative / 7 positive levels + 0); the
  implementation here is symmetric with no zero at even level counts. Fine as
  "an NF-style grid", but the docstring's "mirrors the bitsandbytes
  construction" overstates it, and comparisons labelled "NF" will differ
  slightly from the deployed NF4 datatype.
- **`turboquant_mse_bound`** (`codebooks.py:34`) presents the Panter–Dite
  high-resolution asymptotic as a bound ("Theorem 1 … MSE ≤"). At b = 2 the
  true Lloyd–Max MSE (0.1175) is well below the formula (0.170), so it happens
  to upper-bound at small b, but it should be labelled a high-rate
  approximation with the ≈2.72× Gersho ratio, not a theorem-grade bound.
- **E9's premise is broken by its own config**
  (`configs/e9_spherical_length.yaml`): the spherical codebook is the exact
  marginal *for unit-RMS normalization*, but the config pairs it with
  `scale: mse_search` (0.5–1.5×RMS), off-distribution by construction. Run the
  clean test with `scale: rms`. Separately, at d = 128 the spherical and
  Gaussian grids agree to ~4 decimals (the repo's own screening table shows
  this) — concentration of measure makes the null result predictable. If the
  finite-d effect is the point, test at d = 16–32 (where the correction is
  measurable) or in the KV path with sub-head groups; otherwise this arm can be
  retired.
- **`codebook_dim` defaults to `group_size` even under per-row scaling**
  (`quantize.py:391`): with `scale: turboquant` the normalization dimension is
  `in_features`, not `group_size`, so the spherical marginal is built for the
  wrong d. Cosmetic today (see previous point), but wrong if small-d spherical
  ever matters.
- **QJL naming collision**: `error_comp="qjl"` (the legacy stochastic 1-bit
  residual, the designated E3 loser) and `error_comp="turboquant"` (the
  sign-sketch inner-product corrector) are both called "QJL" in different
  docstrings. A one-line glossary would prevent misreading results.
- **`mean_first_divergence` duplicates `mean_matching_prefix`** — the same
  number is returned under both names, in the per-prompt metrics
  (`eval/trajectory.py:137-138`) and again in the aggregate return dict
  (`eval/trajectory.py:147-148`); the name suggests a different statistic.
- **Butterfly/rotation sign vectors are stored as fp32** (16 KB/layer for 1 bit
  of information per element, also in the GGUF layout). Negligible at current
  scale; inelegant in a format spec that otherwise counts bits honestly.
- **Baseline bpw is nominal** (`baselines/run_baseline.py` records `bits`,
  `group_size`): GPTQ/AWQ artifacts carry zero-points (asymmetric) so their
  true bpw at "4-bit/g128" is ≈4.25 vs RotQuant's 4.125. The competitive
  pipeline's exact-byte matching solves this for GGUF; the internal baseline
  runner should record actual artifact bytes too, or the equal-bits discipline
  quietly favors whichever side has less metadata.
- **KV evaluation contexts are short**: the "long-context" confirmation is
  1,024 tokens with 32–64 continuation tokens. KV-quantization error compounds
  with depth and cache length; the frozen 3.25-bpv map's transfer to 8k–32k
  contexts (where a KV cache actually dominates memory) is untested. The log
  says this honestly — flagging it here because the *value proposition* of KV
  compression lives at exactly the lengths not yet measured.
- **Statistical reporting**: seed-level means ± std are used well, but most KL
  comparisons ride on 64–128 continuation tokens. The paired token-level
  bootstrap machinery already exists in `eval/competitive_run.py`; wiring it
  into the internal KV/logit comparisons (paired per-token deltas, 95% CI)
  would make the 10–20% KL-improvement claims robust at near-zero cost.

---

## 3. Directions worth exploring

Ordered roughly by expected return on the size/quality frontier.

1. **A real QAT budget (see 2.4).** Block-wise reconstruction + end-to-end KD
   at ≥10⁶ tokens. Expected: W4 near-lossless, W3 viable — the largest single
   movement of the frontier available with known techniques.
2. **GPTQ in the product recipe (see 2.5)** with layer-streamed Hessians and
   actorder. Composes with rotation, costs nothing at inference.
3. **Calibrated bias correction (negligible amortized overhead):** fold
   `μ_rot·(W_rot − Q)ᵀ` into each layer's bias. Existing biases cost nothing
   extra; creating one requires `out_features` fp16 values (≈ 0.004 bpw) and a
   fused epilogue or explicit addition. This removes the calibration-mean
   component of quantization error that the sketch arm chases with 100× the
   noise, subject to distribution shift.
4. **Share rotations across same-input projections.** q/k/v (and gate/up)
   consume the same activation but currently get independent per-layer seeds —
   the same x is FWHT-rotated three times with three different rotations per
   block. One shared rotation per input site is a ~3× runtime-rotation saving,
   simplifies fused kernels, and removes a needless difference from the fused
   QuaRot setting. (Per-site seeds still differ across depth, so the
   variance-reduction argument for independent seeds survives.)
5. **Hessian-weighted rotation training.** With consistent rotation, the layer
   output MSE is exactly `tr(Δ·RHRᵀ·Δᵀ)` for weight-space error Δ. The repo
   already collects H for GPTQ; using it as the rotation-training objective
   gives activation-aware training with no stored activations, no
   position-bias (2.8), and none of the gradient pathology of the replay path
   (2.3) — the objective is a deterministic function of θ.
6. **Double-quantize the scales.** fp16 scales are 0.125 bpw at g128 and
   0.25 bpv at the KV g64 — i.e. ~8% of the entire 3.25-bpv cache budget.
   QLoRA-style 8-bit scale quantization with a per-tensor fp scale halves that
   for ~0 quality cost. The config already anticipates this
   (`scale_bits` rejects <16 "until a real codec exists") — it is worth
   building; at W2/W3 it is a bigger lever than any codebook refinement.
7. **KV: keep sinks + a recent window in fp16.** The retrieval oracle already
   models sink/recent sets, but the deployed cache quantizes every position.
   Attention sinks and the newest tokens are precisely where K-quantization
   noise hurts most (KIVI/KVQuant findings); exempting ~4 + 32 positions costs
   O(1) bytes regardless of context length and typically buys back most of the
   K2/K3 loss. This also interacts with 2.11's long-context gap: it is the
   cheapest way to make low-bit K survive depth.
8. **Make E6 a fair fight.** At d=2, vector quantization's granular gain over
   an optimal scalar grid is ~0.17 dB of a 1.53 dB asymptotic ceiling, and
   rotated coordinates are near-independent so there is no memory/shape gain to
   harvest — the arm is nearly powerless by construction. The repo already has
   `nearest_e8`; a finite-rate E8-based codebook (E8P as in QuIP#, 2 bits/dim
   at d=8) or the planned QTIP/trellis baseline is where the scalar-ceiling
   hypothesis (E6) can actually be decided.
9. **Vector-quantize the KV cache instead.** d = head_dim = 128 with per-token
   vectors is TurboQuant's home turf: k ≈ d sketches are affordable, VQ
   codebooks amortize across all positions, and the accumulation over cache
   length averages the added variance (unlike weight rows). Several findings
   above (2.1, 2.11 spherical-d) point the same direction: the finite-d and
   sketch machinery in this repo is better matched to the cache than to
   weights.
10. **Activation quantization decision.** The roadmap already flags it; worth
    reinforcing that E1's central hypothesis ("learned rotations only pull
    ahead at W4A4") is currently untestable — no activation quantizer exists in
    the codebase — so the learned-rotation arms can only ever confirm the null
    half of E1. A minimal W4A8 path (per-token int8 activations after the
    existing rotation) would make E1 falsifiable and unlock compute-bound
    prefill wins.
11. **Learned sign vectors.** The ±1 diagonal before the FWHT is random and
    stays exactly orthogonal for any sign assignment — it is a discrete
    trainable parameter (annealed flips or a straight-through sigmoid) with
    zero storage delta (signs are already stored) and zero runtime delta.
    A cheap, novel-ish middle ground between fixed FWHT and butterfly angles.

---

## 4. Bottom line

The engineering rigor (bit accounting, held-out gates, manifest hashing,
negative-result logging, byte-verified export) is well above the norm and the
core math is implemented correctly almost everywhere it matters. The dominant
scientific risks are concentrated in four places: a sketch correction whose
variance analysis was never done (2.1), rotation training against a
frozen-assignment surrogate that cannot anticipate assignment jumps (2.3),
historical recovery experiments whose budgets were far below serious QAT scale
(2.4), and the strongest known PTQ tool sitting unused in the reviewed release
recipes (2.5).
Addressing 2.4 + 2.5 alone plausibly moves the headline result from
"W4 at +4.7% PPL" toward "W4 near-lossless / W3 at similar loss", which is the
project's stated objective.
