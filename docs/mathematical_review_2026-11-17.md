# Mathematical review and inference optimisation — 17 November 2026

Review of `a926184` against two aims: a rotation-based algorithm for LLM
weights that is as close to lossless as the mathematics allows, and inference
that is as fast as the representation allows.

Everything below was checked by execution in this checkout unless it says
otherwise. Real-weight measurements use `Qwen/Qwen2.5-0.5B` (BF16 source,
hidden size 896 = 7 × 128) on a four-core CPU. No GPU was available, so no
CUDA or Metal kernel was run and nothing here is a GPU speed result.

## 1. Verdict

- **The core mathematics is sound.** The rotation invariant, the Hessian
  transform, GPTQ, the QJL estimator, the refit algebra, packing, KL,
  perplexity, the allocator's dynamic programme and the ternary path were
  each re-derived and checked numerically (section 3).
- **"Lossless" needs splitting into two statements.** The rotation is an exact
  reparameterisation and every stage around the quantiser can be made exact.
  The quantiser itself cannot: at `b` bits per weight no method can beat a
  relative error of `4^-b`, and the scalar grids used here sit 2.4–2.7 times
  above that. A rotation does not help a *truly* lossless coder; it makes the
  output larger (section 2).
- **Eleven defects were fixed**, none of which invalidates an archived
  result. The largest in weight error: the Gaussian Lloyd-Max grids were not
  converged at five bits and above (27% excess MSE at eight bits), the fitted
  ("calibrated") grids were 12% and 64% off at five and six bits, and the
  finite E8 codebook was asymmetric with a non-zero mean (section 4). At the
  model level the Gaussian fix made no difference that eight seeds could
  resolve (section 2.4); it is a correctness fix, not a quality gain.
- **Seed variation is the practical limit on small claims.** On the 0.5B
  model the same recipe varies by 10–14% in KL between rotation seeds
  (section 2.4).
- **CPU inference is 1.9–2.9× faster for one token and 7–9× faster for a
  batch**, with the same result up to floating-point reassociation
  (section 5.1). The GPU gap to a conventional Q4 file is explained by the
  kernel structure, and the exact changes that close most of it are listed in
  order (section 5.2).

## 2. What "lossless" can mean

### 2.1 The floor under any quantiser

For a unit-variance Gaussian coordinate coded at `b` bits, the rate-distortion
function gives `D >= 4^-b`. After a mixing rotation the normalised weights are
close to that source, so this is the floor for every method in this
repository, whatever the codebook or training.

| Bits | Shannon floor `4^-b` | Exact Lloyd-Max | Ratio to floor | Grid used before this review | Its excess |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.5000e-1 | 3.63380e-1 | 1.454 | 3.63380e-1 | 0.00% |
| 2 | 6.2500e-2 | 1.17482e-1 | 1.880 | 1.17482e-1 | 0.00% |
| 3 | 1.5625e-2 | 3.45478e-2 | 2.211 | 3.45478e-2 | 0.00% |
| 4 | 3.9063e-3 | 9.50101e-3 | 2.432 | 9.50105e-3 | 0.00% |
| 5 | 9.7656e-4 | 2.50467e-3 | 2.565 | 2.50648e-3 | 0.07% |
| 6 | 2.4414e-4 | 6.44240e-4 | 2.639 | 6.48244e-4 | 0.62% |
| 7 | 6.1035e-5 | 1.63478e-4 | 2.678 | 1.72217e-4 | 5.35% |
| 8 | 1.5259e-5 | 4.11851e-5 | 2.699 | 5.22933e-5 | 26.97% |

The ratio tends to `sqrt(3)·pi/2 = 2.72`, the Panter–Dite constant quoted in
`turboquant_mse_bound`. So a 5-bit scalar code reproduces a rotated weight
with 0.25% relative squared error at best, and nothing fixed-rate and scalar
can do better. That is small, not zero.

### 2.2 What is exactly lossless, and what was made so

| Stage | Status | Evidence |
|---|---|---|
| Matched weight/activation rotation, in exact arithmetic | Identity | `xR^T (WR^T)^T = xW^T`; float64 error 2e-14 for FWHT |
| The same rotation in floating point | **Not** bit-exact | rotate then inverse-rotate returns 22% of FP32 values and 65% of FP16 values unchanged; relative error 1.1e-7 and 2.0e-4 |
| Integer lifting rotation (`rotquant.lossless.integer_butterfly`) | Bijection | 100% of bits returned for FP16/BF16/FP32 weights, any butterfly angles |
| Packing codes into words | Bijection | `tests/test_pack.py` |
| Entropy coding of the codes (`rans_encode_rows`) | Bijection | bit-identical `PackedTensor` after decode |
| Quantisation | Lossy | section 2.1 |

A butterfly reflection is a sign flip followed by a plane rotation, and a
plane rotation is three shears `x += round(p·y)`. Each shear changes one
coordinate by a quantity computed from the other, so subtracting the same
quantity undoes it whatever the rounding. The lifting multipliers are 24-bit
fixed-point integers, so the transform is defined by integer arithmetic alone.
`floats_to_fixed_point` supplies the exact integer view of a float tensor
(common unit = the last place of the smallest magnitude; negative zeros kept
aside).

### 2.3 A rotation does not help truly lossless compression

With the exact integer rotation in hand the question can be measured. The BF16
weights were coded two ways, both exactly invertible: directly, as two byte
planes with an order-0 model, and after the integer block-Hadamard rotation.

| Matrix | Direct | After rotation | Raw |
|---|---:|---:|---:|
| `layers.0.self_attn.q_proj` | 10.89 bits/weight | 26.90 | 16 |
| `layers.11.mlp.gate_proj` | 10.69 | 29.38 | 16 |
| `layers.11.mlp.down_proj` | 10.72 | 28.26 | 16 |
| `layers.20.self_attn.o_proj` | 10.75 | 27.49 | 16 |

Direct coding saves a third (the sign and exponent are compressible, the
mantissa is not), in line with published lossless coders for BF16 models.
Rotation turns the saving into a 70–85% *expansion*. The reason is general:

- an invertible transform cannot lower the joint entropy;
- a mixing rotation makes every coordinate Gaussian with equal variance, the
  maximum-entropy marginal, so the sum of marginal entropies a scalar coder
  pays goes up;
- floating point is fine near zero and coarse for large values. After mixing,
  every coefficient must be kept to the precision of the smallest weight.

Rotation helps fixed-rate *lossy* coding for the same reason it hurts lossless
coding. A lossless rotation-based compressor for LLM weights is therefore not
a goal worth pursuing; the lossless work that pays is in section 2.5.

### 2.4 How close to lossless the model is

Every linear layer of Qwen2.5-0.5B was quantised without calibration data
(FWHT 128, groups of 128, `mse_search`, exact Gaussian grid, vocabulary left
in full precision) and scored against the FP32 model on 16 × 127 next-token
predictions over text from this repository's documentation. One rotation
seed, 41-point scale search:

| Recipe | Weight NMSE | KL(full precision ‖ quantised), nats | Top-1 agreement | ΔNLL |
|---|---:|---:|---:|---:|
| W4 | 7.7e-3 | 0.1883 | 74.1% | +0.160 |
| W5 | 1.9e-3 | 0.0864 | 86.2% | +0.083 |
| W6 | 4.8e-4 | 0.0237 | 92.7% | +0.019 |
| W8 | 3.1e-5 | 0.0012 | 98.0% | +0.0004 |

Read this as the size of the effect, not as a target: a 0.5B model is far more
sensitive than the 4B model the project evaluates, the text is technical, and
no GPTQ, refit or recovery is applied. It does show that five bits is not
lossless for a small model, that each extra bit divides KL by roughly four
between five and eight bits, and that KL follows the weight error closely.

**Run-to-run variation is large, and it decides what can be claimed.** The
same recipe with a different rotation seed gives a different realisation of
the quantisation error. At W5 over eight seeds (11-point scale search):

| Grid | KL per seed 0–7 | Mean | SD across seeds |
|---|---|---:|---:|
| exact | 0.0685, 0.0915, 0.0797, 0.0784, 0.0710, 0.0741, 0.0685, 0.0740 | 0.0757 | 0.0076 |
| earlier grid | 0.0652, 0.0738, 0.0652, 0.0741, 0.0755, 0.0982, 0.0748, 0.0860 | 0.0766 | 0.0109 |

The paired difference is −0.0009 ± 0.0049 (standard error): no difference.
After the first three seeds the earlier grid had looked 15% better in every
one of them, which is what chance looks like at n = 3 with a seed standard
deviation of 10–14%. At W8 three seeds gave +0.0006 ± 0.0010, again nothing.

Two consequences. The codebook fix is a correctness fix: it lowers 8-bit
weight error by 7–17% and has no model-level effect that this measurement
can resolve. And any comparison of two recipes on a model of this size needs
on the order of eight seeds before a difference below about 15% in KL means
anything; the prompt-level standard error (0.004 here) does not capture this,
because it holds the quantisation realisation fixed.

### 2.5 Where the remaining loss is, in order of size

Measured on real rotated weights (FWHT 128, groups of 128):

| Lever | Effect on weight NMSE | Cost | Status |
|---|---|---|---|
| Codebook shape at 3–6 bits (exact Gaussian, spherical d=128) | within ±1% | none | exhausted; see below |
| Exact instead of unconverged grid at 8 bits | −7% (`mse_search`), −17% (`rms`) | none | fixed |
| Entropy-coding the existing codes | none (bit-exact) | 2.5–5.4% fewer code bytes | coder provided, not yet in the checkpoint format |
| Uniform grid + entropy coding, equal coded rate | −6.7% (W4), −9.5% (W5), −10.1% (W6) | 8-bit codes in memory, decode at load | measured, not implemented |
| The same at equal *file* size against today's fixed-rate file | about −26% at W5 | as above | derived from the two rows above |
| Calibration (GPTQ, activation refit, recovery) | model-dependent | calibration data | already in the repository |

Codebook shape (six matrices, aggregate NMSE, relative to the earlier grid):

| | `rms` exact | `rms` spherical-128 | `mse_search` exact | `mse_search` spherical-128 |
|---|---:|---:|---:|---:|
| W3 | −0.00% | −0.11% | −0.00% | +0.08% |
| W4 | −0.01% | −0.07% | −0.01% | +0.14% |
| W5 | −0.09% | −0.37% | +0.13% | −0.19% |
| W6 | −0.56% | −1.14% | +0.24% | −0.85% |
| W8 | −17.19% | — | −7.08% | — |

Two honest readings. First, the exact Gaussian grid is the correct answer to
"Lloyd-Max for a Gaussian", but under `mse_search` it is 0.1–0.2% *worse* than
the old grid at W5/W6: a group of 128 values normalised by its own RMS lies on
a sphere, its coordinates have thinner tails than a Gaussian, and the old
grid was accidentally narrower. The `spherical` grid with `codebook_dim=128`
is the right model and is the best of the three there. Second, none of this is
a meaningful quality lever below seven bits, and section 2.4 shows that even
the 7% at eight bits is below what a model-level measurement resolves.

The entropy-coding rows rest on one fact: Lloyd-Max levels are not equally
likely. The real `rans_encode_rows` coder reaches within 0.02% of the order-0
entropy:

| Stored | Coded bits/weight | Code bytes saved |
|---|---:|---:|
| W3 | 2.838 | 5.40% |
| W4 | 3.822 | 4.44% |
| W5 | 4.852 | 2.96% |
| W6 | 5.851 | 2.49% |

This changes no code and so no output. It shrinks the file, not the resident
model: the loader would decode to fixed-rate codes. The uniform-grid variant
goes further at equal file size but needs wider codes in memory, which costs
VRAM and decode bandwidth. It is attractive only where download size is the
binding constraint.

## 3. Checked and found correct

- **Rotations.** `rotate_activation` is `xR^T`, `inverse_activation` is `xR`,
  `as_matrix` agrees with both to 4e-15. FWHT: `|RR^T − I|` = 1e-16. Butterfly
  stages are symmetric involutions, so the inverse is the stages in reverse.
  The Cayley map is orthogonal; `solve(I + A, I − A)` equals
  `(I − A)(I + A)^-1` because the factors commute.
- **Hessian transform.** `rotate_hessian` returns `R H R^T`; the ridge
  `λI` is invariant under it.
- **GPTQ.** The upper Cholesky factor of the damped inverse, the per-column
  error `(w − q)/U_ii`, the in-block rank-one updates and the deferred block
  update match the reference algorithm; `H = I` reduces to rounding exactly.
- **Bias and residual corrections.** The length correction enforces
  `<w, ŵ'> = ||w||²`; the mean bias correction adds `E[x]R^T (WR^T − Q)^T`; the
  asymmetric QJL estimator is unbiased with the `sqrt(pi/2)/sqrt(k)` constant
  given the `1/sqrt(k)` column scaling.
- **Activation refit.** The group-scale step is `(eH·c_g)/(c_g^T H_gg c_g)`
  followed by an exact line search; the code-move gain is `2gδ − H_ii δ²`.
- **Rotation training.** The Hessian objective uses the unrotated error
  `(V − Q)R`; ridge and mean removal in the selection gate are algebraically
  right.
- **KV cache.** `Q'K'^T = QK^T`, `(AV')R_V = AV`, and the causal mask offset
  `1 + n_key − n_query` is right for both prefill and decode.
- **Scale storage.** Codes are always assigned against the decoded stored
  scale (FP16 or the 8-bit affine grid), so pack-time and run-time agree.
- **By the independent reviewers.** KL direction, temperature `T²`, masking
  and next-token alignment; perplexity over sliding windows (400 parameter
  combinations, no token scored twice); the allocator's dynamic programme
  (1,010 instances against brute force), its MILP and exchange refinement;
  `_local_error` against the deployed forward; the learned-scale
  straight-through estimator (bit-equal to the packed candidate); LoRA
  scaling; the ternary initialisation (the exact alternating optimum, not the
  0.7 rule), its deployment (bit-equal to the fake-quantised weight) and its
  2-bit accounting; the tied-vocabulary lookup and projection.

## 4. Defects

### Fixed in this revision

| # | Location | Defect | Effect | Fix |
|---|---|---|---|---|
| 1 | `codebooks.lloyd_max_gaussian` | 200 Lloyd steps from a uniform start are not a solver beyond 16 levels; plain Lloyd needs about 2e5 steps at 256 levels | excess MSE 0.07 / 0.62 / 5.3 / 27% at 5–8 bits; the 8-bit grid exceeded the bound the docs quote | Newton on the closed-form cell moments, residual below 1e-12, a few milliseconds; `gaussian_legacy` reproduces the old grid |
| 2 | `codebooks.quantizer_mse` | numerical integration on a grid | error 1e-7 | closed form |
| 3 | `codebooks.lloyd_max_samples` | 100 steps from equal-mass quantiles | held-out MSE 0.4 / 12 / 64% above the optimum at 4 / 5 / 6 bits | prefix-sum iterations, two starts, converged; now within 2% |
| 4 | `codebooks.build_finite_e8_codebook` | candidates from the box `[-2, 2]^8` and its half shift `[-1.5, 2.5]^8` | 512 and 3,808 lattice points of norm² 8 and 10 missing at two bits, 26% of points without their negative, mean up to −0.19 per coordinate; unit-RMS scale not MSE-optimal | complete shells, negation-closed last shell, origin dropped, scale fitted: Gaussian MSE −5.3% (1 bit), −2.9% (2 bits) |
| 5 | `block_train._scale_override` | rotated the FP16/BF16 weight, rounding before the RMS | the deployed block differed from the block the gate had scored: 3% (FP16) and 28% (BF16) of scales | rotate in FP32; now bit-equal |
| 6 | `block_train` scale multipliers | `exp(p).clamp()` has zero gradient outside the bounds | a multiplier that overshoots stays on the bound | projection after each optimiser step |
| 7 | `block_train.distill_packed_model` | "final" losses measured before commit | reported losses were not the deployed model's when adapters were removed or rounded | measured after commit |
| 8 | `dynamic` `protect_top_fraction` | `ceil(0.07 * 100)` is 8 | one layer too many protected for some fractions | guarded ceil |
| 9 | `eval.logit_fidelity` | bootstrap resamples tokens as if independent | intervals too narrow when tokens within a prompt correlate (nominal 95% covering about 55% at correlation 0.1 in simulation) | prompt-resampled interval reported alongside; one prompt gives NaN, not zero width |
| 10 | `native_v2_avx2.cpp` | decode repeated for every input row; scalar code extraction; memory gathers for small codebooks | 2–9× slower than necessary | section 5.1 |
| 11 | docs, comments | GPTQ "uses `H = I`" when uncalibrated (it now rejects); a test comment quoting 0.0136 for a bound that is 0.0425; `turboquant_mse_bound` implying scale bits lower distortion | misleading | corrected |

### Found, confirmed, not fixed in this revision

These were reproduced by the reviewers' scripts. None is on the path of the
shipped configurations. All of them were fixed on 19 November; see section 8.

| Location | Defect | When it bites |
|---|---|---|
| `block_train` code refresh | `refresh_quantization` folds the multiplier into the base scales, so "restore best" and "reject" land on a refreshed state that is neither candidate nor reference | `distill_code_refresh_interval > 0` (no shipped config) |
| `block_train` optimiser | FP16/BF16 parameters are stepped directly by Adam: FP16 angles become non-finite, BF16 updates round to zero | reduced `rotation_storage_dtype` with `distill_train_rotations` |
| `block_train` proxies | learned scales with `length` bias correction, or with the uniform codebook, train a forward that differs from the deployed one | those combinations |
| `block_train` LoRA | `residual_svd` adapters are attached at patch time, so propagated activations include untrained adapters | `distill_lora_init=residual_svd` |
| `dynamic` `random_pareto` | with a `target_bpw` target the random control ignores the budget and usually lands below it, which flatters the guided arm | bpw targets; byte-window targets are unaffected |
| `dynamic` local score | ignores the QJL sketch term for `error_comp="turboquant"` | sketch candidates |
| `dynamic` greedy | adjacent downgrade cannot pass two formats of equal size | mixed palettes with `greedy` |
| `dynamic` bucketed DP | suboptimal when the byte window is narrower than a bucket; the exact MILP only runs when nothing feasible is found | very tight windows |
| `dynamic` relative error | denominator includes the bias | projections with a bias |
| `calibrate` | `collect_hessians` defaults to a 1% ridge and GPTQ adds another 1%; `run_experiment` passes 0 and is unaffected | direct library use |
| `rotate.ButterflyRotation` | trigonometry in FP32 leaves `c² + s²` off by 6e-8, so a float64 reference path is orthogonal only to 2e-7 | float64 diagnostics |

## 5. Inference

### 5.1 CPU kernel (measured)

`native/src/native_v2_avx2.cpp`, one thread, Xeon 2.1 GHz, per-call
validation included, 2560 → 2560 unless stated:

| Case | W4 before | W4 after | W5 before | W5 after | W6 before | W6 after |
|---|---:|---:|---:|---:|---:|---:|
| 1 input row | 4.23 ms | 1.84 ms | 4.40 ms | 2.31 ms | 4.58 ms | 2.21 ms |
| 32 rows | 149.6 ms | 16.1 ms | 142.5 ms | 20.2 ms | 139.3 ms | 17.5 ms |
| 128 rows | 566 ms | 71.8 ms | 544 ms | 80.4 ms | 570 ms | 64.3 ms |
| 1 row, 2560 → 9728 | 16.2 ms | 6.9 ms | 18.5 ms | 8.7 ms | 20.7 ms | 9.0 ms |
| 128 rows, 2560 → 9728 | 2092 ms | 236 ms | 2344 ms | 294 ms | 2303 ms | 249 ms |

Three changes, all exact up to reassociation (relative difference from the
scalar kernel at most 1.3e-5; the conformance tolerance is unchanged):

1. With more than one input row, each packed row is decoded once into a small
   buffer and every input row takes a dense dot product with it. Decode work
   no longer scales with the batch.
2. Eight codes are split into lanes with variable shifts on one word load.
3. Codebooks of up to sixteen entries are looked up with register permutes
   instead of memory gathers.

The build is clean under AddressSanitizer and UBSan over odd shapes (partial
groups, widths not divisible by eight, the last bytes of the buffer). The
scalar and NEON kernels are unchanged; NEON could not be built here.

### 5.2 GPU: where the time goes and what to change

From the archived A100 runs (`research/results/native_*`), W5 backbone with
W6 vocabulary, 128 tokens:

| | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| RotQuant reference kernel | 36.3 | 20.0 |
| RotQuant decode4 | 119.5 | 31.7 |
| RotQuant tile8 (current control) | 160.1 | 33.7 |
| Unsloth UD-Q4 (measured beside the reference kernel) | 3296.4 | 112.9 |
| BF16 | 4926.4 | 89.7 |

The decode4 profile attributes 98.7% of custom-operator time in prefill to
the backbone matmul. Over its 32 decode steps it records 522.9 ms backbone,
335.1 ms vocabulary head and 46.2 ms rotation, i.e. about 16.3, 10.5 and
1.4 ms per token. Reading the kernels in `rotquant-native-v2.patch` explains
both:

- **Prefill has no GEMM.** `kernel_rq3_w5_tiled` decodes each weight once per
  eight-token tile and multiplies inside the decode loop (patch line 327–332).
  This is the structure change 1 above removed on CPU.
- **Decode does per-element byte work.** Every thread re-parses the header,
  extracts codes with byte loads, and rounds both operands through FP16 per
  element (line 294).
- **The head inverse-rotates every vocabulary row every token** to honour the
  "dense-equivalent" contract.
- **Rotation is not the problem.** It is 0.3% of prefill and its decode cost
  is launch latency (200 launches per token).

Ranked changes. The first four keep the result exact or reassociation-only:

1. **Prefill: decode a chunk once, then GEMM.** `rq3_chunk_gemm` in the patch
   already does this with cuBLAS and has never been run on a GPU. Products of
   two FP16-representable values are exact in FP32, so `sgemm` on decoded
   tiles is pure reassociation. Hoist the header parse out of the per-element
   unpack first. This is where the 20× prefill gap is.
2. **Decode backbone.** Pass the parsed header by value, load aligned words,
   build a 32- or 64-entry `round16(book[c]·scale)` table per group, and drop
   the per-element rounding of an input that is already FP16-rounded.
3. **Head.** Use the warp-shuffle FWHT already in the patch, with the codebook
   in shared memory and the scale hoisted.
4. **Launches.** Share one rotation across projections with the same input
   and signs (`q/k/v`, `gate/up`); drop the cast and copy nodes.
5. **Head in the rotated basis** (changes rounding only): rotate the
   activation once and run a plain packed matvec, as
   `PackedVocabulary(projection_mode="rotated")` does in Python. The two
   contracts differ only in where FP16 rounding is applied.

That last change was measured on the real tied vocabulary of the 0.5B model
(151,936 × 896, quantised alone, 256 real final hidden states, FP16 operands
in both contracts):

| | W6 head | W8 head |
|---|---:|---:|
| KL(full precision ‖ dense-equivalent head) | 2.993e-3 | 2.115e-4 |
| KL(full precision ‖ rotated-basis head) | 2.984e-3 | 2.088e-4 |
| KL(dense-equivalent ‖ rotated-basis) | 1.05e-6 | 1.03e-6 |
| KL(unrounded quantised head ‖ each) | 6.9e-7, 7.9e-7 | 7.1e-7, 8.2e-7 |
| Top-1 agreement between the two | 100% | 100% |

The two contracts are 1e-6 nats apart, about 2,800 times less than the W6
head's own quantisation error, and each is equally far from the unrounded
quantised head. Neither is the more faithful; the rotated one is far cheaper,
because it replaces an inverse FWHT of every vocabulary row with one FWHT of
the activation. It is a different FP16 rounding, so it needs its own parity
gate and a held-out KL check before it replaces the current contract.

A byte- or nibble-aligned code layout is not needed: with 128-wide groups a W5
group is exactly 80 bytes and a W6 group 96, already word-aligned.

One compile-line check is worth making before trusting any parity gate: the
affine scale reconstruction relies on separate FP32 multiply and add, so the
CUDA build must not contract them into an FMA.

## 6. Changes in this revision

- `rotquant/codebooks.py`: exact Gaussian solver, closed-form MSE, prefix-sum
  weighted Lloyd, converged sample fit, symmetric E8 codebook,
  `gaussian_legacy`.
- `rotquant/lossless.py` (new): integer lifting rotation, exact fixed-point
  view of float tensors, code entropy, row-wise rANS coder.
- `rotquant/block_train.py`, `rotquant/dynamic.py`,
  `rotquant/eval/statistics.py`, `rotquant/eval/logit_fidelity.py`: defects
  5–9.
- `native/src/native_v2_avx2.cpp`: the kernel of section 5.1.
- `tests/test_lossless.py`, `tests/test_review_mathematics.py`: 45 new tests.

Behaviour that changes for existing configurations:

- `codebook="gaussian"` returns the exact grid. Codes change at five bits and
  above (and by at most 1.5e-3 in centroid value at four bits). Saved
  artifacts carry their own centroids and load unchanged. Use
  `codebook="gaussian_legacy"` to reproduce an earlier quantisation.
- `codebook="calibrated"` and `"e8p"` produce different, better grids.
- `QuantConfig.calibrated_iters` now caps the refinement at 2000 (was 100).
- `evaluate_logit_fidelity` adds `paired_prompt_bootstrap`; the existing
  `paired_token_bootstrap` is unchanged.

## 7. Not done

- No GPU kernel was changed or run. Section 5.2 is analysis.
- The entropy coder is not wired into `save_packed_checkpoint` or the GGUF
  exporter; that is a format revision.
- No full-model recipe was re-quantised or re-evaluated on the Qwen3.5-4B
  protocol. The retained W5/W6 checkpoint is untouched.
- The defects in the second table of section 4 remain open.

## 8. Follow-up, 19 November 2026

The improvement plan was carried out as far as is possible without a GPU.

### 8.1 Open defects

All eleven rows of the second table in section 4 are fixed, each with a
regression test that fails on the earlier code
(`tests/test_recovery_defects.py`, `tests/test_allocator_defects.py`).

| Area | Change | Before → after (reviewers' reproductions) |
|---|---|---|
| Code refresh | Refresh keeps the deployed scales and codebook and only reassigns codes; snapshots hold the packed state with the parameters, so restore-best and reject restore it exactly. Refresh refuses up front for layers it cannot keep exact | rejected run: 20% weight change → 0; refresh with nothing changed: 8% → 0 |
| Reduced-precision training | Distillation steps FP32 masters and writes back the storage-dtype values; block training evaluates angles rounded to their deployed dtype | FP16 angles: non-finite → trained; BF16: 0/384 moved → 384/384; proxy vs packed block 1.5e-2 → 0 |
| LoRA | `residual_svd` adapters are initialised at the start of distillation from the retained source, never during block training | propagated inputs no longer include untrained adapters |
| Proxy vs deployed | learned scales with length correction rejected before mutation; scale-multiplier bounds derived from the quantiser's search interval (Gaussian unchanged at 0.5–1.5); the unlearned proxy rounds scales as stored | uniform: 256/256 groups clamped → 0; proxy error 1.1e-2 → 1.8e-7 |
| `random_pareto` | spends the bpw budget | 4.50–4.75 → 5.0 bpw at a 5.0 target |
| Local score | deployed forward, QJL sketch included; bias-free denominator | k=64 sketch: scored 2.49 vs deployed 15.9 → equal |
| Greedy / random ladders | one candidate per size | 4.0 bpw, target missed → 3.5 reached |
| Narrow byte windows | exact MILP also runs below four buckets; better verified recipe kept | 7/15 suboptimal → 0 |
| Random controls | protected layers drawn at random | inherited the guided set → independent |
| Evaluation | padding masked; token-weighted teacher KL; prompt-level bootstrap (section 4) | — |
| Calibration | Hessians collected undamped by default (GPTQ damps itself) | 1% + 1% → 1% |
| Float64 butterflies | trigonometry in float64 | orthogonal to 1.5e-7 → 1e-13 |

Changes a user will notice: the shipped `qwen35_4b_recovery_cuda` run now
trains LoRA against FP16-rounded values and retains the source weights on the
CPU when `residual_svd` is configured (about 8 GB for a 4B BF16 model); the
dynamic score cache moved to schema 4; refresh-run checkpoints written earlier
fail closed. Still open: a distillation that raises mid-run leaves its partly
trained parameters in the model.

### 8.2 GPU kernels (prepared, not run)

`rq3-v2.cuh` and the registry v2 screen are described in the
[runbook](native_overnight_experiment_plan.md). In short: bit-exact warp
kernels for the backbone and head; a decode-once GEMM for prefill (FP32 and
FP16); the rotated-basis head; and a fused rotation+matmul op that removes one
launch per projection. Correctness so far rests on a lane-level CPU emulation
(bitwise equality of the FP32 sums with the reference kernel), sm_80
compilation with no spills or FMAs, and bitwise CPU equality of fused and
unfused graphs. No speed or GPU-correctness claim is made until the overnight
notebook has run.

The retained checkpoint uses 200 distinct rotations, so sharing one rotation
across projections that read the same input would need a re-export; fusion
was implemented instead.

### 8.3 Entropy-coded checkpoints

`save_packed_checkpoint(..., code_encoding="rans-rows-v1")` stores each
primary scalar code stream as one rANS stream per output row. On two layers of
Qwen2.5-0.5B it saved 3.3–4.7% of code bytes after per-row overhead, and codes
and logits were bit-exact after reload. Loading decodes at about 15 M codes
per second on one core, roughly 24 s for a 4B model. The default output is
unchanged; coded checkpoints are format version 4, and GGUF v2 export reads
them.

### 8.4 Not done

- Step 7 of the plan (a uniform grid with entropy coding, about 26% lower
  weight error at equal file size) was not implemented: it needs 8-bit codes
  in memory, which costs decode bandwidth, and it only pays if the size budget
  is the file rather than the memory in use.
- Step 8 (trellis or vector quantisation) is a research project of its own.
- Metal is unchanged.

