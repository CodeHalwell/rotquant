# How RotQuant works

This document explains the scientific and mathematical ideas behind RotQuant,
then maps each idea to the implementation that exists today. It is intended to
answer four different questions:

1. Why can an orthogonal rotation make quantization easier without changing the
   source model?
2. How are weights, activations, and KV-cache vectors converted into finite-bit
   representations?
3. What is actually optimized during calibration, mixed-precision allocation,
   and recovery?
4. Which statements are mathematical identities, which are results from prior
   research, and which still need experimental confirmation in RotQuant?

The shortest useful mental model is:

> RotQuant changes the coordinate system of a model without changing its
> full-precision function, fits a finite-rate representation in that better
> coordinate system, and then selects and serves the representation under an
> exact byte budget.

It is therefore not one quantizer. It is a pipeline that combines an orthogonal
reparameterization, scalar or vector quantization, optional calibration-aware
error compensation, model-specific rate allocation, a versioned packed format,
and runtime kernels.

## 1. The end-to-end pipeline

For a normal weight-only profile, the data flow is:

```text
source model
   |
   +-- inspect architecture and choose quantizable projections
   |
   +-- optionally collect activation samples, Hessians, or teacher logits
   |
   +-- choose a fixed or learned orthogonal rotation R
   |
   +-- rotate each weight matrix W -> W R^T
   |
   +-- choose scales, codebooks, and integer code indices
   |
   +-- optionally allocate different bit widths to different projections
   |
   +-- pack codes + fp16 scales + codebooks + rotations into a checkpoint
   |
   `-- at inference: rotate x, consume packed W directly, and compute y
```

There are three distinct representations to keep separate:

| Representation | Purpose | Persistent in an optimized artifact? |
|---|---|---|
| Source weight `W` | Full-precision teacher/reference | No |
| Rotated weight `W R^T` | Coordinate system in which codes are fitted | Not as a dense tensor |
| Packed codes, scales, and codebook | Deployable finite-rate representation | Yes |

The Python fallback can transiently reconstruct the rotated weight to test
quality. The native runtime instead streams packed groups into the dot product.
Only the latter can support a speed or packed-resident-memory claim.

## 2. The invariant that makes rotations safe

PyTorch stores a linear layer's weight as
`W in R^(d_out x d_in)` and computes, for row-vector activations,

$$
y = xW^T.
$$

Let `R in R^(d_in x d_in)` be orthogonal:

$$
R^TR = RR^T = I.
$$

Define the rotated activation and weight as

$$
x' = xR^T, \qquad W' = WR^T.
$$

Then

$$
x'{W'}^T
= xR^T(WR^T)^T
= xR^TRW^T
= xW^T.
$$

This equality is exact apart from floating-point round-off. It is the central
correctness invariant in `rotquant.rotate`, `rotquant.patch`, and
`rotquant.linear`.

### The consistency trap

Rotating only the weight is not an equivalent reparameterization:

$$
x(WR^T)^T = xRW^T,
$$

which is generally different from `xW^T`. The activation must enter the same
basis used to encode the weight. RotQuant's normal `consistent` and
`fused_inverse` modes enforce this; `mismatched` exists only as a deliberate
negative control.

The same rule appears in the KV cache. Rotated keys require matching rotated
queries so that `qk^T` is preserved. Rotated values require the attention
result to be mapped back through the inverse rotation.

## 3. Why changing coordinates helps quantization

An orthogonal transform does not remove information or reduce energy:

$$
\lVert xR^T\rVert_2 = \lVert x\rVert_2,
\qquad
\lVert WR^T\rVert_F = \lVert W\rVert_F.
$$

Its benefit comes from changing how that energy is distributed across
coordinates. A scalar quantizer treats coordinates separately, so it is harmed
when a few coordinates contain large outliers while most use only a small part
of the grid. A mixing rotation spreads localized energy across many
coordinates, making their scales and marginal distributions more alike.

For an ideal Haar-random orthogonal matrix and a fixed nonzero vector `v`, the
direction of `Rv` is uniform on the sphere. If

$$
Z = \sqrt{d}\,\frac{(Rv)_1}{\lVert v\rVert_2},
$$

then `Z` has unit variance, support `[-sqrt(d), sqrt(d)]`, and density

$$
f_d(z) =
\frac{\Gamma(d/2)}{\sqrt{\pi d}\,\Gamma((d-1)/2)}
\left(1-\frac{z^2}{d}\right)^{(d-3)/2}.
$$

As `d` grows this coordinate marginal approaches a standard Gaussian. This is
the reason a single Gaussian Lloyd-Max grid is a good data-oblivious default
after sufficiently mixing rotations. RotQuant's `spherical` codebook uses the
finite-dimensional density above instead of the Gaussian limit.

A randomized Hadamard transform is not a Haar-random dense matrix, but its
random signs and mixing stages give similar concentration benefits at much
lower cost. This use of randomized Hadamard transforms for incoherence and
outlier suppression is shared with [QuIP#](https://arxiv.org/abs/2402.04396)
and [QuaRot](https://arxiv.org/abs/2404.00456).

### Rotations implemented by RotQuant

| Rotation | Parameterization | Apply cost | Intended use |
|---|---|---:|---|
| Identity | `R = I` | `O(d)` | Required no-rotation control |
| Randomized block Hadamard | `R = H D_s` | `O(d log B)` | Production default |
| Butterfly | trainable two-coordinate orthogonal stages | `O(d log B)` | Deployable learned rotation |
| Dense random | QR factor of a Gaussian matrix | `O(d^2)` apply, `O(d^3)` construction | Research comparison |
| Dense learned | Cayley transform of a skew matrix | `O(d^2)` apply, `O(d^3)` solve | Small-scale ablation |

Here `B` is the Hadamard block width and `D_s` is a fixed diagonal random-sign
matrix. The normalized Hadamard matrix satisfies `H^T H = I` and is its own
inverse.

The butterfly uses stages of exact two-dimensional orthogonal transforms:

$$
B(\theta) =
\begin{bmatrix}
\cos\theta & \sin\theta \\
\sin\theta & -\cos\theta
\end{bmatrix}.
$$

Every stage is orthogonal for every value of `theta`. At `theta = pi/4`, it is
a normalized Hadamard butterfly, so training starts from the known FWHT
baseline and cannot cheat by shrinking the matrix norm.

The dense learned arm forms a skew-symmetric matrix `A^T = -A` and uses the
Cayley map

$$
R(A) = (I-A)(I+A)^{-1}.
$$

This is orthogonal wherever the inverse exists. It is mathematically clean but
too expensive to be the general large-model runtime representation.

## 4. Scalar quantization

After rotation, a row is divided into groups of `G` weights. For group `g`,
RotQuant stores a positive scale `s_g` and one `b`-bit code per weight. Given a
scalar codebook

$$
\mathcal C = \{c_0, c_1, \ldots, c_{2^b-1}\},
$$

the assignment and reconstruction are

$$
k_i = \arg\min_k \left|\frac{w_i}{s_{g(i)}}-c_k\right|^2,
\qquad
\hat w_i = s_{g(i)}c_{k_i}.
$$

The deployed tensor contains the packed `k_i`, the stored scales, and the
codebook. It does not contain `hat W` as a persistent dense matrix.

### Lloyd-Max codebooks

For a source density `f(z)`, scalar quantization minimizes

$$
D = \mathbb E[(Z-Q(Z))^2]
  = \sum_i \int_{t_{i-1}}^{t_i}(z-c_i)^2 f(z)\,dz.
$$

At a stationary Lloyd-Max solution, the cell boundaries and centroids obey

$$
t_i = \frac{c_i+c_{i+1}}{2},
\qquad
c_i = \mathbb E[Z \mid t_{i-1}<Z\le t_i].
$$

RotQuant implements these alternatives:

- `gaussian`: Lloyd-Max centroids for a standard Gaussian;
- `spherical`: Lloyd-Max centroids for the finite-dimensional sphere marginal;
- `calibrated`: deterministic Lloyd fitting on normalized rotated weights;
- `uniform`: a symmetric uniformly spaced control;
- `nf`: an equal-probability NormalFloat-style control.

The asymptotic Gaussian rate-distortion lower bound at `b` bits per scalar is

$$
D_{\mathrm{Shannon}}(b) = \sigma^2 2^{-2b} = \sigma^2 4^{-b}.
$$

For a unit-variance source, the TurboQuant paper reports the corresponding
distortion expression as

$$
D_{\mathrm{TQ}}(b) \lesssim \frac{\sqrt 3\,\pi}{2}4^{-b},
$$

whose constant is about `2.72` times the Shannon bound. The paper presents an
inequality with this right-hand side, but its derivation uses the Panter–Dite
high-resolution approximation for the larger-bit regime. RotQuant's
`turboquant_mse_bound` helper records that paper-stated, high-rate estimate; it
should not be read as an unconditional finite-bit theorem or as a promise that
every LLM matrix or downstream task will attain that distortion. See
[TurboQuant](https://arxiv.org/abs/2504.19874) for the construction and proof.

### Scale selection

The scale is as important as the centroid locations.

For an RMS group scale,

$$
s_g = \sqrt{\frac{1}{|g|}\sum_{i\in g}w_i^2}.
$$

RotQuant divides a partial final group by its real element count, not by a
zero-padded `G`. The chosen scale is rounded to its declared storage precision
(normally fp16) before final code assignment, ensuring pack-time and runtime
reconstruction agree.

`mse_search` starts from the RMS scale and tests a bounded grid of multipliers:

$$
s_g^* = \arg\min_{s\in\{\alpha_1s_{\mathrm{RMS}},\ldots,
\alpha_ms_{\mathrm{RMS}}\}}
\sum_{i\in g}(w_i-Q_s(w_i))^2.
$$

This is data-free: it optimizes the weight reconstruction itself and needs no
text calibration set. The `turboquant` scale option uses one RMS scale per
output row. It reduces scale metadata from roughly `16/G` bits per weight to
`16/d_in`, at the possible cost of less local adaptation.

### Exact rate accounting

A scalar group with `b`-bit codes and a `p`-bit scale has nominal rate

$$
r = b + \frac{p}{G}\quad\text{bits per weight}.
$$

For W3, fp16 scales, and groups of 128,

$$
r = 3 + \frac{16}{128} = 3.125\ \text{bits/weight}.
$$

That is still only a nominal group calculation. A matrix-level
`QuantizedWeight.bit_budget()` uses the actual retained code, scale, residual,
and sketch buffers, including final 32-bit-word padding. Candidate allocation
uses those packed matrix bytes because codebook and rotation choices are fixed
across its bit-width candidates. An exact artifact comparison goes further and
includes codebooks, rotations, unquantized embeddings, norms, output heads,
multimodal modules, adapters, and all other persistent files.

## 5. Inner-product bias and residual correction

MSE-optimal scalar reconstruction can systematically shorten dot products. For
a centroid quantizer, the reconstruction is a conditional mean, so under its
source model the residual is orthogonal to the reconstruction:

$$
\mathbb E[(W-Q(W))Q(W)] = 0.
$$

Consequently,

$$
\mathbb E[WQ(W)] = \mathbb E[Q(W)^2]
< \mathbb E[W^2]
$$

unless the quantization error is zero.

RotQuant exposes three different responses to residual error. They must not be
conflated.

### Length correction

For each row, the optional zero-extra-code correction is

$$
\alpha = \frac{\lVert w\rVert_2^2}{\langle w,\hat w\rangle},
\qquad
\hat w' = \alpha\hat w.
$$

This enforces `inner(w, hat w') = ||w||^2` when the denominator is safe. The
factor is folded into the row's existing scales, so it does not add a code
stream or change artifact shape. It targets shrinkage, not ordinary MSE, and
can make MSE worse. It is therefore opt-in.

### Deterministic residual quantization

A second scalar pass stores

$$
R_1 = W-Q_1(W),
\qquad
\hat W = Q_1(W)+Q_2(R_1).
$$

This is deterministic and reconstructs the residual itself, but pays for a
second code and scale stream. It is useful as a matched-rate control for more
specialized residual schemes.

### TurboQuant-style QJL residual sketch

The `error_comp="turboquant"` path stores a one-bit random projection of each
residual row. Let `G in R^(d x k)` have independent Gaussian entries scaled by
`1/sqrt(k)`. It stores

$$
z_r = \operatorname{sign}(rG)
$$

and the row norm `||r||_2`. At inference, for rotated activation `x'`, it uses

$$
\widehat{\langle x',r\rangle}
= \frac{\sqrt{\pi/2}}{\sqrt{k}}\lVert r\rVert_2
  \langle x'G,\operatorname{sign}(rG)\rangle.
$$

For a standard Gaussian projection vector `h ~ N(0,I)`,

$$
\mathbb E[(x'^Th)\operatorname{sign}(r^Th)]
= \sqrt{\frac{2}{\pi}}\frac{\langle x',r\rangle}{\lVert r\rVert_2}
$$

up to the explicit projection scaling. The coefficient above therefore makes
the estimator unbiased. Only the stored side is sign-quantized; the activation
projection remains real-valued. This follows the asymmetric QJL construction
used by TurboQuant.

The older `error_comp="qjl"` option in RotQuant is not this estimator. It is a
stochastic one-bit scalar residual control kept for experiments.

RotQuant selectively adapts these ideas; it is not a fork of TurboQuant or
TurboVec. TurboQuant is primarily a data-oblivious online vector/KV method, and
TurboVec is a vector-search system. RotQuant applies related rotation,
finite-dimensional codebook, length-correction, and QJL ideas inside a broader
LLM weight/KV optimization and artifact pipeline.

## 6. GPTQ: activation-aware second-order error feedback

Weight MSE assumes isotropic inputs. Real model activations are not isotropic,
so the same weight error can matter very differently in different directions.

For activation rows collected in `X`, define the empirical second moment

$$
H = \frac{1}{N}X^TX.
$$

For weight error `E = W-hat W`, the layer-output reconstruction objective is

$$
\frac{1}{N}\lVert XE^T\rVert_F^2
= \operatorname{tr}(EHE^T).
$$

This explains why a Hessian-aware quantizer can outperform nearest-centroid
rounding even at the same stored rate: it spends error in directions that the
calibration activations use less.

RotQuant accumulates `H` incrementally instead of storing every activation.
When a weight is rotated, the matching Hessian is also transformed:

$$
H' = RHR^T,
$$

because `X' = XR^T`.

The blocked GPTQ implementation adds diagonal damping, constructs an upper
Cholesky factor of the damped inverse Hessian, quantizes columns sequentially,
and propagates each column's scaled error into the remaining columns. Updates
inside a block happen immediately; the update to later blocks is batched into a
matrix multiplication. This is the same second-order PTQ family introduced by
[GPTQ](https://arxiv.org/abs/2210.17323).

If no calibration Hessian is supplied, RotQuant deliberately uses `H = I` and
reduces exactly to plain rounding. A configuration that says GPTQ but collected
no activations is therefore not evidence for GPTQ.

## 7. Learned rotations and calibration-time recovery

Randomized Hadamard rotation is cheap and robust, but not every random basis is
equally compatible with a particular model and quantizer. This motivates the
learned-rotation family represented by
[SpinQuant](https://arxiv.org/abs/2405.16406).

RotQuant has increasingly global objectives.

### Layer weight objective

The data-free objective is

$$
\mathcal L_{\mathrm{weight}}(\theta)
= \lVert WR(\theta)^T-Q(WR(\theta)^T)\rVert_F^2.
$$

Because `R` is orthogonal, minimizing this is equivalent to minimizing the
unrotated reconstruction norm. The optimizer cannot reduce the loss merely by
shrinking `W`.

Codes and scales are discrete or piecewise constant. RotQuant uses alternating
optimization: recompute assignments for the current rotation, freeze them, and
take an Adam step through the rotation. This is analogous to the assignment and
centroid alternation in Lloyd's algorithm.

### Layer activation objective

For captured inputs `X`, the stronger objective is

$$
\mathcal L_{\mathrm{layer}}(\theta)
= \frac{\lVert XW^T-(XR(\theta)^T)
Q(WR(\theta)^T)^T\rVert_F^2}
{\lVert XW^T\rVert_F^2}.
$$

A trained butterfly is compared with its exact seeded FWHT initialization
under the final deployable quantizer. It is retained only if it clears the
configured held-out improvement margin; otherwise RotQuant restores FWHT.

### Transformer-block objective

Independent layer error misses nonlinearities, residual connections, attention
coupling, and error arriving from earlier layers. Block training replaces all
target projections in one transformer block with fake-quantized butterfly
linears and minimizes

$$
\mathcal L_{\mathrm{block}}
= \frac{\lVert F_{\mathrm{quant}}(h)-F_{\mathrm{source}}(h)\rVert_F^2}
{\lVert F_{\mathrm{source}}(h)\rVert_F^2}.
$$

It can jointly train butterfly angles and bounded group-scale multipliers.
Training, validation, and final packed-candidate selection use separate
captured calls. The optional propagated-input mode feeds later blocks the
already quantized preceding state, exposing accumulated drift during
optimization.

### Model-level distillation

The most global recovery stage can optimize deployed rotations, scale
multipliers, optional low-rank residual adapters, and selected existing
parameters against source logits. Its loss is

$$
\mathcal L
= \lambda_{KL}T^2D_{KL}
\left(p_T^{\mathrm{teacher}}\,\|\,p_T^{\mathrm{student}}\right)
+ \lambda_{CE}\,\mathrm{CE}(p^{\mathrm{student}}, y)
+ \lambda_R\lVert\theta-\theta_0\rVert_2^2.
$$

The temperature factor `T^2` preserves useful gradient scale. A disjoint
selection set decides whether the candidate is committed or the complete
pre-distillation state is restored. Optional low-rank adapters cost real bytes
and must justify those bytes in the final comparison.

## 8. Finite-rate vector quantization

Scalar quantization chooses one centroid per coordinate. Vector quantization
chooses one point for an `m`-dimensional tuple:

$$
k = \arg\min_j \lVert v-c_j\rVert_2^2,
\qquad c_j\in\mathbb R^m.
$$

At `b` bits per weight and vector dimension `m`, an exactly matched codebook
has

$$
K = 2^{bm}
$$

centroids and stores one `bm`-bit index for every `m` weights. Thus a
dimension-2 W3 arm has 64 centroids, stores one 6-bit code per pair, and still
costs 3 code bits per weight before scale metadata.

Vector codebooks can exploit correlations and shape cells more efficiently
than Cartesian scalar bins. That advantage is most interesting at 1--3 bits,
but codebook size and nearest-centroid work grow exponentially with `bm`.

RotQuant's current finite-rate arm fits a deterministic Gaussian codebook with
k-means++ initialization and Lloyd iterations. It is an experimental control,
not a production format. Checkpoint export and native kernels fail closed for
it. The included E8 nearest-lattice routine is only a geometric primitive: an
infinite lattice without a bounded index mapping is not a finite-rate encoder
and cannot be used as a same-size baseline. Competitive low-bit comparisons
still need a real packed method such as [QuIP#](https://arxiv.org/abs/2402.04396)
or [AQLM](https://arxiv.org/abs/2401.06118).

## 9. Model-specific mixed precision

Different projections have different sensitivity. RotQuant's “dynamic” mode
is a static, model-specific mixed-precision recipe, not a precision decision
made dynamically for each inference request.

For projection `l` and candidate bit width `b`, it measures a local normalized
weight or output error and can additionally perturb one projection at a time to
measure teacher-logit KL:

$$
C_l(b) = \lambda_{local}E_l(b)+\lambda_{KL}D_l(b).
$$

The conceptual allocation problem is a multiple-choice knapsack:

$$
\min_{b_1,\ldots,b_L}\sum_l C_l(b_l)
\quad\text{subject to}\quad
\sum_l S_l(b_l)\le B,
$$

where `S_l(b)` is the candidate's actual packed byte count. The present
allocator starts from the highest allowed precision and repeatedly takes the
downgrade with the smallest score penalty per byte saved:

$$
\frac{C_l(b_{lower})-C_l(b_{current})}
{S_l(b_{current})-S_l(b_{lower})}.
$$

Rules can set minimum, maximum, or fixed widths for named projections. A seeded
random downgrade order supplies a matched-format, matched-budget negative
control. Since the greedy procedure is not an exact global knapsack solver,
held-out comparison against uniform and random allocation is mandatory.

## 10. KV-cache quantization

For one attention head,

$$
A = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}+M\right),
\qquad O=AV.
$$

RotQuant stores keys in a rotated basis `K' = KR_K^T` and rotates queries as
`Q' = QR_K^T`. Orthogonality preserves the unquantized logits:

$$
Q'{K'}^T = QR_K^TR_KK^T = QK^T.
$$

Values can use an independent rotation `V' = VR_V^T`. Attention accumulation
happens in the rotated value basis and one inverse is applied to the result:

$$
(AV')R_V = AVR_V^TR_V = AV.
$$

Quantization breaks exact equality, but this construction keeps the error local
to the finite-rate approximation rather than adding a basis mismatch.

The cache can also be tiered by position. For a sequence of length `L`, RotQuant
stores positions `[0, sink_tokens)` and `[L-recent_window, L)` as rotated fp16
rows and packs only the middle. The sets are deduplicated when they overlap.
Protected storage is therefore O(1) in context length, while the packed fraction
tends to one as `L` grows. This is a real split in `QuantizedKV`, not merely a
retrieval reservation.

K and V need not use the same bit width. Key error perturbs logits before a
softmax, so it can change every attention weight. Value error enters linearly
after the weights are chosen. Their observed distributions also differ; for
example, [KIVI](https://arxiv.org/abs/2402.02750) motivates asymmetric grouping
for keys and values. RotQuant therefore supports separate key/value candidates
and static per-layer mixed-bit recipes.

The current Transformers KV path is a fidelity simulator. It quantizes real
post-RoPE cache writes, reconstructs them for the unchanged attention
implementation, and measures next-token KL, top-1 agreement, NLL, NMSE, and
actual packed code/scale cache bytes. It proves semantic integration, not
runtime speed.
A production path must rotate and pack before persistent cache storage and
consume the packed representation inside attention without constructing a full
dense cache.

## 11. Selective KV retrieval

During decode, the current query is already a search vector over cached keys.
This makes a retrieval path possible:

1. scan compact key codes and estimate `qk_i^T`;
2. always retain configured recent and attention-sink positions;
3. select the remaining high-score candidates;
4. gather and dequantize only their value vectors;
5. apply softmax and value accumulation over the selected set.

If `S` is the selected position set and `a_i` are full-attention weights, the
oracle attention-mass coverage is

$$
m(S)=\sum_{i\in S}a_i.
$$

High coverage is necessary but not sufficient: selected value directions and
renormalization also affect the output. RotQuant therefore reports mass,
relative output MSE, cosine similarity, fallback rate, and the fraction of
value vectors read.

The method still scans K linearly in its first form. Its intended saving is to
make that scan compact and SIMD-friendly while avoiding most V traffic. A
page-level shortlist can later reduce key work. This direction is related to
query-aware sparse attention such as [Quest](https://arxiv.org/abs/2406.10774),
but RotQuant's present implementation is an oracle: Python reconstructs keys,
and dense probabilities may be used to measure an upper bound. It does not yet
show that a packed-key index can recover the same candidates.

Diffuse attention is the failure case. A safe runtime needs a confidence or
mass estimator and a dense fallback when the selected set is unreliable.

## 12. Packed inference and where speed comes from

For one scalar-quantized group, a fused linear kernel conceptually computes

$$
y_j = \sum_g s_{jg}
\sum_{i\in g} c_{k_{jgi}}x'_i+b_j,
\qquad x'=xR^T,
$$

while reading packed `k` values and scales. It should not first build an fp16
matrix of every `s*c_k`.

Weight-only decode is often memory-bandwidth bound: each generated token uses
large weights for a small amount of arithmetic. Reducing bytes transferred can
therefore matter more than the extra unpack operations. Prefill has larger
matrix multiplications and higher arithmetic intensity; weight-only packing
does not automatically yield the same benefit, and low-bit activations may be
needed to use integer tensor cores effectively.

This leads to four independent claims:

| Claim | Required evidence |
|---|---|
| Smaller artifact | exact files and byte policy |
| Smaller resident model | process memory with no dense persistent cache |
| Faster operator | packed kernel benchmark against a named baseline |
| Faster server | end-to-end prefill/decode/concurrency benchmark |

RotQuant native-v2 defines a backend-independent scalar block: one little-endian
fp16 scale followed by LSB-first codes for each group. It supports scalar W1--W8
and has a portable C++ streaming reference plus SIMD CPU paths. The Python
fallback and C++ scalar reference establish correctness floors; neither stands
in for the planned CUDA/Triton, Metal, vLLM, SGLang, or llama.cpp serving work.

The generic checkpoint also supports blockwise 8-bit scale metadata. If primary
scales `s_i` are divided into blocks of `B=256`, each block stores fp16 `a` and
`delta` plus uint8 codes

$$
q_i=\operatorname{clip}_{[0,255]}
\operatorname{round}\left(\frac{s_i-a}{\delta}\right),
\qquad \hat s_i=a+q_i\delta.
$$

Assignments are computed against `hat s`, so packing and inference cannot
disagree. Exact code, offset, and step bytes enter the instantiated bit budget.
Existing native-v2 blocks still materialize fp16 scales when exported; a fused
uint8-scale decoder remains a distinct runtime capability.

For W4A8 experiments, activations are rotated first and then quantized with a
signed symmetric per-token scale:

$$
s_x=\frac{\max_i |x_i'|}{2^{b-1}-1},\qquad
\hat x_i'=s_x\,\operatorname{clip}
\left(\operatorname{round}(x_i'/s_x),-q_{max},q_{max}\right).
$$

The portable module immediately dequantizes this value; it defines the quality
semantics and training STE, not a tensor-core speed claim.

The finite E8P comparator keeps exactly `2^(8b)` shaped E8 points for each
8-coordinate product block. At `b=2`, one 16-bit packed index represents eight
weights—exactly 2 bits/weight. Its usual encoder snaps to the nearest E8 lattice
point and uses a deterministic key table; overload points outside the retained
shape fall back to exact finite-codebook search. The same codebook is available
for weights and KV rows.

## 13. How quality is measured

No single metric is adequate.

### Reconstruction metrics

Weight NMSE and layer-output NMSE are fast diagnostics:

$$
\operatorname{NMSE}(u,\hat u)
= \frac{\lVert u-\hat u\rVert_2^2}{\lVert u\rVert_2^2}.
$$

They isolate numerical mechanisms but do not model autoregressive feedback.

### Teacher-forced model metrics

Perplexity is

$$
\operatorname{PPL}
= \exp\left(-\frac{1}{N}\sum_t\log p(x_t\mid x_{<t})\right).
$$

It is a useful regression sentinel, but average log-likelihood can hide changes
to individual predictions. Token-distribution KL is more directly tied to
matching the source model:

$$
D_{KL}(p\|q)=\sum_v p(v)\log\frac{p(v)}{q(v)}.
$$

RotQuant treats source-model probabilities as `p` and quantized-model
probabilities as `q`, and reports distribution tails rather than only a mean.
[Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141) motivates KL
and answer flips because unchanged aggregate accuracy can conceal substantial
behavior changes.

### Free-running trajectory metrics

Small token differences compound during generation. Greedy trajectory tests
therefore measure exact 32-token agreement, matching-prefix length,
first-divergence position, top-1 agreement, looping, empty output, invalid tool
calls, and task outcomes. Agentic/code/math/multilingual/long-document domains
are reported separately so an average cannot conceal a catastrophic domain.

### Experimental controls

A credible comparison fixes:

- source model and tokenizer revisions;
- rendered chat template and exact prompt token IDs;
- calibration and held-out manifests with zero token-sequence overlap;
- decoding settings and auxiliary-module policy;
- actual artifact bytes within the declared tolerance;
- hardware, engine, code revision, and dense-fallback policy;
- paired examples, multiple seeds, and bootstrap intervals.

Selection data chooses the method; final held-out data estimates its quality.
Using the final set to tune codebooks, allocation rules, or thresholds converts
it into training data.

## 14. What current RotQuant evidence says

The following are development results, not universal conclusions or the final
competitive claim:

- Randomized Hadamard rotation prevented catastrophic low-bit degradation on
  the early OPT weight-only experiments.
- Learned butterfly/block recovery improved small-model development results,
  but more optimization steps could overfit and the model-level recovery stage
  sometimes correctly restored step zero.
- On the current Qwen3.5-4B development protocol, the carried-forward simple
  weight profile is Gaussian or calibrated W4 rather than a TurboQuant-style
  row-scale variant.
- A teacher-guided mixed-rate recipe beat its exact-format random allocator in
  the recorded algorithm-lab run, but still needs the hardened rerun and full
  competitive gates.
- Dimension-2 vector W3 beat a matched scalar control locally on the primary
  family, but absolute quality and cross-family transfer were poor. It remains
  research-only.
- A selective-V oracle found a possible bandwidth/quality region, but it used
  only a small number of full-attention layers and oracle selection. It does
  not authorize a retrieval or speed claim.
- Packed checkpoint bytes and native value conformance have been verified for
  the Qwen3.5-4B artifact. Production GPU serving and full 300-prompt quality
  evaluation remain open.

The detailed numbers, provenance, and negative results live in the
[experiment log](experiment_log.md). The acceptance rules for a fair external
comparison live in the [competitive evaluation contract](competitive_eval.md).

## 15. What is proved, implemented, and still hypothetical

| Statement | Status |
|---|---|
| Matched orthogonal weight/activation rotation preserves a linear map | Exact identity and tested |
| FWHT and butterfly transforms are orthogonal | Exact by construction and tested |
| Lloyd-Max conditions minimize scalar MSE locally for the chosen source density | Classical result; implementation source-coding tests pass |
| The Gaussian rate-distortion curve is `sigma^2 2^(-2b)` | Information-theoretic lower bound under its assumptions |
| TurboQuant's rotated scalar construction is within a small constant of that bound | Prior-paper theorem under its assumptions; helper implemented |
| The asymmetric QJL residual estimator is unbiased | Exact in expectation over Gaussian sketches; implementation tested numerically |
| GPTQ optimizes an activation-weighted local error | Prior method and implemented approximation |
| Learned block rotations improve every architecture | Not proved; must pass held-out selection per model |
| Dynamic mixed precision beats uniform quantization | Not guaranteed; current greedy allocator needs controls |
| Finite E8P beats scalar codes at equal rate | Packed comparator implemented; empirical advantage not yet established across models |
| W4A8 improves prefill throughput | Quality semantics implemented; no fused-kernel speed claim yet |
| fp16 sink/recent KV tiers improve long-context quality | Storage semantics implemented; 8k–32k result pending |
| Packed KV retrieval reduces end-to-end latency | Hypothesis; no production kernel claim yet |
| A smaller packed artifact makes inference faster | False in general; requires a specialized runtime and workload measurement |

This separation is intentional. The mathematics constrains what a correct
implementation may do; experiments determine whether a valid mechanism is
useful for a particular model, rate, workload, and device.

## 16. Further reading and implementation map

Primary research lineage:

- [Max, Quantizing for Minimum Distortion (1960)](https://doi.org/10.1109/TIT.1960.1057548)
- [Lloyd, Least Squares Quantization in PCM (1982)](https://doi.org/10.1109/TIT.1982.1056489)
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)
- [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118)
- [KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774)
- [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)

Repository implementation map:

| Topic | Code or specification |
|---|---|
| Rotation invariant and parameterizations | `rotquant/rotate.py` |
| Architecture patching and consistency | `rotquant/patch.py` |
| Codebooks and source-coding bounds | `rotquant/codebooks.py` |
| Scales, GPTQ, residuals, QJL, and accounting | `rotquant/quantize.py` |
| Packed linear semantics | `rotquant/linear.py` |
| Hessian and activation collection | `rotquant/calibrate.py` |
| Layerwise learned rotations | `rotquant/train_rotation.py` |
| Block recovery and distillation | `rotquant/block_train.py` |
| Static mixed-precision allocation | `rotquant/dynamic.py` |
| KV rotations and retrieval oracle | `rotquant/kv_cache.py` |
| Versioned checkpoint bitstream | [packed format v2](packed_format_v2.md) |
| Backend-independent runtime blocks | [native runtime v2](native_runtime_v2.md) |
| Selective cache design | [KV retrieval](kv_retrieval.md) |
| Evaluation and claim policy | [competitive evaluation](competitive_eval.md) |
| Current evidence and negative results | [experiment log](experiment_log.md) |
