# Scientific and mathematical validity review — 2026-09-01

Scope: an independent, line-by-line review of the algorithmic, statistical and
evaluation content of RotQuant at revision `06c1e73`, together with every
result recorded in `docs/experiment_log.md`, `research/results/`,
`paper/data/publication_results.json` and `paper/main.tex`. Unlike the
2026-08-31 review, this pass did not stop at reading: the test suite was run,
each mechanism that could not be settled by derivation was checked
numerically, and the recorded key/value (KV) cache protocol was replicated on
the actual pinned `unsloth/Qwen3.5-4B` weights on CPU under two Transformers
versions.

The short version:

1. The core mathematics (rotation invariant, Hadamard/butterfly/Cayley
   orthogonality, Lloyd–Max grids, GPTQ recursion, packed-bit accounting,
   perplexity and KL definitions) is correct, and the packed-byte accounting
   reproduces the real artifact exactly.
2. **Every KV-cache quality number recorded before 2026-09-01 is invalid.**
   The cache simulator shared linear-attention state between its "source" and
   "packed" decode passes under the Transformers version those Colab runs
   installed, producing a bit-width-independent KL floor of ≈0.5–0.9. The
   defect was reproduced on the real model, fixed on this branch, and guarded
   by a regression test. The headline cache claims in the paper draft and
   publication manifest must be withdrawn until the runs are repeated.
3. The 59.3 % "tensor-file reduction" compares against a source file that
   contains a 241 MB multi-token-prediction head which the Transformers model
   never loads and the export therefore never stores. Like-for-like the
   reduction is 58.3 %.
4. Several smaller issues (a code/scale inconsistency in GPTQ with 8-bit
   scales, an inert learned-sign arm, an over-optimistic tiered-cache
   simulation, statistically weak intervals) are documented below with the
   numbers that establish them.

---

## 1. What was verified and holds up

Checked by derivation against the code and, where marked, by running it.

| Item | Location | Verdict |
|---|---|---|
| `y = xWᵀ = (xRᵀ)(WRᵀ)ᵀ`; FWHT orthonormal and self-inverse; `RandomizedHadamard.inverse_activation` | `rotquant/rotate.py` | Correct; suite test passes. |
| Butterfly stage `[[c,s],[s,-c]]` orthogonal and self-inverse; θ=π/4 reproduces FWHT; reversed-stage inverse | `rotquant/rotate.py` | Correct. |
| Cayley map `(I−A)(I+A)⁻¹` with skew `A` | `rotquant/rotate.py` | Correct. |
| Spherical coordinate density `(1−u²)^{(d−3)/2}`, unit variance after `√d` | `rotquant/codebooks.py` | Correct; d=3 uniform anchor test passes. |
| Lloyd–Max Gaussian grids: 2/3/4/8-bit MSE = 0.1175 / 0.0345 / 0.0095 / 5.2e-5 (run) | `rotquant/codebooks.py` | Correct and monotone; the 8-bit value is 25 % above Panter–Dite because 200 Lloyd iterations from a ±2.5 init do not fully converge at 256 levels. Harmless. |
| Panter–Dite constant `√3π/2 ≈ 2.72` in `turboquant_mse_bound` | `rotquant/codebooks.py` | Correct as a high-rate approximation; the *docstring* still calls it "Theorem 1 … MSE ≤", which `docs/how_rotquant_works.md` already corrects. |
| GPTQ: upper-Cholesky of the damped `H⁻¹`, `err=(w−q)/Hinv[i,i]`, in-block rank-1 and lazy inter-block updates, `H'=RHRᵀ` for rotated inputs, act-order via a static-group permutation | `rotquant/quantize.py`, `rotquant/patch.py` | Correct; matches the reference algorithm; identity-Hessian reduces exactly to rounding (test). Lazy per-group scale refit from the error-fed weights is a legitimate variant. |
| Asymmetric QJL estimator `√(π/2k)‖r‖ (xG)·sign(rG)` | `rotquant/quantize.py`, `rotquant/linear.py` | Exactly unbiased. Its variance is the problem (§3.4). |
| Length correction `α=‖w‖²/⟨w,ŵ⟩` folded into scales | `rotquant/quantize.py` | Correct; codes and rate unchanged. |
| Mean-bias correction `b += μ'(W'−Q)ᵀ` with `μ'=μRᵀ` | `rotquant/linear.py` | Correct sign and basis. |
| LSB-first packing / unpacking, int32 word padding | `rotquant/pack.py` | Correct. |
| Bit accounting: 3,565,158,400 quantisable language-layer weights in the pinned source checkpoint × 4.125 bpw / 8 = 1,838,284,800 bytes, equal to the recorded `packed_weight_bytes` (run against the real safetensors headers) | `rotquant/utils.py`, `scripts/run_experiment.py` | Exact. |
| Sliding-window perplexity: first-token masking, sum-of-NLL / scored tokens, early-stop bookkeeping | `rotquant/eval/perplexity.py` | Correct. |
| Teacher KL direction `KL(source‖candidate)`, `T²` scaling, top-1 counts (0.92710 = 1895/2044 = 4 prompts × 511 targets) | `rotquant/eval/logit_fidelity.py` | Correct. |
| Rotated-K/rotated-Q logit preservation, value-basis accumulation with one inverse rotation | `rotquant/kv_cache.py` | Correct. |
| Hessian-weighted rotation objective `tr(E H Eᵀ)/tr(W H Wᵀ)` with `E=(v−q)R` | `rotquant/train_rotation.py` | Correct. |
| Dynamic allocator's local error is now a summed output distortion (commensurable across layers); greedy per-byte downgrade with uniform restore | `rotquant/dynamic.py`, `rotquant/eval/kv_cache.py` | As intended (the 2026-08-31 review's §2.9 fix is in). |
| Reported aggregates: joint mean 14.5548 (+4.710 %), worst +5.775 %, matched cache ratios 0.803/0.835/0.768 (mean 0.8022, worst 0.8352), `audit_publication.py` checks | `docs/experiment_log.md`, `paper/generated/audit_report.json` | Arithmetic reproduces. The *inputs* to the cache ratios are invalid (§2). |
| Test suite | `tests/` | 428 passed (C++ conformance suite not built here). |

---

## 2. Critical finding: the recorded KV-cache results measure a state-sharing defect, not quantization

### 2.1 The recorded numbers are physically impossible for quantization error

From `docs/experiment_log.md` (Qwen3.5-4B, Gaussian codebook, group 64, fp16
scales, self-teacher KL over 64 continuation tokens):

| Recorded cache recipe | KL |
|---|---:|
| Uniform K8/V8 (first seed-0 matrix) | 0.582664 |
| Uniform K2/V2 (corrected seed-0) | 0.575244 |
| Uniform K4/V4 (corrected seed-0) | 0.548273 |
| Uniform K4/V4, seeds 1 / 2 | 0.843600 / 0.678700 |
| Uniform K2/V2, seeds 1 / 2 | 0.790700 / 0.620100 |
| Source fp16 weights + uniform K4/V4 | 0.8618 |
| 1,024-token uniform K3/V3 / K4/V4 | 1.540281 / 1.553477 |

An 8-bit rotated Gaussian cache reconstructs K/V to a relative error of about
4×10⁻⁵ (measured `prefill_kv_nmse` = 4.7×10⁻⁵), yet it produced the same KL as
a 2-bit cache (NMSE 0.11). K4/V4 is *worse* than K2/V2 in two of three seeds.
No amount of prompt noise explains a 2–8-bit-invariant KL of ≈0.5–0.9 with
top-1 agreement of ≈0.72; the two decode passes must have differed by
something other than the cache codes.

### 2.2 Replication on the real model

`rotquant.eval.kv_cache.evaluate_kv_cache` was run on the pinned
`unsloth/Qwen3.5-4B` weights (bf16, CPU, source weights, prompt 256,
2 prompts × 8 continuation tokens, no fp16 tiers — the old protocol).

| Transformers | Simulator | K8/V8 KL | K4/V4 KL | K2/V2 KL | top-1 (8-bit) | `non_kv_state_bytes` |
|---|---|---:|---:|---:|---:|---:|
| 5.9.0 (pinned in `uv.lock`) | as committed | 5.0×10⁻⁴ | 6.3×10⁻³ | 6.5×10⁻² | 0.94 | 26,738,688 |
| 5.16.1 | as committed | **0.877** | **0.891** | — | 0.69 | **0** |
| 5.16.1 | fixed (this branch) | 3.6×10⁻⁴ | 6.9×10⁻³ | — | 1.00 | 51,904,512 |

Under 5.9.0 the metric is bit-monotone with roughly a decade per two bits.
Under 5.16.1 the *identical* code gives the recorded floor: K8/V8 0.877 is
within noise of the recorded source-weight control (0.8618), the next-token
NLL of the "source" pass itself rises from 2.71 to 3.23 nats, and the
simulator reports no linear-attention state at all. A tiny random hybrid
Qwen3.5 shows the same signature (KL 1.1×10⁻⁶ at 8 bits under 5.9.0; KL 0.1307
at every bit width with top-1 0.000 under 5.16.1).

### 2.3 Mechanism

Transformers 5.9.0 stores each linear-attention layer's `conv_states` and
`recurrent_states` as tensor attributes of the cache layer object. By 5.16.1
they are `dict[int, Tensor]` attributes, and both `update_conv_state` and
`update_recurrent_state` (plus `causal_conv1d_update` in the single-token
decode path) update the stored tensors **in place** with `copy_`.

`_clone_cache` only cloned tensor-valued attributes and shallow-copied
everything else. A shallow-copied dict shares its tensors, so the "packed"
cache and the "source" cache shared every conv and recurrent state. At each
decode step the source pass advanced the shared state, the candidate pass
advanced it again, and both passes then drifted into corrupted trajectories
whose divergence has nothing to do with the K/V code width. `_cache_tensor_bytes`
had the same blind spot, which is why the affected runs report
`non_kv_state_bytes = 0` and why the recorded "deployed cache bytes" for the
1,024-token winner (6,815,744) equal the packed K/V bytes alone.

### 2.4 Which runs were affected

The KV notebooks that produced the seed-0 matrix, the three-seed validation,
the 1,024-token confirmation, the frozen-map transfer study, the whole-system
joint matrix and the matched K4/V4 follow-up all install
`pip install -U "transformers>=5.9,<6"`. Transformers 5.9.0 was released on
2026-05-20 and 5.16.1 on 2026-08-26; those notebooks ran on 2026-08-29/30 and
would have resolved to 5.16.x. The later W4A8/E8 stage pins
`transformers==5.9.0`, recorded 5.9.0 in its raw JSON, reports 26.7 MB of
non-KV state, and gives a plausible KL of 0.030 for 2-bit E8P with fp16 tiers.
That run is not affected. The archived raw JSONs on Drive record
`environment.library_versions.transformers`; confirming 5.16.x there closes
the loop.

Consequently the following are unsupported and should be withdrawn or
re-measured:

- the seed-0 and three-seed uniform-vs-dynamic K/V tables and the
  "dynamic 3.25 bpv reduces KL by 13–17 %" conclusions;
- the 1,024-token confirmation and the claim that key/value sensitivity
  reverses with context length (K2.875/V3.125 → K3.375/V2.625);
- the frozen-map transfer table (24.0 % / 10.6 % reductions);
- the matched-control table (16.5–23.2 % reductions; mean ratio 0.802);
- the `cache_kl` term inside the joint score that selected
  `uniform_w4__frozen_mixed_3.25`, and the "diagnostic winner" status of that
  recipe (its *perplexity* rows are unaffected and remain valid);
- the specific per-layer 2/3/4-bit map embedded as deployment metadata in
  `HallD/qwen35-4b-rotquant-joint` and in the joint native GGUF;
- the abstract, contributions 2–3, §5.3–5.4, Tables `cache-transfer` and
  `matched`, and the appendix map in `paper/main.tex`; the `cache_transfer`
  and `matched_cache` blocks of `paper/data/publication_results.json`.

Weight-only results (WikiText-2/C4 perplexity, teacher KL, top-1,
trajectories, GPTQ promotion, the W4A8 comparison) do not touch the simulator
and stand.

### 2.5 What changed on this branch

- `rotquant/eval/kv_cache.py`: `_clone_cache` now clones every tensor
  reachable through dict/list/tuple attributes of the cache and its layers,
  and fails closed if any storage is still shared between source and clone;
  `_cache_tensor_bytes` counts the same tensors, so `non_kv_state_bytes`
  is correct on both cache layouts.
- `tests/test_kv_cache_bit_monotone.py`: on a tiny hybrid Qwen3.5 the
  simulator must report non-zero recurrent state, KL < 10⁻⁴ at 8 bits, and
  strictly decreasing KL from 2 to 4 to 8 bits. It fails on the previous code
  under 5.16.1 and passes with the fix under both 5.9.0 and 5.16.1.
- Mandatory endpoint check (`KVCacheEvalConfig.endpoint_check_bits=8`,
  `endpoint_max_kl=0.01`): a uniform 8-bit Gaussian cache is evaluated on the
  held-out calls before any candidate or allocator; a run whose endpoint KL
  exceeds the limit raises. `endpoint_check_bits: null` disables it
  explicitly.
- Every Colab notebook pins `transformers==5.9.0`.
- The tiered simulator tracks absolute positions: decode writes are no
  longer their own sequence, sinks are decided by absolute position, and rows
  are packed once when they leave the recent window (§3.3).
- GPTQ with 8-bit scales snaps refit scales onto the frozen affine grid and
  every path retains its encoded scale triple verbatim (§3.2).
- The Hessian rotation objective is gated against seeded FWHT under the exact
  deployed quantizer, and learned-sign logits start at ±0.1 (§3.2).
- The publication manifest, paper draft, README, experiment log and science
  guide mark the cache results as withdrawn and report like-for-like storage
  (§3.1). `docs/roadmap.md` reserves the name QRAT for the future
  quantization-and-rotation-aware training method.

### 2.6 What must happen before any cache claim is made again

1. Pin `transformers` explicitly in every notebook (the new stage already
   does) and keep the storage-sharing guard.
2. Add the 8-bit endpoint to every cache experiment: a K8/V8 row whose KL is
   not ≲10⁻³ invalidates the run before any recipe is compared.
3. Re-run the seed-0 matrix, three-seed validation, cross-context transfer
   and matched controls with the fixed simulator, then re-select the map.
4. Expect much smaller effects. Under the fixed simulator K4/V4 costs KL
   6×10⁻³ and K2/V2 6.5×10⁻² on the source model, so differences between
   3.25-bpv recipes will be at the 10⁻³ level and will need far more than 64
   tokens per seed to resolve (§4).

---

## 3. Other findings

### 3.1 Storage claims must exclude the multi-token-prediction head

The pinned source index totals 9,319,737,856 bytes, all bf16, and includes a
`mtp.*` head of 241,199,104 bytes. `Qwen3_5ForConditionalGeneration` has no
`mtp` submodule, so those tensors are dropped at load and are absent from
`rotquant_model.safetensors`. The unexplained 237,955,440-byte gap between the
4.0277 GB "logical estimate" and the 3.7898 GB export is this head. Like for
like (source without MTP, 9,078,538,752 bytes — which is also the runner's own
`complete_persistent_model_bytes` for the source arm) the reduction is
**58.26 %**, not 59.34 %. The `docs/unsloth_qwen35_4b_comparison.md`
byte comparison should apply the same policy explicitly (the native exporter
already sets `no_mtp`).

### 3.2 The catastrophic "optimized W4" arm: what the code can and cannot explain

Recorded: PPL 243 / 412, KL 3.05, top-1 31 % after bundling learned
Hessian-objective butterflies, shared rotations, learned signs, 8-bit scales
and mean-bias correction. Mechanisms checked here:

- **Learned signs are inert by construction.** Logits start at ±1 and Adam
  with lr 10⁻³ for 200 steps moves them by at most ≈0.2, so no sign can flip
  (measured flip rate 0.0). The arm tested nothing. Initialise logits near
  zero (e.g. ±0.05) or raise the step size if the idea is to be tested.
- **GPTQ with `scale_bits=8` assigns codes against different scales from the
  ones it stores.** Inside `_gptq` each group's refit scale is 8-bit encoded
  as a column block (256 rows of one group); the final `_storage_scales`
  re-encodes the `[out, groups]` matrix row-major in different 256-blocks.
  Measured: the dequantised weight differs from the reconstruction the codes
  were chosen for by 1.1×10⁻³ relative Frobenius norm (0 for every other
  path). Small, but it breaks the "codes are assigned against stored scales"
  invariant and should be fixed before the `scale8_w4` ablation arm runs,
  otherwise that arm conflates the bug with the method.
- **Mean-bias correction did not misbehave in a synthetic sink test**
  (typical-token output MSE ×1.00 with 1/512 tokens carrying a 400× channel
  outlier), so it is not an obvious suspect, but its benefit will also be
  ≈0 for post-norm inputs whose mean is small.
- **The Hessian objective has no deployed-quantizer gate.** Only the
  `activation` objective calls `select_butterfly_checkpoint`; the `hessian`
  objective trains under `assignment_scale: rms` without GPTQ and keeps the
  best *proxy* step (mean best step 104/200, 2.3 % proxy improvement) with no
  packed-versus-FWHT comparison. That cannot by itself produce a 20× PPL
  collapse, but it means a worse-than-FWHT rotation would be deployed
  silently. Add the same fail-closed gate the activation objective has.
- None of these individually explains PPL 243. The planned single-factor
  ablation (`qwen35_4b_w4_factor_ablation_cuda.yaml`, with the fail-fast KL
  gate) is the right next step; run the two fixes above first.

### 3.3 Tiered KV simulation is optimistic beyond the recent window

`quantize_kv` decides tiers from the position inside the tensor it is given.
A single-token decode write has sequence length 1, so with `sink_tokens ≥ 1`
every decode-step write is stored in fp16 (checked: `qweight is None`,
8 fp16 rows) and is never re-quantised when it leaves the window; the last
`recent_window` prefill rows are likewise frozen in fp16 forever. The E8P
smoke (32 continuation tokens ≤ 32-token window) is exact; the corrected 8k
config (64 continuation tokens, 32-token window) will overstate quality and
understate bytes for the second half of each continuation. Track absolute
positions and re-pack rows as they age out.

### 3.4 The TurboQuant weight sketch is a designed loser for weights

The 2026-08-31 review derived that the QJL sketch multiplies squared output
error by ≈πd/(2k). Measured with the shipped `QuantLinear` on random data:
×104 at d=4096, ×270 at d=11008 (theory 100.5 and 270.2), ×3.2 at d=128.
`e3b_turboquant_bias.yaml` is therefore a null control, not a candidate; label
it so.

### 3.5 Statistical reporting

- Paired *token-level* bootstrap intervals (logit fidelity, cache KL) treat
  tokens as independent; tokens within a 512-token C4 document are not, so the
  intervals are too narrow. The stage summary's *prompt-level* intervals use
  `paired_samples: 4`; a percentile bootstrap of four values is not a 95 %
  interval and should not be quoted (e.g. the A8 KL delta
  `[0.00101, 0.00243]`).
- The "±" values for three-seed OPT/Qwen results are population standard
  deviations (n=3), ~18 % below the sample SD; say which.
- 64–128 continuation tokens per seed cannot resolve the 10⁻³-level KL
  differences the corrected cache study will produce; budget ≥10⁴ tokens.
- Selection on `mean_teacher_kl` per byte followed by evaluation on disjoint
  batches is sound; the endpoint-equivalence checks (dynamic-at-8-bit ≡
  uniform-8-bit) are good hygiene, but an *8-bit-near-zero* check is the one
  that would have caught §2.

### 3.6 Points carried over from the 2026-08-31 review that still apply

- E7's `mismatched` control is a single-layer identity violation, not a test
  of cross-layer consistency (`patch.py`).
- Per-row (`turboquant`) scales are mismatched with block-diagonal rotation.
- The frozen-assignment surrogate gradient in butterfly/block training is a
  surrogate; whether STE does better is untested.
- The spherical codebook at d=128 is numerically indistinguishable from the
  Gaussian grid (the repo's own screening shows this); the arm can be retired.
- `normal_float` is not bitsandbytes NF4.

### 3.7 Hypothesis ledger (README E1–E8) versus evidence in the repository

| Hypothesis | Evidence present | Status |
|---|---|---|
| E1 rotation transfers; FWHT ≈ dense | FWHT vs none on OPT-125M/1.3B and Qwen3.5 (large effect). No dense or learned-rotation model result. | Half tested. |
| E1b/E1c learned butterflies | OPT-125M single-seed MPS ablations; Qwen bundled arm catastrophic (confounded). | Inconclusive. |
| E2 Gaussian > uniform grid | Cache only, and that data is invalid (§2). Calibrated ≈ Gaussian on weights. | Untested on weights. |
| E3 deterministic residual > QJL | No model run; the arm is analytically a loser (§3.4). | Untested. |
| E4 `mse_search` > RMS | No model run reported. | Untested. |
| E5 GPTQ helps with real activations | Three seeds, paired: +2.1 % vs +4.2 % WikiText-2, KL 0.021 vs 0.045. Raw records not in the repo. | Supported (development scale). |
| E6 scalar ceiling / vector | d=2 vector W3 beat scalar locally, poor absolute quality; E8P only smoke-tested. | Open. |
| E7 consistency trap | Not run; control is a straw man. | Untested. |
| E8 footprint and speed | Artifact bytes verified; no packed-runtime memory or throughput. | Footprint only. |

---

## 4. Recommended order of work

Items 1, 3 (code side) and 4 are done on this branch (§2.5); the GPU work
remains.

1. Merge the simulator fix, endpoint check and regression tests; the
   notebooks now pin Transformers.
2. Re-run the cache matrix, transfer and matched-control studies with the
   fixed simulator and ≥10⁴ evaluated tokens per seed; re-select and re-embed
   the deployment map; update the experiment log, publication manifest and
   paper. Until then every cache number is marked withdrawn.
3. Run the factor ablation now that the GPTQ/8-bit scale path is exact, the
   learned-sign arm can flip, and the Hessian objective is gated.
4. Storage is now quoted like-for-like (58.26 %); exporting the MTP head
   would allow a true whole-file number instead.
5. Replace token-level intervals with document-level (cluster) bootstraps and
   never report a bootstrap on four samples; the stage summary now flags
   intervals below 20 paired samples as unreliable.
6. Reserve the name QRAT for the future quantization-and-rotation-aware
   training method (roadmap entry added); the properly budgeted recovery
   study in `configs/qwen35_4b_recovery_cuda.yaml` is its first candidate,
   and the STE-through-rotation question from the 2026-08-31 review belongs
   to its design.

## 5. Reproduction notes

- Test suite: `uv sync --extra dev && uv run pytest tests -q` (428 passed;
  the C++ conformance suite needs `cmake`).
- Real-model replication: `uv pip install transformers==5.9.0 accelerate`,
  download `unsloth/Qwen3.5-4B` at revision `3764fa35…`, load with
  `AutoModelForMultimodalLM` in bf16 on CPU, and call
  `evaluate_kv_cache(model, batches, KVCacheEvalConfig(bits=b, group_size=64,
  rotation_block=128, batches=2, prompt_len=256, continuation_len=8,
  skip=0), "cpu")` for b ∈ {8, 4, 2}. Repeat with `transformers==5.16.1`
  (needs `tokenizers>=0.23.1`) to reproduce the floor on the pre-fix code.
- Source checkpoint byte audit: parse the safetensors headers of the two
  shards and sum bytes by prefix; `mtp.*` totals 241,199,104 bytes.
