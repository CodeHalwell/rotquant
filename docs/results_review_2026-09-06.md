# Results review — 2026-09-06: why the Unsloth comparison is being lost

## Verdict

The quantiser is not the reason RotQuant loses to Unsloth's UD-Q4_K_XL at
"matched complete bytes". The byte budget is.

RotQuant's artifact carries the 635.7M-parameter tied vocabulary matrix in
fp16 (1,271,398,400 bytes, 34% of the W4 artifact). Unsloth's GGUF stores the
same tensor as Q6_K (521,472,000 bytes). Because the allocator treats the
fp16 embedding as a fixed cost, matching Unsloth's 3,584,533,344-byte bundle
forces RotQuant's transformer backbone down to about 3.6 bits per weight,
while the Unsloth "Q4" backbone actually sits at 5.31 bits per weight. On
average, every backbone weight in the v3 comparison was quantised with 1.7
fewer bits than its competitor. No allocator, codebook, or group size can
recover that.

Everything below is derived from committed run records, the pinned model
config, and the header of the pinned Unsloth GGUF. Nothing was re-run on a
GPU.

## 1. Byte decomposition of the two artifacts

Qwen3.5-4B (`unsloth/Qwen3.5-4B`, revision `3764fa35…`): hidden 2560, vocab
248,320, tied word embeddings, 32 layers (24 linear-attention, 8 full
attention), intermediate 9,216. RotQuant quantises exactly 3,565,158,400
backbone weights (`fp16_weight_bytes / 2` in the promoted-W4 record). The
Unsloth GGUF's attention, FFN and `ssm_out` tensors sum to the same
3,565,158,400 weights, so the backbones are like for like.

| Component | Unsloth UD-Q4_K_XL | RotQuant scale8 W4 (uniform) | RotQuant v3 Pareto (seed 0) |
|---|---:|---:|---:|
| Tied embedding, 635,699,200 params | **Q6_K, 521,472,000 B (6.56 bpw)** | fp16, 1,271,398,400 B (16 bpw) | fp16, 1,271,398,400 B (16 bpw) |
| Backbone, 3,565,158,400 params | **2,367,959,040 B (5.31 bpw)** | 1,810,866,880 B (4.06 bpw) | 1,618,188,789 B (**3.63 bpw**) |
| Vision tower | 672,423,616 B (`mmproj-F16.gguf`) | 667,028,480 B (fp16) | 667,028,480 B (fp16) |
| `in_proj_a/b`, norms, conv, decay vectors, signs, codebooks | 11,710,464 B | 10,574,656 B | 10,574,656 B |
| Header / tokenizer / config overhead | 10,968,206 B | ~20.44 MB when exported | 20,442,482 B |
| Complete artifact | 3,584,533,344 B | 3,759,868,416 B registered (+~20 MB exported) | 3,587,632,807 B |
| Mean teacher KL (24-prompt C4) | 0.01189 | 0.01680 | 0.03113 |

Sources: `research/results/raw/qwen35_4b_w4a8_e8_8ad3b8e6c809/…_s0.json`
(`packed_weight_bytes`, `registered_model_bytes`, `fp16_weight_bytes`),
`research/results/qwen35_4b_allocator_v3_c9efd2d56774.json`, the model
`config.json` at the pinned revision, and the tensor table in the first
11 MB of `Qwen3.5-4B-UD-Q4_K_XL.gguf` at revision `e87f1764…` (parsed with
`scripts/inspect_gguf_types.py`).

The 1,948,988,736 "registered" bytes that RotQuant never quantises close to
the byte against the pinned config: fp16 embedding 1,271,398,400 (65.2%),
fp16 vision tower 667,028,480 (34.2%), `in_proj_a/b` 7,864,320, norms, conv
and decay vectors 1,923,392, Hadamard sign buffers 774,144. The vision tower
is fp16 on both sides, the small tensors are within 1.2 MB of each other,
and the container overheads net to about 4 MB in Unsloth's favour. The
embedding is the whole difference.

## 2. What "Q4_K_XL" actually contains

Per-layer tensor types from the GGUF header (`-` means the projection does not
exist in that layer type):

| Layer | attn_qkv / attn_gate | ssm_out | attn_q / k / v / o | ffn_gate / up | ffn_down |
|---:|---|---|---|---|---|
| 0, 1, 4, 5, 9, 14, 18, 30 | Q5_K / Q5_K | Q8_0 | – | Q4_K | Q6_K |
| 2, 6 | Q5_K / Q5_K | Q8_0 | – | Q5_K | Q6_K |
| 8, 10, 17, 20, 22, 24, 26, 28, 29 | Q5_K / Q5_K | Q8_0 | – | Q4_K | Q4_K |
| 12, 13, 16, 21, 25 | Q5_K / Q5_K | Q8_0 | – | IQ4_XS | Q5_K |
| 3, 19, 31 (full attention) | – | – | Q5_K / Q5_K / Q6_K / Q4_K | Q5_K | Q6_K |
| 7, 11, 15 (full attention) | – | – | Q4_K / Q4_K / Q6_K / Q4_K | Q4_K | Q4_K |
| 23, 27 (full attention) | – | – | Q4_K / Q4_K / Q6_K / Q4_K | Q4_K | Q6_K |
| `token_embd` | Q6_K | | | | |

Two things follow. First, the name is marketing: the average backbone rate is
5.31 bpw; 84 of the 200 backbone matrices (49% of the weights) are at a
nominal 4-bit rate (Q4_K 4.5 bpw, IQ4_XS 4.25 bpw), 69 are Q5_K, 23 are
Q6_K and 24 are Q8_0. Second, Unsloth's sensitivity choices agree with what allocator v3
measured: the linear-attention output projection is kept at Q8_0 in all 24
layers, `ffn_down` is Q6_K in most layers, and the last layer's MLP is
upgraded. The v3 "stable sensitive region" (layer-31 MLP, early `down`,
linear-attention projections) is the same list. The allocator's ranking
signal is fine; it simply had no bytes to act on it, which is also why
forcing W6/W8 islands "hurt": at 3.6 bpw every upgrade had to be paid for by
a damaging downgrade elsewhere.

## 3. Rate-distortion arithmetic

Within the project's own data, one bit per weight buys roughly 4.4x in mean
KL (uniform W3 0.07377 versus uniform W4 0.01665, same pipeline, seed 0); the
Lloyd-Max distortion ratio between 3 and 4 bits is 3.64x, so KL is slightly
super-linear in weight MSE here. Unsloth's backbone has 1.25 more bits than
uniform scale8 W4 and 1.68 more than the v3 Pareto recipe. On that slope a
1.25-bit advantage is worth around 6x in KL. Unsloth's actual advantage over
uniform W4 is 1.41x, and over the 3.63-bpw Pareto recipe 2.62x. Measured per
bit, the RotQuant primitive (FWHT + Gaussian codebook + MSE scale search +
act-order GPTQ) is already ahead of imatrix K-quants on this model. The 2.62x
loss is entirely explained by where the bytes went.

What the same total budget looks like once the embedding is quantised
(RotQuant's own scale8 accounting, 20.44 MB export overhead, vision and small
tensors unchanged):

| Embedding format | Embedding bytes | Backbone budget at Unsloth's total | Backbone rate |
|---|---:|---:|---:|
| W8 g128 scale8 | 640,743,200 | 2,245,757,326 | 5.04 bpw |
| W6 g128 scale8 | 481,818,400 | 2,404,682,126 | 5.40 bpw |
| W5 g128 scale8 | 402,356,000 | 2,484,144,526 | 5.57 bpw |

Concretely: **uniform W5 backbone + W8 embedding totals 3,595,288,018 bytes,
0.30% above the Unsloth bundle** (inside the registered 1% gate), and
**uniform W5 + W6 embedding is 4.1% below it**, leaving ~150 MB to lift the
`ssm_out`/`down` projections to W6. The Lloyd-Max distortion ratio from 4 to
5 bits is 3.8x; on the in-project KL slope a uniform W5 backbone would be
expected near 0.004–0.0045 mean KL before the embedding's own error is added.
Unsloth is at 0.0119. This is a projection, not a result, but the direction
is not in doubt.

## 4. Consequences for the allocator programme

1. **Allocators v1–v3 were solving the wrong problem.**
   `target_artifact_bytes: 3584533344` with an fp16 embedding *is* a
   3.6-bpw constraint on the backbone (the `target_bpw: 3.5` next to it is
   inert; section 5), and that is a sub-W4 constraint. The Algorithm Lab had already
   found that a 3.625-bpw teacher-guided recipe fails free-running
   diagnostics (worst 32-token agreement 7.81%). The v3 result — a 72.8%
   improvement over random allocation that still loses to both uniform W4
   and Unsloth — is exactly what a correct allocator produces under an
   impossible budget.
2. **Allocator v4 as registered will also lose.** Its palette is W3/W4/W5
   only, chosen because v3 "showed" W6/W8 were useless. That inference does
   not survive the budget correction: at 5.3 bpw the palette must contain W5,
   W6 and W8 (the formats Unsloth actually uses), and the calibrated-codebook
   and group-64 questions are second-order. Do not spend an A100 day on v4 in
   its current form.
3. **The v3 conclusions to keep:** the sensitivity ranking, the exact-byte
   solver, the fingerprinting and paired-confirmation machinery. The
   conclusions to withdraw: "high-precision islands hurt" (true only at
   3.6 bpw) and "provider competitiveness not achieved" (not tested at a fair
   budget).
4. **The uniform W4 framing is also wrong.** "Uniform W4 is 4.89% larger than
   Unsloth" is true only because of the fp16 embedding; its backbone is
   1.25 bits *cheaper* than Unsloth's.

## 5. What the pipeline is missing

- Adapter discovery yields only `nn.Linear` modules
  (`rotquant/adapters.py:70-76`), so `embed_tokens` is never a candidate
  under any config; `PatchConfig` additionally excludes `lm_head`/`embed_out`
  by default (`rotquant/patch.py:64`) and every Qwen config restricts
  `include` to `model.language_model.layers.`. Even if `lm_head` were
  included, `QuantLinear` replaces only the `nn.Linear`; the tied
  `embed_tokens` parameter would keep its own fp16 copy, so the artifact
  would not shrink.
- The allocator then books every non-target tensor as a fixed cost
  (`rotquant/dynamic.py:1563-1586`): `target_bytes = 3,584,533,344 −
  20,750,000`, `fixed_search_bytes = 1,948,214,592`, leaving 1,615,568,752
  bytes for codes, scales, signs and codebooks, i.e. 3.62 bpw. With
  `target_artifact_bytes` set, `target_bpw: 3.5` is inert (`dynamic.py:61-62`,
  `:1576`, `:1595`); the number in the config is documentation, not a
  constraint.
- The GGUF exporter does have a tied-vocabulary path
  (`native_tied_tensor`, `rotquant/gguf.py:47`), but it is fixed at 4-bit,
  `error_comp="none"`, RMS scales, and it is only used by the llama.cpp
  export, never by the Transformers artifact or by any KL measurement. A
  4-bit, RMS-scaled, uncalibrated `lm_head` is the wrong operating point for
  a KL-sensitive output projection; Unsloth uses Q6_K.
- `docs/competitive_eval.md` lists "whether embeddings and output heads are
  included in artifact bytes" as a protocol item, but no comparison note
  decomposes the artifact. The 20 MB export overhead was measured and
  reserved; the 1,271 MB embedding was not looked at.

## 6. Code-level audit

Two independent read-throughs were made for this review: the quantiser core
(`quantize.py`, `codebooks.py`, `rotate.py`, `calibrate.py`, `linear.py`,
`patch.py`, `pack.py`) and the evaluation/accounting path. Neither found a
defect that would inflate the KL of the deployed fp16 + FWHT + Gaussian +
MSE-search + act-order GPTQ recipe. The gap in section 1 is not a bug in the
maths.

### 6.1 Quantiser core: verified correct on the deployed path

| Item | Status | Evidence |
|---|---|---|
| 16-level Gaussian Lloyd-Max codebook | fine | `codebooks.py:87-118`, `:265-306`; measured unit-Gaussian MSE 0.0095011 against an exact fixed-point solve of 0.0095010; midpoint thresholds via `bucketize`. |
| MSE scale search and stored scales | fine | `quantize.py:692-721` (41-point grid, 0.5–1.5 × RMS); codes are assigned against the *decoded* stored scale (`:805-809`, `:830`); GPTQ refits every group from the error-fed weight and re-snaps to the stored grid (`:949-1015`). Bit-identical assign/dequant is covered by `tests/test_scale_storage_consistency.py`. |
| GPTQ Hessian basis | fine | `patch.py:418-421` → `_internal.py:46-60` forms R H Rᵀ with the same `Rotation` object (same seed, sign buffer and block) used for the weight (`linear.py:432`) and the activation (`linear.py:227`); checked numerically against a dense rotation to 1.5e-7. |
| GPTQ update | fine | `quantize.py:1039-1056` damping 1% of mean diag with retry; upper Cholesky of H⁻¹; in-block rank-1 and cross-block error propagation match reference GPTQ; act-order keeps static original groups (`:948`, `:1024-1037`). `tests/test_gptq_identity.py` pins H = I → RTN and blocked == column-wise. |
| Rotation | fine | Block-128 FWHT normalised by 1/√128 on the weight's input dim and identically on the activation; invariance verified to 3e-15 in float64; groups coincide with rotation blocks for every Qwen3.5-4B width. |
| Forward path | fine | `linear.py:226-229`, `:276-299`: one rotation, fp32 dequant, cast to the activation dtype, no inverse rotation, no double rotation. |

Synthetic check (512×512, outlier covariance with condition number 8e7,
g128, W4): the deployed recipe reaches output NMSE 0.00062 (Gaussian
weights) and 0.00053 (heavy-tailed weights) against 0.00441/0.02291 for
unrotated Gaussian-codebook GPTQ and 0.00558/0.02892 for correctly scaled
uniform GPTQ. Rotation plus GPTQ is doing what it is supposed to.

Latent issues found on the way (none active in the runs behind the numbers
above):

1. **bf16 loading re-rounds fp16 scale metadata** (`linear.py:114-143`,
   `checkpoint.py:723`). `load_packed_model(..., dtype=bf16)` casts `scales`,
   `scale_offsets` and `scale_steps` through the `_apply` closure, so codes
   assigned against fp16 scales are dequantised against bf16-rounded ones:
   up to 0.39% relative on 16-bit scales, 2.79% → 2.97% on 8-bit decode. The
   fp16 in-process runs never hit this; a bf16 deployment of a bf16-native
   Qwen would. Keep metadata in fp16/fp32 regardless of the model dtype.
2. **Pure-torch FWHT can overflow fp16** (`rotate.py:150-161`): seven
   unnormalised butterfly stages run in the input dtype before the final
   1/√d, so two entries of 4e4 in one block produce `inf`, while the
   normalised result would fit. Only reachable without
   `fast_hadamard_transform`, and it would surface as NaN, not as mildly
   worse KL.
3. **8-bit scale encoding is linear, not log-domain** (`quantize.py:400-458`):
   blocks of 256 consecutive scales share an fp16 offset and step, so the
   smallest scale in a block carries a relative error of about
   (max/min − 1)/510. Ratio 14 → ≤2.6%; ratio 100 → ~16% (measured). The
   uniform scale8 W4 arm was not measurably worse than fp16 scales, so this
   is not biting today, but the per-block max/min ratio on the real
   `down`/`ssm_out` layers should be checked before scale8 is used on the
   most sensitive projections at W6/W8.
4. **The in-library `uniform` control cannot reach an absmax scale**
   (`codebooks.py:229-233`, `quantize.py:123`, `:706`): the grid spans
   [−1, 1] and the search caps the scale at 1.5 × RMS, i.e. clipping at
   ≤1.5σ. On Gaussian weights that gives NMSE 0.047 versus 0.0099 with a
   [1.5, 4] × RMS search or 0.0138 for plain absmax RTN. Any statement of the
   form "Gaussian codebook beats uniform" (README hypothesis E2) that used
   this arm overstates the codebook's advantage about five-fold; the honest
   gap to a properly scaled int4 grid is 20–40% in MSE. Fix the search range
   for non-Gaussian codebooks before E2 is reported.
5. Minor: `linear.py:432` rotates the weight in the source dtype before the
   fp32 cast (0.07% RMS error in fp16, 0.5% in bf16; energy ≤3e-5, well
   below the 4-bit error, but the cast belongs first). `calib_seq_len: 512`
   with `n_calib: 128` gives 65k calibration tokens, so the Hessian of a
   9,216-wide `down` projection is estimated from ~7 samples per dimension
   against ~28 in the standard 128 × 2048 GPTQ setting; Unsloth's imatrix
   used 80 × 512-token chunks, so this is not a competitive disadvantage,
   but it is a source of seed variance.

### 6.2 Evaluation and byte accounting

| Item | Status | Evidence |
|---|---|---|
| Unsloth comparator never inspects the GGUF | accounting flaw | `scripts/run_unsloth_qwen35_4b_kl.py:56-70`, `:520` record name, bytes and SHA-256 and sum two file sizes; the header's tensor types (Q6_K embedding, 5.31-bpw backbone) are not read, so the comparison note calls a 5.3-bpw artifact a "Q4" one. |
| RotQuant byte identities | fine | `_persistent_registered_tensors` de-duplicates the tied tensor by `id` (`run_experiment.py:276-296`); `complete = registered + packed + codebook` with the fp16 cache excluded (`:745-759`); packed codes/scales/8-bit metadata counted once (`linear.py:443-467`, `quantize.py:301-347`). MTP head absent on both sides (RotQuant registered params = GGUF text + mmproj + 160). |
| Container asymmetries | fine | RotQuant counts the vision tower 5.4 MB smaller than `mmproj-F16` and reserves 9.8 MB more container overhead than the GGUF header; net ≈ 4 MB (0.12%) in Unsloth's favour. |
| KL / top-1 / NLL definitions | fine | Both sides: KL(teacher ‖ student) at T = 1 over the full vocabulary, targets `ids[1:]`, 511 positions per 512-token prompt, logits upcast to fp32 (`rotquant/eval/logit_fidelity.py:88-135`; `run_unsloth_qwen35_4b_kl.py:214`, `:264-269`, `:332`, `:379`). Prompt hashes are enforced equal (`compare_qwen35_dynamic_to_unsloth.py:52-54`); 2,044 = 4 × 511 and 12,264 = 24 × 511 confirm alignment. |
| Asymmetries in the KL | fine, favour Unsloth's number being *pessimistic* | Unsloth's KL includes its Q6_K embedding and Q6_K output-head error; RotQuant's student shares the fp16 embedding and `lm_head` with its teacher. llama.cpp's CUDA K-quant path also quantises activations to Q8_1. Each KL is within its own engine, so the different teachers (fp16 Transformers vs BF16 llama.cpp) are a different reference, not a bias. |
| fp16 fallback path | fine | `_fp_cache = dequantize().to(fp16)` from fp32 centroids × decoded scales (`linear.py:85-95`, `quantize.py:276-299`) and the same activation rotation the export saves; extra error ≈ 1e-5 of the 4-bit MSE. Caveat: the mixed 2–8-bit, uint8-scale recipes have only the Python `reference` backend; the llama.cpp native format is 4-bit/fp16-scale only (`rotquant/gguf.py:29-33`). |
| Perplexity windows | fine, not literature-comparable | Non-overlapping 2048-token windows from position 0, `max_samples: 32` → the first 65,536 tokens of each corpus, hashed and identical for source and student (`rotquant/eval/perplexity.py:61-219`). Unsloth's comparator produces no PPL. |
| `quantizable_parameters` | minor | Double-counts the tied embedding through `lm_head` (`adapters.py:122-124`); diagnostic only, not used in any budget. |

## 7. Recommended next experiment (cheap, decisive)

No new runtime is needed to *measure* the effect. In the fp16 fallback path
the embedding can be fake-quantised in place and its bytes counted at the
chosen rate:

1. Quantise `embed_tokens.weight` (the tied matrix) with the existing
   `Quantizer` at W8 g128 scale8, no rotation, no GPTQ; write the
   dequantised values back into the parameter; add `8.0635 × 635,699,200 / 8`
   bytes to `registered_model_bytes` in place of the fp16 count. Repeat at
   W6 with `rotation=fwht` (hidden 2560 is 20 × 128) and MSE scale search.
   With a rotation the dequantised rows are in the rotated basis, so apply
   the inverse rotation (`Rotation.inverse_activation`, as
   `native_tied_tensor`'s consumer does) before writing them back:
   `embed_tokens.weight ← Q(W R) Rᵀ`. A plain embedding lookup has no
   activation-rotation counterpart, and the tied `lm_head` computes
   `x · (Q(W R) Rᵀ)ᵀ = (x Rᵀ) · Q(W R)ᵀ`, which is exactly what a rotated
   `QuantLinear` would compute, so one un-rotated tensor serves both uses.
2. Re-run the existing seed-0 screen with `quant.bits: 5` uniform, then the
   allocator with `candidate_bits: [4, 5, 6, 8]` and the corrected
   `registered` budget. Keep uniform scale8 W4 and the Unsloth anchor as the
   controls.
3. Report KL, top-1, the KL tail and the 32-token trajectories exactly as
   before. If uniform W5 + W8 embedding lands under the Unsloth anchor at
   ≤1% more bytes, that is the first fair matched-size result the project has
   had.
4. Only then implement the packed representation: a `QuantEmbedding` that
   shares the packed codes with the `lm_head` `QuantLinear` (one block-FWHT
   per looked-up token undoes the row rotation if a rotation is used, or use
   no rotation at 8-bit), extend `native_tied_tensor` beyond 4-bit, and
   count the tied bytes once.

## 8. Verification record

- `uv sync --extra dev --extra eval` in a fresh container, then
  `pytest tests/ -q`: **592 passed, 17 skipped** (NEON-only native cases on
  x86), 63 s. The v4 preflight test imports `transformers` unconditionally
  and fails rather than skips without the eval extra; every other Transformers
  test skips cleanly.
- `unsloth/Qwen3.5-4B` `config.json` at the pinned revision: vocab 248,320,
  hidden 2560, `tie_word_embeddings: true`, 32 layers, 8 full-attention.
- `Qwen3.5-4B-UD-Q4_K_XL.gguf` at revision `e87f1764…`: 426 tensors,
  `general.file_type = 15`, imatrix from `unsloth_calibration_Qwen3.5-4B.txt`
  (80 chunks). Tensor bytes from the header sum to 2,901,141,504; plus the
  10,968,206-byte header this reproduces the file size to within alignment
  padding, so the type table above is complete. The script's "backbone"
  figure (5.331 bpw over 3,569,876,992 params) additionally counts the small
  fp16/fp32 `ssm_alpha`/`ssm_beta`/`ssm_conv1d` matrices; over the
  3,565,158,400 weights RotQuant quantises it is 5.314 bpw.
- All 16 `configs/qwen35_*.yaml` files set
  `include: [model.language_model.layers.]`, so no Qwen run has ever
  quantised the vocabulary matrix in the Transformers path.
- RotQuant byte identities reproduce exactly: `packed_weight_bytes =
  3,565,158,400 × 4.125 / 8`; the scale8 saving of 27,417,920 bytes equals
  one byte per 128 weights less the fp16 block metadata of the 8-bit scale
  encoder; `complete_persistent_model_bytes = packed + registered + codebook`.
- No GPU, no Colab and no model download beyond the two headers above.
