# RotQuant experiment log

This is the durable research ledger for the project. Raw local runs remain in
`results/*.json`; this document records the comparison that was made, what was
learned, negative results, and the decision that followed. Results produced in
external notebooks are recorded here even when their raw artifacts live on
Google Drive.

## Reporting rules

- Compare methods only when model, tokenizer, data split, sequence length, and
  sample count match.
- Report packed component bytes separately from complete-model bytes.
- Mark MPS `fallback=true` runs as **quality-only**: they materialize source-dtype
  weights and are not packed-memory or throughput measurements.
- A retained adapter is not evidence of recovery unless its selected checkpoint
  moves away from initialization and improves a held-out/deployed metric.
- Record unsuccessful runs. They determine the next experiment just as much as
  successful runs do.
- Treat WikiText-2 subsets as development measurements until confirmed by full
  perplexity, downstream tasks, long-context tests, and multiple seeds.

## Result summary

| Date | Model / device | Method | Evaluation | Result | Interpretation |
|---|---|---|---|---:|---|
| 2026-08-29 | tiny random Llama / CPU | FWHT 5-bit smoke | WikiText-2, 128 tokens/window | PPL 32140.2637 | Pipeline completed; PPL is intentionally meaningless for a random model. |
| 2026-08-29 | OPT-125M / CPU | FWHT 3-bit | WikiText-2, 256, 64 samples, seeds 0/1/2 | **112.2128 ± 1.3688** | Stable result; rotation makes 3-bit quantization viable. |
| 2026-08-29 | OPT-125M / CPU | No rotation, 3-bit | WikiText-2, 256, 64 samples, seed 0 | 793.3127 | Strong negative control: rotation is essential at 3-bit. |
| 2026-08-29 | OPT-1.3B / MPS | Source FP16 | WikiText-2, 256, 32 samples, seed 0 | **30.3330** | Matched source reference. |
| 2026-08-29 | OPT-1.3B / MPS | FWHT 3-bit | Same, seeds 0/1/2 | **58.8901 ± 4.2042** | Much better than unrotated, but still a substantial source-quality loss. |
| 2026-08-29 | OPT-1.3B / MPS | No rotation, 3-bit | Same, seed 0 | 313.5290 | Confirms the rotation effect at larger scale. |
| 2026-08-29 | OPT-125M / MPS | Block butterfly, 12 steps | WikiText-2, 256, 64 samples | 105.8485 | Learned block rotations improved on the FWHT development result. |
| 2026-08-29 | OPT-125M / MPS | Block butterfly, 18 steps | Same | 106.0078 | More steps did not improve deployed PPL; several blocks overfit and restored FWHT. |
| 2026-08-29 | OPT-125M / MPS | Block butterfly with validation | Same | 107.7451 | Cleaner selection protocol, but less improvement; local held-out reconstruction is imperfectly correlated with PPL. |
| 2026-08-29 | OPT-125M / MPS | Butterfly + learned scales | Same | **92.8261** | Large gain; every block passed the held-out gate. Scale/clipping adaptation mattered more than extra rotation steps. |
| 2026-08-29 | OPT-125M / MPS | Propagated butterfly + scales | Same | **83.2201** | Propagating quantized inputs reduced accumulated cross-layer mismatch. |
| 2026-08-29 | OPT-125M / MPS | End-to-end distillation | Same | 83.2201 | Negative result: best checkpoint was step 0 and the block model was restored. |
| 2026-08-29 | OPT-125M / MPS | LoRA-QAT recovery | Same | **76.0044** | Positive small-model result: selected step 10/12 reduced held-out distillation loss and improved PPL. |
| 2026-08-29 | Qwen3.5-4B / MPS | Source FP16 | WikiText-2, 256, 32 samples | **17.8463** | Matched source reference. |
| 2026-08-29 | Qwen3.5-4B / MPS | FWHT 3-bit | Same | 22.6234 (+26.8%) | Too much quality loss for the project goal. |
| 2026-08-29 | Qwen3.5-4B / MPS | FWHT 4-bit | Same | **18.9015 (+5.9%)** | Promising quality-only fallback result across 200 language layers. |
| 2026-08-29 | Qwen3.5-4B / CUDA pilot | Source | WikiText-2 subset | 17.8458 | Pilot source reference. |
| 2026-08-29 | Qwen3.5-4B / CUDA pilot | 4-bit RotQuant + attempted LoRA-QAT | Same | 18.7075 (+4.83%) | Passed 5% quality gate, but LoRA was not retained; this measured RotQuant/block recovery rather than useful LoRA. |
| 2026-08-30 | Qwen3.5-4B / CUDA joint matrix + matched follow-up | Uniform W4 + frozen mixed 3.25-bpv K/V | WikiText-2, 256, 64 samples, seeds 0/1/2 | **14.5548 mean PPL (+4.71%)** | Passed the matched development release gates at 56.78% estimated weight reduction; worst candidate/control cache-KL ratio was 0.835. |
| 2026-08-30 | Qwen3.5-4B / exported checkpoints | Pinned source vs joint RotQuant artifact | Exact tensor and snapshot bytes; native value conformance | **59.336% tensor-file reduction** | 200 packed projections, no LoRA, and all native packed codes/scales/rotations verified against the exact producer revision. |

### Qwen3.5-4B CUDA trial matrix

These trials used WikiText-2 with sequence length 256 and 128 evaluation
samples. Artifacts were written under
`rotquant/qwen35_lora_matrix/e9e2da7c1bd8` on Google Drive.

| Trial | Seed | PPL | Relative PPL | Estimated complete size | Reduction | Adapter | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Source CUDA | 0 | **15.3652** | +0.00% | 9.320 GB | 0.00% | 0 MB | Matched source baseline. |
| Plain 4-bit FWHT | 0 | 16.1549 | +5.14% | 4.028 GB | 56.78% | 0 MB | Simple competitive control. |
| Block-only | 0 | **15.9433** | **+3.76%** | 4.033 GB | 56.73% | 0 MB | Best seed-0 size/quality result. |
| Rank-4 medium | 0 | 15.9433 | +3.76% | 4.048 GB | 56.57% | 15.24 MB | No useful LoRA: early stop after 6 steps, best step 0. |
| 3-bit rank-4 medium | 0 | 19.1128 | +24.39% | 3.603 GB | 61.34% | 15.24 MB | Size gain is too small for the quality loss; best step 0. |
| Rank-4 large | 0 | 15.9433 | +3.76% | 4.048 GB | 56.57% | 15.24 MB | More data/steps did not move from step 0. |
| Rank-8 large | 0 | 15.9433 | +3.76% | 4.063 GB | 56.40% | 30.47 MB | More rank also did not move from step 0. |
| Block-only confirmation | 1 | 15.9924 | +4.08% | 4.032 GB | 56.74% | 0 MB | Replicated. |
| Block-only confirmation | 2 | 15.9342 | +3.70% | 4.034 GB | 56.72% | 0 MB | Replicated. |

Across the three block-only seeds, PPL is approximately **15.9566 ± 0.0256**
(population standard deviation). The measured result is reproducible on this
development subset. Complete-model sizes are estimates and must be replaced by
actual exported checkpoint and resident-memory measurements for publication.

## Detailed observations and decisions

### Baseline rotation experiments

- OPT-125M FWHT seed results were 110.5817, 113.9311, and 112.1257.
- The patched linear weights used 3.125 effective stored bits/weight before
  rotation or adapter overhead, or a nominal 5.12x reduction from FP16.
- The eight-sample development runs (98.9451 FWHT versus 711.1379 unrotated)
  correctly predicted the direction but understated the stable 64-sample PPL.
- Decision: retain the no-rotation arm as a required control and avoid drawing
  conclusions from eight-sample PPL.

### Learned rotation and scale sequence

1. Increasing butterfly optimization from 12 to 18 steps produced 105.8485 to
   106.0078, demonstrating that additional optimization of local reconstruction
   does not guarantee better language-model quality.
2. Disjoint validation/selection made the protocol more defensible but yielded
   107.7451. Several trained blocks were correctly restored to FWHT.
3. Learning deployable group scales reduced PPL to 92.8261.
4. Feeding each later block the already-quantized preceding state reduced PPL
   further to 83.2201.
5. The first end-to-end distillation arm selected step 0 and made no deployed
   change. Rank-4 LoRA-QAT later reached 76.0044 on OPT-125M.

Decision: optimize and gate global behavior, retain step-0 rollback, initialize
recovery from the actual quantization residual, refresh discrete codes during
training, and evaluate multi-token trajectories rather than trusting local MSE.

### Qwen3.5-4B recovery matrix

- Plain FWHT was already within 5.14% of source PPL at an estimated 56.78% total
  size reduction.
- Block training recovered 1.38 percentage points of relative PPL degradation
  without an adapter.
- Rank 4, more calibration batches, more steps, and rank 8 all failed to move
  away from zero-initialized adapters. `lora_retained=true` described serialized
  parameters, not a learned improvement.
- The 3-bit arm reduced estimated total size only another 4.61 percentage points
  while degrading PPL by 24.39%.

Decision: the current winner is block-only 4-bit. Do not pay adapter bytes unless
held-out selection and deployed PPL both show a real gain. Pursue mixed 2/3/4-bit
allocation rather than uniform 3-bit, and use residual-SVD initialization plus
periodic code refresh for the next recovery study.

### Runtime and export observations

- MPS fallback evaluation is fast after patching but stores materialized FP16
  weights; it does not demonstrate deployment compression.
- The first native GGUF server loaded successfully, but prompt ingestion ran at
  roughly 0.83 tokens/s because the RotQuant path lacked a fused Metal
  implementation and used an impractical 262,144-token context split across four
  slots.
- Native tied-vocabulary support now represents input lookup and output logits
  from one packed RotQuant matrix. For Qwen3.5-4B, the original tied FP16 matrix
  is about 1,212.5 MiB and its 4-bit packed representation is about 312 MiB,
  before small metadata/scale overhead: roughly 900 MiB additional savings over
  leaving it FP16.
- The modified llama.cpp tree compiles with the CPU and Metal row-decode paths.
  Native throughput and output-parity benchmarks remain outstanding.

Decision: all performance claims require the native packed path; fallback runs
remain explicitly quality-only.

## Implementation-validation record

| Date | Change | Validation | Status |
|---|---|---|---|
| 2026-08-29 | Dynamic per-layer mixed-precision allocator | Dynamic/block/runner tests | Passed |
| 2026-08-29 | Held-out multi-token trajectory metrics | Trajectory integration tests | Passed |
| 2026-08-29 | Residual-SVD recovery and code refresh | Recovery/block tests | Passed |
| 2026-08-29 | Tied packed vocabulary export and native loading | Python GGUF tests and complete llama.cpp build | Passed |
| 2026-08-29 | Rotated quantized KV simulation and rotation training | 18 targeted KV/GGUF/block tests | Passed |
| 2026-08-29 | Real Transformers KV-cache boundary smoke | tiny random Llama, 16-token prefill + 2 decode tokens | Passed: top-1 1.0, KL 8.43e-7, logical cache 4096 -> 1024 bytes |
| 2026-08-29 | Held-out dynamic K/V rate allocation | Unit suite plus real runner smoke with disjoint selection/eval batches | Passed: exact 6.0 effective bpv target, logical cache 4096 -> 768 bytes, top-1 1.0, KL 1.24e-6 |
| 2026-08-29 | Direct prefill K/V NMSE and whole-system CUDA matrix | 131 tests; 98 parsed weight/KV configurations; 47-cell nbformat/AST validation | Structurally passed; Qwen CUDA execution pending |
| 2026-08-29 | Strict frozen K/V recipes and cross-context transfer notebook | 134 tests; ruff; 33-cell nbformat/AST validation | Structurally passed; six Qwen CUDA transfer trials pending |
| 2026-08-29 | Frozen mixed-context K/V transfer and joint-matrix promotion | Exact 3.25-bpv replay; short/long held-out KL; updated 47-cell joint notebook validation | Passed: universal mixed map selected; whole-system CUDA execution pending |

The first frozen replay used an intentionally strict 1% KL-only gate. Exact
recipe storage reproduced at 3.25 bpv in both contexts; short KL drifted 0.61%
and long KL drifted 1.04%, causing a boundary failure despite no evidence of a
recipe mismatch. The replay protocol now allows 2% KL drift while also requiring
cosine agreement within 0.005, top-1 agreement within 0.03125, and exact bpv.
This keeps the gate blocking on material output changes without treating normal
cross-runtime CUDA drift as a failed serialization.

The first full Qwen3.5-4B dynamic-K/V MPS attempt was manually stopped during
the one-time weight patch at 48/200 projections. It produced no quality result;
the experiment was moved to the CUDA matrix notebook because repeating broad
cache trials after separate MPS weight rebuilds wastes time and does not provide
native packed performance data.

### Qwen3.5-4B seed-0 K/V matrix protocol correction

The first CUDA notebook matrix completed its seed-0 development table, but its
uniform and dynamic rows were not evaluated on the same C4 calls. Uniform rows
used the first four loaded batches, while dynamic rows reserved those four for
recipe selection and reported the following four. The mismatch was exposed by
the nominally identical 8-bit recipes: uniform K8/V8 reported KL 0.582664 while
dynamic 8.25 bpv reported KL 0.563029. The apparent seed-0 Pareto frontier of
dynamic 2.25 and 3.25 bpv is therefore **invalid for cross-profile ranking**.

The run remains useful as a protocol diagnostic and as provisional evidence
that the allocator reaches exact byte targets. It must not support a quality
claim. `eval_offset_batches` now reserves the same selection prefix for uniform
controls, dynamic trials, source controls, and long-context confirmation. A
regression test requires uniform 8-bit and a dynamic allocator restricted to
8-bit to produce identical held-out metrics.

### Corrected Qwen3.5-4B seed-0 K/V matrix

The rerun with matched held-out C4 calls passed both endpoint equivalence
checks: dynamic 2.25 bpv exactly reproduced uniform K2/V2, and dynamic 8.25 bpv
exactly reproduced uniform K8/V8. This confirms that the corrected uniform and
dynamic paths used the same evaluation calls and equivalent deployed recipes.

On fixed 4-bit FWHT RotQuant weights, dynamic 3.25 bpv was the seed-0 winner:

| Cache recipe | Effective bpv | KL | Cosine | Top-1 | NLL delta |
|---|---:|---:|---:|---:|---:|
| Uniform K2/V2 | 2.25 | 0.575244 | 0.909170 | 0.718750 | +0.1134 |
| Uniform K2/V3 | 2.75 | 0.538010 | 0.916070 | 0.718750 | +0.0669 |
| Uniform K2/V4 | 3.25 | 0.517738 | 0.918358 | 0.703125 | +0.0727 |
| Dynamic mixed K/V | 3.25 | **0.448988** | **0.920411** | **0.750000** | **+0.0612** |
| Uniform K4/V4 | 4.25 | 0.548273 | 0.916469 | 0.718750 | +0.0413 |
| Dynamic mixed K/V | 4.188 | 0.457670 | 0.919841 | 0.734375 | +0.0850 |

At the exact 3.25-bpv budget, dynamic allocation reduced KL by 13.3% relative
to the best same-size uniform recipe, K2/V4. It also used fewer bytes and lower
KL than every higher dynamic budget in this seed. The source-weight control
improved from KL 0.8618 with uniform K4/V4 to 0.7186 with dynamic allocation at
the same 4.25-bpv budget.

Uniform and NF cache codebooks at K4/V4 improved KL and top-1 agreement over
the Gaussian cache codebook, but increased held-out NLL delta substantially.
This is a real multi-objective tradeoff rather than an unconditional codebook
win. Decision: promote dynamic 3.25 bpv to seeds 1/2 and 1,024-token validation;
do not spend more cache bits unless those checks overturn the seed-0 frontier.

### Qwen3.5-4B three-seed K/V validation

Seeds 1 and 2 confirmed the seed-0 ranking. Dynamic 3.25 bpv had the lowest
held-out KL in every seed. Seed 2's discrete allocator used 3.125 effective bpv,
slightly below the nominal budget. Dynamic 2.25 bpv again exactly reproduced
uniform K2/V2 in both seeds, providing two more endpoint-equivalence checks.

| Cache recipe | Seed 0 KL | Seed 1 KL | Seed 2 KL | Mean KL | Sample std |
|---|---:|---:|---:|---:|---:|
| Uniform K2/V2 | 0.575244 | 0.790700 | 0.620100 | 0.662015 | 0.113679 |
| Uniform K2/V3 | 0.538010 | 0.922300 | 0.554800 | 0.671703 | 0.217185 |
| Dynamic 3.25 bpv | **0.448988** | **0.711100** | **0.545300** | **0.568463** | 0.132582 |
| Uniform K3/V3 | 0.572863 | 0.886000 | 0.594300 | 0.684388 | 0.174930 |
| Uniform K4/V4 | 0.548273 | 0.843600 | 0.678700 | 0.690191 | 0.147998 |

Across seeds, dynamic 3.25 bpv reduced mean KL by 14.1% relative to the best
uniform mean (K2/V2) and by 16.9% relative to same-budget uniform K3/V3. The
paired per-seed wins are more important than the absolute cross-seed KL shift,
because each seed selects different held-out calls. Decision: retain dynamic
3.25 bpv as the K/V candidate and proceed to the pre-registered 1,024-token
validation before treating it as the final cache profile.

### Qwen3.5-4B 1,024-token K/V confirmation

The completed external matrix is pinned to Git SHA
`bf94f2045a902beee7940eaf3a29d6e3b31db660` and contains 32 seed-0 profiles,
seeds 0/1/2 for the promoted candidates, and the long-context confirmation.

The long-context confirmation used four 1,024-token selection calls and four
disjoint evaluation calls, with 32 continuation tokens. At equal 3.25-bpv
storage, the dynamically allocated cache reduced held-out KL from 1.540281 for
uniform K3/V3 to 1.345134, a 12.7% improvement. It also beat uniform K4/V4
(KL 1.553477) by 13.4% while reducing packed cache storage from 8.913 MB to
6.816 MB (23.5% fewer bytes). Cosine agreement improved from 0.873417 to
0.879871 and top-1 agreement from 0.492188 to 0.500000 relative to K3/V3.

The long-context allocator selected mean K3.375/V2.625, whereas the seed-0
256-token allocator selected mean K2.875/V3.125. This reversal is evidence that
key/value sensitivity changes with context length. It is also an important
protocol distinction: this long run recalibrated the allocation on disjoint
long-context selection data, so it validates the allocation method at long
context but does not yet prove that one frozen 256-token recipe transfers to
1,024 tokens. A frozen-recipe cross-context trial is required before choosing a
single deployment map.

### Qwen3.5-4B frozen K/V transfer result

The frozen transfer run at Git SHA `49d3cf1182f0` replayed both context-specific
recipes at exactly 3.25 bpv and then evaluated the short, long, and mixed maps on
both held-out contexts. Replay KL drift was 0.61% at 256 tokens and 1.04% at
1,024 tokens; exact storage matched and both passed the multi-metric replay gate.

| Frozen map | 256-token KL | 1,024-token KL | Worst regret vs replayed context optimum |
|---|---:|---:|---:|
| Short-context map | 0.445652 | 1.430826 | +5.28% |
| Long-context map | 0.496695 | 1.359091 | +11.45% |
| Mixed-context map | **0.435155** | 1.376698 | **+1.30%** |
| Uniform K3/V3 | 0.572441 | 1.540281 | n/a |

At identical 3.25-bpv storage, the mixed map reduced KL by 24.0% relative to
uniform K3/V3 at 256 tokens and by 10.6% at 1,024 tokens. It improved on the
replayed short-specific map by 2.36% and trailed the replayed long-specific map
by only 1.30%. Decision: promote the saved mixed recipe as the universal frozen
K/V candidate in the whole-system weight-plus-cache matrix, while retaining
dynamic per-weight allocation as an ablation.

The first two launches of the promoted whole-system matrix exited during the
initial source-weight control before writing a result, so neither attempt
carries a quality measurement. The first exposed only `CalledProcessError`; the
new persistent log on the second identified a Qwen3.5 multimodal RoPE shape
mismatch. Trajectory generation had populated the wrapper's `rope_deltas`, then
the independent K/V evaluator supplied a one-token query with a full-cache
attention mask and allowed the wrapper to infer positions. It consequently
constructed 257 positions for one query token. Cached text decode now supplies
the explicit absolute one-token `position_ids`. A tiny real Qwen3.5 multimodal
regression reproduces the stale-RoPE condition and passes with the fix.

### Qwen3.5-4B whole-system joint matrix

The corrected CUDA run at Git SHA `8d9676f109fa` completed its seed-0 weight
screen with a matched source PPL of 13.9001 (WikiText-2, 64 samples, 256-token
sequences). Only two quantized recipes passed the predeclared 10% seed-0 gate:

| Weight recipe | PPL | Relative PPL | Estimated complete weights | Reduction | Effective weight bpw |
|---|---:|---:|---:|---:|---:|
| Uniform W4 | **14.7029** | **+5.78%** | 4.0277 GB | 56.78% | 4.125 |
| Dynamic 4.125-bpw | 14.9976 | +7.90% | **4.0238 GB** | **56.83%** | 4.116 |

Dynamic 4.125-bpw saved only 3.93 MB (0.098%) relative to uniform W4 while
raising PPL by a further 2.00%, so the seed-0 screen currently favors uniform
W4. Uniform W3 (+28.74%), dynamic 2.75 (+66.53%), dynamic 3.25 (+21.20%),
dynamic 3.625 (+12.07%), and guarded dynamic 3.25 (+36.02%) all failed the 10%
gate. The joint stage therefore retained only `uniform_w4` and
`dynamic_w4.125` rather than filling the nominal three-finalist allowance.

K/V reconstruction NMSE remained approximately 0.009 for every weight recipe.
The cache KL values use each weight model as its own full-cache teacher, so they
measure that model's sensitivity to K/V quantization and must not be interpreted
as cross-weight accuracy against the FP16 source. Likewise, trajectory agreement
is based on only 32 continuation tokens at this screening stage.

The full K/V cross, recovery ladder, three-seed validation, and 1,024-token
prefill confirmation subsequently completed. At seed 0, the best joint recipe
was uniform W4 plus the transferred frozen mixed Gaussian K/V map:

| Joint recipe | PPL | Relative PPL | Cache KL | Effective K/V bpv | Total estimated system GB | Joint score |
|---|---:|---:|---:|---:|---:|---:|
| Uniform W4 + frozen mixed | **14.7029** | **+5.78%** | **0.4352** | 3.25 | 4.0294 | **0.1020** |
| Uniform W4 + dynamic Gaussian | 14.7029 | +5.78% | 0.4502 | 3.25 | 4.0294 | 0.1035 |
| Dynamic 4.125-bpw + frozen mixed | 14.9976 | +7.90% | 0.5515 | 3.25 | 4.0255 | 0.1341 |
| Dynamic 4.125-bpw + dynamic uniform | 14.9976 | +7.90% | 0.5627 | 3.188 | 4.0254 | 0.1359 |

Uniform W4 plus the frozen map combined an estimated 4.0277-GB complete weight
artifact (56.78% below the 9.320-GB source estimate) with a 3.25-bpv K/V cache.
It beat the dynamic-weight finalist despite using only 3.93 MB more estimated
weight storage. The fixed transferred K/V map also beat the recalibrated dynamic
allocator at the same nominal K/V budget, so dynamic allocation did not justify
its substantial calibration cost in this run.

The recovery stage was applied only to the first, smallest-system finalist,
`dynamic_w4.125__dynamic_u_3.25`, rather than to the lowest-joint-loss uniform-W4
candidate. Block-and-scale recovery reduced its PPL from 14.9976 to 14.8905
(-0.1071, or -0.71%) while adding about 7.14 MB of estimated weight state. The
LoRA-QAT arm selected step 0: train, validation, and held-out losses were
unchanged, the block checkpoint was restored, and no adapter was retained. This
is a negative LoRA result and does not test whether block recovery can improve
the eventual uniform-W4 winner.

Three-seed validation selected uniform W4 plus frozen mixed K/V as the diagnostic
winner:

| Finalist | Seed PPL values | Mean PPL | Mean relative PPL | Worst relative PPL | Mean / worst cache KL |
|---|---|---:|---:|---:|---:|
| Uniform W4 + frozen mixed | 14.7029, 14.6091, 14.3524 | **14.5548** | **+4.71%** | **+5.78%** | **0.550312 / 0.700824** |
| Dynamic 4.125-bpw + dynamic uniform | 14.9976, 14.9588, 14.5860 | 14.8475 | +6.82% | +7.90% | 0.641834 / 0.763350 |

The uniform/frozen recipe passes the predeclared 5% mean-PPL and 10%
worst-PPL gates. It was not declared release-ready because its worst cache KL
of 0.700824 exceeded 0.568819. That cache threshold, however, was calculated as
1.05 times the *seed-0* uniform-W4/K4/V4 control KL, then applied to the
*worst of three seeds* for each finalist. Its mean cache KL of 0.550312 is 3.25%
below that threshold, while its worst value is 23.21% above it. This unmatched
comparison makes the release failure methodologically inconclusive. A matched
uniform-W4/K4/V4 control must be run at seeds 1 and 2 before assigning release
status; compare candidate and control within seed, or compare like-for-like
mean and worst summaries.

At 1,024-token prefill, the winner's frozen 3.25-bpv cache produced KL 0.822912,
top-1 agreement 0.59375, and 6,815,744 deployed cache bytes. Source weights with
the same frozen cache produced KL 1.093824 and top-1 0.546875. These are
self-teacher cache-sensitivity measurements, not evidence that quantized weights
are more accurate than source weights. No long-context perplexity or retrieval
task was evaluated.

Decision at this point: retain `uniform_w4__frozen_mixed_3.25` as the diagnostic
whole-system winner, pending matched multi-seed K4/V4 controls. The focused,
conditional experiment was implemented in
`notebooks/qwen35_4b_joint_release_followup_colab.ipynb`.

### Qwen3.5-4B matched release follow-up

The follow-up at Git SHA `206ea94454b3` reused the completed candidate rows and
ran only the missing uniform-W4/K4/V4 controls for seeds 1 and 2. This replaced
the earlier seed-0-to-worst-seed gate with a within-weight, within-seed cache
comparison:

| Seed | Frozen 3.25-bpv KL | Matched K4/V4 KL | Candidate/control ratio | KL reduction vs control |
|---:|---:|---:|---:|---:|
| 0 | 0.435155 | 0.541733 | 0.803 | 19.7% |
| 1 | 0.700824 | 0.839113 | 0.835 | 16.5% |
| 2 | 0.514957 | 0.670372 | 0.768 | 23.2% |

The mean matched ratio was 0.802209 and the worst was 0.835197, comfortably
below the predeclared 1.05 gate on every seed. Together with mean PPL 14.5548
(+4.71%) and worst-seed PPL degradation +5.78%, the unrecovered recipe passes
all matched development release gates.

Block-and-scale recovery was then tested on the actual winner at seed 0. It
raised PPL from 14.702918 to 14.713017, a 0.0687% degradation, while increasing
estimated complete weight size from 4.027706 GB to 4.036566 GB (+8.86 MB) and
reducing estimated weight savings from 56.783% to 56.688%. Its matched cache-KL
ratio of 0.798873 passed, the candidate/control PPL values matched exactly, and
no adapter was present, but the predeclared PPL-improvement gate failed. The
notebook correctly rejected recovery and skipped its four additional seed
trials.

Decision: promote the simpler unrecovered
`uniform_w4__frozen_mixed_3.25` recipe to artifact export and broader quality
validation. Retire this block-recovery profile for the winner and retain the
existing prohibition on the step-0 LoRA-QAT profile. This is a development
release decision only: complete sizes remain logical estimates under CUDA
fallback, and long-context perplexity, retrieval, packed runtime memory, and
throughput remain unverified.

The dedicated export and verification workflow is now implemented in
`notebooks/qwen35_4b_joint_winner_export_colab.ipynb`. It fails closed on stale
release inputs or weight-code drift, pins and records the Hub revision used for
reconstruction, embeds the frozen K/V map as deployment metadata, requires the
seed-0 PPL to match the released row, audits actual artifact bytes and tensor
keys, writes checksums, and reloads without persistent fallback caches in a
fresh process. The completed export is audited in the following entry.

### Qwen3.5-4B exported artifact audit

The winner export was subsequently published as the private checkpoint
`HallD/qwen35-4b-rotquant-joint` at revision
`986ed9839cd3396d538aa567ba254ba13403ea9c`. The deployment manifest pins the
source checkpoint `unsloth/Qwen3.5-4B` at revision
`3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`, records the matched release
metrics, contains the exact frozen 3.25-bpv K/V map, and reports no retained
LoRA adapter.

The cached checkpoints now provide a like-for-like file audit:

| Artifact | Tensor bytes | Complete snapshot bytes |
|---|---:|---:|
| Pinned source checkpoint | 9,319,737,856 | 9,348,509,644 |
| Packed RotQuant checkpoint | 3,789,750,560 | 3,809,973,748 |
| Reduction | **59.336%** | **59.245%** |

The packed checkpoint contains 200 quantized projections. The earlier 4.0277-GB
and 56.78% figures remain useful logical CUDA estimates, but actual tensor-file
bytes now supersede them for storage reporting. Resident and peak transient
memory are still unmeasured.

The joint native GGUF is 3,146,308,736 bytes with SHA-256
`ac70480575ce94d482402ac575e75e12bd01125a85cc32711af13bbd4491ecba`.
`scripts/verify_rotquant_gguf.py` compared it with the exact joint producer
revision and passed all 200 projections: every packed code, fp16 scale, and
rotation value matched. The native artifact omits the vision tower, so its file
size is not used as a like-for-like multimodal storage comparison.

Decision: actual exported checkpoint storage and native weight conformance are
now verified. Keep resident-memory, fused-cache execution, full quality, and
throughput claims open.

## Algorithm Lab initial selection run (2026-08-31)

The archived `9dbe985` Colab bundle contains 59 complete records with no failed
trials. Qwen3.5-4B was the primary family and Qwen2.5-3B the transfer family.
These are developmental subsets, not the 300-prompt competitive suite.

- Calibrated W4 increased Qwen3.5 WikiText-2/C4 PPL by 4.58%/3.24% while
  reducing complete persistent model bytes by 58.26%. On Qwen2.5 the changes
  were +10.32%/+7.03% and -66.67% bytes.
- Gaussian W4 was nearly quality-equivalent and materially cheaper to produce,
  so it remains the fast baseline; calibrated W4 is the quality preset
  candidate.
- Teacher-guided 3.625-bpw allocation beat the exact-format random allocator
  for every recorded seed/dataset/family comparison. Its Qwen3.5 PPL changes
  were +11.12%/+7.00% at -60.73% bytes; Qwen2.5 was +18.03%/+14.40% at -69.50%.
  It is the compact preset candidate, subject to the hardened confidence gate.
- Dimension-2 vector W3 beat its exact-rate scalar control on the primary
  family but both had poor absolute quality, and transfer was catastrophic.
  Vector formats remain research-only.
- TurboQuant-style scale changes did not earn promotion in this matrix.
- The dense-attention selective-V oracle found a potentially useful region: a
  90% mass target read 51.6% of value rows at about 2.95% extra error versus
  full RotQuant V; 95% read 67.2% at about 0.97% extra error. This covered only
  two full-attention layers and used oracle selection, so it does not authorize
  a runtime speed claim.

A later add-on run printed 16 further results (75 total) but its raw records
could not be persisted. Treat those observations as internal decision evidence
only: local allocation and its guarded variant did not improve on the teacher
recipe, dynamic vector 2.75-bpw transferred catastrophically, and spherical
codebooks did not displace the calibrated/Gaussian W4 choices. They cannot be
used in a release claim until reproduced from content-addressed records.

Decision: carry forward three provisional recipes—calibrated W4 (`quality`),
Gaussian W4 (`fast`), and unguarded teacher-guided 3.625-bpw (`compact`)—but do
not expose them as validated public presets until the rerun passes the repaired
selection/confidence policy. The competitive follow-up uses exact deployed-size
GGUF/Unsloth controls, KL distribution tails, and the disjoint 300-prompt,
32-token suite specified in `docs/competitive_eval.md`.

### Algorithm Lab repaired confirmation run (2026-09-01)

The complete rerun at Git SHA `5f81a739b4a75738e9b994bdf93eecb9a68479f7`
is archived under experiment identity `5f81a739b4a7/6fa343e0f6c1`. It ran on an
NVIDIA A100-SXM4-40GB with PyTorch 2.11.0/CUDA 12.8 and a source-built
fast-hadamard-transform 1.1.0. The delivered bundle contains 61 valid compact
records and 61 matching raw records, with no failure artifacts: 23 primary
screens, 27 three-seed candidate validations plus their source reference, nine
cross-family candidates plus their source reference, and the selective-V
oracle. This supersedes the unavailable add-on observations above.

The repaired primary-model validation made the free-running distinction clear:

| Qwen3.5-4B recipe | WikiText-2 mean / worst delta | C4 worst delta | Top-1 agreement | 32-token agreement / worst | Mean matching prefix | Status |
|---|---:|---:|---:|---:|---:|---|
| Calibrated W4 | +4.58% / +6.04% | +3.43% | 86.51% | **52.08% / 39.84%** | **15.67** | runtime candidate |
| Gaussian W4 | +4.63% / +5.80% | +3.31% | 84.66% | 37.24% / 25.00% | 11.67 | runtime candidate |
| Teacher 3.625 bpw | +11.12% / +11.94% | +7.57% | 82.80% | 20.31% / **7.81%** | 6.17 | diagnostics failed |
| TurboQuant-scale W4 | +10.85% / +10.96% | +9.82% | 82.28% | 14.84% / 7.81% | 4.00 | diagnostics failed |
| Vector W3 | +26.35% / +26.89% | +19.83% | 74.87% | 9.64% / 5.47% | 2.92 | research quality failed |

Calibrated and Gaussian W4 have the same 3,789,608,768-byte logical complete
model size (58.26% below the 9,078,531,392-byte source accounting). Calibration
therefore buys trajectory stability rather than storage. It costs more to
produce—about 166 seconds per complete trial versus 73 seconds for Gaussian—but
does not change the deployed scalar format. Both passed the primary diagnostic
gate. `runtime candidate` means only that they advance to packed runtime work;
the CUDA fallback materialized dense weights and is not runtime evidence.

Teacher-guided 3.625-bpw allocation continued to beat its exact-rate random
control on paired WikiText-2 and C4 NLL for every primary seed and on Qwen2.5.
Nevertheless, its worst primary 32-token agreement was 7.81%, and its Qwen2.5
worst-layer NMSE was 0.781. The compact preset is therefore rejected: a good
teacher-forced allocation score did not preserve free-running behavior. The
dimension-2 vector W3 control similarly improved paired PPL while worsening
several source-fidelity metrics, and dynamic vector 2.75 bpw collapsed on
Qwen2.5 (PPL +3,108%, top-1 10.32%). No sub-W4 recipe advances.

Calibrated W4 was also the strongest transfer recipe on Qwen2.5-3B: WikiText-2
+10.32%, C4 +7.03%, top-1 84.13%, 32-token agreement 60.16%, exact trajectory
25%, and a 19-token mean prefix. Gaussian W4 remains the cheaper fast-conversion
control. The dense-attention selective-V result reproduced the earlier useful
oracle points: a 90% mass gate read 51.56% of V rows at 2.95% additional error
over dense RotQuant V, while 95% read 67.19% at 0.97% additional error. This is
still only two layers with dense candidate selection.

Decision: freeze calibrated W4 as the quality control and Gaussian W4 as the
fast control for the streamed-GPTQ stage. Drop the provisional compact preset,
TurboQuant scale arm, and vector formats from the production ladder. The next
candidate must beat these W4 controls on held-out KL/top-1 and 32-token
trajectories as well as PPL. The registered 300-prompt competitive contract was
not run and remains mandatory before any comparison with Unsloth.

## Native K/V cache benchmark notes

Command family:

```bash
third_party/llama.cpp/build-rotquant/bin/llama-bench \
  -m qwen35-4b-rotquant-native.gguf -ngl 99 -fa on \
  -ctk <f16-or-q4_0> -ctv <f16-or-q4_0> -p 0 -n 16 -d 512 -r 3
```

Apple M5 Max, native RotQuant weight artifact:

| K cache | V cache | Test | Throughput | Interpretation |
|---|---|---|---:|---|
| F16 | F16 | pp64, 2 repeats | 20.57 ± 0.27 tok/s | Short-prompt reference. |
| Q4_0 | Q4_0 | pp64, 2 repeats | 22.59 ± 0.77 tok/s | Preliminary prompt improvement; context is too short for a claim. |
| F16 | F16 | tg16, 2 repeats | 14.67 ± 1.46 tok/s | Short-decode reference. |
| Q4_0 | Q4_0 | tg16, 2 repeats | 17.15 ± 2.01 tok/s | High variance; motivated the depth-controlled repeat. |
| F16 | F16 | tg16 at depth 512, 3 repeats | 14.72 ± 2.88 tok/s | Depth-controlled reference. |
| Q4_0 | Q4_0 | tg16 at depth 512, 3 repeats | 12.13 ± 0.45 tok/s | Generic Q4 cache dequantization overhead still exceeds bandwidth savings at this depth. |

Mixed single-side Q4 results were also slower in the two-repeat short run. Do
not claim a cache speedup from these data. The next native experiment must find
the context-length crossover with more repeats and compare the generic Q4_0
codebook against an exact Gaussian RotQuant cache kernel.

## Open experiments

### 2026-09-01 — scientific-review implementation pass (no model result yet)

Implemented the review's high-value prerequisites before spending another
Qwen3.5-4B run: randomized bounded-memory calibration sampling; disjoint source
rows; streamed actorder GPTQ; larger resumable recovery; mean-bias correction;
shared and jointly trained projection rotations; Hessian-weighted rotation
training; learned hard signs; real uint8 scale metadata; per-token W4A8;
position-tiered KV storage; finite-rate E8P weight/KV codebooks; and paired
bootstrap intervals. Added focused CUDA profiles for the compositional W4A8/E8
trial and the 8k tiered-KV confirmation. These are implementation milestones,
not quality or speed results; promotion still depends on the recorded held-out
gates.

- The Algorithm Lab and real-attention selective-V stages are now complete; see
  the repaired confirmation entry above. Do not rerun the broad screen. Run the
  focused streamed-GPTQ stage against calibrated/Gaussian W4 next, retaining
  the same paired WikiText-2/C4, KL/top-1, and trajectory gates.

- Measure packed checkpoint and native process resident memory and peak
  transient memory; artifact and tensor-file bytes are now verified.
- Measure longer trajectory agreement, long-context perplexity, and retrieval
  for the matched-gate winner and its controls.
- Formalize that trajectory work as a pinned, calibration-disjoint
  300-prompt/32-token suite across agentic, code, maths, multilingual, and
  long-document prompts, with source, same-size GGUF, and Unsloth controls. The
  algorithm-lab notebook's four C4 prompts are only an early drift gate.
- Revisit dynamic weight allocation only if a sensitivity objective can beat
  uniform W4 at a meaningful byte delta; the current 3.93-MB saving is not worth
  the observed PPL loss.
- Fuse rotate, quantize, and persistent KV-cache write before HBM/DRAM storage;
  fuse cache read, dequantization, and attention on CUDA/Metal.
- Benchmark the packed tied vocabulary and native linear kernels against source
  and standard GGUF quantization.
- Implement the canonical Transformers quantization-config contract, then a
  vLLM out-of-tree quantization plugin and the corresponding SGLang method; see
  `docs/serving_backends.md` for acceptance gates.

### 2026-09-01 — focused 4B promotion harness (no new model result yet)

Recorded the completed Algorithm Lab as the baseline and replaced the next
broad rerun with a five-arm W4 ladder: source FP16, Gaussian W4, calibrated W4,
and streamed GPTQ applied independently to each codebook. The pinned 4B GPTQ
profile uses four disk-offloaded Hessians per calibration replay on the A100
40 GB development target; the 27B profile must return to one per pass unless a
measured memory trace supports a larger group. Resumption is keyed by both the
fully resolved configuration and Git revision.

The generated `notebooks/qwen35_4b_optimization_stage_colab.ipynb` keeps
W4A8/E8, million-token recovery, and four 8k-context KV confirmations as
separate opt-ins, so an end-to-end default run does not accidentally launch
every costly stage. A source/candidate competitive collector now persists full
FP16 teacher distributions once and emits the registered per-token KL, top-1,
source/candidate trajectory, and structured-failure records for Transformers
or RotQuant checkpoints. No quality numbers were produced by this
implementation pass.

### 2026-09-01 — three-seed Qwen3.5-4B W4 promotion

The focused W4 ladder completed at code revision
`f1f2fb1734d5076a4f1f6916adb2b7e2ad02f3fb`, model revision
`3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`, on an A100 40 GB. All arms used
the same pinned, disjoint manifests. GPTQ used 128 C4 sequences of 512 tokens
(65,536 calibration tokens), four disk-offloaded Hessians per pass, actorder,
and scale recomputation. Evaluation used 64 paired windows on each of
WikiText-2 and C4, 1,016 held-out teacher-forced tokens, and eight held-out
32-token trajectories per rotation seed.

| Three-seed mean | Gaussian W4 | Gaussian + GPTQ | Calibrated W4 | Calibrated + GPTQ |
|---|---:|---:|---:|---:|
| WikiText-2 PPL increase | +4.16% | **+2.11%** | +3.93% | **+2.07%** |
| C4 PPL increase | +3.98% | **+1.82%** | +4.35% | **+1.94%** |
| Mean teacher KL | 0.04467 | **0.02119** | 0.04482 | **0.02322** |
| Top-1 agreement | 88.62% | **92.13%** | 89.07% | **92.26%** |
| 32-token agreement | 36.46% | **52.47%** | 43.62% | **55.86%** |
| Exact source trajectories | 0/24 | **5/24** | 0/24 | **3/24** |

GPTQ improved WikiText-2, C4, teacher KL, and top-1 agreement against its
matched non-GPTQ control in every seed at zero additional inference bits.
Promote streamed GPTQ into the W4 recipe. Gaussian is the provisional default:
its KL was lower in all three seeds (8.75% lower on the three-seed mean), its
patch was 26% faster, and it retained more exact trajectories. Calibrated GPTQ
remains a challenger because its mean top-1 and aligned-token agreement were
slightly higher; none of the head-to-head PPL, top-1, or trajectory differences
is decisive with only eight unique trajectory prompts.

The post-promotion W4A8/E8 stage now starts from an exact Gaussian FWHT+GPTQ
W4/g128 arm. One paired branch tests optimized weight-only composition; a
separate branch changes only A8 and then adds E8 KV. This avoids conflating the
effects or repeating the 200-step learned-rotation fit three times. The
300-prompt competitive suite remains the final Gaussian-versus-calibrated
codebook gate.

## Entry template

For each new run, append:

1. Date, git revision/dirty state, model revision, device, dependency versions.
2. Full command or config plus overrides and random seed.
3. Data split, calibration/evaluation separation, sequence length, and samples.
4. Quality, exact bytes, latency, throughput, memory, and training time as
   applicable.
5. Whether selection changed the deployed checkpoint.
6. Interpretation, confounders, and the next decision.
