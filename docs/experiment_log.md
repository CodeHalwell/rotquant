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

- Evaluate dynamic mixed precision on Qwen3.5-4B at the exact byte budgets of
  uniform 4-bit and the failed uniform 3-bit trial.
- Measure trajectory agreement and global KL on disjoint prompts for all
  candidate recipes.
- Evaluate real per-head K/V cache quantization, including attention-output error,
  long-context perplexity/retrieval, exact cache bytes, and K/V asymmetric bits.
- Run `qwen35_4b_dynamic_kv_mps.yaml` and compare its disjoint global KL/NLL at
  exactly the uniform-4-bit byte budget; repeat the selected recipe on CUDA.
- Fuse rotate, quantize, and persistent KV-cache write before HBM/DRAM storage;
  fuse cache read, dequantization, and attention on CUDA/Metal.
- Benchmark the packed tied vocabulary and native linear kernels against source
  and standard GGUF quantization.
- Replace estimated complete sizes with actual artifacts and resident-memory
  measurements.
- Implement the canonical Transformers quantization-config contract, then a
  vLLM out-of-tree quantization plugin and the corresponding SGLang method; see
  `docs/serving_backends.md` for acceptance gates.

## Entry template

For each new run, append:

1. Date, git revision/dirty state, model revision, device, dependency versions.
2. Full command or config plus overrides and random seed.
3. Data split, calibration/evaluation separation, sequence length, and samples.
4. Quality, exact bytes, latency, throughput, memory, and training time as
   applicable.
5. Whether selection changed the deployed checkpoint.
6. Interpretation, confounders, and the next decision.
