# RotQuant project notes

September 11 follow-up: the [first A100 throughput pilot](../research/results/native_pilot_2026_09_11/README.md)
passed 13 stages; 2048-token timing exceeded an undersized cap. Completed 128/512
contexts measured ~36.4 prefill / 19.8 decode tok/s. The next runnable notebook is
the [native optimisation study](native_gpu_optimization.md): corrected budgets,
targeted contexts, separate operator profiles, an opt-in tiled-prefill candidate
and same-bridge BF16/UD-Q4 controls. Recipe/quality gates are unchanged. No new
kernel speedup or promotion is claimed; its CUDA validation remains to be run.

Latest follow-up (10 September): the user's A100 run at `06a4379c7107` passed
native CUDA and retained W5/W6 saved-probe parity, after the public-task notebook
was stopped for reference-runtime cost. See the [results review](../research/results/native_cuda_2026_09_10/README.md)
and [`native_gpu_validation.md`](native_gpu_validation.md). Full graph/GPU
implementation is no longer pending; native throughput, broader parity and
matched-baseline task evaluation are. The original September 9 inventory below
is historical; no saved artifact or old task outcome was replaced.

Working reference notes, written 9 September 2026 against `85e1254` (origin/main)
plus the [same-day review](project_review_2026-09-09.md). They are a companion
to, not a replacement for, the three documents that carry authority:

- [`roadmap.md`](roadmap.md): the plan and the current gate status;
- [`experiment_log.md`](experiment_log.md): the durable ledger of every run;
- [`how_rotquant_works.md`](how_rotquant_works.md): the mathematics.

Everything below was read from the repository, the archived records under
`research/results/`, GitHub Actions, or verified by running the code in a fresh
locked environment. Nothing on Google Drive was inspected. Where a number is
quoted, the section says which record it comes from. Dates are 2026.

Contents

1. [Orientation: which document answers what](#1-orientation-which-document-answers-what)
2. [What RotQuant is, in plain terms](#2-what-rotquant-is-in-plain-terms)
3. [Timeline](#3-timeline)
4. [Repository map](#4-repository-map)
5. [How a run works](#5-how-a-run-works)
6. [Formats, runtimes and what can execute where](#6-formats-runtimes-and-what-can-execute-where)
7. [Results ledger with canonical numbers](#7-results-ledger-with-canonical-numbers)
8. [Negative results and lessons](#8-negative-results-and-lessons)
9. [Defect and open-item register](#9-defect-and-open-item-register)
10. [Evaluation protocol notes](#10-evaluation-protocol-notes)
11. [Operations cookbook](#11-operations-cookbook)
12. [Decisions log](#12-decisions-log)
13. [Open questions and hypotheses](#13-open-questions-and-hypotheses)
14. [Next steps](#14-next-steps)
15. [Glossary](#15-glossary)

## 1. Orientation: which document answers what

The `docs/` directory has 35 files. Most were written for one experiment or one
review and several have been superseded. Read them in this order.

| Question | Document | Status |
|---|---|---|
| What is the plan and what is the gate status right now? | `roadmap.md` (Stage 2 header and its dated updates) | Current; the top entries supersede the historical ones below them |
| What has every run measured and decided? | `experiment_log.md` | Current ledger; newest entries at the top, older narrative below the result table |
| What are the current headline numbers? | `fresh_quality_results_2026-09-09.md` | Current |
| What is the next GPU run? | `native_runtime_v3.md`, then `public_tasks_run_2026-09-09.md` | Public-task run stopped; native model parity and speed/cost preflight first |
| Why is the Unsloth comparison structured the way it is? | `results_review_2026-09-06.md` | Current; the byte-budget finding |
| What is the maths and what is only hypothesis? | `how_rotquant_works.md` (sections 14 and 15 especially) | Current |
| What may a competitive claim say? | `competitive_eval.md`, `competitive_data.md` | Current contract; the 300-prompt suite it describes has never been built |
| What are the file formats? | `packed_format_v1.md`, `v2`, `v3`, `native_runtime_v2.md` | Current specifications |
| Which engines can serve an artifact? | `serving_backends.md`, `integrations/llama.cpp/README.md`, `native/README.md` | Partly stale: both still describe the withdrawn 3.25-bpv KV map as implemented |
| What defects are known? | `project_deep_dive_2026-09-08.md` §4, `project_review_2026-09-09.md` §3 | Current |
| What was wrong scientifically and what was fixed? | `scientific_review_2026-08-31.md`, `scientific_validity_review_2026-09-01.md`, `change_report_2026-09-01.md` | Historical but still the best explanation of the KV withdrawal and the scale-encoder fixes |
| Why were allocators v2/v3/v4 built? | `dynamic_allocator_v2.md`, `v3`, `v4`, `qwen35_dynamic_mixed_experiment.md` | Historical; all conclusions revised by the byte-budget finding |
| Why W5 with a compressed vocabulary? | `vocabulary_budget_plan_2026-09-06.md`, `vocabulary_results_2026-09-07.md` | Current rationale |
| How were the artifacts exported and checked? | `packed_vocabulary_validation_run.md`, `fresh_quality_run_2026-09-08.md` | Runbooks; the "operational notes" section of the latter is required reading before any Colab session |
| What research is queued? | `hybrid_attention_quantization_research_2026-09-09.md`, `unsloth_full_frontier_plan_2026-09-09.md`, `performance_plan_2026-09-05.md` | Plans only |
| Odd ones out | `kv_retrieval.md`, `turboquant_extensions.md` | Design notes; the retrieval oracle and the TurboQuant arms are parked |
| The paper | `paper/main.tex`, `paper/data/publication_results.json` | Stale: still framed around the withdrawn joint weight-plus-KV result |

The README is a 666-line front page that accumulated the run-by-run narrative.
Its "next run" paragraph is now correct, but it still states a Llama-2-7B/13B
confirmation bar that no result has met, and its layout block omits `docs/`,
`notebooks/`, `native/` and `research/`.

## 2. What RotQuant is, in plain terms

**One sentence.** RotQuant changes the coordinate system of each linear layer
with an orthogonal rotation that leaves the model's function unchanged, fits a
low-bit scalar code in that better coordinate system, then packs, verifies and
tries to serve exactly that representation under an exact byte budget.

**Why rotating helps.** A scalar quantiser handles each coordinate on its own,
so a few outlier coordinates force a coarse grid on everything else. A mixing
rotation spreads energy across coordinates, and after enough mixing every
coordinate looks close to Gaussian. A single Gaussian Lloyd-Max grid is then a
good data-free default. The rotation is free in exact arithmetic because
`x Wᵀ = (x Rᵀ)(W Rᵀ)ᵀ` for any orthogonal `R`; the activation must be rotated
with the same `R` as the weight, which the code calls `consistent` mode.

**The deployed recipe** (backbone and vocabulary differ):

| Component | Backbone projections (200 matrices) | Tied vocabulary (one matrix) |
|---|---|---|
| Rotation | Randomised block-128 fast Walsh-Hadamard transform (FWHT), seeded, fixed | Same, seeded |
| Codebook | Gaussian Lloyd-Max, 5 bits (32 levels) | Gaussian Lloyd-Max, 6 or 8 bits |
| Scales | Group of 128, MSE search over 0.5–1.5 × RMS, stored as 8-bit codes in blocks of 256 with fp16 offset and step | Group of 128, fp16 |
| Error compensation | Act-order GPTQ from streamed C4 Hessians (128 sequences × 512 tokens) | None |
| Activations | fp16, unquantised | fp16 |
| Execution | Tiled Python reference path (`QuantLinear`) | Packed shared owner for embedding and head, `dense_equivalent` projection mode |

The names `W5/W6` and `W5/W8` mean a 5-bit backbone with a 6-bit or 8-bit
vocabulary. They are not uniform 4-bit models and should never be described as
"W4".

**The model.** `unsloth/Qwen3.5-4B` at revision `3764fa35…`: a multimodal
checkpoint whose language path is a hybrid of 24 Gated DeltaNet (linear
attention) blocks and 8 full-attention blocks (every fourth), hidden size
2,560, MLP width 9,216, vocabulary 248,320 with tied input and output
embeddings. Only the language path is quantised; the vision tower (about
667 MB fp16) and the tiny linear-attention gate projections `in_proj_a/b` stay
at source precision and are counted in every byte total.

| Quantity | Value | Source |
|---|---:|---|
| Backbone weights quantised | 3,565,158,400 | `results_review_2026-09-06.md` §1 |
| Tied vocabulary parameters | 635,699,200 | same |
| Source tensors the model loads (excludes the MTP head) | 9,078,538,752 bytes | `scientific_validity_review_2026-09-01.md` §3.1 |
| Source safetensors index (includes a 241,199,104-byte MTP head that is never loaded) | 9,319,737,856 bytes | same |

**The external target.** Unsloth's released `Qwen3.5-4B-UD-Q4_K_XL.gguf`
(2,912,109,728 bytes) plus its `mmproj-F16.gguf` projector (672,423,616 bytes),
3,584,533,344 bytes in total, at repository revision `e87f1764…`. Despite the
name it is a 5.31 bits-per-weight backbone with a Q6_K vocabulary. Unsloth's
own documentation describes the Qwen3.5 uploads as Dynamic 2.0, so a claim
about Dynamic 3.0 cannot be made from this artifact.

**What "success" means here.** Lower KL divergence from the source model at
equal or fewer bytes, confirmed across seeds and on inputs that were not used
to choose the recipe, then task accuracy on public benchmarks, then a runtime
that actually consumes the packed bytes. The project has the first, is about
to test the second, and has not started the third for the winning recipe.

## 3. Timeline

Twelve days of visible history: 104 commits, pull requests #6 to #18, two
commit authors (Daniel Halwell 71, Claude sessions 32 at `85e1254`), plus
Codex and Copilot review passes. Roughly a dozen A100 Colab campaigns.

| Date | What happened | Key commits / records |
|---|---|---|
| 29 Aug | Repository history starts (the first commit already contains 87 files, so the code predates it). OPT-125M/1.3B rotation results; Qwen3.5-4B 3/4-bit on Apple Silicon; CUDA pilot with LoRA-QAT; KV matrix, three-seed KV validation, 1,024-token confirmation, frozen-map transfer | `49d3cf1`, `8d9676f` (Qwen cached decode positions fix) |
| 30 Aug | Whole-system joint matrix; uniform W4 plus frozen mixed 3.25-bpv cache selected; matched release follow-up; winner exported to the private Hub repo `HallD/qwen35-4b-rotquant-joint`; production runtime and public API | `891664c`, `eddb5ae`, `1656499`, PR #6 |
| 31 Aug | Roadmap; Algorithm Lab; licensing, packaging and Python CI; competitive data pipeline; science guide; first scientific review; Algorithm Lab initial run | PRs #7 to #15, `dfd3b04`, `0f652b7` |
| 1 Sep | Scientific-review implementation pass; focused 4B optimisation stage; three-seed W4 ladder promotes streamed GPTQ; W4A8/E8 composition (bundled "optimized W4" arm collapses); Unsloth KL comparator; KV simulator state-sharing bug found; all cache results withdrawn | `c476509`, `f1f2fb1`, `8ad3b8e`, `0ac733a`, `4051bdb` |
| 2 Sep | Validity fixes merged (endpoint check, tiered ageing, exact scales, subnormal step, shared-site gate, Hessian gate); Unsloth 4-prompt anchor and provisional 8k E8P run recorded | PR #16, `c94cd57`, `0e4822e` |
| 3 Sep | Full W4 factor ablation decided (scale8 kept as control); dynamic mixed-precision run: learned signs shelved, allocator v1 rejected; allocator v2 built | `2911e7a`, `3dbae03`, `2dce3aa` |
| 4 Sep | Allocator v2 result (−25.1 % KL vs random); allocator v3 built and run (−72.8 % vs random, export within 0.087 % of the provider, still 2.62× its KL); project review with nine defects | `c9efd2d`, `project_review_2026-09-04.md` |
| 5 Sep | Review fixes; performance plan; allocator v4 prepared and run (neither finalist promoted) | `8ba3b75` |
| 6 Sep | Results review: the fp16 tied vocabulary explains the gap; GGUF header inspector; vocabulary-budget experiment prepared and run (completed 22:48 UTC) | PR #17, `1e92f78`, `ce6c8ec` |
| 7 Sep | Vocabulary results: W5/W6 and W5/W8 finalists; packed validation notebook; first CUDA run fails the W6 reload gate; RoPE fp32 buffer defect found and fixed; revalidation notebook | `8f10ee6`, `89d25f3` |
| 8 Sep | Revalidation passes with zero probe error; fresh-quality experiment prepared; two failed attempts (missing `generation_config.json`; Hindi tokenizer gate); reuse path; Python CI red at `a4ac776`; deep dive | `bdf6595`, `733bb3d`, `a4ac776`, `6c90054`, `36eb548` |
| 9 Sep | Deep dive merged (PR #18); fresh-quality run completes and is archived; public-task gate prepared; memory-bandwidth requirement, hybrid-attention research note and full Unsloth inventory recorded; this review | `d4292d6`, `5a98b99`, `85e1254`, `1aaa1a7` |

Pattern worth noticing: every second day a review found a defect that would
have invalidated a result (KV state sharing, subnormal scale step, GPTQ scale
mismatch, shared-site gate, RoPE buffer downcast, missing generation config,
shallow-clone CI). The reviews earned their keep. The cost is that five review
documents and about 560 KB of docs accumulated in ten days.

## 4. Repository map

### 4.1 Top level

| Path | Contents | Size |
|---|---|---|
| `rotquant/` | Library: 24 modules, 16,609 lines including `eval/` | |
| `rotquant/eval/` | Fixed evaluation protocol: 15 modules | |
| `scripts/` | 55 command-line tools: runners, notebook builders, selectors, archivers, exporters | 18,507 lines |
| `tests/` | 58 files, 469 test functions, 746 collected cases (729 pass, 17 NEON-only skips on x86) | 12,107 lines |
| `configs/` | 40 YAML experiment cells | |
| `notebooks/` | 19 generated Colab notebooks | |
| `native/` | Dependency-free C++17 runtime with C ABI, CMake, conformance tests | 1,628 lines |
| `integrations/llama.cpp/` | 99 KB patch against pinned llama.cpp `17252c76` plus README | |
| `baselines/` | GPTQ/AWQ/AQLM wrappers through the same harness (unverified against Transformers 5) | |
| `research/` | Compact result records, raw archived evidence, the development eval suite, the Unsloth inventory | 111 MB in the working tree, 7.5 MiB packed |
| `paper/` | LaTeX draft, publication manifest, audit outputs | stale |
| `docs/` | 35 documents | about 650 KB |
| `.github/workflows/` | `python-ci.yml`, `native-runtime.yml`, `llama-cpp-patch.yml` | |

### 4.2 Library modules

| Module | Lines | What it does |
|---|---:|---|
| `api.py` | 185 | `RotQuantConfig`, `inspect_model`, `optimize_model`, `save_pretrained`, `from_pretrained`: the model-oriented public surface |
| `adapters.py` | 321 | Architecture discovery and replacement hooks. Registered adapters: `dense-decoder`, `moe-decoder`, `hybrid-decoder`, `multimodal`, `encoder-decoder`, `encoder`, `generic-linear` |
| `rotate.py` | 600 | `Identity`, `RandomizedHadamard` (block FWHT with random signs), `ButterflyRotation` (trainable, starts at FWHT), `DenseOrthogonal`, `LearnedRotation` (Cayley map); `fwht` with the CUDA `fast-hadamard-transform` kernel or a pure-torch fallback |
| `codebooks.py` | 655 | Scalar codebooks (`gaussian`, `spherical`, `calibrated`, `uniform`, `nf`), Lloyd-Max fitting, dimension-2 vector codebooks, finite E8P codebooks, source-coding bounds |
| `quantize.py` | 1,158 | The single `Quantizer`: scale search, code assignment, GPTQ, residual and QJL error compensation, 8-bit scale encoding, exact `BitBudget` accounting |
| `pack.py` | 130 | LSB-first int32 bitstream packing and unpacking, 1–16-bit codes |
| `linear.py` | 474 | `QuantLinear`: keeps codes packed, rotates the activation per forward, dequantises per layer on the reference path; optional fp16 dense fallback cache for quality-only runs; per-token A8 semantics |
| `patch.py` | 499 | Walks a model, replaces `nn.Linear` with `QuantLinear`, enforces rotation consistency, shares rotations across sibling projections, wires calibration and rotation training |
| `calibrate.py` | 467 | Activation capture, incremental Hessians, streamed collection with a disk-offloaded `DiskHessianStore` |
| `train_rotation.py` | 411 | Layerwise butterfly training (weight, activation and Hessian objectives) with a deployed-quantiser gate against seeded FWHT |
| `block_train.py` | 1,496 | Transformer-block reconstruction, learned scales, propagated inputs, end-to-end distillation, LoRA recovery, streamed variant |
| `dynamic.py` | 1,843 | Static mixed-precision allocation: candidate scoring with the deployed quantiser, marginal-KL, bucketed multiple-choice knapsack, MILP repair, random controls, content-addressed score cache |
| `vocabulary.py` | 370 | Tied-vocabulary PTQ: dense quality prototype and the packed shared owner used by embedding and head |
| `checkpoint.py` | 797 | Pickle-free packed checkpoint writer and loader, transactional publication, hashes, v1–v3 |
| `format.py` | 383 | Executable format contract: `FORMAT_VERSION`, `PackingContract`, manifest validation |
| `native.py` | 358 | Native-v2 block layout encoder and NumPy reference kernels |
| `native_ffi.py` | 338 | ctypes binding to the C ABI; can register the compiled kernel in the `KernelRegistry` |
| `runtime.py` | 187 | Fail-closed kernel registry keyed by backend, operation, bits and group size |
| `gguf.py` | 345 | RotQuant-GGUF v1 packing primitives (4-bit, fp16 scales, tied vocabulary at 4-bit RMS) |
| `kv_cache.py` | 693 | Rotated K/V quantisation, tiers, E8P cache codes, the selective retrieval oracle |
| `validation.py` | 196 | Bounded packed-checkpoint conformance probes and gate values |
| `utils.py` | 235 | Seeding, logging, memory probes, `BitBudget`, `write_result`, `environment_record` |
| `_internal.py` | 91 | `rotate_hessian`, `encoded_storage_scales`: shared primitives |

`rotquant/eval/`: `perplexity` (WikiText-2 and C4 with pinned revisions),
`logit_fidelity` (teacher KL, top-1, NLL over full vocabularies),
`trajectory` (32-token greedy agreement), `kv_cache` (the Transformers cache
simulator with endpoint check), `layer_mse`, `quantization`
(`compare_quantizers` at matched bits), `statistics` (paired bootstrap),
`promotion` (shared quality guards), `competition`, `competitive_collect`,
`competitive_run`, `data_manifest` (the 300-prompt contract machinery),
`fresh_tasks` (the 96 authored tasks and their strict oracles), `throughput`,
`zeroshot` (an `lm-eval` wrapper that no archived run has ever used).

### 4.3 Scripts by role

| Role | Scripts |
|---|---|
| Generic runner | `run_experiment.py` (config → quantise → eval → JSON), `aggregate.py` |
| Qwen3.5 stage runners | `run_qwen35_next_stage.py` (W4 ladder, ablations, allocators v1–v4), `run_qwen35_vocabulary_budget.py`, `run_qwen35_packed_validation.py`, `run_qwen35_fresh_eval.py`, `run_qwen35_public_tasks.py`, `run_unsloth_qwen35_4b_kl.py` |
| Notebook builders | `build_qwen35_*_notebook.py` (nine), `build_algorithm_lab_notebook.py`; the older builders inherit each other by string rewriting (next-stage → dynamic-mixed → v2 → v3 → v4), the newer ones are generated directly |
| Selection and assessment | `select_qwen35_*_finalists.py` (five), `assess_qwen35_*.py` (three), `algorithmic_selection.py`, `algorithmic_trials.py` |
| Archiving | `archive_allocator_v4.py`, `archive_vocabulary_screen.py`, `archive_packed_revalidation.py`, `archive_fresh_quality.py`: copy compact JSON byte-for-byte and write `evidence_index.json` with SHA-256 pairs |
| Preflights | `preflight_allocator_v4.py`, `preflight_vocabulary_budget.py`, `preflight_packed_validation.py`: synthetic or tiny-model checks that run before a Colab session spends GPU time |
| Artifacts | `export_rotquant_gguf.py`, `verify_rotquant_gguf.py`, `inspect_gguf_types.py`, `generate_packed.py`, `build_rotquant_llama_cpp.sh`, `serve_rotquant_gguf.sh` |
| Competitive pipeline | `build_competitive_manifest.py`, `build_competitive_protocol.py`, `build_competitive_run_metadata.py`, `collect_competitive_transformers.py`, `aggregate_competitive_run.py`, `compare_competitive_runs.py`, `compare_qwen35_dynamic_to_unsloth.py` |
| Publication | `audit_publication.py`, `run_publication_suite.py` |
| Benchmarks | `benchmark_native_reference.py`, `benchmark_quantizer_variants.py`, `benchmark_rotquant_kv.sh` |
| Misc | `colab_runtime.py` (live subprocess wrapper with heartbeats), `public_task_suite.py` |

### 4.4 Configs

| Group | Files | Note |
|---|---|---|
| Hypothesis cells E1–E9 | `e1_rotation` … `e9_spherical_length` (17 files) | Written for Llama-2-7B in `_base.yaml`; most never ran on the target model. E3b and E4b are analytically designed losers (see §8) |
| Qwen3.5-4B stages | `qwen35_4b_*` (18 YAML files) | Live: `vocabulary_budget_cuda` (defines the frozen W5 recipe) and `packed_validation_cuda`. Retired: `mps`, `kv_mps`, `dynamic_kv_*`, `joint_cuda`, `lora_qat_cuda`, `gptq_cuda`, `w4a8_e8_trials`, `w4_factor_ablation`, `sign_replication`, `dynamic_mixed`, `allocator_v2/v3/v4`, `recovery_cuda` (the properly budgeted recovery profile, never run), `long_context_kv_cuda` (provisional cache run) |
| Publication | `publication_dense_cuda`, `publication_qwen35_joint_cuda` | The latter still embeds the withdrawn 3.25-bpv KV map |
| Frozen KV map | `qwen35_4b_frozen_mixed_kv_3p25.json` | Withdrawn |
| Smoke | `smoke_cpu.yaml` | Tiny random Llama, CPU, under a minute; the only config CI-adjacent |
| Algorithm Lab | `algorithm_lab_cuda.yaml` | Completed |

### 4.5 Notebooks

| Notebook | Cells | Status |
|---|---:|---|
| `qwen35_4b_public_tasks_colab` | 22 | **Stopped: reference path too slow; preserve for provenance** |
| `qwen35_4b_fresh_quality_colab` | 24 | Completed 9 Sep (producer `d4292d6`) |
| `qwen35_4b_packed_revalidation_colab` | 18 | Completed 8 Sep (loader `89d25f3`) |
| `qwen35_4b_packed_validation_colab` | 18 | Completed 7 Sep (`8f10ee6`); W6 reload gate failed, later revalidated |
| `qwen35_4b_vocabulary_budget_colab` | 22 | Completed 6 Sep (`ce6c8ec`) |
| `qwen35_4b_allocator_v4_colab` | 27 | Completed (`8ba3b75`); no promotion |
| `qwen35_4b_allocator_v3_colab` | 27 | Completed (`c9efd2d`) |
| `qwen35_4b_allocator_v2_colab` | 25 | Completed (`2dce3aa`) |
| `qwen35_4b_dynamic_mixed_precision_colab` | 29 | Completed (`3dbae03`); allocator v1 rejected, signs shelved |
| `qwen35_4b_optimization_stage_colab` | 29 | Completed (`f1f2fb1`, `8ad3b8e`, `2911e7a`, `06c1e73` eras) |
| `rotquant_algorithm_lab_colab` | 49 | Completed (`9dbe985`, `5f81a73`) |
| `qwen35_4b_joint_winner_export_colab`, `qwen35_4b_export_colab` | 28, 20 | Completed; produced the old W4 artifacts |
| `qwen35_4b_joint_release_followup_colab`, `qwen35_4b_joint_rotquant_kv_colab`, `qwen35_4b_kv_cache_matrix_colab`, `qwen35_4b_kv_frozen_transfer_colab` | 42, 47, 36, 33 | Retired; their cache conclusions are withdrawn |
| `qwen35_4b_lora_qat_colab`, `qwen35_4b_lora_trial_matrix_colab` | 28, 48 | Retired; negative LoRA results |

### 4.6 Tests

Groups, with the number of test functions in brackets: quantiser and rotation
foundations (`test_rotation_invariance` 4, `test_gptq_identity` 4,
`test_source_coding` 8, `test_turboquant` 35, `test_scale_storage_consistency`
8, `test_vector_quantization` 7, `test_calibrated_codebook` 4); formats and
runtime (`test_format` 5, `test_pack` 9, `test_checkpoint` 13, `test_native` 9,
`test_native_cpp` 8 which builds the C++ library with CMake, `test_runtime` 4,
`test_gguf` 8, `test_inspect_gguf_types` 8); calibration and training
(`test_calibrate_vector` 7, `test_train_rotation` 22, `test_block_train` 13);
allocation (`test_dynamic` 19, `test_qwen35_allocator_v2/v3/v4` 4/5/9,
`test_qwen35_dynamic_experiment` 4); KV cache (`test_kv_cache` 13,
`test_kv_cache_eval` 22, `test_kv_cache_bit_monotone` 1); runner and
experiments (`test_runner` 48, `test_qwen35_next_stage` 11,
`test_vocabulary*` 18, `test_packed_validation` 11, `test_fresh_quality` 19,
`test_fresh_quality_reuse` 8, `test_public_tasks` 20); competitive pipeline
(`test_competition_eval` 4, `test_competitive_*` 16, `test_data_manifest` 11);
API (`test_api_adapters` 9, `test_review_runtime` 3, `test_integration` 3).

Known gaps (deep dive §4.2): no reload test without an explicit `dtype`, no
fp16 in-process model in `test_checkpoint.py`, no fp16/bf16 orthogonality test
for butterflies, no "GPTQ without Hessian raises" test, no non-dividing block
test, no LoRA save/load round trip, and the shared-rotation reload branch is
never executed.

### 4.7 Native runtime and llama.cpp integration

- `native/src/native_v2.cpp` (scalar and dispatch), `native_v2_avx2.cpp`,
  `native_v2_c.cpp` (C ABI), `cli.cpp`, `benchmark.cpp`; headers
  `native_v2.h`, `native_v2_c.h`, `native_export.h`.
- C ABI version 1, format version 2: `rq_native_v2_abi_version`,
  `rq_native_v2_format_version`, `rq_native_v2_kernel_count`,
  `rq_native_v2_kernel_capability_at`, `rq_native_v2_resolve_kernel`,
  `rq_native_v2_dequantize`, `rq_native_v2_matmul`, `rq_native_v2_last_error`.
- Kernels: portable scalar (always), ARM NEON (compiled on ARM), x86 AVX2
  (advertised only after CPUID and OS vector-state checks). On the ARM
  development host the 256×1024, batch-4, group-128 microbenchmark ran at
  0.21–0.24 ms with NEON against 0.60–0.69 ms scalar.
- The llama.cpp patch touches 19 files: a new `src/llama-rotquant.{h,cpp}`,
  the Qwen3.5 model graph, the model loader, the KV cache, and the Metal
  backend including `kernels/rotquant.metal`. It recognises `*.rqweight` and
  `*.rqrotation` tensors and evaluates them through a `ggml_custom` operator.
  A stock llama.cpp build fails closed on such a file.
- Scope is **4-bit only**: Gaussian codebook, group 128, fp16 scales, FWHT or
  butterfly rotations of block 128, tied vocabulary at 4-bit with RMS scales.
  It cannot load the W5 recipe, 8-bit scales or the packed W6/W8 vocabulary.

## 5. How a run works

### 5.1 The generic runner

`scripts/run_experiment.py` deep-merges an experiment YAML over
`configs/_base.yaml`, applies `--set key=value` overrides, derives a run id,
then executes stage functions in order: `resolve_device_dtype` (which, contrary
to the fail-closed rule, downgrades `device: cuda` to CPU with a warning when
CUDA is absent), `load_hf_model`, `build_calib_loader` (pinned dataset
revisions, eligibility by sequence length, `skip` offsets),
`_prepare_calibration` (streamed Hessians, disk offload, token caches),
`_capture_references` (teacher logits and trajectories before mutation),
`_apply_quantization` (patch, GPTQ, optional rotation training, allocation),
`_run_evaluations` (perplexity, logit fidelity suites, trajectories, KV cache,
throughput), `_export_checkpoint`. It writes one JSON per run with an
`environment_record` (versions, git SHA, dirty flag, GPU name).

### 5.2 The Colab conventions

Every experiment since 31 August follows the same shape, which is worth
internalising because the identity discipline is also what makes runs brittle:

1. **Publish first, then pin.** The notebook fetches `REPO_REF = "main"`,
   resolves it to an immutable SHA, prints it, and expects that SHA to be
   pinned before any resume. `source_identity()` is the HEAD SHA plus a hash
   of every `.py` under `rotquant/` and `scripts/`.
2. **Revision-named Drive roots.** Outputs go under
   `/content/drive/MyDrive/rotquant/<experiment>/<sha12>/…`, with `logs/`,
   `progress.json`, atomic per-arm or per-prompt JSON records with SHA-256
   sidecars, and a `manifest.json` that freezes inputs before inference.
3. **Resume by hash, never by name.** Completed records are reused only when
   code, configuration, runtime (torch, CUDA, Python, GPU name) and data
   identities match. `FORCE_RERUN=False` is the default and should stay so.
4. **Live output.** Subprocesses print PID and a `tail -f` command, 30-second
   notebook heartbeats and 60-second GPU heartbeats; a browser disconnect is
   not an interruption, an explicit cell interrupt kills the process group.
5. **Compact download.** A ZIP of JSON, checksums and logs comes back;
   weights, probe tensors and full-vocabulary reference logits stay on Drive.
6. **Archive in one commit, after the summary.** `archive_*.py` copies the
   compact records byte-for-byte and writes an `evidence_index.json`. The
   fresh-eval runner hashes every JSON under `research/results/raw` into its
   frozen protocol, so archiving mid-run breaks the run.

### 5.3 Identity couplings to remember

- A push to `main` between sessions changes HEAD and therefore forces a new
  output root for any run whose notebook tracks `main`.
- Reuse of the seed-0 fresh-quality records binds the runtime: `torch
  2.11.0+cu128`, Python 3.13, an A100-SXM4-40GB. A different SKU refuses.
- `tests/test_fresh_quality_reuse.py` still protects the frozen checkpoint,
  quantizer, model/scoring paths and five runner ASTs against `733bb3d`.
  The September 9 native preparation narrowly exempts the isolated matrix
  modules and audits the **exact** L5 guard additions to the two legacy
  exporters; other edits to those modules require re-review too. It does not
  authorize reusing old records with a native model backend.
- The public-task runner binds its receipts to `source_identity()` too, so
  the same "merge, pin, then run" rule applies to the next session.

### 5.4 Drive layout referenced by the live runbooks

```text
/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f/   b5_v6_s0, b5_v8_s0 (original tensors)
/content/drive/MyDrive/rotquant/qwen35_fresh_quality/d4292d6fdec6/replicas/   b5_v6_s1, b5_v8_s1, b5_v6_s2, b5_v8_s2
/content/drive/MyDrive/rotquant/qwen35_fresh_quality/733bb3d2e477/fresh/     reused seed-0 and source records
```

Planning figures from the runbooks: about 45 GB Drive for packed validation,
140 GB for fresh quality with two replication seeds, 35 GB local plus 1 GB
Drive for the public-task run; the llama-cpp-python CUDA build took about 29
minutes; source Hessian collection about 39 minutes; a full-logit reference set
is roughly 15 GB.

## 6. Formats, runtimes and what can execute where

### 6.1 Checkpoint formats

| Version | Adds | Written by |
|---|---|---|
| v1 | `rotquant_config.json` manifest written last, `rotquant_model.safetensors` (embeddings, norms, biases, rotations), `rotquant_packed.safetensors` (codes, scales, codebooks); int32 LSB-first bitstream, 1–16-bit codes, kernel profiles 1–8 | Legacy artifacts; still readable |
| v2 | Vector codebooks, blockwise uint8 scales with fp16 offset/step, shared `rotation_id`, butterfly `storage_dtype`, `activation_bits`, deduplicated codebooks, `generation_id`, `files_sha256`, transactional publication with a `.NAME.rotquant-previous` recovery directory | Default for backbone-only artifacts |
| v3 | One shared `PackedVocabulary` referenced by embedding and head, `projection_mode` (`rotated` or `dense_equivalent`), removal of the dense vocabulary parameter; loader keeps framework fp32 buffers (RoPE) at their constructor dtype | The W5/W6 and W5/W8 artifacts |

### 6.2 Native-v2 block layout

Per group: one little-endian fp16 scale then `ceil(group_size × bits / 8)`
LSB-first code bytes; partial final groups padded with zero codes; the
codebook travels as `2^bits` fp32 centroids. For 4-bit, group 128 this is
byte-identical to RotQuant-GGUF v1. Residual, sketch, per-row-scale and 8-bit
scale layouts fail closed.

### 6.3 Compatibility matrix

| Recipe | Python `QuantLinear` (reference, per-layer dequant) | Native-v2 C++ | RotQuant-GGUF v1 / llama.cpp patch |
|---|---|---|---|
| W4, fp16 scales, FWHT or butterfly, tied vocab fp16 or 4-bit RMS | Yes | Yes (byte-exact) | Yes (CPU scalar, Metal) |
| W4 with 8-bit scales (`scale8`) | Yes | Rejected (L5 fixed; formerly lossy) | Rejected (L5 fixed; formerly lossy) |
| W5 backbone, 8-bit scales | Yes | No layout for 5-bit with 8-bit scales | No |
| Packed W6/W8 tied vocabulary (checkpoint v3) | Yes, tiled | No | No |
| Any activation quantisation (A8) | Yes (dequantised immediately) | No | No |
| KV cache codes | Simulator only | No | 3.25-bpv map implemented in the patch, quality withdrawn |

The consequence: the recipe with the best quality evidence has no full-model
runtime outside Python, and no native resident-memory or throughput measurement
exists for it. The 3.44/3.60 GB figures are file sizes.

The new [native-v3 matrix primitive](native_runtime_v3.md) can exactly encode
and reconstruct W1–W8 with scale8/16 in scalar C++. This adds matrix conformance,
not graph-level support to the GGUF/Metal columns above; rotations, shared
vocabulary semantics and final FP16 rounding remain operator responsibilities.
The stopped public-task log is now a slow **reference-path** timing observation,
not a native serving benchmark.

### 6.4 Serving backends

Transformers: loads through `load_packed_model`, works for `forward` and
`generate`, no fused kernel. llama.cpp: experimental W4 fork only. vLLM and
SGLang: nothing implemented; the roadmap wants a `QuantizationConfig` plus
linear method plus Triton/CUDA GEMV. Unsloth: a training-side producer only;
its GGUF export requantises and therefore does not preserve RotQuant.

### 6.5 Native throughput figures on record

Apple M5 Max, old W4 joint artifact: `llama-server` reported about 23 prompt
tokens/s for a 128-token prefill and 47 tokens/s decode for a short request;
`llama-bench` reported 14.67 ± 1.46 tokens/s for `tg16` and 20.57 ± 0.27 for
`pp64`. The two decode figures were never reconciled. Generic Q4_0 cache
quantisation was slower than fp16 cache at depth 512. None of this concerns
the W5 recipe.

## 7. Results ledger with canonical numbers

All KL values are `KL(source ‖ candidate)` at temperature 1 over the full
vocabulary, averaged over scored positions. "Development suite" means the 24
C4 prompts of 512 tokens (12,264 positions) plus the 25 authored diverse
prompts, both used repeatedly for selection since 3 September.

### 7.1 Early results (29–30 August)

| Model | Method | Metric | Result |
|---|---|---|---|
| OPT-125M | FWHT 3-bit vs no rotation | WikiText-2 PPL, 64 samples | 112.2 ± 1.4 vs 793.3 |
| OPT-1.3B | FWHT 3-bit vs no rotation vs source | same | 58.9 ± 4.2 vs 313.5 vs 30.3 |
| OPT-125M | butterfly + learned scales + propagated inputs; LoRA-QAT | same | 83.2; 76.0 |
| Qwen3.5-4B | FWHT 3-bit / 4-bit vs source (MPS) | WikiText-2 PPL, 32 samples | 22.62 / 18.90 vs 17.85 |
| Qwen3.5-4B | uniform W4 + frozen mixed cache, three seeds | WikiText-2, 64 samples | 14.5548 mean PPL, +4.71 % vs 13.9001 |
| Qwen3.5-4B | exported W4 joint artifact vs loaded source tensors | bytes | 58.26 % smaller (3,789,750,560 vs 9,078,538,752) |

The weight halves of these stand; every cache number is withdrawn.

### 7.2 Algorithm Lab (31 August, 1 September)

Calibrated W4 and Gaussian W4 both pass the free-running diagnostics
(32-token agreement 52.1 % and 37.2 %); the teacher-guided 3.625-bpw
compact recipe, TurboQuant-scale W4 and dimension-2 vector W3 fail them
(worst 32-token agreement 7.8 %, 7.8 %, 5.5 %). Selective-V oracle: a 90 %
mass gate reads 51.6 % of value rows at 2.95 % extra error on two layers.
Records: `9dbe985` bundle, `5f81a739b4a7/6fa343e0f6c1` identity.

### 7.3 W4 ladder and GPTQ promotion (1 September, code `f1f2fb1`)

| Three-seed mean | Gaussian W4 | Gaussian + GPTQ | Calibrated W4 | Calibrated + GPTQ |
|---|---:|---:|---:|---:|
| WikiText-2 PPL increase | +4.16 % | +2.11 % | +3.93 % | +2.07 % |
| C4 PPL increase | +3.98 % | +1.82 % | +4.35 % | +1.94 % |
| Mean teacher KL (4 prompts) | 0.04467 | 0.02119 | 0.04482 | 0.02322 |
| Top-1 agreement | 88.62 % | 92.13 % | 89.07 % | 92.26 % |
| 32-token agreement | 36.46 % | 52.47 % | 43.62 % | 55.86 % |

GPTQ wins every metric in every seed at zero inference bits. Gaussian FWHT +
GPTQ W4/g128 became the promoted recipe. Raw records exist only on Drive.

### 7.4 W4A8/E8 composition (1 September, code `8ad3b8e`, seed 0)

| Arm | WikiText-2 | C4 | Mean / p95 KL | Top-1 | Decision |
|---|---:|---:|---:|---:|---|
| Promoted W4 | +2.73 % | +1.77 % | 0.02226 / 0.0706 | 92.71 % | retain |
| Bundled "optimized" W4 (learned butterflies, shared rotations, signs, scale8, mean bias, all at once) | +2,436 % | +2,840 % | 3.047 / 8.06 | 31.2 % | reject; confounded by the subnormal scale-step and GPTQ scale-mismatch defects fixed on 2 September |
| Promoted W4 + per-token A8 | +2.95 % | +1.85 % | 0.02380 / 0.0776 | 92.71 % | quality viable, +6.9 % KL, no speed evidence |

Record: `research/results/qwen35_4b_w4a8_e8_8ad3b8e6c809.json`.

### 7.5 Unsloth anchors before the fresh run

| Date | Inputs | Teacher and engine | Unsloth KL | Unsloth top-1 | RotQuant comparator |
|---|---|---|---:|---:|---|
| 2 Sep | 4 C4 prompts, 2,044 positions | BF16 GGUF in llama.cpp | 0.012944 | 93.69 % | promoted W4 0.022258, 92.71 %, 5.66 % larger |
| 3 Sep | 24 C4 prompts, 12,264 positions | BF16 GGUF in llama.cpp | 0.011888 | 94.09 % | uniform scale8 W4 0.016654, 93.42 %, 4.89 % larger |

Both are cross-engine and cross-teacher anchors, useful for direction only.

### 7.6 Factor ablation (3 September, code `2911e7a`, three seeds, 4-prompt suite)

| Arm | Mean KL | Top-1 | WikiText-2 | 32-token agreement | Complete bytes |
|---|---:|---:|---:|---:|---:|
| Promoted W4 | 0.022541 | 92.71 % | 9.7914 | 52.47 % | 3,787,286,336 |
| Scale8 W4 | 0.021921 | 92.55 % | 9.7985 | 52.08 % | 3,759,868,736 |
| Shared FWHT W4 | 0.023411 | 92.91 % | 9.7801 | 54.04 % | 3,787,163,456 |

Scale8 kept as the lower-byte uniform control; sharing neutral; learned signs
sent to replication. Record: `qwen35_4b_full_ablation_2911e7af539f.json`.

### 7.7 Allocators v1–v4 (3–6 September)

| Generation | Control it had to beat | Outcome at the fp16-vocabulary budget (3,584,533,344 bytes) | Record |
|---|---|---|---|
| v1 (RMS, no-GPTQ proxy) | matched random | Lost: KL 0.1381 vs random 0.1155 vs uniform scale8 W4 0.01665 | `qwen35_4b_dynamic_mixed_3dbae035f0d8.json` |
| v2 (deployed-quantiser scoring, marginal KL, Pareto DP) | random W3/W4 | −25.1 % KL (0.02587 vs 0.03453); export 1.57 % over the provider bytes; two identical finalists | `qwen35_4b_allocator_v2_2dce3aa43029.json` |
| v3 (exact artifact bytes, pair exchange, islands) | exact broad random | −72.8 % KL (0.03113 vs 0.11434); export 3,587,632,807 bytes, 0.087 % over; still 85 % above uniform W4 and 2.62× Unsloth; islands hurt; refinement a no-op | `qwen35_4b_allocator_v3_c9efd2d56774.json` |
| v4 (named format palette) | bits-only v3 | Full-format KL 0.03134 vs bits-only 0.03113; trajectory 45.8 % vs 36.8 %; no promotion | `raw/qwen35_4b_allocator_v4_8ba3b751bb82/` |

Learned signs (fp32 and fp16 angle storage) were shelved at the same time:
three-seed KL 0.016697 / 0.016931 versus 0.017052 for promoted W4, but 32-token
agreement fell from 58.9 % to 47.4 % / 50.0 % and bytes rose by 5–11 MB.

### 7.8 The byte decomposition (6 September)

| Component | Unsloth UD-Q4_K_XL | RotQuant scale8 W4 | RotQuant v3 Pareto |
|---|---:|---:|---:|
| Tied embedding (635.7 M params) | Q6_K, 521,472,000 B (6.56 bpw) | fp16, 1,271,398,400 B | fp16, 1,271,398,400 B |
| Backbone (3,565 M params) | 2,367,959,040 B (5.31 bpw) | 1,810,866,880 B (4.06 bpw) | 1,618,188,789 B (3.63 bpw) |
| Vision tower | 672,423,616 B | 667,028,480 B | 667,028,480 B |
| Small tensors, signs, codebooks | 11,710,464 B | 10,574,656 B | 10,574,656 B |
| Container | 10,968,206 B | ~20.4 MB | 20,442,482 B |
| Complete | 3,584,533,344 B | 3,759,868,416 B | 3,587,632,807 B |
| Mean KL (24-prompt) | 0.01189 | 0.01680 | 0.03113 |

Per bit, the RotQuant primitive was already ahead of imatrix K-quants on this
model: one bit per weight buys about 4.4× in KL in the project's own data, so
the provider's 1.25-bit backbone advantage should have been worth about 6× and
was worth 1.41×. Everything after this table follows from it.

### 7.9 Vocabulary screen (6 September, code `ce6c8ec`, seed 0, development suite)

| Backbone / vocabulary | Primary KL | Top-1 | 32-token match | Projected GB |
|---|---:|---:|---:|---:|
| FP16 / FP16 | ≈0 | 100 % | 100 % | 9.099 |
| FP16 / W8 | 0.000067 | 99.41 % | 94.0 % | 8.474 |
| FP16 / W6 | 0.000831 | 97.96 % | 89.4 % | 8.315 |
| W4 / FP16 | 0.016497 | 93.35 % | 33.6 % | 3.781 |
| W4 / W8 | 0.016552 | 93.35 % | 33.6 % | 3.155 |
| W4 / W6 | 0.017301 | 93.00 % | 34.1 % | 2.996 |
| W5 / FP16 | 0.003969 | 96.63 % | 80.6 % | 4.226 |
| W5 / W8 | 0.004037 | 96.60 % | 79.5 % | 3.601 |
| W5 / W6 | 0.004741 | 96.05 % | 81.6 % | 3.442 |

W6 vocabulary alone costs 0.0008 KL; the W4 to W5 backbone step recovers
0.0125. Archive: `raw/qwen35_vocabulary_budget_ce6c8ec861a2/`.

### 7.10 Packed export and revalidation (7–8 September)

Exported artifacts: W5/W6 3,441,544,638 bytes (143 MB under the provider
bundle), W5/W8 3,600,470,206 bytes (0.445 % over, inside the 1 % ceiling). The
first fresh-process reload of W6 failed its strict gate (mean KL 1.18e-5,
max abs logit error 0.023) because the loader's whole-model fp16 cast rounded
Qwen's fp32 RoPE frequency buffers; after the fix both artifacts reload with
zero probe error on 16 positions and four short generations, and their
development quality is 0.0047409438 / 0.0040373128 KL. Archive:
`raw/qwen35_packed_revalidation_89d25f3/` (keeps the failed record too).

### 7.11 Fresh quality (9 September, producer `d4292d6`, A100-SXM4-40GB)

24 new C4 documents at skip 32,768, common HF FP16 teacher, frozen HF token IDs
for every arm, BF16-GGUF bridge, `frozen-hf` input policy for the two GGUF
arms (four Hindi prompts segment differently under llama.cpp's `qwen35`
pre-tokeniser; recorded, not hidden).

| Arm | Bytes | C4 KL | Top-1 | Authored tasks correct of 96 |
|---|---:|---:|---:|---:|
| HF source FP16 | — | 0 | 100 % | 77 |
| BF16 GGUF bridge | — | 0.00002712 | 99.58 % | 77 |
| Unsloth UD-Q4_K_XL | 3,584,533,344 | 0.013379 | 94.04 % | 74 |
| W5/W6 seed 0 / 1 / 2 | 3,441,544,638 | 0.005939 / 0.005770 / 0.005633 | 96.02 / 95.91 / 95.95 % | 71 / 80 / 73 |
| W5/W8 seed 0 / 1 / 2 | 3,600,470,206 | 0.005210 / 0.004959 / 0.004901 | 96.33 / 96.32 / 96.51 % | 69 / 81 / 73 |

Paired document-bootstrap contrasts (right minus left, 24 documents, 4,000
draws), from `fresh/summary.json`:

| Contrast | KL delta | 95 % interval | Top-1 delta |
|---|---:|---|---:|
| Unsloth − W5/W6 (seeds 0, 1, 2) | +0.0074, +0.0076, +0.0077 | about [+0.0064, +0.0089] | −2.0, −1.9, −1.9 points |
| Unsloth − W5/W8 (seeds 0, 1, 2) | +0.0082, +0.0084, +0.0085 | about [+0.0071, +0.0097] | −2.3, −2.3, −2.5 points |
| W5/W8 − W5/W6 (matched seed) | −0.00073, −0.00081, −0.00073 | excludes zero | +0.3, +0.4, +0.6 points |

Per-domain authored tasks (24 each; multilingual arithmetic and tool
selection are at their ceilings for every arm, 24/24 and 18/24):

| Arm | Code tracing | JSON structure | Per-domain notes |
|---|---:|---:|---|
| Source FP16 | 13 | 22 | the source itself gets 11 traces wrong |
| Unsloth | 8 | 24 | |
| W5/W6 s0 / s1 / s2 | 9 / 14 / 13 | 20 / 24 / 18 | |
| W5/W8 s0 / s1 / s2 | 9 / 15 / 13 | 18 / 24 / 18 | |

Every JSON failure is a correct object wrapped in Markdown fences; every tool
failure chose the right tool with an `"invoice INV-13"` query the oracle
rejected; the `condition` family never exercised its positive branch. The
task suite therefore measures formatting habits and seed variance more than
capability. Archive: `raw/qwen35_fresh_quality_d4292d6fdec6/` (2,281 files,
1,125 SHA-256 pairs; no tensors).

### 7.12 KV cache track

Everything measured before 1 September is withdrawn: under Transformers 5.16
the simulator shared linear-attention state between its two decode passes, so
K8/V8, K4/V4 and K2/V2 all reported KL 0.5–0.9 and top-1 about 0.72. On the
fixed simulator with the pinned 5.9.0 the source model gives K8/V8 5.0e-4,
K4/V4 6.3e-3, K2/V2 6.5e-2 (CPU, 256-token prompt, 16 tokens). The one
post-pin run, four 8k-prefill prompts with 2-bit E8P and fp16 sink/recent
tiers, gave cache KL 0.0192 (source), 0.0200 (W4), 0.0207 (W4A8) at a
prefill-accounted 2.188 bits per value and 7.31× raw compression; it predates
tier ageing (deployed figure closer to 6.98×) and the endpoint check, so it is
provisional. Nothing has been re-measured since.

### 7.13 The one table to quote

| Claim | Number | Conditions |
|---|---:|---|
| W5/W6 C4 KL vs Unsloth | 0.00563–0.00594 vs 0.01338 (−56 to −58 %) | 24 fresh documents, common FP16 teacher, three quantisation seeds |
| W5/W8 C4 KL vs Unsloth | 0.00490–0.00521 vs 0.01338 (−61 to −63 %) | same |
| Bytes | W6 −3.99 %, W8 +0.45 % relative to 3,584,533,344 | complete artifacts including vision and container |
| Top-1 agreement | W6 95.9–96.0 %, W8 96.3–96.5 %, Unsloth 94.0 % | same inputs |
| Development-suite KL (older, for continuity) | W5/W6 0.00474, W5/W8 0.00404, scale8 W4 0.0165–0.0168, Unsloth 0.01189 | 24 development prompts; Unsloth on a BF16-GGUF teacher |
| Reload | zero probe error, six artifacts | 16 positions, four generations each |

### 7.14 Why there are several "uniform W4" KL values

| Value | Recipe | Suite | Seeds | Record |
|---:|---|---|---|---|
| 0.02226 | promoted W4, fp16 scales | 4 prompts, 2,044 positions | 0 | `w4a8_e8_8ad3b8e` |
| 0.02254 | promoted W4 | 4 prompts | 3 | `full_ablation_2911e7a` |
| 0.02192 | scale8 W4 | 4 prompts | 3 | same |
| 0.01705 | promoted W4 | 24 prompts | 3 | `dynamic_mixed_3dbae03` |
| 0.01665 | scale8 W4 | 24 prompts | 0 | same |
| 0.01680 | scale8 W4 | 24 prompts | 3 | `allocator_v2`, `allocator_v3` |
| 0.01650 | scale8 W4, later code era | 24 prompts | 0 | vocabulary screen |

The differences are the suite and the number of seeds, not contradictions.
Complete bytes for scale8 W4 also appear as 3,759,868,416 and 3,759,868,736:
`registered_model_bytes` moved by 320 bytes between the `8ad3b8e` and
`8ba3b75` code eras, an undocumented accounting change.

## 8. Negative results and lessons

### 8.1 Things that did not work, and why

| Idea | What was tried | Why it lost | Status |
|---|---|---|---|
| Learned butterfly rotations | Layerwise, block and Hessian objectives, 200 steps, calibration-scale budgets | Local proxy gains of about 2 % did not transfer; one arm was confounded by since-fixed scale bugs; deployed fp16 angles are not exactly orthogonal (L1) | Shelved; QRAT reserved for a properly budgeted attempt |
| Learned signs | Three seeds, fp32 and fp16 angles | Inert at first (±1 logits could not flip), then small KL gains with a trajectory regression and more bytes | Shelved |
| Block recovery on the winner | Block-and-scale reconstruction | PPL 14.7029 → 14.7130 with 8.9 MB more state | Retired |
| LoRA-QAT | Rank 4/8, up to 32 steps on a few thousand tokens | Best step was always 0; the 31 August review calls the budget three orders of magnitude below the literature | Underpowered, not disproved |
| End-to-end distillation | 512 tokens, 12 steps | Selected step 0 | Same |
| TurboQuant QJL sketch for weights | 64 sign bits per row | Adds about πd/2k more squared error than it removes: ×104 at d = 4096 | Null control only |
| Per-row TurboQuant scales | Under block-128 rotation | Inter-block energy variation survives a block rotation; the arm tests a weaker setting than TurboQuant's | Retired |
| Spherical codebook | d = 128 | Indistinguishable from Gaussian to four decimals | Retire |
| Dimension-2 vector codebook | W3 | Beat scalar locally, poor absolute quality, catastrophic transfer to Qwen2.5-3B | Research only |
| Teacher-guided 3.625-bpw compact recipe | Algorithm Lab | Won teacher-forced NLL, worst 32-token agreement 7.8 % | Rejected: free-running behaviour is the gate |
| Allocators v1–v4 | At a budget with an fp16 vocabulary | The budget forced a 3.6-bpw backbone against a 5.3-bpw one; no allocator can recover 1.7 bits per weight | Sensitivity ranking and solver kept; conclusions about islands withdrawn |
| Forced W6/W8 islands | Top 1–5 % of layers | At 3.6 bpw every upgrade was paid for by a damaging downgrade | Untested at the corrected budget |
| A8 activations | Per-token int8 after rotation | +6.9 % KL, no native A8 GEMM to show a speed benefit | Decision pending since 31 August |
| E7 "mismatched" control | Rotate weight, feed raw activation | Breaks a single layer's identity; tests a triviality | Reframe or drop |
| KV cache mixed maps | Everything before 1 September | Simulator defect | Withdrawn |
| Cross-engine provider anchors | 4- and 24-prompt BF16-GGUF teachers | Different teacher and engine from the RotQuant side | Replaced by the common-teacher design |

### 8.2 Methodological lessons the project learned the hard way

- **Match the controls or the result is uninterpretable.** The first KV
  matrix evaluated uniform and dynamic rows on different calls; the release
  gate compared a seed-0 control with a worst-of-three candidate.
- **Insist on an endpoint that must be near zero.** An 8-bit cache that does
  not reproduce the source is a bug, not a result. The same principle is now
  wanted for weights (an identity-quantiser arm).
- **Pin everything.** An unpinned `transformers>=5.9,<6` resolved to 5.16 and
  silently changed the cache layout. Every notebook now pins 5.9.0.
- **Development suites get spent.** The 24 + 25 prompts selected every recipe
  since 3 September; the fresh run had to draw new documents and exclude every
  archived token hash.
- **Bytes must be decomposed, not summed.** The 1.27 GB vocabulary hid in a
  "registered bytes" line item for a week.
- **Seeds vary the quantisation RNG only.** Calibration rows and test documents
  are shared; three seeds are not three datasets and must not be pooled.
- **Bootstraps need clusters.** Token-level intervals inside a document are too
  narrow; a percentile bootstrap of four prompts is not an interval; family
  bootstraps over four families have at most 35 distinct resamples.
- **Fail closed, and say so in the record.** Silent CPU fallback, silent block
  shrinking, silent H = I, silent metric switches are the recurring class of
  defect in §9.
- **KL is not task accuracy.** Lower C4 KL did not order the authored-task
  scores; seed 1 won both recipes by a margin larger than the recipe effect.
- **Cross-engine numbers need a bridge.** The BF16-GGUF bridge (KL 0.000027,
  identical task outputs) is what makes the common-teacher provider comparison
  defensible.

## 9. Defect and open-item register

Status as of `85e1254`. "Open" means verified still present in the code today.

### 9.1 Library queue behind the reuse freeze (deep dive §4.2)

| # | Where | Defect | Status |
|---|---|---|---|
| L1 | `rotate.py:371` | Butterfly cos/sin computed in the storage dtype; fp16 angles give a non-orthogonal transform (gain 1.0033) and the gate scores fp32 before storage commit | Open |
| L2 | `checkpoint.py:771-787` | `model.to(dtype)` rounds fp32 butterfly angles before restoring them; `LearnedRotation.theta` and `DenseOrthogonal.R` never restored | Open |
| L3 | `checkpoint.py:313-317` | `torch_dtype` inferred from the first floating tensor; a v3 artifact with fp32 theta reloads as fp32 | Open |
| L4 | `quantize.py:958`, `calibrate.py:181-188`, `linear.py:402-403` | GPTQ without a Hessian warns and falls back to rounding; `refresh_quantization()` re-packs GPTQ layers as RTN | Open |
| L5 | `native.py`, `gguf.py`, `native_v3.py` | Legacy exporters now reject non-FP16 scales; separate native-v3 matrix storage preserves compressed scales exactly | Fixed for export; full-model v3 integration pending |
| L6 | `rotate.py:255` | A block that does not divide the dimension is silently replaced by the largest power-of-two divisor | Open |
| L7 | `linear.py:114-149`, `patch.py:437-438` | In-process `.half()` recasts rotation parameters | Open |
| L8 | `quantize.py:137-146` | `replace(cfg, codebook=…)` keeps Gaussian search bounds, so a uniform codebook clips at 1.5σ (the E2 confound) | Open |
| L9 | `run_experiment.py:723-724`, `linear.py:449-474` | Rotation buffers outside the byte counters; shared theta counted per sibling | Open (by reading) |
| L10 | `calibrate.py:50-54`, `quantize.py:117` | Double damping on the public API path | Open (by reading) |
| L11 | `checkpoint.py:739-744` | LoRA storage dtype not preserved on reload | Open |
| L12 | `quantize.py:400-410` | No finiteness check on fp16 scale storage | Open |

None affects the archived W5 artifacts (fp16 model dtype, parameter-free FWHT).

### 9.2 Allocator, KV simulator, block training, statistics (deep dive §4.4)

`_pareto_selection` in `target_bpw` mode does not optimise the objective (149
of 300 random instances suboptimal); the KV simulator cannot prove decode rows
went through it (only `cache.update` is patched); non-tiered 8-bit-scale cache
writes are packed per chunk; missing activations silently switch the
allocator's local metric; streamed block training trains on the deployed
input rather than the source's own output and reports zero drift; the endpoint
check has trivial-pass modes and a 25× loose threshold; the candidate-score
cache key omits dtype, device and code revision; solver optimality is
unquantified (an exact MILP audit is cheap); three bootstrap implementations
with different RNGs. All open.

### 9.3 Native, GGUF, CI, packaging (deep dive §4.5)

The llama.cpp patch workflow only checks that the patch applies; the patch
already fails to apply at upstream HEAD in four files; no Windows/MSVC job;
CI downloads the CUDA torch stack for lint; NEON never compared against the
Python oracle in CI; `inspect_gguf_types.py` lacks ggml types 40–42 and prices
`.rqweight` at 8 bpw; no wheel or sdist build; no tag; the `baselines` extra
pins packages that predate Transformers 5 and is untested. The C ABI enum
undefined behaviour was fixed on 8 September. All others open.

### 9.4 Documentation and claims

Withdrawn cache numbers still shipped by `paper/generated/audit_report.json`
(reports the 24.0 %/10.6 % reductions and 0.80/0.84 ratios as derived values),
`configs/publication_qwen35_joint_cuda.yaml`, `docs/serving_backends.md` and
the llama.cpp README; the README's obsolete confirmation bar; CHANGELOG
duplicated headings; `CITATION.cff` describes a 0.1.0 release that has no tag;
the three-seed GPTQ promotion, the OPT ladder and all native throughput numbers
exist only on Drive or in prose; the fresh-quality archive carries four
identical 1.2-million-line `tokenizer.json` copies.

### 9.5 Fixed in the last week, for the record

Shared rotations under `inference_mode` (R1), missing `nbformat` in CI (R2),
confirmation promoting regressed seeds (R3), estimate-only export passing the
byte gate (R4), scoring/deployment mismatch for shared rotations and A8 (R5),
Pareto pruning discarding the only feasible state (R6), attention parents
reading child weights (R7), interrupted overwrite leaving a mixed checkpoint
(R8), whole-prefill activation retention (R9); the RoPE fp32 buffer downcast;
the missing `generation_config.json`; the Hindi tokenizer gate; the shallow
clone in CI; the C ABI enum; the NLL columns in the fresh summary; the
llama.cpp build timeout.

## 10. Evaluation protocol notes

### 10.1 Metrics

| Metric | Definition | Used for |
|---|---|---|
| Teacher KL | `KL(p_source ‖ p_candidate)` per position at T = 1 over the full vocabulary; mean, median, p95, max | primary fidelity |
| Top-1 agreement | fraction of positions where argmax matches the source | secondary |
| NLL delta | candidate minus source negative log-likelihood, token weighted | secondary |
| Perplexity | non-overlapping 2,048-token windows from position 0 on pinned WikiText-2 and C4, `max_samples: 32` (the first 65,536 tokens) | regression sentinel, not literature-comparable |
| Trajectory agreement | greedy 32-token continuations: token agreement, exact-trajectory rate, matching prefix | free-running behaviour |
| Task success | strict typed oracles on authored tasks; public checkers on GSM8K, CRUXEval-O, IFEval | usefulness |
| Layer NMSE, KV NMSE | reconstruction diagnostics | mechanism checks |

### 10.2 Suites

| Suite | Size | Status |
|---|---|---|
| Primary development C4 | 24 × 512 tokens at skip 4,096, 12,264 positions | Spent (used for every selection since 3 September) |
| Diverse development | 25 authored snippets, 5 per domain (agentic, code, maths, multilingual, long document), plus 25 trajectory prompts | Spent |
| Fresh C4 | 24 × 512 tokens at skip 32,768, disjoint from every archived hash | Used once (9 September); now development data |
| Authored tasks | 96: 24 each of multilingual arithmetic, Python code tracing, JSON structure, tool selection | Frozen; two families need repair before reuse |
| Public tasks | GSM8K test (1,319, MIT), CRUXEval test (800, MIT), IFEval (541, Apache-2.0); 128 hash-selected per benchmark with seed 20260909, 384 per arm, nine arms, 3,456 generations; caps 1,024 / 512 / 2,048 new tokens; prompts over 2,048 tokens fail preparation; Google's IFEval checkers pinned at `e6890f85`, NLTK 3.9.2 | Prepared, never run |
| 300-prompt competitive contract | 60 per domain, 32 greedy tokens, exact token IDs, disjoint calibration | Specified since 31 August, never built; five prerequisites untouched |
| Zero-shot bundle | ARC, BoolQ, PIQA, WinoGrande, HellaSwag via `lm-eval` | Wired, never run; `lm_eval: not-installed` in every archived environment |

### 10.3 Gates and thresholds

- **Promotion guards** (`rotquant.eval.promotion`): primary and diverse KL and
  each PPL ratio ≤ 1.02, top-1 loss ≤ 1 point, trajectory loss ≤ 2 points,
  finite metrics, paired samples ≥ 20 for a reliable interval.
- **Fail-fast** during evaluation: mean KL > 0.25 or top-1 < 75 % stops an arm.
- **Packed reload probes**: prototype→packed max abs logit error ≤ 0.125, mean
  ≤ 0.005, KL ≤ 1e-4, top-1 ≥ 99 %; packed→fresh reload ≤ 0.002, ≤ 0.0002,
  ≤ 1e-6, 100 %, exact short generations required.
- **Size**: same-size pair requires ≤ 1 % byte mismatch, two-sided; an
  under-budget artifact is a frontier point, not a matched pair.
- **KV endpoint**: uniform 8-bit cache must give KL ≤ 0.01 on the held-out
  calls (observed 4e-4; the threshold is loose).
- **Claim language** (`competitive_eval.md`): any competitive statement names
  model revision, artifact pair, exact bytes, prompt fingerprint, metrics,
  interval, engine, hardware and code revision; "best" and "same size" are
  blocked until the raw records and protocol are public.

### 10.4 Statistics rules

Paired bootstrap over documents or families, 4,000 seeded draws, two-sided
95 %; `interval_reliable` false below 20 paired samples; report seed-specific
results and never pool seeds as independent prompts; three-seed "±" values in
the log are population standard deviations; KL differences are not additive
(do not subtract bridge KL from candidate KL); a non-significant loss is not
equivalence.

## 11. Operations cookbook

### 11.1 Local

```bash
uv sync --locked --extra dev --extra eval      # 5.2 GB on Linux: the lock pins torch 2.12.0+cu130
uv run --no-sync ruff check .                  # locked ruff is 0.16.5; an older ruff reports E402 on the sys.path bootstraps
ROTQUANT_REQUIRE_GIT_HISTORY=1 uv run --no-sync pytest tests/ -q   # 729 passed, 17 skipped, about 2.5 minutes; needs full git history and cmake
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DROTQUANT_NATIVE_WARNINGS_AS_ERRORS=ON
cmake --build build/native --parallel && ctest --test-dir build/native --output-on-failure
build/native/rotquant-native-cli --capabilities
uv run --no-sync python scripts/run_experiment.py configs/smoke_cpu.yaml --output-dir /tmp/smoke   # PPL 32140.2637 on the random model
```

Dry runs and preflights that need no GPU:

```bash
uv run --no-sync python scripts/preflight_packed_validation.py --device cpu
uv run --no-sync python scripts/run_qwen35_packed_validation.py --output-dir /tmp/plan --dry-run
uv run --no-sync python scripts/run_qwen35_fresh_eval.py --output-dir /tmp/plan --dry-run
uv run --no-sync python scripts/build_qwen35_public_tasks_notebook.py   # regenerates the notebook; must be byte-identical
```

GGUF tooling (old W4 artifact only):

```bash
scripts/build_rotquant_llama_cpp.sh                       # clones llama.cpp at 17252c76, applies the patch, builds CLI, server, bench
uv run python scripts/export_rotquant_gguf.py <checkpoint> out.gguf --llama-cpp-dir third_party/llama.cpp
uv run python scripts/verify_rotquant_gguf.py <checkpoint> out.gguf --llama-cpp-dir third_party/llama.cpp
uv run python scripts/inspect_gguf_types.py <file-or-url>   # per-layer ggml types and byte shares from the header alone
scripts/serve_rotquant_gguf.sh out.gguf 8085
```

### 11.2 Colab, before pressing run

1. Merge everything to `main`; confirm Python CI is green at that SHA.
2. Open the notebook, run the checkout cell, copy the printed SHA into
   `REPO_REF`, and leave it there for the life of the run.
3. Do not push to `main` and do not commit under `research/results/raw`
   until the summary exists.
4. First session: smoke scope (`SMOKE_RUN=True` for the public-task
   notebook), a separate root. Then the standard scope on a new root.
5. Check the runtime against the previous manifest if reuse is intended
   (torch, CUDA, Python, GPU name).
6. On disconnect: reconnect, read the printed PID and `tail -f` path, never
   start a second worker on the same root.
7. Afterwards: download the compact ZIP, run the matching `archive_*.py`,
   commit archive, results note, log entry and roadmap pointer together.

### 11.3 Things that will bite

- A resumed run after a `main` push: "consumer code changed", new root needed.
- Committing an archive mid-run: "frozen protocol changed", every phase refuses.
- A different Colab GPU SKU: reuse refuses; budget for rerunning `source`
  and seed 0.
- Importing llama.cpp before torch in a diagnostic: NCCL symbol clash; torch
  first.
- The llama-cpp-python CUDA build is about 29 minutes; the timeout is 3,600 s.
- Running `pytest` while committing: the packed-validation identity test
  hashes HEAD and fails on a moving tree.
- A shallow clone skips the reuse guard locally; CI sets
  `ROTQUANT_REQUIRE_GIT_HISTORY=1` so it cannot skip there.

## 12. Decisions log

| Date | Decision | Basis |
|---|---|---|
| 29 Aug | Rotation is mandatory at 3 bits; keep a no-rotation arm as a required control | OPT results |
| 30 Aug | Block-only W4 is the winner; do not pay adapter bytes without a held-out gain | Qwen CUDA matrix |
| 31 Aug | Stage 2 is canonical GPU serving; reliability gates are entry gates | Roadmap |
| 1 Sep | Promote streamed act-order GPTQ into W4; Gaussian is the default codebook, calibrated the challenger | Three-seed ladder |
| 1 Sep | Withdraw every cache-quality result; require an 8-bit endpoint | Simulator defect |
| 1 Sep | The Unsloth comparison is against a static mixed-format recipe; mixed precision is necessary for a fair frontier but not RotQuant's contribution | Competitive decision entry |
| 2 Sep | Reject the bundled optimised-W4 arm without blaming a factor; decompose | Confounds found |
| 3 Sep | Keep scale8 W4 as the lower-byte control; shelve learned signs; retire allocator v1 | Ablation and dynamic run |
| 4 Sep | Promote allocator v3's recipe only as the bits-only baseline; islands do not promote | v3 result |
| 5 Sep | Reliability fixes before the next sweep; one measured packed operator after a quality winner | Project review |
| 6 Sep | The vocabulary budget comes first; do not spend an A100 day on v4 as registered | Results review |
| 7 Sep | Advance W5/W6 first with W5/W8 as comparator; verify real files before any search | Vocabulary screen |
| 8 Sep | Fresh inputs and seeds 1/2 before anything else; the library is frozen until that run completes | Deep dive |
| 9 Sep | Keep W5/W6 as the under-budget candidate and W5/W8 as the fidelity alternative; no promotion from the authored tasks; run the public-task gate; then one measured serving path | Fresh-quality result |
| 9 Sep | Compression must reduce inference memory traffic, not only file size; fused decode/scale/matmul with device-resident static data is a runtime acceptance requirement | User priority recorded in the roadmap |
| 9 Sep | Extend the external target to every published Unsloth variant (21 for 4B, 24 for 27B), curve not point; 27B is separately budgeted | Frontier plan |

## 13. Open questions and hypotheses

1. **Does the C4 KL advantage survive public tasks?** GSM8K, CRUXEval-O and
   IFEval on nine arms. A 128-example subset can detect large regressions
   only; a one- or two-point effect needs a preregistered larger run.
2. **How much of the W5 advantage is the vocabulary format and how much the
   backbone?** The screen says the W4→W5 step is worth 0.0125 KL and W6
   vocabulary costs 0.0008, but only at seed 0 on the development suite.
3. **Would spending the 143 MB of W6 headroom on `ssm_out` and `down` help?**
   The provider keeps them at Q8_0/Q6_K; allocator v3's sensitivity ranking
   agrees; untested at the corrected budget.
4. **Does a larger calibration set reduce seed variance?** 65,536 tokens give
   the 9,216-wide `down` Hessians about seven samples per dimension.
5. **Attention-type sensitivity.** Are Gated DeltaNet projections more robust
   to W4 than full-attention projections on this model? The September note
   specifies isolation arms with MLP and vocabulary precision fixed.
6. **Can learned rotations help at W5/W6 where they did not at W4?** The
   trainer removes GPTQ from its proxy while the gate uses GPTQ; measure the
   mismatch before spending GPU time, and fix L1/L2 first.
7. **Is E8P worth anything on the fixed simulator?** Repeat the 8k check with
   the endpoint recorded, one fixed FP16 teacher, at least 20 prompts and
   10,000 tokens per arm, or park the track.
8. **Activation quantisation.** Declare A8 out of scope for the kernel contract
   until a native A8 GEMM exists, or build one; the decision has been pending
   since 31 August and blocks the operator interface.
9. **What does the W5 recipe cost to serve?** Unknown until a runtime exists:
   resident bytes, prefill and decode tokens/s, DRAM traffic.
10. **Where on the 21-variant curve does RotQuant sit?** One point has been
    measured; the provider's own IQ2 to Q8 range is 1.52 to 5.95 GB.

## 14. Next steps

The September 10 CUDA result supersedes the original review's §4 ordering. Follow
[`native_runtime_v3.md`](native_runtime_v3.md): exact full-model GGUF/CPU graph
and shared vocabulary, numerical conformance, packed Metal/CUDA execution,
measured speed/memory/cost preflight, then a new runtime-bound public-task run.
The matrix format/CPU floor, L5, full graph and packed GPU path are implemented;
the user's A100 W5/W6 probe run passed. Next: persistent compatible runtime/export
reuse, bounded timing/VRAM, retained W5/W8 and broader-context parity, then small
native public tasks. Unrelated review defects are not closed by this run.
Research branches and all-variant 4B/27B
comparisons wait for this serving path. No version tag or release is implied.

## 15. Glossary

| Term | Meaning here |
|---|---|
| bpw, bpv | bits per weight, bits per cache value, including scale metadata: W4/g128 with fp16 scales is 4.125 bpw; with 8-bit scales about 4.06 |
| FWHT | fast Walsh-Hadamard transform; `RandomizedHadamard` applies random signs then a normalised block-128 Hadamard |
| Butterfly | trainable orthogonal transform built from two-coordinate rotations, initialised at π/4 to equal FWHT |
| Gaussian codebook | Lloyd-Max centroids for a standard normal; 2-bit MSE 0.1175, 3-bit 0.0345, 4-bit 0.0095 |
| MSE search | choose each group scale from a 41-point grid between 0.5 and 1.5 × RMS to minimise reconstruction error |
| GPTQ, act-order | column-sequential quantisation with error feedback through the inverse Hessian; act-order visits columns by decreasing `diag(H)` |
| scale8 | 8-bit double-quantised scales: blocks of 256 scales share an fp16 offset and step |
| Tied vocabulary | one matrix serves as input embedding and output head; 635.7 M parameters on Qwen3.5-4B |
| `dense_equivalent` | vocabulary projection mode that inverse-rotates and rounds one row tile to the execution dtype before multiplying |
| Consistent mode | the activation is rotated with the same rotation used to encode the weight; `mismatched` is the negative control |
| Shared rotation | one rotation object for sibling projections that consume the same activation (q/k/v, gate/up) |
| Teacher KL | `KL(source ‖ candidate)` over the full vocabulary at the same position |
| Trajectory | greedy 32-token continuation compared token by token with the source |
| Endpoint check | a near-lossless configuration (8-bit cache) must reproduce the source before any candidate is scored |
| Reuse receipt | record binding adopted results to producer and consumer `source_identity()` plus runtime identity |
| Compact bundle | the JSON/CSV/log ZIP that returns from Colab; no tensors |
| Evidence index | `evidence_index.json`, SHA-256 of every archived original file |
| UD-Q4_K_XL | Unsloth's Dynamic 2.0-era mixed-format GGUF for Qwen3.5-4B; 5.31 bpw backbone, Q6_K vocabulary |
| Same-size pair | complete artifact bytes within 1 % of each other, two-sided |
| Native-v2 | backend-neutral block layout: fp16 scale plus packed codes per group, 1–8 bits |
| RotQuant-GGUF v1 | the llama.cpp fork's 4-bit format with `.rqweight` and `.rqrotation` tensors |
| MTP head | Qwen's multi-token-prediction tensors, 241 MB in the source index, never loaded, excluded from every like-for-like byte count |
| QRAT | reserved name for a future quantisation-and-rotation-aware training method at a real training budget; nothing current qualifies |
| Reuse freeze | the CI test that pins `rotquant/` to revision `733bb3d` while adopted records depend on it |
