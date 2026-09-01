# RotQuant implementation roadmap

PR #6 established the production-facing Python API, packed checkpoint v1,
native-v2 CPU runtime, experimental llama.cpp path, and the 1--8-bit storage
contract. The next stage turns that foundation into a dependable GPU-serving
path. Correct artifact production and reconstruction are entry gates, not
postponed cleanup.

## Stage 2: canonical GPU serving

### 2.0 Reliability gate

Current gate status:

- [x] reconstruct the model with the adapter recorded in the checkpoint before
  selecting the Transformers auto-model class or replacing modules;
- [x] use the adapter's conversion and replacement hooks while loading custom
  projection types;
- [x] reject `optimize_model()` calls whose include/exclude filters select no
  modules, rather than returning a full-precision model with a quantized report;
- [x] cover decoder-only, encoder, encoder-decoder, and custom-adapter round
  trips, plus empty-selection failures, in CI (`python-ci.yml` runs the full
  pytest suite, including the cross-language native conformance tests, on
  Python 3.10/3.11/3.12; previously no workflow executed any Python test);
- [ ] add the small multimodal round-trip fixture without relying on a remote
  checkpoint or GPU.

Acceptance: a fresh-process checkpoint round trip preserves packed bytes and
reference logits for every supported loader class, and every successful
optimization report records at least one patched module.

### 2.0b Infrastructure track

Every Stage 2 acceptance gate below depends on infrastructure that must be
named, not assumed:

- [x] Python CI (pytest across supported versions + ruff) on every push and PR;
- [x] native sanitizer job (ASan/UBSan) and a llama.cpp patch-applies check;
- [ ] **decide and provision the GPU CI mechanism.** Stages 2.2 and 2.3
  require per-bit-width CUDA conformance, memory, and throughput gates plus
  vLLM server conformance; none of these run on hosted free runners. Options:
  a self-hosted GPU runner, a scheduled cloud job (Modal/Lambda/RunPod), or a
  documented manual pre-merge checklist with pinned benchmark manifests.
  This decision blocks Stage 2.2 kernel acceptance, so it is made before
  kernel work starts, not after;
- [ ] Windows/MSVC native build job (the MSVC CPUID, `/W4 /WX`, and dllexport
  paths are currently unverified; the exported `std::vector` surface in
  `native_v2.h` is expected to need attention there);
- [ ] wheel builds that bundle the native library (cibuildwheel) once the
  packaging milestone below is scheduled.

### 2.0c Release engineering (pulled forward from Stage 5)

Licensing and installability are prerequisites for anyone trying the project
during Stages 2--4, not end-stage polish:

- [x] LICENSE (MIT), CITATION.cff, CONTRIBUTING.md, CHANGELOG.md;
- [x] complete packaging metadata, `rotquant.__version__` recorded in result
  provenance, `py.typed`, and the eval package namespaced as `rotquant.eval`;
- [ ] first tagged release and PyPI publication (sdist + pure-Python wheel;
  native wheels follow the infrastructure track);
- [ ] model cards and eval disclosure for released Hub artifacts, and a
  public location for the summarized results behind every published claim
  (raw JSONs currently live only in Google Drive).

### Parallel track: algorithm laboratory

Algorithmic changes advance through a controlled funnel before kernel work is
specialized around them:

- [x] add exact-rate dimension-2 vector controls at W1--W3 while keeping the
  scalar packed layout unchanged;
- [x] add deterministic weight-calibrated scalar codebooks and the existing
  spherical, length-corrected, and TurboQuant scale controls to one matrix;
- [x] compare local reconstruction, teacher-KL, and guarded mixed-bit allocation
  objectives with exact packed-byte accounting;
- [x] add a real-attention selective-V oracle that reports output error,
  selected attention mass, fallback rate, and effective V reads;
- [x] harden the Colab with a source/W4 sentinel, paired per-window NLL and
  bootstrap intervals, C4 confirmation, held-out logit/layer/trajectory
  diagnostics, immutable token hashes, complete model-byte accounting, and
  atomic resumable records;
- [x] add seeded random-allocation and scalar 2.75-bpw controls so dynamic and
  vector candidates must beat a matched-format, matched-rate baseline;
- [x] execute the staged
  [algorithm-lab Colab](../notebooks/rotquant_algorithm_lab_colab.ipynb) on the
  primary model, then promote only Pareto winners to three seeds and a second
  architecture family (Qwen3.5-4B primary and Qwen2.5-3B transfer; see the
  experiment log for positive and negative results);
- [x] rerun the repaired three-seed/cross-family confidence protocol with
  KL/top-1 and 32-token trajectories; calibrated W4 and Gaussian W4 are the only
  runtime candidates, while the 3.625-bpw compact and all sub-W4 recipes fail
  free-running or transfer quality;
- [x] formalize the exact-size external-artifact and competitive-claim gates in
  [the competitive evaluation contract](competitive_eval.md), including
  content-addressed prompt/calibration protocols and KL distribution tails;
- [x] implement the source-agnostic
  [competitive data pipeline](competitive_data.md): immutable source/license
  manifests, post-template token identities, exact/near leakage checks, fixed
  domain quotas, structured run failures, and paired/domain bootstrap reports;
- [x] add the focused, commit/config-aware
  [Qwen3.5-4B optimization Colab](../notebooks/qwen35_4b_optimization_stage_colab.ipynb)
  for matched calibrated/Gaussian W4 versus streamed GPTQ, with W4A8/E8,
  serious recovery, and 8k KV as separate opt-in stages; add a resumable
  source-Transformers/RotQuant collector for the registered divergence records;
- [ ] build a pinned, calibration-disjoint 300-prompt/32-token free-running
  divergence suite spanning agentic, code, maths, multilingual, and long-document
  prompts; compare the source, RotQuant, same-size GGUF, and Unsloth baselines;
- [ ] build and publish the diverse calibration mixture and generated
  importance/Hessian artifacts, with chat-template correctness, row/token
  hashes, licensing, deduplication, and calibration-size ablations;
- [ ] complete the Qwen3.5-4B development ladder in
  [the competitive evaluation contract](competitive_eval.md): frozen presets,
  the 300-prompt suite, Qwen2.5 transfer canary, every target-rate artifact,
  production-operator conformance, and a measured 27B cost forecast;
- [ ] make optimization resumable and layer-streamed on Qwen3.5-4B under an
  artificial 27B-oriented memory cap. GPU memory must be bounded to the active
  block and calibration workspace; no second full-model allocation is allowed;
- [ ] freeze and version the optimizer, calibration/eval manifests, byte policy,
  and promotion thresholds before reading Qwen3.8-27B final quality results;
- [ ] run locked Qwen3.8-27B W2/W3/W4 or nearest mixed-rate anchors against
  same-size standard GGUF and Unsloth artifacts before expanding the full
  frontier;
- [ ] run exact-size 1.5/2/2.5/3/4/5/6/8-bit frontier comparisons against
  standard llama.cpp GGUF and current Unsloth artifacts; freeze the RotQuant
  allocator before final-model evaluation and require paired intervals;
- [ ] implement packed-key candidate generation only if the dense-attention
  oracle shows a useful V-read/quality region;
- [ ] **meet the project's own confirmation bar at scale.** The README requires
  >=3 seeds on at least Llama-2-7B and 13B, on WikiText-2 and C4, surviving the
  zero-shot bundle; no current headline result meets it (results so far are
  OPT-125M/1.3B and Qwen3.5-4B on development subsets). This includes making
  the quantization pipeline itself scale: GPTQ Hessians cost ~25 GB VRAM on a
  7B and block training replays whole transformer blocks, so 70B-class models
  need the layer streaming / CPU-offload path now scheduled on the 4B ladder;
- [ ] integrate at least one trellis/lattice state-of-the-art baseline
  (QuIP#, QTIP, or HIGGS) with real checkpoint loading and exact rate
  accounting before publishing W1--W3 comparisons; the dimension-2 vector
  control is a protocol check, not a competitive baseline;
- [ ] add task-level generation quality (e.g. GSM8K, IFEval-style
  instruction following) to the release gates. The released artifact is a
  chat-capable model; perplexity plus the zero-shot bundle can miss
  instruction-following regressions entirely.

### Scope decision required: activation quantization

The rotation lineage this project builds on (QuaRot / SpinQuant / TurboQuant)
is motivated by W4A4/W4A8, and hypothesis E1 itself states that learned
rotations only pull ahead once activations are quantized. Weight-only
quantization mainly accelerates bandwidth-bound decode; prefill is
compute-bound and needs low-bit activations to use integer tensor cores. The
Stage 2.2 operator contract must either:

- add an activation-quantized serving stage (quantize-activation epilogue
  fused into the rotation, int-times-int GEMM path, per-token or per-tensor
  activation scales in the artifact schema), or
- explicitly declare activation quantization out of scope so the kernel
  contract is not silently designed into a weights-only corner.

This is a research-direction decision for the maintainer; the roadmap only
records that it must be made before the kernel interface freezes.

Research-only representations fail closed at checkpoint and native-runtime
boundaries. A winning algorithm earns a format and kernel proposal; adding an
experimental Python path alone does not expand production support.

### 2.1 Canonical Transformers artifact

- add `config.json.quantization_config.quant_method = "rotquant"`;
- replace ordinal packed tensor names with original module-prefix names;
- make packed tensors independently shardable, with logical shapes, codebooks,
  scales, rotations, mixed precision, tied weights, and retained adapters in a
  versioned schema;
- implement the Transformers quantizer/loader extension while retaining a v1
  compatibility reader and a deterministic v1-to-v2 conversion tool;
- publish small conformance fixtures for each architecture tier.

Acceptance: stock Transformers plus the RotQuant integration loads a sharded
artifact without reconstructing dense weights in persistent memory, and the v1
and v2 loaders produce equivalent selected-layer outputs and greedy tokens.

### 2.2 Fused GPU kernel contract

Build one backend-neutral operator boundary for packed linear layers, split into
decode GEMV and prefill/batched GEMM. CUDA/Triton is the first production
implementation; Mojo remains an experimental implementation behind the same
conformance and benchmark gates until it demonstrates a deployment advantage.

Kernel profiles roll out in measured groups:

1. W4 establishes loading, rotation fusion, tensor-parallel partitioning, and
   benchmark infrastructure.
2. W2/W3 add genuinely low-bit unpack and dot-product paths.
3. W1 adds its own binary profile rather than emulating a wider format.
4. W5--W8 add byte/word-aligned profiles selected independently by workload and
   hardware.

Each bit width needs a specialized dispatch entry. Unsupported hardware or
shapes must fail closed or use an explicitly reported reference path; silently
materializing and caching a dense matrix is not an optimized implementation.

Acceptance per bit width: byte-exact decode against native-v2, bounded layer and
logit error against the Python loader, no persistent dense cache, reported peak
memory, and separate prefill/decode throughput over short and long contexts.

### 2.3 vLLM out-of-tree plugin

- implement the RotQuant quantization configuration and linear method;
- load sharded packed tensors directly into tensor-parallel workers;
- route GEMV/GEMM through the Stage 2.2 operator boundary;
- support continuous batching, CUDA graphs, prefix caching, and tensor
  parallelism without format conversion;
- add source-fp16, high-precision quantized, GPTQ/AWQ, and RotQuant comparisons
  using the same prompts, batch sizes, context lengths, and memory accounting.

Acceptance: the vLLM server passes fixed layer/logit/token conformance and shows
packed resident memory with reproducible latency, throughput, and quality
results. Loading alone does not count as backend support.

### 2.4 Architecture coverage tiers

Runtime support is claimed by tier rather than by an open-ended model list:

- Tier A: dense decoder-only families;
- Tier B: MoE and hybrid recurrent/attention decoders;
- Tier C: encoder-decoder and encoder-only families;
- Tier D: multimodal models, including vision/projector modules and processors.

An adapter may advertise discovery before its runtime tier is complete, but the
support report must distinguish discovery, checkpoint round trip, quality
validation, and fused-kernel availability. Each tier needs at least two model
families where practical so implementation is not accidentally checkpoint
specific.

### 2.5 Packed KV and selective retrieval

After the weight-only server is stable, fuse K/V quantization before persistent
cache storage and consume packed cache blocks inside attention. The selective
retrieval oracle then becomes a runtime experiment: scan compressed keys, keep
recent-token and attention-sink reservations, gather only candidate value rows,
and use a confidence gate to fall back to dense attention when recall is unsafe.

Acceptance: no full-cache dequantization buffer, exact accounting of K scan and
V row reads, long-context quality/recall gates, and a demonstrated
context-length crossover rather than a short-context speed claim.

Reference semantics completed on 2026-09-01: true fp16 sink/recent storage,
8-bit double-quantized scale metadata, finite 2-bit/dimension E8P cache codes,
and paired bootstrap intervals. The remaining work in this stage is
kernelization and long-context promotion, not deciding what the format means.

## Stage 2 definition of done

Stage 2 is complete when:

1. the reliability gate is enforced in CI;
2. the canonical sharded artifact has a stable compatibility contract;
3. vLLM serves at least Tier A through specialized W1--W8 dispatch, with every
   claimed profile passing correctness, memory, and performance gates;
4. remaining architecture and KV gaps are explicit capability results rather
   than inferred from adapter discovery;
5. benchmark manifests pin model revisions, code revisions, dependencies,
   devices, workload shapes, and dense-fallback policy.

## Later stages

- Stage 3 ports the proven operator and artifact contracts to SGLang and
  upstreamable llama.cpp/Metal integrations.
- Stage 4 promotes packed KV kernels and selective retrieval only where the
  quality and context-crossover gates pass.
- Stage 5 focuses on autotuning, additional accelerators, stable public APIs,
  documentation, and release qualification comparable to mature optimization
  libraries. (Licensing, packaging metadata, and the first-release checklist
  moved forward into Stage 2.0c: they gate adoption, not polish.)
