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
  trips, plus empty-selection failures, in CI;
- [ ] add the small multimodal round-trip fixture without relying on a remote
  checkpoint or GPU.

Acceptance: a fresh-process checkpoint round trip preserves packed bytes and
reference logits for every supported loader class, and every successful
optimization report records at least one patched module.

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
  packaging, documentation, and release qualification comparable to mature
  optimization libraries.
