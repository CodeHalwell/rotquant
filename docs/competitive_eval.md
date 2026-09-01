# RotQuant competitive evaluation contract

RotQuant is competitive only when a released artifact improves the
quality/size frontier and remains practical to create and serve. A Python
fallback experiment is evidence about an algorithm, not a competitive product
claim.

The immediate external target is Unsloth Dynamic 3.0: model-specific PTQ,
calibration-aware mixed quantization, GGUF artifacts from roughly 1--8 bits,
held-out KL/top-1 and 300-prompt greedy-trajectory evaluation, and deployment
through llama.cpp. Their public releases and importance matrices are valid
artifact baselines even where their allocator is not publicly reproducible.

### What the Unsloth comparison is testing

The publicly visible core of Dynamic GGUF is a static, model-specific
mixed-format recipe: calibration and importance information determine which
tensors receive different GGUF quantization formats, higher precision, or an
exception. It is not a precision choice made dynamically for each request, and
the GGUF releases execute through existing llama.cpp operators rather than
requiring a new Dynamic-specific arithmetic primitive. The exact Dynamic 3.0
selection objective and all of its recipe details are not public, so the
comparison must use released artifacts rather than treating it as a known
simple per-layer bit-width heuristic.

Mixed precision is therefore required for a fair attempt at the same-size
frontier, but it is not RotQuant's complete contribution. The intended
composition is:

$$
\text{structured rotations} + \text{GPTQ error compensation}
+ \text{codebook/scale choice} + \text{mixed-format allocation}.
$$

Every competitive rate must preserve the following ablation ladder at matched
deployed bytes:

1. standard uniform quantization;
2. standard mixed-format quantization without RotQuant;
3. uniform RotQuant with the frozen error-compensation recipe;
4. mixed-format RotQuant with the same frozen quantizer; and
5. the nearest-size released Unsloth artifact.

This separates gains from allocation from gains within each allocated format.
Uniform Gaussian W4+GPTQ remains the locked W4 control until a mixed recipe
beats it on calibration-disjoint KL and trajectories at the same actual bytes.
Mixed allocation is evidence-gated at W4, expected to become increasingly
important at W3, and is a required design component for credible W2/W1
artifacts. Weight allocation, activation precision, and KV-cache precision are
three separate experiments and must not be conflated in one arm.

## Four independent gates

| Gate | Required evidence | A failure means |
|---|---|---|
| Optimizer | A frozen recipe beats uniform RotQuant and random/matched-format allocation on held-out data. | The algorithm is not selected. |
| Artifact | Exact deployed bytes, independently loadable shards, checksums, and no persistent dense fallback. | No storage or portability claim. |
| Quality | Same source revision, tokenizer/chat template, prompt manifest, decoding policy, auxiliary-head policy, and <=1% byte mismatch against external artifacts. | No provider comparison. |
| Runtime | Packed resident memory plus prefill/decode latency, throughput, and peak memory on named hardware. | Quality-only result; no speed claim. |

Passing one gate never implies another. Results must label the implementation
as Python fallback, RotQuant native, llama.cpp, vLLM, or another engine.

## Registered quality protocol

Before looking at final scores, freeze a content-addressed protocol using
`rotquant.eval.competition.CompetitiveEvalProtocol`:

- exact model and tokenizer revisions;
- the final chat-template hash and thinking/tool-use settings;
- a replayable calibration manifest and a disjoint held-out prompt manifest,
  plus public metadata-only summaries where source terms prevent token
  redistribution (see the [competitive data pipeline](competitive_data.md));
- whether MTP, vision projectors, embeddings, and output heads are included in
  artifact bytes;
- greedy decoding for exactly 32 generated tokens;
- 300 held-out prompts balanced across agentic/tool use, code, maths,
  multilingual text, and long-document tasks;
- exact prompt token IDs, not only rendered text, so engine/template drift is
  detectable;
- one SHA-256 identity per calibration and held-out token sequence, with an
  empty intersection verified by the protocol constructor. Unequal aggregate
  manifest hashes alone are not accepted as evidence of disjointness.

The 300-prompt suite reports:

1. teacher-forced KL divergence (mean, median, p95, maximum, and paired
   bootstrap interval);
2. teacher-forced top-1 token agreement;
3. free-running 32-token agreement, exact-trajectory rate, and mean matching
   prefix/first-divergence position;
4. domain-level results and worst-domain regression;
5. task outcomes for code execution, tool-call syntax/arguments, maths, and
   instruction following;
6. failures such as empty output, invalid tool calls, language switching, and
   repeated-token loops.

Perplexity on pinned WikiText-2 and C4 remains a regression sentinel, not the
headline competitive metric. The four-prompt C4 trajectory check in the
Algorithm Lab is also only an early drift gate.

## Exact-size comparison matrix

For each target rate, compare the full source model, uniform RotQuant,
calibrated RotQuant, the frozen dynamic RotQuant recipe, a standard llama.cpp
GGUF, and the nearest-size Unsloth release. Use actual artifact bytes and pair
artifacts only within 1%; interpolate a frontier rather than comparing visibly
different sizes.

The first release matrix should cover approximate effective rates of 1.5, 2,
2.5, 3, 4, 5, 6, and 8 bits per quantized weight. Integer RotQuant kernels may
serve mixed layer allocations at fractional model-wide rates. All metadata,
codebooks, scales, rotations, retained high-precision tensors, and auxiliary
modules count toward deployed bytes.

## Calibration programme

Build a reproducible, licensed, content-addressed calibration mixture rather
than optimizing only on Wikipedia-like text. It should contain tokenizer-correct
chat transcripts and balanced samples of:

- agentic/tool calling and structured outputs;
- executable code and repository-scale code context;
- mathematical reasoning;
- multilingual and non-Latin text;
- long documents and retrieval-style context;
- ordinary conversation and factual prose.

Publish the row IDs, transformations, token hashes, domain weights, filtering
rules, and generated importance/Hessian artifacts. Deduplicate against every
held-out prompt and benchmark source. Run a calibration-size ablation so gains
can be attributed separately to the dataset and allocator.

## Model coverage

Freeze allocator hyperparameters on development models, then confirm without
retuning across the architecture tiers in the roadmap:

- at least two dense decoder families;
- a hybrid recurrent/attention decoder;
- an MoE family;
- an encoder or encoder-decoder family for library generality;
- a multimodal family, with projector/vision-byte policy reported explicitly.

The flagship provider comparison should use a currently contested model and
the same public source revision as the external artifacts. Smaller models are
for iteration speed and CI, not the final claim.

### Qwen development ladder

Qwen3.5-4B is the primary development model. It is small enough for frequent
full sweeps while exercising the hybrid decoder, multimodal wrapper, chat
template, and architecture-specific exclusions needed by the flagship target.
Qwen2.5-3B remains a cheap transfer canary: a candidate that is catastrophic on
that family does not advance merely because it wins on Qwen3.5-4B.

Development on the 4B model may change the calibration mixture, importance
objective, bit-allocation algorithm, codebooks, kernels, artifact schema, and
evaluation implementation. Before the first Qwen3.8-27B quality result is
examined, freeze and version:

- calibration and held-out dataset manifests;
- tokenizer/chat-template processing;
- allocation objective, candidate bit widths, rate tolerance, and all search
  hyperparameters;
- preset definitions and promotion thresholds;
- artifact byte policy, including vision/projector and MTP tensors;
- benchmark metrics, bootstrap procedure, and failure taxonomy.

The frozen optimizer may still collect model-specific importance statistics and
produce a different layer map on Qwen3.8-27B. Changing the objective or
thresholds in response to its final scores is tuning on the test model and
requires a new protocol version plus a fresh final evaluation.

The 4B stage is complete only when:

1. the quality, fast, and compact candidates pass the registered 300-prompt
   suite and the Qwen2.5 transfer canary;
2. target rates across the planned 1--8-bit frontier export and reload without
   persistent dense weights;
3. quantization is resumable and layer-streamed under an artificial memory cap,
   with GPU state bounded to the active block/calibration workspace rather than
   a second full model;
4. the same packed artifacts execute through the intended production operator
   boundary, with correctness, memory, prefill, and decode measurements;
5. a measured per-layer cost model provides a credible Qwen3.8-27B time, VRAM,
   host-RAM, and storage forecast.

Qwen3.8-27B is then the locked flagship confirmation model. Start with the
nearest-size W2/W3/W4 or mixed-rate anchors against standard GGUF and Unsloth,
including source-normalized quality. Fill the complete 1.5/2/2.5/3/4/5/6/8-bit
frontier only after the pipeline and comparison protocol pass those anchors.
Implementation bug fixes are allowed, but invalidate affected source and
candidate results equally and must trigger a complete rerun under a new code
revision.

## Runtime protocol

Benchmark warm and cold load, prompt processing, decode, time to first token,
inter-token latency, throughput under concurrency, resident/peak memory, and
energy where available. Report short and long contexts and separate CPU,
CUDA, and Metal results. Compare through the same serving interface where
possible and normalize quality to each engine's own full-precision source to
make engine numerical drift visible.

## Claim language

A competitive statement must name the model revision, artifact pair, exact
bytes, prompt-manifest fingerprint, metrics, confidence interval, engine,
hardware, and code revision. “Best”, “same size”, or percentage-improvement
language is blocked until the corresponding raw records and protocol manifest
are public.
