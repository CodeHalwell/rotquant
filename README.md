# rotquant-eval

A GPU-oriented assessment harness for **TurboQuant-style rotation + weight
compression**. The implemented experiment cells test the following hypotheses on
real models with fixed metrics:

1. **Rotation transfers** to weight-only quant; FWHT ≈ dense random-orthogonal for
   weight-only, while learned rotations only pull ahead once activations are also
   quantised (W4A4). *(E1)*
   **E1b** trains an exactly-orthogonal butterfly initialized from FWHT against
   bounded source-model activation samples, then freezes and packs it.
   **E1c** jointly trains every butterfly inside a transformer block against the
   complete FP block output and accepts it on a disjoint calibration sequence.
2. **QJL must go** — a deterministic residual pass beats the stochastic 1-bit QJL
   residual at equal bits. *(E3)*
3. **Gaussian MSE-optimal grid > uniform** at the same bit budget. *(E2)*
4. **Data-free scale-search** is a free win over RMS scales at identical bits. *(E4)*
5. **GPTQ helps — with real activations.** *(E5)*
6. **Scalar has a hard ceiling** at low bit rates. A finite-rate dimension-2
   vector arm now provides an exact-rate W1--W3 research comparison; model-level
   results are still required before drawing the vector conclusion. *(E6)*
7. **The consistency trap** — rotating weights without the matching activation
   basis change causes cross-layer drift. *(E7)*
8. **Footprint & speed** — packed `QuantLinear` vs fp16 fallback. *(E8)*

## Layout

```
rotquant/       core library  (api, adapters, format, quantize, pack, linear, rotate, patch, calibration)
rotquant/eval/  fixed eval protocol (perplexity, zeroshot, layer_mse)
baselines/      working GPTQ/AWQ/AQLM wrappers through the same evaluation harness
tests/          correctness tests that must pass before trusting any experiment
scripts/        run_experiment.py (config -> quantise -> eval -> JSON), aggregate.py
configs/        one YAML per experiment cell (E1..E8)
results/        JSON per run + generated tables/figures
```

The chronological [experiment log](docs/experiment_log.md) records successful
and negative results, methodological caveats, and the decision taken after each
trial. Update it alongside raw JSON results so development history remains
paper-ready and reproducible.

The [science and mathematics guide](docs/how_rotquant_works.md) derives the
rotation invariant, scalar and vector codebooks, exact rate accounting, GPTQ,
learned/block objectives, mixed-precision allocation, KV compression, and
selective retrieval. It also separates mathematical identities and prior-paper
results from RotQuant's current experimental evidence and open hypotheses.

The [serving-backend matrix](docs/serving_backends.md) tracks which runtimes
preserve the packed RotQuant representation, the extension point for each
backend, and the conformance gates required before claiming support.

The [implementation roadmap](docs/roadmap.md) defines the next canonical GPU
serving stage: checkpoint reliability gates, a sharded Transformers artifact,
specialized W1--W8 kernels, vLLM integration, architecture tiers, and packed-KV
acceptance criteria.

The [competitive evaluation contract](docs/competitive_eval.md) defines what it
takes to compare RotQuant with Dynamic GGUF providers: exact deployed-size
matching, disjoint calibration and 300-prompt manifests, KL distribution tails,
32-token greedy divergence, task failures, and runtime evidence. A fallback PPL
win alone is explicitly not treated as a competitive product claim.

The pinned [Unsloth Qwen3.5-4B comparison note](docs/unsloth_qwen35_4b_comparison.md)
records the exact released artifacts, hashes, byte mismatch, same-engine KL
method, and the boundary between the next development result and a Dynamic 3.0
competitive claim.

The [competitive data pipeline](docs/competitive_data.md) turns that contract
into immutable calibration/held-out manifests, leakage checks, engine-neutral
observations, and paired domain/bootstrap reports without redistributing gated
source tokens.

The [packed checkpoint v2 specification](docs/packed_format_v2.md) retains the
v1 word layout while adding finite vector codebooks, true uint8 scale metadata,
shared rotations, and activation-quantization metadata with fail-closed
compatibility rules.

The [native runtime v2 contract](docs/native_runtime_v2.md) generalises native
weight blocks and fail-closed kernel dispatch across 1–8 bits while preserving
the deployed 4-bit GGUF v1 bytes exactly.

The [algorithm-lab Colab](notebooks/rotquant_algorithm_lab_colab.ipynb) runs the
completed research funnel: exact-rate scalar/vector controls, calibrated and
TurboQuant-style variants, mixed-bit allocation, cross-family replication, and
a real-attention selective-V upper-bound oracle. The focused
[Qwen3.5-4B optimization Colab](notebooks/qwen35_4b_optimization_stage_colab.ipynb)
now runs the promoted W4 controls against streamed GPTQ, with W4A8/E8,
million-token recovery, and 8k-context KV stages as explicit resumable opt-ins.

## Install

```bash
# Core + dev deps (CPU, no GPU required):
uv sync --extra dev

# Add the eval stack (transformers, datasets, lm-eval, …) — installs on CPU too:
uv sync --extra eval

# On the GPU box only: the CUDA FWHT kernel (source build, needs nvcc; the
# pure-torch fallback is used automatically when it's absent):
uv pip install fast-hadamard-transform --no-build-isolation

# Add implemented baseline packages (gptqmodel, autoawq, aqlm):
uv sync --extra baselines
```

Run commands inside the managed venv with `uv run <cmd>`, or activate it first:

```bash
source .venv/bin/activate
```

> **GPU / CUDA PyTorch:** `uv sync` installs the default (CPU) torch wheel. For a
> CUDA-enabled build, follow the [PyTorch install selector](https://pytorch.org/get-started/locally/)
> and either use `uv pip install` with the appropriate `--extra-index-url`, or add a
> `[tool.uv.sources]` override in `pyproject.toml` pointing at the CUDA wheel index.

The **core foundation + correctness tests run on CPU with just `torch`, `numpy`,
`scipy`** — no GPU, model download, or CUDA kernel needed.

## Library API

The production-facing API validates kernel-targeted profiles from 1 through 8
bits, resolves an architecture adapter, and optimises the supplied model in
place so a large model is never duplicated implicitly:

```python
from rotquant import RotQuantConfig, inspect_model, optimize_model

support = inspect_model(model)
print(support.to_dict())

model = optimize_model(
    model,
    RotQuantConfig(bits=4, group_size=128, rotation="fwht"),
)
```

Finite-dimensional TurboQuant codebooks and its inexpensive self-dot correction
are opt-in so they can be evaluated against the Gaussian/MSE default at exactly
the same packed rate:

```python
model = optimize_model(
    model,
    RotQuantConfig(
        bits=3,
        group_size=128,
        rotation_block=128,
        codebook="spherical",
        codebook_dim=128,
        bias_correction="length",
    ),
)
```

`spherical` uses the exact unit-RMS coordinate marginal for the requested
dimension instead of its asymptotic Gaussian approximation. `length` folds a
rowwise self-dot correction into the existing fp16 scales, so codes, artifact
shape and bits/weight do not change. It deliberately remains optional: it fixes
inner-product shrinkage but can increase ordinary reconstruction MSE. Use
`rotquant.eval.quantization.compare_quantizers` and the KV attention-logit metrics to
choose it from held-out results rather than enabling it globally.

For calibrated experiments, `fit_scalar_codebook(normalized_samples, levels)`
returns a normal packed `ScalarCodebook`; pass it to
`Quantizer(config, codebook=...)`. No runtime or checkpoint format extension is
needed because centroid values already travel with native-v2 artifacts.

`QuantConfig(codebook="vector", vector_dim=2)` is the finite-rate vector
research arm. Its packed index rate exactly matches the requested bits per
weight, but it intentionally fails closed for stable checkpoints and native
kernels until a versioned vector-codebook contract is implemented.

Use `save_pretrained(model, output_dir, ...)` and
`from_pretrained(output_dir, ...)` for the pickle-free packed checkpoint. New
artifacts embed the exact int32/LSB-first packing contract; legacy v1 artifacts
without that explicit metadata remain loadable. Architecture discovery is
extensible through `register_model_adapter`, while unfamiliar models safely
fall back to ordinary `nn.Linear` discovery rather than being labelled as a
validated architecture.

The portable native reference can be benchmarked independently of a serving
backend:

```bash
uv run python scripts/benchmark_native_reference.py --bits 1 2 3 4 5 6 7 8
```

A dependency-free C++17 implementation now provides compiled native-v2
dequantization and streaming matmul for all 1–8-bit profiles. It includes the
portable scalar correctness floor plus capability-gated ARM NEON and x86 AVX2
paths, with no unreported SIMD or dense fallback:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
build/native/rotquant-native-cli --capabilities
build/native/rotquant-native-bench --iterations 10
```

See [`native/README.md`](native/README.md) for embedding and installation.
The runtime can be built as either a static C++ library or a shared library with
a versioned C ABI for llama.cpp-style integrations and foreign-function
bindings. `NativeRuntimeLibrary(path)` loads that ABI explicitly and can
register its resolved scalar, NEON, or AVX2 implementation as a fail-closed
`KernelRegistry` backend.

## Selective KV retrieval

`retrieval_rotquant_decode` is a quality oracle for a decode-only cache path
that scans compressed keys but gathers only a selected set of value vectors.
Its candidate budget can reserve recent positions and attention sinks, while
`kv_retrieval_metrics` reports full-precision attention-mass coverage, output
error and the fraction of V rows a fused runtime would read. The Python oracle
materialises keys for correctness; it is not a throughput claim. See
[`docs/kv_retrieval.md`](docs/kv_retrieval.md) for the intended packed runtime.

## Correctness first

These are cheap and catch the bugs that silently invalidate results. Don't trust
any experiment until they pass:

```bash
pytest tests/ -q
```

CI runs the full suite (including the cross-language native conformance
tests) on Python 3.10/3.11/3.12 plus ruff lint on every push and pull
request; the native workflow additionally builds with ASan/UBSan and checks
that the pinned llama.cpp patch still applies.

* `test_rotation_invariance` — rotating the activation then matmul equals
  dequant-then-matmul (~1e-3), and every rotation is orthogonal.
* `test_gptq_identity` — GPTQ with `H = I` reduces **exactly** to plain rounding.
* `test_source_coding` — scalar Lloyd-Max on a unit Gaussian gives ≈0.1175 MSE at
  2-bit and ≈0.0345 at 3-bit; the Shannon bound `2^(-2R)` comes out at
  0.0625 / 0.0156; the bits/weight accounting assertion holds.

## Running an experiment

For the complete staged algorithm trial on Google Colab, open
[`notebooks/rotquant_algorithm_lab_colab.ipynb`](notebooks/rotquant_algorithm_lab_colab.ipynb).
The notebook performs a cheap synthetic rate/correctness preflight first, then
requires `CONFIRM_EXPENSIVE_RUN = True` before model downloads or GPU trials.
Results are written as content-addressed records in Google Drive, so interrupted
runs resume without silently mixing configurations.

```bash
# Plumbing smoke test first (tiny random model, CPU, <1 min):
python scripts/run_experiment.py configs/smoke_cpu.yaml

python scripts/run_experiment.py configs/e5_gptq.yaml --output-dir results
python scripts/aggregate.py --results-dir results --out results/summary.md
```

Each experiment YAML is deep-merged on top of `configs/_base.yaml` (experiment
keys win), so per-experiment files only state what they change. `--model`,
`--seed` and `--device` override the merged config from the CLI — seed and model
sweeps never require editing YAML:

```bash
for s in 0 1 2; do
  python scripts/run_experiment.py configs/e1_rotation.yaml --seed $s
done
python scripts/run_experiment.py configs/e1_rotation.yaml --model meta-llama/Llama-2-13b-hf

# Any dotted config key is sweepable with --set (YAML-typed values), so the
# sweeps described in the config comments are one-liners:
for r in none dense fwht; do
  python scripts/run_experiment.py configs/e1_rotation.yaml --set patch.rotation=$r
done
# A true source-model reference: unlike rotation=none, this skips quantisation.
python scripts/run_experiment.py configs/e1_rotation.yaml --set patch.enabled=false
# The E1 learned arm needs its per-layer theta training enabled (data-free
# alternating minimisation of rotated-domain quant MSE via the Cayley map;
# without it, theta stays at ~identity and the arm is a no-rotation control):
python scripts/run_experiment.py configs/e1_rotation.yaml \
  --set patch.rotation=learned --set 'patch.train_rotation={steps: 200, lr: 0.001}'
# Practical activation-aware structured training (starts exactly from FWHT):
python scripts/run_experiment.py configs/e1b_butterfly.yaml
# Stronger joint transformer-block reconstruction. Training, validation-based
# checkpoint choice, and the exact packed-vs-FWHT gate use disjoint calls:
python scripts/run_experiment.py configs/e1c_block_butterfly.yaml
# Joint block rotation plus learned deployable group scales/clipping:
python scripts/run_experiment.py configs/e1d_block_scale.yaml
# Propagation-aware joint rotation/scale training:
python scripts/run_experiment.py configs/e1e_propagated_block_scale.yaml
# End-to-end teacher-logit/LM-loss tuning of the packed model:
python scripts/run_experiment.py configs/e1f_end_to_end_distill.yaml
# Quantization-aware rank-4 LoRA recovery with exact adapter accounting:
python scripts/run_experiment.py configs/e1g_lora_qat.yaml
python scripts/run_experiment.py configs/e4_scale_group.yaml --set quant.group_size=64
python scripts/run_experiment.py configs/e8_footprint.yaml --set patch.fallback=true
```

### Qwen3.5-4B on Apple Silicon

`unsloth/Qwen3.5-4B` is a unified image/text checkpoint rather than a plain
`AutoModelForCausalLM`. The dedicated config selects Transformers'
multimodal loader but deliberately evaluates and quantises the language path
only. It leaves the vision tower and the small linear-attention state gates at
source precision:

```bash
# Matched source-model reference (short text-only sanity check):
uv run python scripts/run_experiment.py configs/qwen35_4b_mps.yaml \
  --set patch.enabled=false

# Matched FWHT 3-bit language-backbone trial:
uv run python scripts/run_experiment.py configs/qwen35_4b_mps.yaml
```

MPS automatically uses the fp16 fallback described below, so these are quality
runs rather than compressed-memory or throughput measurements. Vision serving
will additionally require the checkpoint's `AutoProcessor`; WikiText/C4 text
perplexity uses its tokenizer directly.

> **Cache-result validity notice (2026-09-01).** Every Qwen3.5-4B cache KL
> recorded before 2026-09-01 (uniform-versus-mixed tables, the three-seed
> validation, the 1,024-token confirmation, frozen-map transfer, and the
> matched K4/V4 controls) was measured with a simulator that shared
> linear-attention state between its two decode passes under the Transformers
> release those notebooks installed. The results were independent of the K/V
> bit width and have been withdrawn; see
> [`docs/scientific_validity_review_2026-09-01.md`](docs/scientific_validity_review_2026-09-01.md).
> The simulator now clones all cache state, fails closed on shared storage,
> and rejects any run whose uniform 8-bit cache does not reproduce the fp16
> cache (`eval.kv_cache.endpoint_check_bits`). Notebooks pin
> `transformers==5.9.0`. The notebooks below still run, but their earlier
> conclusions must be re-established.

Rotation-aware cache experiments use true post-RoPE K/V states. The uniform
control and held-out dynamic allocator are separate configs:

```bash
# Uniform 4-bit K/V quality and exact logical bytes.
uv run python scripts/run_experiment.py configs/qwen35_4b_kv_mps.yaml

# Same 4.25-bpv target, but allow 3/4/8-bit K and V independently per layer.
uv run python scripts/run_experiment.py configs/qwen35_4b_dynamic_kv_mps.yaml
```

The dynamic selector uses C4 sequences disjoint from final evaluation, scores
one K/V state at a time by global teacher KL per exact byte saved, measures the
joint recipe, and restores the best same-budget uniform recipe if interactions
make it worse. For native Metal cache throughput across context depths, run
`scripts/benchmark_rotquant_kv.sh` after building the pinned llama.cpp fork.
The joint release GGUF embeds the frozen 3.25-bpv map selected by the earlier
cache study; that selection is withdrawn pending re-measurement. The patched
runtime stores its eight full-attention K/V layers in true Gaussian 2/3/4-bit
rows with fp16 group scales and rejects non-flash-attention execution rather
than silently replacing the recipe.

For the full CUDA study, open
[`notebooks/qwen35_4b_kv_cache_matrix_colab.ipynb`](notebooks/qwen35_4b_kv_cache_matrix_colab.ipynb).
It loads the 4-bit weight model only once per seed, runs all 16 K/V precision
pairs plus codebook/group/rotation/dynamic-budget ablations, persists every
trial to Drive, validates Pareto candidates across three seeds, and confirms the
3-bit- and 4-bit-budget winners at 1,024-token context.

After that matrix completes, open
[`notebooks/qwen35_4b_kv_frozen_transfer_colab.ipynb`](notebooks/qwen35_4b_kv_frozen_transfer_colab.ipynb)
to test deployment-map transfer without rerunning candidate scoring. It replays
the saved short- and long-context recipes, evaluates each fixed map on both
held-out contexts, constructs a mixed-context map from selection metrics only,
and recommends one universal map only when it beats uniform K3/V3 at no more
exact bytes in both contexts. Frozen recipes require an exact layer match, so a
missing or stale layer cannot silently fall back to uniform precision.

For whole-system co-design, open
[`notebooks/qwen35_4b_joint_rotquant_kv_colab.ipynb`](notebooks/qwen35_4b_joint_rotquant_kv_colab.ipynb).
It screens uniform and mixed 2/3/4/8-bit weight recipes, crosses the viable
weights with asymmetric and dynamic K/V budgets, and imports the exact saved
3.25-bpv mixed-context frozen recipe selected by the transfer study. It
optionally applies block-scale and LoRA-QAT recovery, then validates the joint
model-plus-cache Pareto winners over three seeds and a 1,024-token prefill.
Direct K/V reconstruction NMSE is reported alongside end-to-end KL to
distinguish error cancellation from a quantizer-quality problem.

After the joint matrix selects a diagnostic winner, open
[`notebooks/qwen35_4b_joint_release_followup_colab.ipynb`](notebooks/qwen35_4b_joint_release_followup_colab.ipynb).
It reuses the completed three-seed candidate rows, fills only the missing
same-seed K4/V4 controls, and compares cache KL within weight recipe and seed.
It then tests block-and-scale recovery on the actual uniform-W4/frozen-map
winner and spends four additional confirmation runs only when seed-0 recovery
passes its predeclared PPL, cache, size, and no-adapter gates.

Once those matched gates pass, open
[`notebooks/qwen35_4b_joint_winner_export_colab.ipynb`](notebooks/qwen35_4b_joint_winner_export_colab.ipynb).
It reconstructs the released uniform-W4/FWHT seed-0 weights, pins the current
Hub revision, embeds the frozen mixed 3.25-bpv Gaussian K/V map (selection
withdrawn; see the validity review) in
the packed manifest, confirms the released seed-0 perplexity, audits actual
safetensors bytes and forbidden fallback keys, writes SHA-256 checksums, and
reloads the checkpoint in a fresh process before it can be published.

For CUDA LoRA-QAT quality recovery, open
[`notebooks/qwen35_4b_lora_qat_colab.ipynb`](notebooks/qwen35_4b_lora_qat_colab.ipynb)
in Colab. It checks the GPU, establishes a matched CUDA source baseline, runs a
bounded smoke trial, then exposes the full 4-bit rank-4 recovery experiment from
`configs/qwen35_4b_lora_qat_cuda.yaml`. The CUDA config deliberately enables the
cached fp16 fallback to accelerate training on high-memory GPUs; its peak VRAM
is not the packed deployment footprint.

After the first rank-4 result, use
[`notebooks/qwen35_4b_lora_trial_matrix_colab.ipynb`](notebooks/qwen35_4b_lora_trial_matrix_colab.ipynb)
for the release decision. It runs matched source/FWHT/block-only controls,
increases disjoint LoRA calibration data before rank, conditionally promotes to
rank 8, validates the selected 4-bit recipe across three seeds, and writes a
persistent quality/size report. It also includes one seed-0 3-bit LoRA-QAT probe
on the same medium-data recipe. Expensive fallback arms are skipped when an
earlier recipe already passes its held-out and deployed-PPL gates.

### Packed checkpoint export and Transformers loading

`QuantLinear` codes are not part of a normal PyTorch `state_dict`. Export the
selected model explicitly during its final reconstruction run:

```bash
python scripts/run_experiment.py configs/qwen35_4b_lora_qat_cuda.yaml \
  --device cuda --seed 0 \
  --set patch.train_rotation.distill_steps=0 \
  --set eval.perplexity=false \
  --set eval.zeroshot=false \
  --export-dir /path/to/qwen35-4b-rotquant \
  --export-processor
```

Use `--export-deployment-metadata deployment.json` to embed a plain JSON object
in `rotquant_config.json`. This is used by the joint-winner export notebook to
keep its K/V cache map and release provenance with the weight
artifact. Metadata is declarative: consumers must still apply the K/V recipe
in their cache runtime.

The directory is self-contained and pickle-free. It stores ordinary
Transformers state, packed codes/scales/codebooks, rotation parameters, model
configuration, tokenizer, and (when requested) multimodal processor metadata.
The quality-only fp16 fallback cache is never serialized. Reload it as a normal
Transformers model object:

```python
import torch
from transformers import AutoProcessor, AutoTokenizer
from rotquant.checkpoint import load_packed_model

checkpoint = "/path/to/qwen35-4b-rotquant"
model = load_packed_model(checkpoint, device="cuda", dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
```

For a text-generation smoke test, run
`python scripts/generate_packed.py /path/to/qwen35-4b-rotquant --device cuda`.
The returned model supports normal Transformers `forward` and `generate` calls.
RotQuant still lacks a fused packed matmul, so the default compressed path
transiently dequantizes each layer and prioritizes storage over throughput.
Passing `fallback=True` caches fp16 weights and is faster, but forfeits the
runtime-memory reduction. vLLM, SGLang, and Unsloth do not understand this
custom artifact without a backend plugin.

For llama.cpp, the repository now includes an experimental native GGUF v1
exporter and pinned runtime patch. It preserves the packed 4-bit Gaussian
codes, fp16 group scales, and learned butterfly rotations exactly—there is no
dense reconstruction or second quantization pass. See
[`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md) for the
build, export, conformance-check, and OpenAI-compatible serving commands. The
patched runtime now includes both a portable CPU reference and native Apple
Metal kernels for the structured rotation and packed Gaussian-codebook
matvec. The Metal path keeps RotQuant tensors on the GPU and never
materializes dense projection weights.

To reconstruct only the already-selected Qwen artifact in Colab without
repeating the trial matrix, open
[`notebooks/qwen35_4b_export_colab.ipynb`](notebooks/qwen35_4b_export_colab.ipynb).
That notebook preserves the earlier block-only export workflow. For the newer
whole-system release winner, use
[`notebooks/qwen35_4b_joint_winner_export_colab.ipynb`](notebooks/qwen35_4b_joint_winner_export_colab.ipynb).

Each run writes `results/<run_id>.json` with the config, git SHA, library
versions, GPU, all metrics (including true bits/weight and packed-vs-fp16
footprint for every run; `eval: {throughput: ...}` adds greedy-decode tokens/s
and peak generation VRAM for E8), and wall-clock
(`rotquant.utils.environment_record`). Derived run ids get a `_s<seed>` suffix
and CLI-overridden runs get the overridden values appended, so seed, device,
`--model`, and `--set` sweeps do not overwrite each other; an explicit `run_id:` in the
YAML is used verbatim when no CLI override modifies the run. Overlong IDs are
deterministically shortened with a digest while retaining the seed suffix.
`aggregate.py`
emits both the markdown table and a tidy CSV next to it.

Quantisation targets every `nn.Linear` **except** `lm_head`/`embed_out` (the
convention all baselines follow); override with `patch: {exclude: []}`. GPT-2
style models (transformers `Conv1D`) are not supported and are flagged loudly.

> **GPTQ memory note:** calibration accumulates one fp32 `[in, in]` Hessian per
> quantised linear *on the GPU* (~25 GB extra for Llama-2-7B, all layers).
> Finalised Hessians are offloaded to CPU before patching, but plan VRAM for the
> accumulation phase or restrict `patch: {include: [...]}`.

> **Learned-rotation cost note:** each training step solves the O(d³) Cayley map
> per layer, so `train_rotation: {steps: 200}` on a 7B model is roughly an hour
> of GPU time on top of patching. Per-run training aggregates land under
> `metrics.rotation_train` (mean rotated-domain quant-MSE before/after).

> **Structured-training note:** `rotation: butterfly` replaces the dense Cayley
> map with `d/2 * log2(block)` trainable angles. It begins exactly at the seeded
> block-FWHT, remains orthogonal, and applies in O(d log(block)). With
> `objective: activation`, the runner captures at most `max_tokens` source-model
> inputs per linear for optimization and `selection_tokens` disjoint inputs for
> exact final-quantizer checkpoint selection.
> The best calibration checkpoint is restored, so a short run cannot knowingly
> finish worse than its FWHT initialization. Packed results separately report
> rotation-parameter bytes and effective bits/weight. This gate is layer-local,
> so a run must still beat matched FWHT perplexity before being scaled up.

> **Block-training note:** `objective: block` replays captured transformer-block
> calls and jointly trains all structured rotations inside each block. Its
> `train_batches`, `validation_batches`, and `selection_batches` are disjoint:
> validation chooses and can early-stop the proxy checkpoint, while selection
> performs the one-time exact packed-candidate versus FWHT gate. Rejected blocks
> are emitted as parameter-free FWHT; accepted blocks retain only their trained
> butterfly angles. This captures attention, residual, norm, and MLP interactions
> that independent linear reconstruction misses.

> **Learned-scale note:** `learn_scales: true` initializes every group at the
> configured exact scale-search result, then jointly optimizes bounded scale
> multipliers and butterfly angles. Accepted candidates write those values into
> the same fp16 scale slots already charged by the quantizer, so learning scales
> adds no packed storage. The untouched final gate still compares against an
> independently packed FWHT + `mse_search` reference.

> **Propagation-aware note:** `propagate_quantized_inputs: true` trains block 0
> normally, exactly packs the selected candidate, and replays its real outputs
> as block 1 inputs. This repeats through the network while full-precision
> outputs remain the teacher targets. Reported input-drift metrics make the
> accumulated-error signal explicit.

> **End-to-end distillation note:** `distill_steps > 0` keeps packed 3-bit code
> indices fixed and tunes only retained butterfly angles and existing group
> scales against source-model logits plus optional next-token loss. Distillation
> has separate train, validation, and final-gate sequences. Scale changes are
> rounded and committed into the original fp16 slots, so this stage adds no
> deployment tensors or bits.

> **LoRA-QAT note:** `distill_lora_rank > 0` adds zero-output adapters in each
> packed linear's deployed rotated basis. The global held-out gate either retains
> all adapters as fp16 parameters or removes them completely. Result JSONs report
> `adapter_parameter_bytes`; effective bpw and compression include those bytes.

> **Apple MPS note:** this project has no fused packed MPS matmul. MPS runs
> automatically enable `patch.fallback=true`, perform integer packing on CPU, and
> cache dequantized fp16 weights for quality evaluation. Their reported packed-byte
> accounting remains useful, but do not use MPS runs for packed throughput or peak
> memory comparisons. Unquantized `patch.enabled=false` reference runs do not need
> or enable this fallback.

### Baselines

```bash
# gptq/awq quantise the model here (C4 calibration):
python baselines/run_baseline.py --backend gptq --model meta-llama/Llama-2-7b-hf --bits 4 --zeroshot
# aqlm (and pre-quantised gptq/awq checkpoints) load as-is:
python baselines/run_baseline.py --backend aqlm --model ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf --bits 2 --prequantized
```

These implemented baselines go through the **identical** perplexity/zero-shot
harness. QuIP#/QTIP/HIGGS are not exposed as runnable choices until their
checkpoint loaders and rate accounting are integrated.

## Methodology rigour

* **Seeds & repeats.** Run E1/E5/E6 with ≥3 seeds; report mean ± std (random
  rotations alone can swing zero-shot by double digits).
* **Equal-bits discipline.** Compare at matched *true* bits/weight. Reported bpw
  comes from retained code, scale, residual, norm, and sketch buffers, including
  partial groups and int32 word padding. A config can additionally set
  `claimed_bpw` to make `BitBudget.assert_matches` enforce an expected rate.
* **One variable at a time.** Each matrix row changes a single factor vs a fixed base.
* **Separate quality from footprint.** Use the fp16 `fallback` path for fast quality
  sweeps on small models; report all memory/throughput numbers from the packed path.

A finding is **confirmed** when it holds across ≥3 seeds, on at least Llama-2-7B and
13B, on both WikiText-2 and C4, *and* survives the zero-shot bundle.

## Status

Fully implemented and CPU-tested: `rotate`, scalar `codebooks`, `pack`, `quantize`,
`linear`, `calibrate`, `patch`, `train_rotation` (the E1 learned arm), and the
correctness suites (`pytest tests/`).
The full `run_experiment.py` pipeline — config merge, quantise, patch, GPTQ
calibration on streamed C4, layer-MSE drift, perplexity, result JSON,
aggregation — is smoke-tested end-to-end on CPU via `configs/smoke_cpu.yaml`.
A GPU (+ HF access for gated models) is needed for real-model numbers, the
zero-shot bundle at scale, and the CUDA FWHT kernel.

E6 now includes a deterministic dimension-2 finite-rate vector control at W1,
W2, and W3. It is an algorithmic research path, not a deployable artifact:
vector checkpoints and native vector kernels deliberately fail closed.
`nearest_e8` remains a tested lattice primitive, not a finite-rate packed codec;
no E8, QuIP#, QTIP, or HIGGS result should be reported from this repository yet.

## License and citation

MIT — see [`LICENSE`](LICENSE). If you use this software in research, cite it
via [`CITATION.cff`](CITATION.cff). Contributions are welcome under the
conventions in [`CONTRIBUTING.md`](CONTRIBUTING.md); software changes are
tracked in [`CHANGELOG.md`](CHANGELOG.md).
