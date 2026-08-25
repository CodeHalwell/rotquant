# rotquant-eval

A full, GPU-grade assessment harness for **TurboQuant-style rotation + weight
compression**. The goal is to confirm or refute, on real models with real metrics,
a set of findings about rotation-based weight-only quantisation:

1. **Rotation transfers** to weight-only quant; FWHT ≈ dense random-orthogonal for
   weight-only, while learned rotations only pull ahead once activations are also
   quantised (W4A4). *(E1)*
2. **QJL must go** — a deterministic residual pass beats the stochastic 1-bit QJL
   residual at equal bits. *(E3)*
3. **Gaussian MSE-optimal grid > uniform** at the same bit budget. *(E2)*
4. **Data-free scale-search** is a free win over RMS scales at identical bits. *(E4)*
5. **GPTQ helps — with real activations.** *(E5)*
6. **Scalar has a hard ceiling** (~2× the rate-distortion bound at 3-bit); a
   vector/trellis residual is required to approach 2-bit usability. *(E6)*
7. **The consistency trap** — rotating weights without the matching activation
   basis change causes cross-layer drift. *(E7)*
8. **Footprint & speed** — packed `QuantLinear` vs fp16 fallback. *(E8)*

## Layout

```
rotquant/      core library  (rotate, codebooks, quantize, pack, linear, calibrate, patch, utils)
eval/          fixed eval protocol (perplexity, zeroshot, layer_mse)
baselines/     wrappers around GPTQ/AWQ/AQLM/QuIP#/QTIP/HIGGS through the same harness
tests/         correctness tests that must pass before trusting any experiment
scripts/       run_experiment.py (config -> quantise -> eval -> JSON), aggregate.py
configs/        one YAML per experiment cell (E1..E8)
results/       JSON per run + generated tables/figures
```

## Install

```bash
# Core + dev deps (CPU, no GPU required):
uv sync --extra dev

# Add the eval stack (transformers, datasets, lm-eval, …) — installs on CPU too:
uv sync --extra eval

# On the GPU box only: the CUDA FWHT kernel (source build, needs nvcc; the
# pure-torch fallback is used automatically when it's absent):
uv pip install fast-hadamard-transform --no-build-isolation

# Add baseline comparison packages (gptqmodel, autoawq, aqlm, flute-kernel):
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

## Correctness first

These are cheap and catch the bugs that silently invalidate results. Don't trust
any experiment until they pass:

```bash
pytest tests/ -q
```

* `test_rotation_invariance` — rotating the activation then matmul equals
  dequant-then-matmul (~1e-3), and every rotation is orthogonal.
* `test_gptq_identity` — GPTQ with `H = I` reduces **exactly** to plain rounding.
* `test_source_coding` — scalar Lloyd-Max on a unit Gaussian gives ≈0.1175 MSE at
  2-bit and ≈0.0345 at 3-bit; the Shannon bound `2^(-2R)` comes out at
  0.0625 / 0.0156; the bits/weight accounting assertion holds.

## Running an experiment

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
```

Each run writes `results/<run_id>.json` with the config, git SHA, library
versions, GPU, all metrics (including true bits/weight and packed-vs-fp16
footprint for every run), and wall-clock (`rotquant.utils.environment_record`).
Derived run ids get a `_s<seed>` suffix so seed sweeps never overwrite each
other; an explicit `run_id:` in the YAML is used verbatim.

Quantisation targets every `nn.Linear` **except** `lm_head`/`embed_out` (the
convention all baselines follow); override with `patch: {exclude: []}`. GPT-2
style models (transformers `Conv1D`) are not supported and are flagged loudly.

> **GPTQ memory note:** calibration accumulates one fp32 `[in, in]` Hessian per
> quantised linear *on the GPU* (~25 GB extra for Llama-2-7B, all layers).
> Finalised Hessians are offloaded to CPU before patching, but plan VRAM for the
> accumulation phase or restrict `patch: {include: [...]}`.

### Baselines

```bash
# gptq/awq quantise the model here (C4 calibration):
python baselines/run_baseline.py --backend gptq --model meta-llama/Llama-2-7b-hf --bits 4 --zeroshot
# aqlm (and pre-quantised gptq/awq checkpoints) load as-is:
python baselines/run_baseline.py --backend aqlm --model ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf --bits 2 --prequantized
```

Baselines go through the **identical** perplexity/zero-shot harness so a finding
is only counted once placed next to GPTQ/AWQ at 3–4 bit and QuIP#/AQLM/QTIP at 2 bit.

## Methodology rigour

* **Seeds & repeats.** Run E1/E5/E6 with ≥3 seeds; report mean ± std (random
  rotations alone can swing zero-shot by double digits).
* **Equal-bits discipline.** Compare at matched *true* bits/weight (scales +
  metadata included); `BitBudget.assert_matches` enforces this for every config.
* **One variable at a time.** Each matrix row changes a single factor vs a fixed base.
* **Separate quality from footprint.** Use the fp16 `fallback` path for fast quality
  sweeps on small models; report all memory/throughput numbers from the packed path.

A finding is **confirmed** when it holds across ≥3 seeds, on at least Llama-2-7B and
13B, on both WikiText-2 and C4, *and* survives the zero-shot bundle.

## Status

Fully implemented and CPU-tested: `rotate`, `codebooks`, `pack`, `quantize`,
`linear`, `calibrate`, `patch`, and the correctness suites (`pytest tests/`).
The full `run_experiment.py` pipeline — config merge, quantise, patch, GPTQ
calibration on streamed C4, layer-MSE drift, perplexity, result JSON,
aggregation — is smoke-tested end-to-end on CPU via `configs/smoke_cpu.yaml`.
A GPU (+ HF access for gated models) is needed for real-model numbers, the
zero-shot bundle at scale, and the CUDA FWHT kernel.
