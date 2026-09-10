# Native GPU validation: retained Qwen3.5-4B

Status: 10 September 2026. The private llama.cpp integration now has a
whole-Qwen graph for the retained W5/scale8 backbone and shared W6/W8 vocabulary,
with scalar CPU, Metal and CUDA packed operators. CPU and Metal have been
compiled/executed locally. **CUDA has not been compiled or executed here, and
the actual retained 4B checkpoint has not been tested in this runtime.** This
Mac has no NVIDIA toolkit; local result bundles lack the saved model weights.

This is an experimental correctness-first runtime, not a production inference
library or a demonstrated speedup. Do not restart the public-task sweep yet.

## Colab: what to run

Open [the new notebook](../notebooks/qwen35_4b_native_gpu_validation_colab.ipynb),
not the old public-task or Python-reference notebook. It must first be published
to GitHub before a Colab `main` checkout can contain it.

1. Select a CUDA GPU runtime. The previous A100 40GB is a sensible test device;
   it is not a newly measured minimum-memory requirement.
2. Point `SOURCE_ROOT` at the original Drive preparation root. Each arm must
   contain `checkpoint/`, `prepared.json`, `preparation.json`, and
   `packed_probes.safetensors`. Result-only download bundles are insufficient.
3. Start with `ARMS = ("b5_v6_s0",)`, `RUN_RETAINED_MODEL = True`, and
   `RUN_TIMING = False`. Add W8 after the W6 test. Pin `REPO_REF` to the published
   commit if possible; the notebook resolves it once and never updates mid-run.
4. Run cells in order. Each stage writes a persistent log, prints its PID/tail
   command, streams progress and emits a heartbeat every 30 seconds. The default
   software time cap is 60 minutes, with per-phase limits. It kills subprocess
   groups on interruption/timeout; it does **not** terminate Colab billing.
5. Share the new reports archive, then disconnect/delete the paid runtime.

Stages are: environment checks → pinned native build → CPU/CUDA operator
conformance → W6/W8 tiny whole-model conformance → lossless retained export →
canonical saved-probe parity → optional bounded timing. A failed gate stops the
run. The retained cell explicitly checks previous reports, so jumping ahead
does not bypass them. No quantization/training, external-Hadamard build,
`llama-cpp-python` wheel, competitor download or public-task sweep is needed.

Exports go to a new local directory; reports/logs go to a new Drive root. Saved
checkpoint files and old experiment identities are read-only. The notebook
does not import results from previous native binaries or resume old quality
receipts. After losing a runtime, use a new run tag and repeat the gates.

Before the retained export, an additional small offline Transformers-to-GGUF
check exercises the real converter and compares CPU/GPU logits with the canonical
random model. Its fake tokenizer is explicitly outside production validation.
Gate receipts must match the current runtime library hashes; rebuilding cannot
silently reuse old passes.

## Build and local test commands

The reproducible build script checks the exact pinned base, patch SHA and every
modified source file. It rejects unrelated edits rather than resetting them.
Keep source/build directories separate. The consolidated v2 patch includes v1.

```bash
python scripts/build_rq3_runtime.py \
  --source-dir build/llama-rqv3-source --build-dir build/llama-rqv3-metal \
  --backend Metal --jobs 4
python scripts/check_rq3_gpu.py \
  --library build/llama-rqv3-metal/bin/librotquant_ggml_test.dylib \
  --backend CPU --output build/operators-cpu.json
GGML_METAL_TENSOR_DISABLE=1 python scripts/check_rq3_gpu.py \
  --library build/llama-rqv3-metal/bin/librotquant_ggml_test.dylib \
  --backend MTL0 --output build/operators-metal.json
python scripts/make_rq3_model_fixture.py \
  --llama-dir build/llama-rqv3-source --vocabulary-bits 6 \
  --output build/tiny-w6.gguf
GGML_METAL_TENSOR_DISABLE=1 python scripts/check_rq3_model.py \
  --library build/llama-rqv3-metal/bin/librotquant_ggml_test.dylib \
  --model build/tiny-w6.gguf --backend MTL0 --output build/model-w6.json
```

Repeat with vocabulary bits 8 and new output paths. For Linux/CUDA, use
`--backend CUDA` for the builder, `CUDA0` for tests and
`bin/librotquant_ggml_test.so`. Missing CUDA/nvcc is an error, never a CPU fallback.
These are private experimental bindings, not a supported public C/Python API.

### M5 limitation discovered during testing

The pinned upstream M5 tensor-API matmul path produced incorrect values in the
small dense alpha/beta projections of the random fixture at longer prefills.
For a 64-token probe the input embedding was identical and first normalization
differed by only 2.4e-7, but dense alpha/beta outputs differed by about 2.9/3.1.
Disabling that newer path with the **explicit** upstream
`GGML_METAL_TENSOR_DISABLE=1` setting restored conformance without changing
RotQuant weights, packed kernels or thresholds. Computation still runs on Metal
through the simdgroup path; this is not CPU fallback. This setting is recorded
in test reports. Do not interpret the failure as a quantization-quality result
or assume an untested M5 default is safe. No upstream bug-fix claim is made.

## What executes and what stays packed

The loader validates native-v3 matrix headers/payloads, saved ±1 signs and
bijective GDN row/column maps. GGUF v2 explicitly requires FWHT128, FP16
execution boundaries and one shared dense-equivalent vocabulary owner.

Backbone execution rounds inputs to FP16, applies the saved signed FWHT, rounds
rotated activations, reconstructs affine scales using separate FP32 multiply
and add, decodes/rounds weights and accumulates the dot product. Vocabulary
lookup and head execution inverse-rotate decoded weight tiles and round the
weights to FP16 **before** use. Moving that rounding across the dot product
would be a different model and is not used.

GPU kernels use bounded 128-value on-chip tiles. Codes, scales, codebooks,
signs and GDN maps remain compact in GPU storage; neither a full dense backbone
nor a full dense vocabulary is materialized. The input embedding is assigned
to the GPU when all layers are offloaded, and the output head aliases it.
Dense norms, the excluded small projections, attention, recurrent state and
FP16 KV cache execute through llama.cpp's existing GPU operators. Codebooks
still need memory/cache accesses; this does not claim zero bandwidth overhead.

`ROTQUANT_REQUIRE_GPU=1` checks actual scheduler assignments before graph
execution. Non-view compute nodes assigned to a non-GPU backend are rejected.
CPU is allowed only for the explicit small-fixture reference pass. Allocated
CPU staging buffers are not by themselves evidence of CPU arithmetic fallback.
Full static-transfer/scratch high-water profiling remains a later audit.

## Evidence and acceptance boundaries

- Operator tests cover W5/scale8, W6/scale16 and W8/scale16, 256/512 input
  widths, 137 output rows, 1/7/33 tokens, non-identity GDN maps and lookup rows.
  FP16 matmul gates are atol 0.001 / rtol 0.002; lookup atol is 0.000125.
- Whole-model fixtures contain two Qwen layers: linear and full attention.
  Prompts have 1/4/17/64 tokens, each followed by eight cached steps. CPU/GPU
  max/mean absolute error, teacher-to-candidate KL, top-1 agreement and exact
  greedy traces must all pass. This is random-model plumbing, not 4B quality.
- Real-model parity consumes the original frozen IDs/logits/greedy traces;
  no native retokenization. Source checkpoint, preparation, export and probe
  hashes are checked. Extra multimodal inputs or nontrivial masks are rejected.
  Cross-engine gates reuse the predeclared prototype bounds (max 0.125, mean
  0.005, KL 1e-4, top-1 ≥99%) and additionally require exact short generation.
  It is **not** labelled same-runtime reload parity or next-token task accuracy.
- Optional timings use contexts 128/512/2048 with 16 cached decode steps,
  one warmup and two measured repetitions. Synchronization is included. The
  reserved context capacity is reported separately. `nvidia-smi` samples this
  process's VRAM because Torch allocator counters exclude GGML allocations;
  samples are not an exact instantaneous peak. Metal VRAM is unavailable here.
- Complete payload accounting includes the unchanged non-text sidecar. The
  runtime is text-only: a counted sidecar is not a working vision projector.
  Do not use llama.cpp's guessed parameter/BPW printout for opaque I8 blobs.

Local results are recorded in [the experiment ledger](experiment_log.md).
After real 4B/CUDA parity, collect bounded timings, profile transfers/scratch,
and budget a new runtime-bound public-task comparison with matched baselines.
No new quality promotion or serving-speed claim follows from this implementation.
