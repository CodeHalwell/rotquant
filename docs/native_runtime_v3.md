# Native-v3 matrix contract and W5 serving milestone

Status, 10 September 2026: **the exact matrix format, full Qwen graph and
CPU/Metal/CUDA operators are implemented; CPU/Metal conformance and the user's
A100 CUDA/retained-W5/W6 saved-probe checks have passed.** The
[September 10 evidence review](../research/results/native_cuda_2026_09_10/README.md)
does not establish throughput, long-context behavior or retained W5/W8 parity. Use the
[new native-GPU notebook and run guide](native_gpu_validation.md), not the
old Python-reference quality sweep. This matrix format is not checkpoint
v3, a GGUF format version, or an extension that stock llama.cpp understands.
Existing native-v2/GGUF-v1 bytes and supported 16-bit-scale paths are unchanged.
The legacy exporters now reject compressed/32-bit scales instead of silently
rounding them to FP16 (review defect L5).

## Why this comes before another Colab

The user stopped the public-task run. Its partial log showed about 0.85 generated
tokens/s on the first W5/W6 arm, versus about 15.4 for FP16 on the same first 30
GSM8K prompts. Those are logged generation-plus-scoring rates, not a controlled
native kernel benchmark. Repeated Python dequantization is not an acceptable
release runtime. Keep the saved checkpoints and partial evidence; do not resume
new native execution into those old receipts. See the [ledger](experiment_log.md).

The retained recipe is still W5/g128/Gaussian/GPTQ with affine uint8 scales,
fixed FWHT, and a single shared W6 or W8 vocabulary with FP16 scales. No new
quantization, rotation training, allocation, cache compression or promotion is
part of this milestone.

## Wire format

All multibyte values are little-endian. There is one 64-byte header, then five
contiguous arrays with **no alignment padding**. Readers must validate sizes
before accessing arrays; arbitrary dimensions in untrusted headers must not
cause overflow or allocation. Python and C++ reject non-finite/negative scale
metadata, non-finite centroids, decoded-weight FP32 overflow, unsupported
layouts, trailing bytes and nonzero unused code bits.

| Header offset | Type | Meaning |
|---:|---|---|
| 0 | 8 bytes | `RQNATV3\0` |
| 8, 12 | u32, u32 | format version 3, header length 64 |
| 16, 20 | u32, u32 | code width 1–8, group size >0 |
| 24, 32 | u64, u64 | output/input features, positive signed-64-bit range |
| 40, 44 | u32, u32 | scale width 8/16; metadata block size >=2 for scale8, zero for scale16 |
| 48, 52 | u32, u32 | flags and reserved, both zero |
| 56 | u64 | exact payload length |

For `O` rows, `I` columns, width `b`, group `g`, let `S = O * ceil(I/g)`.
The payload is:

1. `ceil(O*I*b/32)` original int32 code words. Row-major, continuous LSB-first
   packing; **no per-row or per-group padding**. Only the last word has unused
   bits, all zero. Copy words from the canonical checkpoint; do not unpack and
   requantize them.
2. `S` scale entries in row-major order: uint8 codes or FP16 scales.
3. For scale8 only: `ceil(S/block)` FP16 offsets.
4. For scale8 only: the same number of FP16 steps.
5. `2**b` FP32 scalar centroids, preserved bit-for-bit. Non-Gaussian scalar
   tables work too; vector, residual and sketch formats are out of scope.

Scale8 reconstruction must use **two separate FP32 operations**, not FMA:

```text
product = fp32(scale_code) * fp32(step)
scale = fp32(offset) + product
weight = fp32(centroid) * scale
```

Do not cast that reconstructed scale to FP16. A W5/g128 regression fixture
exhibits >0.001 absolute weight drift after such a cast. The native-v3 C++
translation unit disables FP contraction; both native-v3 decoders agree exactly
with canonical Torch dequantization in the tested fixtures. Zero steps and FP16
subnormal steps are valid. Bits, scales and centroids remain compact in storage.

These weights are in **rotated coordinates**, and this primitive returns FP32.
It does not implement `QuantLinear`'s activation rotation or FP16 weight rounding,
nor the vocabulary's inverse rotation followed by FP16 `dense_equivalent`
rounding. Matrix conformance cannot substitute for those model-level checks.

## Local check — no GPU or model download

From the repository root, using the existing development environment:

```bash
cmake -S native -B build/native-v3 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build/native-v3 --parallel
ctest --test-dir build/native-v3 --output-on-failure
.venv/bin/python scripts/check_native_v3.py --library build/native-v3/librotquant_native.dylib
```

Linux uses `librotquant_native.so`; select the actual DLL path on Windows.
The script prints progress for all 16 width/scale combinations, then a JSON
report with the library SHA-256, code revision, dirty flag and matrix results.
`--output PATH` optionally saves a local report without overwriting an existing
one. `--require-model-runtime` currently returns **exit 2**, even if matrix tests
pass: GGUF graph execution, GPU parity and the speed/cost checks are missing.
This command is a readiness report, not an interlock retrofitted into old Colabs.

Python API (deliberately not registered as a full-model backend):

```python
from rotquant.native_v3 import encode_native_v3
from rotquant.native_v3_ffi import NativeV3Runtime

matrix = encode_native_v3(qweight)  # original compact tensors; no re-quantization
runtime = NativeV3Runtime("build/native-v3/librotquant_native.dylib")
handle = runtime.prepare(matrix)   # one owned compact copy, retained across calls
y = handle.matmul(x)              # contiguous FP32 [batch, in_features]
```

The scalar C ABI validates compact storage each call and streams decoded values
directly into accumulation without a dense weight cache. It is a correctness
floor, **not a tuned GEMV/GEMM**. The explicit `dequantize_rows` diagnostic does
allocate the requested dense rows. Preparation/serialization may temporarily
hold more than one packed copy; those costs must not be hidden in load/peak
memory reporting. No automatic Metal/CUDA fallback exists in this binding.

## GGUF-v2 export work (10 September)

`scripts/export_rotquant_gguf_v2.py` now assembles the retained W5/scale8 backbone
and W6/W8 shared vocabulary into an experimental GGUF-v2 container. It requires
the original integrity-verified checkpoint, not a result bundle containing only
`rotquant_config.json`. It uses the pinned Qwen converter for the dense tensors,
tokenizer and model metadata, and copies the native-v3 packed payload unchanged.

- `*.rqv3` contains the original native-v3 bytes; `*.rqsign` stores the saved
  int8 FWHT signs. Optional `*.rqrow` and `*.rqcol` are int32 Qwen GDN maps.
  Permuting affine uint8 scale codes in isolation would change the weights;
  instead, the runtime must apply these row and input-column maps.
- `token_embd.rqv3` contains all original, word-aligned vocabulary chunks with
  one scalar codebook. Its embedding and output-head uses must share storage.
  The new runtime implements inverse-FWHT/FP16 vocabulary rounding explicitly.
- Non-text tensors excluded by the text converter are saved unchanged in
  `auxiliary.safetensors` and included in the payload byte total. That sidecar
  is not a working multimodal projector. Nothing is discarded to inflate a
  compression claim.
- Export goes to a new directory and records source/file/tensor hashes in
  `export.json`. Existing artifacts are never overwritten. Unsupported bias,
  LoRA, activation quantization and learned rotations fail closed.

Small synthetic tests cover the matrix assembly and converter adapter,
including exact saved codes/scales, deduplicated signs, vocabulary sharing,
GDN maps and unsupported-recipe rejection. Explicit per-layer hybrid topology
is preserved instead of assuming the pinned converter's default interval four.
Separate native operator, whole-model and offline HF-to-GGUF tests now execute
on CPU/Metal; see the [run guide](native_gpu_validation.md) for their boundaries.
They do **not** establish successful full-size conversion or 4B/CUDA parity.
The export receipt still says `runtime_validated: false`: successful export is
not execution evidence, and the old GGUF-v1 runtime cannot load this container.

The development machine has an M5 Max with 64 GiB unified memory. Metal testing
is possible locally; NVIDIA CUDA execution is not. Locally available retained
result bundles currently contain manifests but no model/packed safetensors.
The original W5/W6 or W5/W8 checkpoint is needed for actual 4B validation.

## Remaining work, in implementation order

1. **Validate the implemented exporter and graph on the actual checkpoint.**
   The contract and CPU/Metal/CUDA operators now exist. Verify original W5 codes and uint8 affine
   scales; one owner for W6/W8 embedding/head storage; original rotation IDs,
   scales and centroids. Retain Qwen GDN value-head permutations and model
   metadata. Count auxiliary/vision tensors, headers and serialization overhead.
   Do not route through the old `--quantize-tied-embedding` W4 conversion.
2. **Model conformance before speed.** Compare canonical checkpoint, exported
   CPU graph, then accelerator on the same fixed tensors, token IDs and dtype.
   Test individual projections, inverse-rotated vocabulary rows, tied head,
   logits and greedy continuations, including partial groups and GDN shapes.
   Require bit-exact payload preservation and separately registered numerical
   tolerances before observing results. Use FP16 cache; do not revive withdrawn
   KV claims. Keep FP32 model buffers intact. Close any relevant L1/L2/L3/L7
   rotation/loading defects before supporting learned-rotation variants.
3. **Validate CUDA, then optimize the correctness-first packed kernels.**
   Decode/scale/round/multiply already execute in bounded tiles; static packed
   tensors and one shared vocabulary stay on the GPU. No complete dense weight
   matrix is written per token. Benchmark
   the actual W5/scale8 and W6/W8 workload, not old W4 numbers. Build and execute
   the llama.cpp patch in CI; `git apply --check` is insufficient. A standard
   `GGML_CUDA=ON` build does not provide custom RotQuant operators.
4. **Mandatory short model preflight.** On the intended device, pin runtime
   build/library hashes, artifact hashes, inputs, precision and cache. Measure
   warm prefill/decode at batch 1 and contexts 128/512/2048, plus load time and
   allocated/reserved peak memory. Audit static transfers, dense caches and
   scratch high-water marks. Sample the public-task token caps, estimate total
   session time from measured prompt/token rates, include setup + at least 25%
   contingency, and require an explicit user budget. Fail if any graph op falls
   back, parity fails, the budget is exceeded, or persistent allocation grows.
   No speed threshold should be invented after seeing the performance.
5. **New runtime-bound public-task run.** Only then freeze a new evaluation
   identity/root, run the small smoke, then the agreed bounded sweep. Keep
   source, provider and BF16 bridge controls in the same engine where possible;
   measure their differences instead of assuming engine identity removes every
   confound. Report per-benchmark paired intervals/flips and truncation. Preserve
   the old run as incomplete reference-path evidence, never merge its records
   into native outcomes. All-provider 4B and then 27B remain later milestones.

No paid sweep, retained 4B GGUF artifact, 4B Metal/CUDA performance result or new
quality result is produced by this work. Random tiny-model GGUFs are test fixtures,
not retained-model evidence. The remaining L1–L12 queue is
not declared closed: only the lossy-export blocker L5 is addressed here.

## Historical matrix-only verification record (9 September)

Development checkout based on `0054735324928733f75d9574e74a40f1dc41eda8`
(working-tree changes), macOS arm64, AppleClang 21, Torch 2.12.0 and NumPy 2.4.6:

- Full Python suite with `ROTQUANT_REQUIRE_GIT_HISTORY=1`: 778 passed, 17
  architecture-specific AVX2 skips; the historical reuse audit ran. Two existing
  SWIG deprecation warnings remain, not numerical/native failures.
- `ruff check .` and `git diff --check`: clean.
- Release shared CMake build with warnings-as-errors: all three CTest suites
  pass. Separate Debug ASan + UBSan build: all three pass.
- Cross-language coverage includes all widths 1–8 and scale8/16, partial groups,
  crossing code words/rows/scale blocks, subnormal and zero-step scales,
  malformed headers/payload mutations, output non-mutation on rejection, and
  W5/scale8, W6/scale16, W8/scale16 Gaussian/GPTQ/g128 fixtures.
- The bounded local readiness command passes 16 matrix cases and returns exit
  2 under `--require-model-runtime`, correctly leaving the paid model run blocked.

These are local results, not a claim that remote CI or a real 4B GPU model run
has passed. No model weights were downloaded or re-quantized for verification;
quantization tests use small synthetic tensors.
