# Native runtime v2

Native v2 is the backend-independent matrix representation for RotQuant's
kernel-targeted 1–8-bit scalar profiles. It complements the existing GGUF v1
integration: v1 remains the deployed, fixed 4-bit llama.cpp format, while v2 is
the general contract new CPU, Metal, CUDA and experimental Mojo kernels should
implement.

## Block layout

Each matrix row is divided into logical groups. Every group stores:

1. One little-endian fp16 scale.
2. `ceil(group_size * bits / 8)` LSB-first code bytes.

The code stream follows the same ordering as the generic checkpoint format.
Partial final groups are padded with zero code indices; `in_features` remains
authoritative and padding is never returned by the decoder. A matrix carries
its scalar codebook as `2**bits` float32 centroids, so the runtime is not tied
to Gaussian or uniform profiles.

`rotquant.native.NativeLayout.to_manifest()` is the machine-readable binary
contract. `NativeEncodedMatrix` combines it with dimensions, codebook and raw
row bytes. Residual, sketch and per-row-scale encodings deliberately fail
closed until separate native layouts are specified for them.

## Kernel dispatch

`rotquant.runtime` registers implementations by:

- Backend.
- Operation.
- Supported bit widths.
- Supported group sizes.
- Priority.

There is no implicit backend fallback. Requesting `backend="metal"` before a
matching Metal kernel is registered raises an error rather than timing NumPy or
a reconstructed dense matrix. The initial `reference` backend covers
dequantization and matrix multiplication for all 1–8-bit layouts. Its dispatched
matmul streams one decoded group at a time and never reconstructs the complete
dense weight; a separate dense oracle exists only for numerical comparison.

Production backends should add distinct operations for fused matrix-vector,
matrix-matrix, embedding lookup and cache attention as those kernels land.

## Current compatibility

For 4-bit codes with groups of 128, native v2 produces exactly the same
scale-and-nibble block bytes as RotQuant-GGUF v1. GGUF v1 additionally fixes the
Gaussian codebook and 128-wide butterfly representation, so existing llama.cpp
artifacts and kernels remain unchanged.

The required gates for any new backend are:

1. Exact index, fp16-scale and codebook decoding for every supported width.
2. Matmul output agreement with the reference implementation.
3. No hidden dense persistent weight or fallback allocation.
4. Reported capability metadata matches actual dispatch.
5. Benchmarks separate packing, decoding and fused execution costs.

## Reference benchmark

Run the complete profile matrix with:

```bash
uv run python scripts/benchmark_native_reference.py \
  --output results/native_reference.json
```

The benchmark is a correctness and regression baseline, not a production
performance claim. Its NumPy timings give future kernels an exact named target
and its self-check prevents a fast but numerically different path from being
accepted accidentally.

## Portable C++ runtime

The first compiled implementation lives under `native/`. It is an installable,
dependency-free C++17 static library with a streaming matmul and a separate
dequantization oracle for all 1–8-bit scalar layouts. The matmul reads one
packed group at a time and does not allocate or persist a dense weight. Build,
test, inspect, and benchmark it with:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
build/native/rotquant-native-cli --capabilities
build/native/rotquant-native-bench --iterations 10
```

The C++ self-test covers every bit width and partial final groups. The Python
test suite additionally emits randomized native-v2 blocks, executes them
through the raw-file CLI, and compares the result with the Python streaming
oracle. This makes the Python encoder and C++ consumer a cross-language format
gate rather than two implementations that can drift together unnoticed.

The portable `scalar` kernel is always available as the correctness and
profiling floor. ARM builds also compile an `arm-neon` path for dequantization
and streaming matmul across all eight widths; `auto` selects it only when it is
present. The benchmark can force `scalar`, `neon`, or `avx2` and records
numerical drift against scalar for every timed case. x86 builds compile AVX2
separately and advertise it only after runtime CPUID and OS vector-state checks;
forced use is rejected rather than falling back when those checks fail.

The NEON inner loops are specialized at compile time for each width. Four codes
are extracted from one packed word per vector step, including the unaligned
3/5/6/7-bit streams, so kernels do not create a byte-per-index side buffer. On
the current ARM development host, the matched 256×1024, batch-4, group-128
microbenchmark runs at roughly 0.21–0.24 ms across the eight widths, versus
roughly 0.60–0.69 ms for scalar. These are local engineering measurements for
regression guidance, not model-serving or publication claims.

The same implementation is available through the version-1 C ABI in
`rotquant/native_v2_c.h`. Static and shared CMake builds are supported. ABI
calls return explicit status codes, keep diagnostic text thread-local, enumerate
only compiled kernels, and never allow a C++ exception to cross the boundary.
This is the integration surface intended for runtime forks and foreign-function
bindings; the raw-file CLI remains a conformance tool rather than an artifact
loader.

`rotquant.native_ffi.NativeRuntimeLibrary` is the first consumer of that ABI.
It takes an explicit shared-library path, verifies ABI version 1 and native
format version 2, exposes capability and numerical calls, and can register the
resolved implementation under an explicit `KernelRegistry` backend. Array
dtype and contiguity checks are strict so benchmark results cannot conceal a
conversion or copy before native execution.

The x86 AVX2 kernel processes eight codes per packed word and uses hardware
float32 codebook gathers. It compiles as a separate ISA-targeted translation
unit, keeping the rest of the library runnable on older x86 CPUs. The current
ARM host can cross-build and execute the generic x86 binary through Rosetta,
which correctly withholds AVX2 because that execution environment does not
expose it. Native CI provides the actual x86 SIMD execution gate.
