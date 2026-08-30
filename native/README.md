# RotQuant native runtime

This directory contains the backend-neutral C++17 implementation of the
RotQuant native-v2 block contract. It is intentionally independent of PyTorch,
NumPy, GGUF, and a particular model architecture. It includes a portable scalar
correctness baseline plus ARM NEON and x86 AVX2 streaming kernels for 1–8-bit
weights. SIMD implementations are compiled and advertised only on supported
targets.

Build and test it with:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

Inspect compiled capabilities and run the JSON benchmark matrix:

```bash
build/native/rotquant-native-cli --capabilities
build/native/rotquant-native-bench \
  --out-features 128 --in-features 512 --group-size 128 \
  --batch 4 --iterations 10 --kernel auto
```

The library can also be embedded with `add_subdirectory(native)` and linked as
`RotQuant::native`, or installed as a CMake package:

```bash
cmake --install build/native --prefix /path/to/prefix
```

Consumers should call `available_kernels()` before selecting a forced kernel.
`CpuKernel::automatic` prefers a runtime-available AVX2 implementation, then
compiled NEON, and otherwise selects scalar. The AVX2 object is compiled with
its own ISA flag, leaving scalar code compatible with older x86 CPUs; CPUID and
OS vector-state checks run before it is advertised. Explicitly requesting an
unavailable kernel throws and never falls back silently. Use `--kernel scalar`,
`--kernel neon`, or `--kernel avx2` for matched local benchmark runs; every case
reports scalar-relative maximum absolute and relative-L2 error alongside timing.

The NEON implementation dispatches once on bit width and then runs a
compile-time-specialized decoder. Each inner step reads one packed word,
extracts four indices, gathers their arbitrary float32 codebook values, and
performs a vector multiply-accumulate. This covers non-byte-aligned 3/5/6/7-bit
formats without repacking them into a second persistent representation.

The AVX2 implementation follows the same format contract with eight codes per
packed word and hardware float32 codebook gathers. It is built in an isolated
translation unit so enabling it does not raise the minimum ISA of the library's
scalar path. Native CI runs the compiled C/C++ conformance and exercises the
best reported SIMD kernel on x86 Linux and macOS hosts.

## C ABI

[`include/rotquant/native_v2_c.h`](include/rotquant/native_v2_c.h) exposes ABI
version 1 without C++ exceptions or STL types. It provides:

- Runtime and format version queries.
- Compiled-kernel enumeration and fail-closed resolution.
- Dequantization and streaming matmul over caller-owned buffers.
- Stable status codes and thread-local error details.

The default build produces a static library. Pass `-DBUILD_SHARED_LIBS=ON` for
a shared library suitable for Python `ctypes`, Rust FFI, Mojo, or a runtime
plugin:

```bash
cmake -S native -B build/native-shared \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build/native-shared --parallel
ctest --test-dir build/native-shared --output-on-failure
```

The test suite compiles one conformance executable as C—not C++—and separately
loads the shared library from Python before comparing all eight widths against
the Python streaming oracle. `rq_native_v2_last_error()` is local to the calling
thread and remains valid until a later call on that thread changes it.

Python callers can load and register the shared library explicitly:

```python
from rotquant import KERNELS, NativeRuntimeLibrary, run_kernel

native = NativeRuntimeLibrary("build/native-shared/librotquant_native.dylib")
native.register_kernels(KERNELS, backend="native-cpu")
output = run_kernel("matmul", encoded, values, backend="native-cpu")
```

The binding validates ABI and format versions before registration. It requires
already-contiguous float32 inputs and native-v2 buffers; it never converts an
input dtype, discovers a library implicitly, or substitutes the reference
backend. Registry callables retain the loaded library for their lifetime.

`rotquant-native-cli` is a raw-file conformance bridge, not a checkpoint
container. It reads packed `uint8` qdata plus native-endian float32 codebook and
input files, then writes native-endian float32 output. Persistent artifacts
must carry the full v2 manifest described in
[`../docs/native_runtime_v2.md`](../docs/native_runtime_v2.md).
