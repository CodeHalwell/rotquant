"""Cross-language conformance for the portable native-v2 C++ runtime."""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rotquant.native import (
    NativeEncodedMatrix,
    NativeLayout,
    pack_native_blocks,
    reference_dequantize,
    reference_streaming_matmul,
)
from rotquant.native_ffi import NativeRuntimeError, NativeRuntimeLibrary
from rotquant.runtime import KernelRegistry, run_kernel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def native_cpp_binaries(tmp_path_factory):
    if shutil.which("cmake") is None:
        pytest.skip("CMake is not available")
    build_dir = tmp_path_factory.mktemp("rotquant-native-build")
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "native"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if configure.returncode != 0:
        pytest.fail(f"native CMake configure failed:\n{configure.stdout}\n{configure.stderr}")
    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"native C++ build failed:\n{build.stdout}\n{build.stderr}")

    def executable(name: str) -> Path:
        matches = [
            path
            for path in build_dir.rglob(name + (".exe" if os.name == "nt" else ""))
            if path.is_file()
        ]
        if len(matches) != 1:
            pytest.fail(f"expected one {name} executable, found {matches}")
        return matches[0]

    libraries = [
        path
        for path in build_dir.rglob("*rotquant_native*")
        if path.is_file() and path.suffix in {".dll", ".dylib", ".so"}
    ]
    if len(libraries) != 1:
        pytest.fail(f"expected one RotQuant shared library, found {libraries}")

    return {
        "bench": executable("rotquant-native-bench"),
        "cli": executable("rotquant-native-cli"),
        "conformance": executable("rotquant-native-conformance"),
        "library": libraries[0],
    }


class CNativeLayout(ctypes.Structure):
    _fields_ = [
        ("bits", ctypes.c_uint32),
        ("group_size", ctypes.c_uint32),
    ]


@pytest.fixture(scope="session")
def native_c_abi(native_cpp_binaries):
    library = ctypes.CDLL(str(native_cpp_binaries["library"]))
    library.rq_native_v2_abi_version.argtypes = []
    library.rq_native_v2_abi_version.restype = ctypes.c_uint32
    library.rq_native_v2_last_error.argtypes = []
    library.rq_native_v2_last_error.restype = ctypes.c_char_p
    library.rq_native_v2_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        CNativeLayout,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    library.rq_native_v2_matmul.restype = ctypes.c_int
    return library


def test_cpp_self_conformance(native_cpp_binaries):
    result = subprocess.run(
        [str(native_cpp_binaries["conformance"])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "bits 1..8" in result.stdout


@pytest.mark.parametrize("bits", range(1, 9))
def test_shared_c_abi_matches_python_native_v2(native_c_abi, bits):
    rng = np.random.default_rng(500 + bits)
    layout = NativeLayout(bits, 13)
    out_features = 4
    in_features = 29
    batch = 2
    indices = rng.integers(
        0, 1 << bits, size=(out_features, in_features), dtype=np.uint16
    )
    scales = rng.uniform(
        0.01,
        0.5,
        size=(out_features, layout.groups_for(in_features)),
    ).astype(np.float16)
    codebook = np.ascontiguousarray(
        rng.normal(size=(1 << bits,)).astype(np.float32)
    )
    values = np.ascontiguousarray(
        rng.normal(size=(batch, in_features)).astype(np.float32)
    )
    encoded = NativeEncodedMatrix(
        qdata=pack_native_blocks(indices, scales, layout),
        codebook=codebook,
        layout=layout,
        in_features=in_features,
        out_features=out_features,
    )
    expected = reference_streaming_matmul(values, encoded)
    qdata = np.ascontiguousarray(encoded.qdata).view(np.uint8)
    output = np.empty((batch, out_features), dtype=np.float32)

    assert native_c_abi.rq_native_v2_abi_version() == 1
    status = native_c_abi.rq_native_v2_matmul(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.size,
        batch,
        qdata.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        qdata.size,
        codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        codebook.size,
        out_features,
        in_features,
        CNativeLayout(bits, layout.group_size),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.size,
        0,
    )
    error = native_c_abi.rq_native_v2_last_error().decode()
    assert status == 0, error
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


def test_python_native_library_registers_fail_closed_backend(
    native_cpp_binaries,
):
    native_library = NativeRuntimeLibrary(native_cpp_binaries["library"])
    capabilities = native_library.capabilities()
    kernels = {capability.kernel for capability in capabilities}
    assert "scalar" in kernels
    assert native_library.resolve_kernel("auto") in kernels
    if "avx2" not in kernels:
        with pytest.raises(NativeRuntimeError, match="not compiled: avx2"):
            native_library.resolve_kernel("avx2")

    rng = np.random.default_rng(900)
    layout = NativeLayout(5, 11)
    out_features = 6
    in_features = 23
    indices = rng.integers(
        0, 1 << layout.bits, size=(out_features, in_features), dtype=np.uint8
    )
    scales = rng.uniform(
        0.05,
        0.5,
        size=(out_features, layout.groups_for(in_features)),
    ).astype(np.float16)
    encoded = NativeEncodedMatrix(
        qdata=pack_native_blocks(indices, scales, layout),
        codebook=np.ascontiguousarray(
            rng.normal(size=(1 << layout.bits,)).astype(np.float32)
        ),
        layout=layout,
        in_features=in_features,
        out_features=out_features,
    )
    values = np.ascontiguousarray(
        rng.normal(size=(3, in_features)).astype(np.float32)
    )
    registry = KernelRegistry()
    specs = native_library.register_kernels(registry, backend="native-test")
    assert {spec.operation for spec in specs} == {"dequantize", "matmul"}

    actual_weight = run_kernel(
        "dequantize", encoded, backend="native-test", registry=registry
    )
    actual_output = run_kernel(
        "matmul", encoded, values, backend="native-test", registry=registry
    )
    np.testing.assert_allclose(
        actual_weight, reference_dequantize(encoded), rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        actual_output,
        reference_streaming_matmul(values, encoded),
        rtol=2e-5,
        atol=2e-5,
    )

    with pytest.raises(TypeError, match="float32"):
        native_library.matmul(encoded, values.astype(np.float64))
    with pytest.raises(RuntimeError, match="backend='missing'"):
        run_kernel("matmul", encoded, values, backend="missing", registry=registry)


def test_cpp_capabilities_are_fail_closed(native_cpp_binaries):
    result = subprocess.run(
        [str(native_cpp_binaries["cli"]), "--capabilities"],
        capture_output=True,
        text=True,
        check=True,
    )
    capabilities = json.loads(result.stdout)
    assert capabilities["format_version"] == 2
    kernels = {entry["kernel"]: entry for entry in capabilities["kernels"]}
    assert kernels["scalar"] == {
        "name": "portable-scalar",
        "kernel": "scalar",
        "min_bits": 1,
        "max_bits": 8,
        "group_size": 0,
    }
    if platform.machine().lower() in {"arm64", "aarch64"}:
        assert kernels["neon"] == {
            "name": "arm-neon",
            "kernel": "neon",
            "min_bits": 1,
            "max_bits": 8,
            "group_size": 0,
        }
        assert "avx2" not in kernels
    if "avx2" in kernels:
        assert kernels["avx2"] == {
            "name": "x86-avx2",
            "kernel": "avx2",
            "min_bits": 1,
            "max_bits": 8,
            "group_size": 0,
        }


@pytest.mark.parametrize("kernel", ["scalar", "neon", "avx2"])
def test_cpp_benchmark_reports_correctness_and_timing(
    native_cpp_binaries, kernel
):
    capabilities = subprocess.run(
        [str(native_cpp_binaries["cli"]), "--capabilities"],
        capture_output=True,
        text=True,
        check=True,
    )
    compiled_kernels = {
        entry["kernel"] for entry in json.loads(capabilities.stdout)["kernels"]
    }
    if kernel not in compiled_kernels:
        pytest.skip(f"native C++ kernel is not compiled: {kernel}")

    result = subprocess.run(
        [
            str(native_cpp_binaries["bench"]),
            "--out-features",
            "13",
            "--in-features",
            "65",
            "--group-size",
            "32",
            "--batch",
            "2",
            "--iterations",
            "2",
            "--kernel",
            kernel,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    benchmark = json.loads(result.stdout)
    assert benchmark["format_version"] == 2
    assert [case["bits"] for case in benchmark["cases"]] == list(range(1, 9))
    assert all(case["kernel"] == kernel for case in benchmark["cases"])
    assert all(case["matmul_ms"] > 0 for case in benchmark["cases"])
    assert all(
        case["relative_l2_error_vs_scalar"] < 1e-4
        for case in benchmark["cases"]
    )


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("kernel", ["scalar", "neon", "avx2"])
@pytest.mark.parametrize(
    ("out_features", "in_features", "group_size", "batch"),
    [(5, 17, 7, 3), (7, 1025, 128, 2)],
    ids=["partial-small", "partial-long"],
)
def test_cpp_kernel_matches_python_native_v2(
    native_cpp_binaries,
    tmp_path,
    bits,
    kernel,
    out_features,
    in_features,
    group_size,
    batch,
):
    capabilities = subprocess.run(
        [str(native_cpp_binaries["cli"]), "--capabilities"],
        capture_output=True,
        text=True,
        check=True,
    )
    compiled_kernels = {
        entry["kernel"] for entry in json.loads(capabilities.stdout)["kernels"]
    }
    if kernel not in compiled_kernels:
        pytest.skip(f"native C++ kernel is not compiled: {kernel}")

    rng = np.random.default_rng(100 + bits)
    layout = NativeLayout(bits, group_size)
    indices = rng.integers(
        0, 1 << bits, size=(out_features, in_features), dtype=np.uint16
    )
    scales = rng.uniform(
        0.01,
        0.5,
        size=(out_features, layout.groups_for(in_features)),
    ).astype(np.float16)
    codebook = rng.normal(size=(1 << bits,)).astype(np.float32)
    values = rng.normal(size=(batch, in_features)).astype(np.float32)
    encoded = NativeEncodedMatrix(
        qdata=pack_native_blocks(indices, scales, layout),
        codebook=codebook,
        layout=layout,
        in_features=in_features,
        out_features=out_features,
    )
    expected = reference_streaming_matmul(values, encoded)

    qdata_path = tmp_path / "qdata.bin"
    codebook_path = tmp_path / "codebook.bin"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    encoded.qdata.tofile(qdata_path)
    encoded.codebook.tofile(codebook_path)
    values.tofile(input_path)
    result = subprocess.run(
        [
            str(native_cpp_binaries["cli"]),
            "--qdata",
            str(qdata_path),
            "--codebook",
            str(codebook_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--bits",
            str(bits),
            "--group-size",
            str(layout.group_size),
            "--in-features",
            str(in_features),
            "--out-features",
            str(out_features),
            "--batch",
            str(batch),
            "--kernel",
            kernel,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    actual = np.fromfile(output_path, dtype=np.float32).reshape(batch, out_features)
    relative_tolerance = 2e-6 if in_features < 100 else 2e-5
    absolute_tolerance = 2e-6 if in_features < 100 else 2e-4
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )


def test_cpp_unimplemented_simd_request_does_not_fallback(
    native_cpp_binaries, tmp_path
):
    capabilities = subprocess.run(
        [str(native_cpp_binaries["cli"]), "--capabilities"],
        capture_output=True,
        text=True,
        check=True,
    )
    compiled_kernels = {
        entry["kernel"] for entry in json.loads(capabilities.stdout)["kernels"]
    }
    unavailable_kernel = "avx2" if "avx2" not in compiled_kernels else "neon"
    if unavailable_kernel in compiled_kernels:
        pytest.skip("all named SIMD kernels are compiled")

    layout = NativeLayout(4, 8)
    qdata = pack_native_blocks(
        np.zeros((1, 8), dtype=np.uint8),
        np.ones((1, 1), dtype=np.float16),
        layout,
    )
    paths = {
        "qdata": tmp_path / "qdata.bin",
        "codebook": tmp_path / "codebook.bin",
        "input": tmp_path / "input.bin",
        "output": tmp_path / "output.bin",
    }
    qdata.tofile(paths["qdata"])
    np.zeros(16, dtype=np.float32).tofile(paths["codebook"])
    np.zeros(8, dtype=np.float32).tofile(paths["input"])
    result = subprocess.run(
        [
            str(native_cpp_binaries["cli"]),
            "--qdata",
            str(paths["qdata"]),
            "--codebook",
            str(paths["codebook"]),
            "--input",
            str(paths["input"]),
            "--output",
            str(paths["output"]),
            "--bits",
            "4",
            "--group-size",
            "8",
            "--in-features",
            "8",
            "--out-features",
            "1",
            "--batch",
            "1",
            "--kernel",
            unavailable_kernel,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert f"not compiled: {unavailable_kernel}" in result.stderr
    assert not paths["output"].exists()

    benchmark = subprocess.run(
        [
            str(native_cpp_binaries["bench"]),
            "--out-features",
            "2",
            "--in-features",
            "8",
            "--group-size",
            "8",
            "--batch",
            "1",
            "--iterations",
            "1",
            "--kernel",
            unavailable_kernel,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert benchmark.returncode == 2
    assert benchmark.stdout == ""
    assert f"not compiled: {unavailable_kernel}" in benchmark.stderr


@pytest.mark.parametrize("corruption", ["truncated-qdata", "nan-codebook"])
def test_cpp_rejects_malformed_native_v2_payload(
    native_cpp_binaries, tmp_path, corruption
):
    layout = NativeLayout(4, 8)
    qdata = pack_native_blocks(
        np.zeros((1, 8), dtype=np.uint8),
        np.ones((1, 1), dtype=np.float16),
        layout,
    )
    codebook = np.zeros(16, dtype=np.float32)
    if corruption == "truncated-qdata":
        qdata = qdata[:, :-1]
    else:
        codebook[3] = np.nan

    paths = {
        "qdata": tmp_path / "qdata.bin",
        "codebook": tmp_path / "codebook.bin",
        "input": tmp_path / "input.bin",
        "output": tmp_path / "output.bin",
    }
    qdata.tofile(paths["qdata"])
    codebook.tofile(paths["codebook"])
    np.zeros(8, dtype=np.float32).tofile(paths["input"])
    result = subprocess.run(
        [
            str(native_cpp_binaries["cli"]),
            "--qdata",
            str(paths["qdata"]),
            "--codebook",
            str(paths["codebook"]),
            "--input",
            str(paths["input"]),
            "--output",
            str(paths["output"]),
            "--bits",
            "4",
            "--group-size",
            "8",
            "--in-features",
            "8",
            "--out-features",
            "1",
            "--batch",
            "1",
            "--kernel",
            "scalar",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not paths["output"].exists()
