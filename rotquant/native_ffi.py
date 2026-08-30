"""Explicit Python binding for the versioned RotQuant native-v2 C ABI."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native import NATIVE_FORMAT_VERSION, NativeEncodedMatrix
from .runtime import KernelRegistry, KernelSpec

NATIVE_C_ABI_VERSION = 1

_STATUS_OK = 0
_KERNEL_VALUES = {"auto": 0, "scalar": 1, "neon": 2, "avx2": 3}
_KERNEL_NAMES = {value: name for name, value in _KERNEL_VALUES.items()}
_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
_BYTE_POINTER = ctypes.POINTER(ctypes.c_uint8)


class NativeRuntimeError(RuntimeError):
    """A native C ABI call returned a non-success status."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"RotQuant native runtime status {status}: {detail}")


@dataclass(frozen=True, slots=True)
class NativeKernelCapability:
    """One kernel reported by the loaded native library."""

    name: str
    kernel: str
    min_bits: int
    max_bits: int
    group_size: int | None

    @property
    def bits(self) -> frozenset[int]:
        return frozenset(range(self.min_bits, self.max_bits + 1))


class _CLayout(ctypes.Structure):
    _fields_ = [
        ("bits", ctypes.c_uint32),
        ("group_size", ctypes.c_uint32),
    ]


class _CCapability(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("kernel", ctypes.c_int),
        ("min_bits", ctypes.c_uint32),
        ("max_bits", ctypes.c_uint32),
        ("group_size", ctypes.c_uint32),
    ]


class NativeRuntimeLibrary:
    """Loaded native-v2 shared library with checked NumPy entry points.

    Loading is always explicit: callers supply the library path and decide when
    to register its kernels. Inputs must already be contiguous float32/native-v2
    buffers, preventing an ostensibly native benchmark from hiding conversion
    or dense-weight allocations.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"native runtime library not found: {self.path}")
        self._library = ctypes.CDLL(str(self.path))
        self._configure_signatures()
        abi_version = int(self._library.rq_native_v2_abi_version())
        if abi_version != NATIVE_C_ABI_VERSION:
            raise RuntimeError(
                f"native runtime ABI {abi_version} is incompatible with "
                f"Python ABI {NATIVE_C_ABI_VERSION}"
            )
        format_version = int(self._library.rq_native_v2_format_version())
        if format_version != NATIVE_FORMAT_VERSION:
            raise RuntimeError(
                f"native runtime format {format_version} is incompatible with "
                f"Python format {NATIVE_FORMAT_VERSION}"
            )

    def _configure_signatures(self) -> None:
        library = self._library
        library.rq_native_v2_abi_version.argtypes = []
        library.rq_native_v2_abi_version.restype = ctypes.c_uint32
        library.rq_native_v2_format_version.argtypes = []
        library.rq_native_v2_format_version.restype = ctypes.c_uint32
        library.rq_native_v2_last_error.argtypes = []
        library.rq_native_v2_last_error.restype = ctypes.c_char_p
        library.rq_native_v2_kernel_count.argtypes = []
        library.rq_native_v2_kernel_count.restype = ctypes.c_size_t
        library.rq_native_v2_kernel_capability_at.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(_CCapability),
        ]
        library.rq_native_v2_kernel_capability_at.restype = ctypes.c_int
        library.rq_native_v2_resolve_kernel.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        library.rq_native_v2_resolve_kernel.restype = ctypes.c_int
        library.rq_native_v2_dequantize.argtypes = [
            _BYTE_POINTER,
            ctypes.c_size_t,
            _FLOAT_POINTER,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _CLayout,
            _FLOAT_POINTER,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.rq_native_v2_dequantize.restype = ctypes.c_int
        library.rq_native_v2_matmul.argtypes = [
            _FLOAT_POINTER,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _BYTE_POINTER,
            ctypes.c_size_t,
            _FLOAT_POINTER,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _CLayout,
            _FLOAT_POINTER,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.rq_native_v2_matmul.restype = ctypes.c_int

    def _last_error(self) -> str:
        detail = self._library.rq_native_v2_last_error()
        return "unknown native runtime error" if detail is None else detail.decode()

    def _check(self, status: int) -> None:
        if status != _STATUS_OK:
            raise NativeRuntimeError(status, self._last_error())

    @staticmethod
    def _kernel_value(kernel: str) -> int:
        try:
            return _KERNEL_VALUES[kernel]
        except KeyError as error:
            raise ValueError(f"unknown native runtime kernel: {kernel!r}") from error

    @staticmethod
    def _validate_encoded_buffers(encoded: NativeEncodedMatrix) -> None:
        if not isinstance(encoded, NativeEncodedMatrix):
            raise TypeError("encoded must be a NativeEncodedMatrix")
        if not encoded.qdata.flags.c_contiguous:
            raise ValueError("native qdata must be C-contiguous")
        if not encoded.codebook.flags.c_contiguous:
            raise ValueError("native codebook must be C-contiguous")

    def capabilities(self) -> tuple[NativeKernelCapability, ...]:
        count = int(self._library.rq_native_v2_kernel_count())
        if count == 0:
            detail = self._last_error()
            raise NativeRuntimeError(4, detail or "native runtime reported no kernels")
        capabilities: list[NativeKernelCapability] = []
        for index in range(count):
            capability = _CCapability()
            self._check(
                self._library.rq_native_v2_kernel_capability_at(
                    index, ctypes.byref(capability)
                )
            )
            name = capability.name
            if name is None or capability.kernel not in _KERNEL_NAMES:
                raise RuntimeError("native runtime returned malformed capability data")
            capabilities.append(
                NativeKernelCapability(
                    name=name.decode(),
                    kernel=_KERNEL_NAMES[capability.kernel],
                    min_bits=int(capability.min_bits),
                    max_bits=int(capability.max_bits),
                    group_size=(
                        None
                        if capability.group_size == 0
                        else int(capability.group_size)
                    ),
                )
            )
        return tuple(capabilities)

    def resolve_kernel(self, kernel: str = "auto") -> str:
        resolved = ctypes.c_int()
        self._check(
            self._library.rq_native_v2_resolve_kernel(
                self._kernel_value(kernel), ctypes.byref(resolved)
            )
        )
        if resolved.value not in _KERNEL_NAMES:
            raise RuntimeError("native runtime resolved an unknown kernel")
        return _KERNEL_NAMES[resolved.value]

    def dequantize(
        self,
        encoded: NativeEncodedMatrix,
        *,
        kernel: str = "auto",
    ) -> np.ndarray:
        self._validate_encoded_buffers(encoded)
        qdata = encoded.qdata.view(np.uint8)
        output = np.empty(
            (encoded.out_features, encoded.in_features), dtype=np.float32
        )
        self._check(
            self._library.rq_native_v2_dequantize(
                qdata.ctypes.data_as(_BYTE_POINTER),
                qdata.size,
                encoded.codebook.ctypes.data_as(_FLOAT_POINTER),
                encoded.codebook.size,
                encoded.out_features,
                encoded.in_features,
                _CLayout(encoded.layout.bits, encoded.layout.group_size),
                output.ctypes.data_as(_FLOAT_POINTER),
                output.size,
                self._kernel_value(kernel),
            )
        )
        return output

    def matmul(
        self,
        encoded: NativeEncodedMatrix,
        values: np.ndarray,
        *,
        kernel: str = "auto",
    ) -> np.ndarray:
        self._validate_encoded_buffers(encoded)
        if not isinstance(values, np.ndarray) or values.dtype != np.float32:
            raise TypeError("native matmul input must be a float32 NumPy array")
        if values.ndim != 2 or values.shape[1] != encoded.in_features:
            raise ValueError(
                "native matmul input must have shape [batch, in_features]"
            )
        if not values.flags.c_contiguous:
            raise ValueError("native matmul input must be C-contiguous")
        qdata = encoded.qdata.view(np.uint8)
        output = np.empty((values.shape[0], encoded.out_features), dtype=np.float32)
        self._check(
            self._library.rq_native_v2_matmul(
                values.ctypes.data_as(_FLOAT_POINTER),
                values.size,
                values.shape[0],
                qdata.ctypes.data_as(_BYTE_POINTER),
                qdata.size,
                encoded.codebook.ctypes.data_as(_FLOAT_POINTER),
                encoded.codebook.size,
                encoded.out_features,
                encoded.in_features,
                _CLayout(encoded.layout.bits, encoded.layout.group_size),
                output.ctypes.data_as(_FLOAT_POINTER),
                output.size,
                self._kernel_value(kernel),
            )
        )
        return output

    def register_kernels(
        self,
        registry: KernelRegistry,
        *,
        backend: str = "native-cpu",
        kernel: str = "auto",
        replace: bool = False,
    ) -> tuple[KernelSpec, KernelSpec]:
        """Register this library's resolved dequantize and matmul kernels."""

        if not isinstance(registry, KernelRegistry):
            raise TypeError("registry must be a KernelRegistry")
        resolved = self.resolve_kernel(kernel)
        capability = next(
            (
                item
                for item in self.capabilities()
                if item.kernel == resolved
            ),
            None,
        )
        if capability is None:
            raise RuntimeError(
                f"resolved kernel {resolved!r} is absent from capabilities"
            )
        group_sizes = (
            None
            if capability.group_size is None
            else frozenset({capability.group_size})
        )

        def dequantize_kernel(encoded: NativeEncodedMatrix) -> np.ndarray:
            return self.dequantize(encoded, kernel=resolved)

        def matmul_kernel(
            encoded: NativeEncodedMatrix, values: np.ndarray
        ) -> np.ndarray:
            return self.matmul(encoded, values, kernel=resolved)

        dequantize_spec = KernelSpec(
            name=f"{backend}-{resolved}-dequantize",
            backend=backend,
            operation="dequantize",
            bits=capability.bits,
            group_sizes=group_sizes,
            implementation=dequantize_kernel,
            priority=100,
        )
        matmul_spec = KernelSpec(
            name=f"{backend}-{resolved}-matmul",
            backend=backend,
            operation="matmul",
            bits=capability.bits,
            group_sizes=group_sizes,
            implementation=matmul_kernel,
            priority=100,
        )
        registry.register(dequantize_spec, replace=replace)
        registry.register(matmul_spec, replace=replace)
        return dequantize_spec, matmul_spec


__all__ = [
    "NATIVE_C_ABI_VERSION",
    "NativeKernelCapability",
    "NativeRuntimeError",
    "NativeRuntimeLibrary",
]
