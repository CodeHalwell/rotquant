"""Explicit scalar CPU binding for native-v3 matrices, not a model backend."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from .native_v3 import NATIVE_V3_VERSION, NativeV3Matrix, positive_integer

_FLOAT = ctypes.POINTER(ctypes.c_float)
_BYTE = ctypes.POINTER(ctypes.c_uint8)
_SIZE = ctypes.c_size_t


class NativeV3Runtime:
    """Load explicitly; no GPU dispatch, fallback or implicit model conversion."""

    def __init__(self, library: str | Path):
        self.path = Path(library).resolve(strict=True)
        self.library = ctypes.CDLL(str(self.path))
        for name in ("abi_version", "format_version"):
            function = getattr(self.library, "rq_native_v3_" + name)
            function.argtypes, function.restype = [], ctypes.c_uint32
        if (self.library.rq_native_v3_abi_version() != 1
                or self.library.rq_native_v3_format_version() != NATIVE_V3_VERSION):
            raise RuntimeError("incompatible native-v3 ABI or format")
        self.library.rq_native_v3_last_error.argtypes = []
        self.library.rq_native_v3_last_error.restype = ctypes.c_char_p
        self.library.rq_native_v3_dequantize.argtypes = [_BYTE, _SIZE, _FLOAT, _SIZE, _SIZE, _SIZE]
        self.library.rq_native_v3_dequantize.restype = ctypes.c_int
        self.library.rq_native_v3_matmul.argtypes = [_BYTE, _SIZE, _FLOAT, _SIZE, _SIZE, _FLOAT, _SIZE]
        self.library.rq_native_v3_matmul.restype = ctypes.c_int

    def check(self, status):
        if status:
            detail = self.library.rq_native_v3_last_error().decode("utf-8", errors="replace")
            raise RuntimeError(f"native-v3 status {status}: {detail}")

    def prepare(self, matrix: NativeV3Matrix):
        return PreparedNativeV3Matrix(self, matrix)


class PreparedNativeV3Matrix:
    """Own one compact buffer copy for repeated calls; never cache dense weights.

    Preparation is an export-time operation. Each matmul allocates only its
    output; the C++ reference still validates compact scales each invocation.
    This is a conformance floor, not an optimized production matmul.
    """

    def __init__(self, runtime: NativeV3Runtime, matrix: NativeV3Matrix):
        if not isinstance(matrix, NativeV3Matrix):
            raise TypeError("expected a NativeV3Matrix")
        payload = matrix.to_bytes()
        self.runtime, self.layout = runtime, matrix.layout
        self.persistent_bytes = len(payload)
        self._buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)

    def dequantize_rows(self, start=0, count=None):
        count = self.layout.out_features - start if count is None else count
        positive_integer(count, "count")
        if type(start) is not int or start < 0 or start + count > self.layout.out_features:
            raise ValueError("invalid row range")
        output = np.empty((count, self.layout.in_features), dtype=np.float32)
        self.runtime.check(self.runtime.library.rq_native_v3_dequantize(
            self._buffer, self.persistent_bytes, output.ctypes.data_as(_FLOAT),
            output.size, start, count))
        return output

    def matmul(self, values: np.ndarray):
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32"):
            raise TypeError("input must already be float32; no hidden conversion")
        if (values.ndim != 2 or not values.flags.c_contiguous or not values.flags.aligned
                or values.shape[0] < 1
                or values.shape[1] != self.layout.in_features):
            raise ValueError("expected aligned contiguous [batch, in_features] input")
        output = np.empty((values.shape[0], self.layout.out_features), dtype=np.float32)
        self.runtime.check(self.runtime.library.rq_native_v3_matmul(
            self._buffer, self.persistent_bytes, values.ctypes.data_as(_FLOAT), values.size,
            values.shape[0], output.ctypes.data_as(_FLOAT), output.size))
        return output
