"""Explicit native test bindings. Not registered as a production RotQuant backend."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np
import torch  # noqa: F401 - load Torch/CUDA dependencies before the native library.

from rotquant.native_v3 import NativeV3Matrix


def validate_operator_layout(matrix, mode):
    matrix.validate()
    if (matrix.layout.group_size not in (32, 64, 128) or matrix.layout.in_features % 128
            or (mode and matrix.layout.group_size != 128)):
        raise ValueError("operator tests require g32/g64/g128 aligned inputs, vocabulary g128")
    if mode not in (0, 1, 2):
        raise ValueError("invalid operator mode")


def validate_operator_maps(matrix, signs, rows, columns):
    signs = np.ascontiguousarray(signs)
    if signs.dtype != np.int8 or signs.shape != (matrix.layout.in_features,):
        raise ValueError("invalid sign tensor")
    for value, count in ((rows, matrix.layout.out_features), (columns, matrix.layout.in_features)):
        if value is not None and (value.dtype != np.int32 or value.shape != (count,) or not value.flags.c_contiguous):
            raise ValueError("invalid permutation tensor")
    return signs


def validate_operator(matrix, signs, inputs, mode, rows, columns):
    validate_operator_layout(matrix, mode)
    inputs = np.ascontiguousarray(inputs)
    if inputs.dtype != (np.int32 if mode == 2 else np.float32):
        raise ValueError("expected int32 token IDs or float32 activations")
    if mode == 2:
        if inputs.ndim != 1:
            raise ValueError("token IDs must be one-dimensional")
    elif inputs.ndim != 2 or inputs.shape[1] != matrix.layout.in_features or not np.isfinite(inputs).all():
        raise ValueError("invalid activation shape/values")
    return inputs, validate_operator_maps(matrix, signs, rows, columns)


class NativeTests:
    def __init__(self, library: Path):
        self.library = Path(library).resolve(strict=True)
        self.lib = C.CDLL(str(self.library))
        self.lib.rq3_test_error.restype = C.c_char_p
        self.lib.rq3_model_error.restype = C.c_char_p
        self.lib.rq3_test_eval.argtypes = [C.c_char_p, C.c_void_p, C.c_size_t,
            C.c_int64, C.c_int64, C.c_int64, C.c_int, *([C.c_void_p] * 5)]
        self.lib.rq3_test_eval.restype = C.c_int
        self.lib.rq3_model_open.argtypes = [C.c_char_p, C.c_char_p, C.c_uint32]
        self.lib.rq3_model_open.restype = C.c_void_p
        self.lib.rq3_model_vocab.argtypes = [C.c_void_p]
        self.lib.rq3_model_vocab.restype = C.c_int
        self.lib.rq3_model_eval.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_bool, C.c_void_p]
        self.lib.rq3_model_eval.restype = C.c_int
        self.lib.rq3_model_eval_selected.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_bool, C.c_int32, C.c_void_p]
        self.lib.rq3_model_eval_selected.restype = C.c_int
        self.lib.rq3_model_is_eog.argtypes = [C.c_void_p, C.c_int32]
        self.lib.rq3_model_is_eog.restype = C.c_bool
        self.lib.rq3_model_custom_ops.argtypes = [C.c_void_p]
        self.lib.rq3_model_custom_ops.restype = C.c_uint64
        self.lib.rq3_model_close.argtypes = [C.c_void_p]
        self.lib.rq3_model_close.restype = None
        self.capabilities = getattr(self.lib, "rq3_model_capabilities", None)
        self.compact = getattr(self.lib, "rq3_model_greedy", None)
        if self.capabilities:
            self.capabilities.argtypes, self.capabilities.restype = [C.c_void_p], C.c_uint32
        if self.compact:
            self.compact.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_bool, C.c_void_p]
            self.compact.restype = C.c_int


    def operator(self, backend, matrix: NativeV3Matrix, signs, inputs, mode=0, rows=None, columns=None, *, benchmark=None):
        inputs, signs = validate_operator(matrix, signs, inputs, mode, rows, columns)
        raw = np.frombuffer(matrix.to_bytes(), dtype=np.uint8)
        output = np.empty((len(inputs), matrix.layout.in_features if mode == 2 else matrix.layout.out_features), dtype=np.float32)
        ptr = lambda x: None if x is None else x.ctypes.data
        arguments = [backend.encode(), ptr(raw), raw.nbytes, matrix.layout.out_features,
            matrix.layout.in_features, len(inputs), mode, ptr(inputs), ptr(signs), ptr(rows), ptr(columns), ptr(output)]
        return self.execute_operator(arguments, output, benchmark)

    def execute_operator(self, arguments, output, benchmark):
        seconds = None
        if benchmark is not None:
            repeats, iterations = benchmark
            if (type(repeats) is not int or type(iterations) is not int
                    or not 1 <= repeats <= 20 or not 1 <= iterations <= 100):
                raise ValueError("bounded integer benchmark repetitions/iterations required")
            function = self.lib.rq3_test_benchmark
            function.argtypes = [*self.lib.rq3_test_eval.argtypes, C.c_int, C.c_int, C.c_void_p]
            function.restype = C.c_int
            seconds = np.empty(repeats, dtype=np.float64)
            status = function(*arguments, repeats, iterations, seconds.ctypes.data)
        else:
            status = self.lib.rq3_test_eval(*arguments)
        if status:
            raise RuntimeError(self.lib.rq3_test_error().decode())
        if seconds is not None:
            if not np.isfinite(seconds).all() or (seconds <= 0).any():
                raise RuntimeError("invalid native benchmark durations")
            return output, seconds.tolist()
        return output

    def greedy_operator(self, backend, logits):
        values = np.ascontiguousarray(logits, dtype=np.float32)
        if values.ndim != 1 or not 0 < len(values) <= 1048576:
            raise ValueError("invalid greedy fixture shape")
        function = self.lib.rq3_test_greedy
        function.argtypes = [C.c_char_p, C.c_void_p, C.c_int64, C.c_void_p]
        function.restype = C.c_int
        token = C.c_int32(-1)
        if function(backend.encode(), values.ctypes.data, len(values), C.byref(token)):
            raise RuntimeError(self.lib.rq3_test_error().decode())
        return token.value

    def model(self, path: Path, backend: str, context=512):
        return NativeModel(self, path, backend, context)


class NativeModel:
    def __init__(self, runtime, path, backend, context):
        self.runtime = runtime
        self.backend = backend
        self.handle = runtime.lib.rq3_model_open(os.fsencode(Path(path).resolve(strict=True)), backend.encode(), context)
        if not self.handle:
            raise RuntimeError(runtime.lib.rq3_model_error().decode())
        self.vocab = runtime.lib.rq3_model_vocab(self.handle)
        capabilities = getattr(runtime, "capabilities", None)
        flags = capabilities(self.handle) if capabilities else 0
        self.supports_no_logits = bool(flags & 1)
        self.supports_compact_greedy = bool(flags & 2) and getattr(runtime, "compact", None) is not None


    def evaluate(self, ids, *, reset=False, selected=1):
        if not self.handle:
            raise RuntimeError("model already closed")
        ids = np.asarray(ids)
        if (ids.dtype.kind not in "iu" or ids.ndim != 1 or not 0 < len(ids) <= 8192
                or (ids < 0).any() or (ids >= self.vocab).any() or type(selected) is not int or not 0 <= selected <= min(16, len(ids))):
            raise ValueError("invalid bounded token probe")
        if selected == 0 and not self.supports_no_logits:
            raise RuntimeError("runtime lacks graph-level no-logits prefill")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        output = np.empty((selected, self.vocab), dtype=np.float32)
        if self.runtime.lib.rq3_model_eval_selected(self.handle, ids.ctypes.data, len(ids), reset, selected, output.ctypes.data):
            raise RuntimeError(self.runtime.lib.rq3_model_error().decode())
        return output[0] if selected == 1 else output

    def greedy(self, ids, *, reset=False):
        if not self.handle or not self.supports_compact_greedy:
            raise RuntimeError("compact finite-checked sampling unavailable")
        values = np.asarray(ids)
        if (values.dtype.kind not in "iu" or values.ndim != 1 or not 0 < len(values) <= 8192
                or (values < 0).any() or (values >= self.vocab).any()):
            raise ValueError("invalid compact token probe")
        values = np.ascontiguousarray(values, dtype=np.int32)
        token = C.c_int32(-1)
        if self.runtime.compact(self.handle, values.ctypes.data, len(values), reset, C.byref(token)):
            raise RuntimeError(self.runtime.lib.rq3_model_error().decode())
        if not 0 <= token.value < self.vocab:
            raise RuntimeError("native sampler returned an invalid token")
        return token.value

    def is_eog(self, token):
        if not self.handle or not 0 <= token < self.vocab:
            raise ValueError("invalid model/token")
        return bool(self.runtime.lib.rq3_model_is_eog(self.handle, token))

    @property
    def inference_diagnostics(self):
        function = getattr(self.runtime.lib, "rq3_model_diagnostics", None)
        if not function:
            raise RuntimeError("native inference diagnostics unavailable")
        function.argtypes, function.restype = [C.c_void_p, C.c_void_p], C.c_int
        values = (C.c_uint64 * 4)()
        if not self.handle or function(self.handle, values):
            raise RuntimeError("native inference diagnostics failed")
        return dict(zip(("full_logit_rows", "compact_tokens", "no_logits_batches", "head_ops"), values))

    @property
    def custom_ops(self):
        return self.runtime.lib.rq3_model_custom_ops(self.handle)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.handle:
            self.runtime.lib.rq3_model_close(self.handle)
            self.handle = None
