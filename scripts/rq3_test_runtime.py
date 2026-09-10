"""Explicit native test bindings. Not registered as a production RotQuant backend."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np
import torch  # noqa: F401 - load Torch/CUDA dependencies before the native library.

from rotquant.native_v3 import NativeV3Matrix


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

    def operator(self, backend, matrix: NativeV3Matrix, signs, inputs, mode=0, rows=None, columns=None):
        matrix.validate()
        if matrix.layout.group_size != 128 or matrix.layout.in_features % 128:
            raise ValueError("operator tests require g128/aligned inputs")
        if mode not in (0, 1, 2):
            raise ValueError("invalid operator mode")
        inputs = np.ascontiguousarray(inputs)
        if inputs.dtype != (np.int32 if mode == 2 else np.float32):
            raise ValueError("expected int32 token IDs or float32 activations")
        if mode == 2:
            if inputs.ndim != 1:
                raise ValueError("token IDs must be one-dimensional")
        elif inputs.ndim != 2 or inputs.shape[1] != matrix.layout.in_features or not np.isfinite(inputs).all():
            raise ValueError("invalid activation shape/values")
        signs = np.ascontiguousarray(signs)
        if signs.dtype != np.int8 or signs.shape != (matrix.layout.in_features,):
            raise ValueError("invalid sign tensor")
        for value, count in ((rows, matrix.layout.out_features), (columns, matrix.layout.in_features)):
            if value is not None and (value.dtype != np.int32 or value.shape != (count,) or not value.flags.c_contiguous):
                raise ValueError("invalid permutation tensor")
        raw = np.frombuffer(matrix.to_bytes(), dtype=np.uint8)
        output = np.empty((len(inputs), matrix.layout.in_features if mode == 2 else matrix.layout.out_features), dtype=np.float32)
        ptr = lambda x: None if x is None else x.ctypes.data
        status = self.lib.rq3_test_eval(backend.encode(), ptr(raw), raw.nbytes, matrix.layout.out_features,
            matrix.layout.in_features, len(inputs), mode, ptr(inputs), ptr(signs), ptr(rows), ptr(columns), ptr(output))
        if status:
            raise RuntimeError(self.lib.rq3_test_error().decode())
        return output

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

    def evaluate(self, ids, *, reset=False, selected=1):
        if not self.handle:
            raise RuntimeError("model already closed")
        ids = np.asarray(ids)
        if (ids.dtype.kind not in "iu" or ids.ndim != 1 or not 0 < len(ids) <= 8192
                or (ids < 0).any() or (ids >= self.vocab).any() or not 1 <= selected <= min(16, len(ids))):
            raise ValueError("invalid bounded token probe")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        output = np.empty((selected, self.vocab), dtype=np.float32)
        if self.runtime.lib.rq3_model_eval_selected(self.handle, ids.ctypes.data, len(ids), reset, selected, output.ctypes.data):
            raise RuntimeError(self.runtime.lib.rq3_model_error().decode())
        return output[0] if selected == 1 else output

    def is_eog(self, token):
        if not self.handle or not 0 <= token < self.vocab:
            raise ValueError("invalid model/token")
        return bool(self.runtime.lib.rq3_model_is_eog(self.handle, token))

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
