"""Snapshots of explicitly instrumented CUDA custom operators, not serving rates."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

from scripts.native_hashing import digest

NAMES = ("rotation", "matrix", "vocabulary_head", "embedding")


class _DlInfo(C.Structure):
    _fields_ = [("filename", C.c_char_p), ("base", C.c_void_p),
                ("symbol", C.c_char_p), ("address", C.c_void_p)]


def symbol_owner(function):
    """Identify the mapped object supplying a symbol, without loading an alias."""
    resolve = C.CDLL(None).dladdr
    resolve.argtypes = [C.c_void_p, C.POINTER(_DlInfo)]
    resolve.restype = C.c_int
    info = _DlInfo()
    if not resolve(C.cast(function, C.c_void_p), C.byref(info)) or not info.filename:
        raise RuntimeError("Cannot identify the loaded CUDA diagnostic provider")
    return Path(os.fsdecode(info.filename)).resolve(strict=True)


class CudaDiagnostics:
    def __init__(self, library):
        # Resolve through the execution library's dependency handle. Cached
        # SONAME aliases are regular copies: opening libggml-cuda.so directly
        # can instantiate a second DSO with untouched process-local counters.
        path = Path(library).resolve(strict=True)
        self.library = C.CDLL(str(path))
        self.read = self.library.ggml_cuda_rq3_diagnostics
        self.read.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
        self.read.restype = C.c_int
        self.decode_calls = self.library.ggml_cuda_rq3_decode_dispatches
        self.decode_calls.argtypes = []
        self.decode_calls.restype = C.c_uint64
        provider = symbol_owner(self.read)
        if (symbol_owner(self.decode_calls) != provider or provider.parent != path.parent
                or not (provider.name == "libggml-cuda.so" or provider.name.startswith("libggml-cuda.so."))):
            raise RuntimeError("CUDA diagnostic symbols are not owned by this runtime's CUDA dependency")
        self.binding = {"lookup": "execution-library dependency handle", "library": str(path),
                        "provider": str(provider), "provider_sha256": digest(provider)}

    def snapshot(self):
        ms, calls, tiled = (C.c_double * 4)(), (C.c_uint64 * 4)(), C.c_uint64()
        if self.read(ms, calls, C.byref(tiled)) != 1:
            raise RuntimeError("Native CUDA diagnostic snapshot failed")
        return {"operators": {name: {"milliseconds": ms[i], "host_dispatches": calls[i]}
                              for i, name in enumerate(NAMES)},
                "tiled_host_dispatches": tiled.value,
                "decode_host_dispatches": self.decode_calls(),
                "binding": self.binding,
                "instrumented": os.environ.get("ROTQUANT_RQ3_PROFILE") == "1"}


def difference(before, after):
    return {"operators": {name: {key: after["operators"][name][key] - before["operators"][name][key]
                                for key in ("milliseconds", "host_dispatches")} for name in NAMES},
            "tiled_host_dispatches": after["tiled_host_dispatches"] - before["tiled_host_dispatches"],
            "decode_host_dispatches": after.get("decode_host_dispatches", 0) - before.get("decode_host_dispatches", 0),
            "instrumented": after["instrumented"]}
