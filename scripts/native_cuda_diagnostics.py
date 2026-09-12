"""Snapshots of explicitly instrumented CUDA custom operators, not serving rates."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

NAMES = ("rotation", "matrix", "vocabulary_head", "embedding")


class CudaDiagnostics:
    def __init__(self, library):
        path = Path(library).resolve().parent / "libggml-cuda.so"
        self.library = C.CDLL(str(path))
        self.read = self.library.ggml_cuda_rq3_diagnostics
        self.read.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
        self.read.restype = C.c_int

    def snapshot(self):
        ms, calls, tiled = (C.c_double * 4)(), (C.c_uint64 * 4)(), C.c_uint64()
        if self.read(ms, calls, C.byref(tiled)) != 1:
            raise RuntimeError("Native CUDA diagnostic snapshot failed")
        return {"operators": {name: {"milliseconds": ms[i], "host_dispatches": calls[i]}
                              for i, name in enumerate(NAMES)},
                "tiled_host_dispatches": tiled.value,
                "instrumented": os.environ.get("ROTQUANT_RQ3_PROFILE") == "1"}


def difference(before, after):
    return {"operators": {name: {key: after["operators"][name][key] - before["operators"][name][key]
                                for key in ("milliseconds", "host_dispatches")} for name in NAMES},
            "tiled_host_dispatches": after["tiled_host_dispatches"] - before["tiled_host_dispatches"],
            "instrumented": after["instrumented"]}
