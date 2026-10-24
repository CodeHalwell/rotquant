"""Versioned opt-in inference controls and device-executed dispatch receipts."""
from __future__ import annotations

import ctypes as C
import os
from dataclasses import asdict, dataclass
from pathlib import Path

KERNELS = ("reference", "decode4", "tile8", "head-warp", "sgemm-128", "sgemm-512",
           "hgemm-128", "hgemm-512", "hgemm-2048", "hgemm-512-head", "ternary-g128", "ternary-w2a8", "tensorcore-fused")
FORMATS = ("w2_g128_s16", "w4_s8", "w6", "other")
PATHS = ("reference", "decode4", "tiled4", "tile8", "head-warp", "staged-gemm",
         "ternary-g128", "ternary-w2a8", "tensorcore-fused")


def arithmetic_contract(kernel):
    if kernel == "ternary-w2a8":
        return "fused-fwht128-half-boundaries-group-q8-roundeven-dp4a-new-numerical-lane"
    return "same-half-operands-reordered" if "gemm" in kernel or kernel == "tensorcore-fused" else "exact-half-operands"


@dataclass(frozen=True)
class InferenceProfile:
    kernel: str = "reference"
    graph_prefill: bool = False
    compact_greedy: bool = False
    arithmetic: str = "w2a16"
    version: int = 2

    def __post_init__(self):
        if (self.kernel not in KERNELS or type(self.version) is not int or self.version != 2
                or self.arithmetic not in ("w2a16", "w2a8")
                or (self.kernel == "ternary-w2a8") != (self.arithmetic == "w2a8")):
            raise ValueError("unsupported inference kernel/arithmetic/version; W2A8 is a separate numerical lane")
        if type(self.graph_prefill) is not bool or type(self.compact_greedy) is not bool:
            raise ValueError("inference capability choices must be booleans")

    @classmethod
    def from_mapping(cls, values=None):
        if values is not None and not isinstance(values, dict):
            raise ValueError("inference profile must be a mapping")
        return cls(**(values or {}))

    def identity(self):
        return {"protocol": "rotquant-inference-v2", **asdict(self),
                "arithmetic_contract": arithmetic_contract(self.kernel),
                "dispatch_evidence": "device-executed-counts-including-CUDA-graph-replays"}

    def environment(self):
        return {"ROTQUANT_RQ3_KERNEL": self.kernel}

    def apply(self):
        os.environ.update(self.environment())

    def wrap(self, model):
        from scripts.w4_research_measure import ChunkedNativeModel
        return ChunkedNativeModel(model, graph_prefill=self.graph_prefill,
                                  compact_greedy=self.compact_greedy)


class DispatchReceipt:
    def __init__(self, library):
        self.library = C.CDLL(str(library))
        self.read = self.library.ggml_cuda_rq3_dispatch_v2
        self.read.argtypes, self.read.restype = [C.c_void_p, C.c_int], C.c_int
        from scripts.native_cuda_diagnostics import symbol_owner
        path = Path(library).resolve(strict=True)
        provider = symbol_owner(self.read)
        if provider.parent != path.parent or not provider.name.startswith("libggml-cuda.so"):
            raise RuntimeError("dispatch receipt belongs to a different CUDA runtime")
        from scripts.native_hashing import digest
        self.binding = {"provider": str(provider), "provider_sha256": digest(provider)}

    def snapshot(self, *, reset=False):
        values = (C.c_uint64 * (len(FORMATS) * len(PATHS)))()
        if self.read(values, int(reset)) != 1:
            raise RuntimeError("device dispatch receipt unavailable")
        return {fmt: dict(zip(PATHS, map(int, values[i * len(PATHS):(i + 1) * len(PATHS)])))
                for i, fmt in enumerate(FORMATS)}


def require_dispatch(receipt, expected):
    for fmt, paths in expected.items():
        if fmt not in FORMATS or not paths or any(path not in PATHS for path in paths):
            raise ValueError("invalid format-specific dispatch expectation")
        if any(receipt.get(fmt, {}).get(path, 0) < 1 for path in paths):
            raise RuntimeError(f"requested kernel did not execute for {fmt}: {paths}")
    return receipt


def matrix_dispatch_path(profile, *, ternary=False):
    kernel = profile.kernel
    if ternary and kernel in ("ternary-g128", "ternary-w2a8"):
        return kernel
    if kernel == "tensorcore-fused":
        return "tensorcore-fused"
    if "gemm" in kernel:
        return "staged-gemm"
    if kernel in ("tile8", "head-warp", "ternary-w2a8"):
        return "tile8"
    return "tiled4" if kernel == "decode4" else "decode4" if kernel == "ternary-g128" else "reference"


def matrix_dispatch_paths(profile, *, ternary=False):
    prefill = matrix_dispatch_path(profile, ternary=ternary)
    return [prefill] if prefill in ("reference", "decode4", "ternary-g128", "ternary-w2a8") else [prefill, "decode4"]


def expected_model_dispatch(profile, label):
    if label == "fp16":
        return {}
    head = "head-warp" if profile.kernel in ("head-warp", "hgemm-512-head", "tensorcore-fused") else "reference"
    if label.startswith("ternary"):
        return {"w2_g128_s16": matrix_dispatch_paths(profile, ternary=True),
                "w4_s8": matrix_dispatch_paths(profile), "w6": [head]}
    formats = {"w4": "w4_s8", "w5": "other", "w6": "w6"}
    prefix = label.split("_", 1)[0]
    if prefix not in formats:
        raise ValueError(f"unknown model format label: {label}")
    result = {formats[prefix]: matrix_dispatch_paths(profile)}
    result["w6"] = list(dict.fromkeys([*result.get("w6", []), head]))
    return result
