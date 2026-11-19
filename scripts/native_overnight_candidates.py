"""Frozen experiment registry: no defaults or existing exact gates are changed.

Version 2 adds the kernels of ``rq3-v2.cuh`` and screens only those. The
version-1 candidates stay selectable by name (their CUDA code is unchanged)
but are superseded: every v1 GEMM has a v2 counterpart with the same
arithmetic contract and a faster decode, and ``v2-head`` supersedes
``head-warp``.

Lanes:

* ``exact``: outputs must be bitwise equal to the reference kernel.
* ``reordered``: same operands, different summation order; held to the
  canonical operator tolerances against the reference.
* ``rounding``: the same linear map with FP16 rounding in a different place
  (the rotated-basis head); held to the bar the reference itself meets against
  the unrounded canonical result.
"""
from __future__ import annotations

REGISTRY_VERSION = "rq3-overnight-candidates-v2"
CONTROL = "w5s8-tile8"
SCRATCH_LIMIT = 256 * 1024**2

# ``fused`` builds the backbone graph with one RQ3_FUSED op per projection
# (ROTQUANT_RQ3_FUSE=1) instead of a rotation op followed by a matmul op.
_DEFAULTS = {"rows": 0, "half": False, "head": False, "matmul_v2": False, "gemm_v2": 0,
             "head_v2": False, "head_rotated": False, "fused": False}


def _spec(identifier, lane, role, speed_tokens=(), **values):
    # Lists, not tuples: specs are stored in run controls, which are compared
    # with their JSON round trip on resume.
    return {"id": identifier, **_DEFAULTS, **values, "lane": lane, "role": role,
            "speed_tokens": list(speed_tokens)}


CANDIDATES = {
    # Version 1 (rq3-experiment.cuh). Kept for reproducibility; not screened.
    "head-warp": _spec(1, "exact", "head", (1, 16), head=True),
    "sgemm-128": _spec(2, "reordered", "backbone", (16, 128), rows=128),
    "sgemm-512": _spec(3, "reordered", "backbone", (16, 128), rows=512),
    "hgemm-128": _spec(4, "reordered", "backbone", (16, 128), rows=128, half=True),
    "hgemm-512": _spec(5, "reordered", "backbone", (16, 128), rows=512, half=True),
    "hgemm-2048": _spec(6, "reordered", "backbone", (16, 128), rows=2048, half=True),
    "hgemm-512-head": _spec(7, "reordered", "combined", (16, 128), rows=512, half=True, head=True),
    # Version 2 (rq3-v2.cuh).
    "v2-matmul": _spec(8, "exact", "backbone", (1, 16, 128), matmul_v2=True, fused=True),
    "v2-head": _spec(9, "exact", "head", (1, 16), head_v2=True),
    "v2-sgemm": _spec(10, "reordered", "backbone", (1, 16, 128), matmul_v2=True, gemm_v2=1, fused=True),
    "v2-hgemm": _spec(11, "reordered", "backbone", (1, 16, 128), matmul_v2=True, gemm_v2=2, fused=True),
    "v2-head-rotated": _spec(12, "rounding", "head", (1, 16), head_rotated=True),
    "v2-exact": _spec(13, "exact", "combined", (1, 16, 128), matmul_v2=True, head_v2=True, fused=True),
    "v2-fast": _spec(14, "rounding", "combined", (1, 16, 128), matmul_v2=True, gemm_v2=2, head_rotated=True,
                     fused=True),
    "v2-fast-sgemm": _spec(15, "rounding", "combined", (1, 16, 128), matmul_v2=True, gemm_v2=1,
                           head_rotated=True, fused=True),
}
SCREEN_CANDIDATES = ("v2-matmul", "v2-head", "v2-sgemm", "v2-hgemm", "v2-head-rotated")
# A combination is screened only when every component is eligible, and at most
# one combination per group goes forward, in this order of preference.
COMBINATIONS = {
    "v2-exact": {"group": "exact", "parts": ["v2-matmul", "v2-head"]},
    "v2-fast": {"group": "fast", "parts": ["v2-hgemm", "v2-head-rotated"]},
    "v2-fast-sgemm": {"group": "fast", "parts": ["v2-sgemm", "v2-head-rotated"]},
}
# These are the EXISTING canonical operator tolerances, not tuned to GPU results.
# A rejection is evidence, never an invitation to loosen them mid-run.
OPERATOR_RTOL, OPERATOR_ATOL = .002, .001
# Rounding lane: aggregate error against the unrounded canonical result may be
# at most this multiple of the reference kernel's own error on the same case.
ROUNDING_ERROR_RATIO = 2.0


def requested_scratch(candidate, rows, width, tokens, mode=0):
    """Bytes the CUDA dispatch requests from the pool; mirrors rq3.cu exactly."""
    if min(rows, width, tokens) <= 0:
        raise ValueError("Invalid dimensions")
    spec = CANDIDATES[candidate]
    if mode == 1 and spec["head_rotated"]:
        return width * tokens * 4
    if mode != 0:
        return 0
    if spec["gemm_v2"] and tokens >= 2:
        item = 2 if spec["gemm_v2"] == 2 else 4
        inputs, row = width * tokens * item, width * item
        if inputs + row + tokens * 4 > SCRATCH_LIMIT:
            raise ValueError("Shape exceeds bounded scratch contract")
        if inputs + rows * row <= SCRATCH_LIMIT:
            return inputs + rows * row
        chunk = (SCRATCH_LIMIT - inputs) // (row + tokens * 4)
        return inputs + chunk * row + chunk * tokens * 4
    if not spec["rows"] or tokens < 4:
        return 0
    item = 2 if spec["half"] else 4
    inputs, per_row = width * tokens * item, width * item + tokens * 4
    if inputs + per_row > SCRATCH_LIMIT:
        raise ValueError("Shape exceeds bounded scratch contract")
    return inputs + min(rows, spec["rows"], (SCRATCH_LIMIT-inputs)//per_row) * per_row


def experimental_dispatch(candidate, mode, tokens):
    """Whether the candidate's own code handles this operator call."""
    spec = CANDIDATES[candidate]
    if mode == 1:
        return bool(spec["head"] or spec["head_v2"] or spec["head_rotated"])
    if mode == 0:
        return bool(spec["matmul_v2"] or spec["gemm_v2"] or (spec["rows"] > 0 and tokens >= 4))
    return False


def operator_lane(candidate, mode, tokens):
    """The parity rule for one operator call: exact, reordered or rounding."""
    spec = CANDIDATES[candidate]
    if not experimental_dispatch(candidate, mode, tokens):
        return "exact"
    if not (spec["matmul_v2"] or spec["gemm_v2"] or spec["head_v2"] or spec["head_rotated"]):
        return spec["lane"]
    if mode == 1:
        return "rounding" if spec["head_rotated"] else "exact"
    return "reordered" if spec["gemm_v2"] and tokens >= 2 else "exact"


class ExperimentDiagnostics:
    """Resolve through the actual executing dependency, never load a second DSO."""
    def __init__(self, library):
        import ctypes as C

        from scripts.native_cuda_diagnostics import CudaDiagnostics, symbol_owner
        self.base = CudaDiagnostics(library)
        self.read = self.base.library.ggml_cuda_rq3_experiment_stat
        self.read.argtypes, self.read.restype = [C.c_int], C.c_uint64
        if str(symbol_owner(self.read)) != self.base.binding["provider"]:
            raise ValueError("Experimental counter provider mismatch")

    def snapshot(self, candidate):
        return {"calls": self.read(CANDIDATES[candidate]["id"]), "scratch_requested_peak_bytes": self.read(0),
                "binding": self.base.binding}

    def require(self, before, candidate, *, mode=None, tokens=None):
        after = self.snapshot(candidate)
        expected = mode is None or experimental_dispatch(candidate, mode, tokens)
        calls = after["calls"] - before["calls"]
        if (expected and calls <= 0) or (not expected and calls != 0):
            raise ValueError("Candidate dispatch missing or applied to an unsupported operation")
        if after["scratch_requested_peak_bytes"] > SCRATCH_LIMIT:
            raise ValueError("Candidate exceeded requested scratch ceiling")
        return {**after, "calls": calls, "expected": bool(expected)}
