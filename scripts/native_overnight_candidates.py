"""Frozen experiment registry: no defaults or existing exact gates are changed."""
from __future__ import annotations

REGISTRY_VERSION = "rq3-overnight-candidates-v1"
CONTROL = "w5s8-tile8"
SCRATCH_LIMIT = 256 * 1024**2
CANDIDATES = {
    "head-warp": {"id": 1, "rows": 0, "half": False, "head": True, "lane": "exact"},
    "sgemm-128": {"id": 2, "rows": 128, "half": False, "head": False, "lane": "reordered"},
    "sgemm-512": {"id": 3, "rows": 512, "half": False, "head": False, "lane": "reordered"},
    "hgemm-128": {"id": 4, "rows": 128, "half": True, "head": False, "lane": "reordered"},
    "hgemm-512": {"id": 5, "rows": 512, "half": True, "head": False, "lane": "reordered"},
    "hgemm-2048": {"id": 6, "rows": 2048, "half": True, "head": False, "lane": "reordered"},
    "hgemm-512-head": {"id": 7, "rows": 512, "half": True, "head": True, "lane": "reordered"},
}
SCREEN_CANDIDATES = tuple(k for k in CANDIDATES if k != "hgemm-512-head")
# These are the EXISTING canonical operator tolerances, not tuned to GPU results.
# A rejection is evidence, never an invitation to loosen them mid-run.
OPERATOR_RTOL, OPERATOR_ATOL = .002, .001


def requested_scratch(candidate, rows, width, tokens):
    if min(rows, width, tokens) <= 0:
        raise ValueError("Invalid dimensions")
    spec = CANDIDATES[candidate]
    if not spec["rows"] or tokens < 4:
        return 0
    item = 2 if spec["half"] else 4
    inputs, per_row = width * tokens * item, width * item + tokens * 4
    if inputs + per_row > SCRATCH_LIMIT:
        raise ValueError("Shape exceeds bounded scratch contract")
    return inputs + min(rows, spec["rows"], (SCRATCH_LIMIT-inputs)//per_row) * per_row


def experimental_dispatch(candidate, mode, tokens):
    spec = CANDIDATES[candidate]
    return (mode == 1 and spec["head"]) or (mode == 0 and spec["rows"] > 0 and tokens >= 4)


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
