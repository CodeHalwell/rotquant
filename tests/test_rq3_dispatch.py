"""Dispatch-gate logic with explicit mocks; never evidence of CUDA execution."""
import copy
import json
import os
import sys
import types

import numpy as np
import pytest

from scripts import check_rq3_dispatch as probe
from scripts import check_rq3_gpu as operators
from scripts import native_cuda_diagnostics as diagnostics


def delta(decode=0, tiled=0, matrix=1):
    return {"operators": {name: {"milliseconds": 0., "host_dispatches": matrix} for name in diagnostics.NAMES},
            "decode_host_dispatches": decode, "tiled_host_dispatches": tiled, "instrumented": False}


@pytest.mark.parametrize("kernel,tokens,decode,tiled", [
    ("reference", 1, 0, 0), ("reference", 4, 0, 0), ("tiled4", 1, 0, 0),
    ("tiled4", 4, 0, 1), ("decode4", 1, 1, 0), ("decode4", 4, 0, 1),
])
def test_expected_dispatch_and_missing_matrix(kernel, tokens, decode, tiled):
    probe.require_dispatch(delta(decode, tiled), kernel, tokens)
    with pytest.raises(ValueError, match="dispatch"):
        probe.require_dispatch(delta(decode, tiled, matrix=0), kernel, tokens)


@pytest.mark.parametrize("kernel,tokens,decode,tiled", [
    ("decode4", 1, 0, 0), ("decode4", 4, 1, 0), ("decode4", 1, 1, 1),
    ("reference", 1, 1, 0), ("reference", 4, 0, 1), ("tiled4", 4, 0, -1),
])
def test_wrong_or_stale_dispatch_fails(kernel, tokens, decode, tiled):
    with pytest.raises(ValueError, match="dispatch"):
        probe.require_dispatch(delta(decode, tiled), kernel, tokens)


@pytest.mark.parametrize("fault", [None, "no-counters", "different-output", "fallback"])
def test_probe_rejects_faults_and_restores_environment(tmp_path, monkeypatch, fault):
    state = delta(matrix=100)  # Old counters alone cannot satisfy the gate.
    state["decode_host_dispatches"] = state["tiled_host_dispatches"] = 50
    class Counters:
        def __init__(self):
            self.binding = {"mock": True}
        def snapshot(self):
            return copy.deepcopy(state)
    def execute(backend, matrix, signs, inputs):
        assert backend == "CUDA0"
        if fault == "fallback":
            raise RuntimeError("GPU unavailable, no fallback")
        selected = os.environ["ROTQUANT_RQ3_KERNEL"]
        if fault != "no-counters":
            state["operators"]["matrix"]["host_dispatches"] += 1
            if selected == "decode4":
                state["decode_host_dispatches" if len(inputs) == 1 else "tiled_host_dispatches"] += 1
        return np.full((len(inputs), 5), int(fault == "different-output" and selected == "decode4"), dtype=np.float32)
    monkeypatch.setattr(probe, "NativeTests", lambda _: types.SimpleNamespace(operator=execute))
    monkeypatch.setattr(probe, "CudaDiagnostics", lambda _: Counters())
    monkeypatch.setattr(probe, "runtime_identity", lambda _: {"mock": True})
    monkeypatch.setenv("ROTQUANT_RQ3_KERNEL", "original")
    monkeypatch.setenv("ROTQUANT_RQ3_PROFILE", "original")
    if fault:
        with pytest.raises(probe.DispatchProbeError) as error:
            probe.check(tmp_path / "lib", "decode4")
        assert not error.value.evidence["passed"]
        assert "error" in error.value.evidence
    else:
        result = probe.check(tmp_path / "lib", "decode4")
        assert result["passed"] and result["exact_reference"]
        assert len(result["cases"]) == 4
        assert result["cases"][1]["delta"]["decode_host_dispatches"] == 1
    assert os.environ["ROTQUANT_RQ3_KERNEL"] == os.environ["ROTQUANT_RQ3_PROFILE"] == "original"


def test_operator_cli_persists_dispatch_failure(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    evidence = {"passed": False, "binding": {"provider": "mock"}, "cases": []}
    def failure(*a):
        raise probe.DispatchProbeError("counter failed", evidence)
    monkeypatch.setattr(operators, "check", failure)
    monkeypatch.setattr(sys, "argv", ["check", "--library", "mock.so", "--backend", "CUDA0", "--output", str(output)])
    with pytest.raises(probe.DispatchProbeError):
        operators.main()
    report = json.loads(output.read_text())
    assert not report["passed"] and report["dispatch_probe"] == evidence


def test_foreign_diagnostic_provider_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "librotquant.so"
    root.touch()
    def read(*a):
        pass
    def decode():
        pass
    monkeypatch.setattr(diagnostics.C, "CDLL", lambda _: types.SimpleNamespace(
        ggml_cuda_rq3_diagnostics=read, ggml_cuda_rq3_decode_dispatches=decode))
    monkeypatch.setattr(diagnostics, "symbol_owner", lambda _: tmp_path / "foreign/libggml-cuda.so.0")
    with pytest.raises(RuntimeError, match="not owned"):
        diagnostics.CudaDiagnostics(root)
