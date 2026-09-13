"""Conventional bridge gates; synthetic/mock evidence is never a GPU result."""
from __future__ import annotations

import copy
import json
import os
import struct
import subprocess
import types

import numpy as np
import pytest

from scripts import check_conventional_model as control
from scripts.native_hashing import digest


def capture(drift=0.):
    logits = np.tile(np.arange(256, dtype=np.float32) / 256 * drift, (36, 1))
    logits[:, 0] = 2.
    return {"logits": logits, "traces": np.zeros((4, 8), dtype=np.int32)}


def test_q4_drift_is_not_mistaken_for_bridge_error():
    public = {"CPU": capture(), "CUDA0": capture(.04)}
    result = control.assess(copy.deepcopy(public), public, "CUDA0", "q4_0")
    assert result["passed"]
    assert not result["cross_backend"]["public_api"]["passed"]
    assert not result["cross_backend"]["private_bridge"]["passed"]
    assert all(r["passed"] for r in result["same_backend"].values())
    assert "diagnostic" in result["cross_backend_numerics_role"]


def test_bf16_retains_its_cross_backend_numeric_gate():
    public = {"CPU": capture(), "CUDA0": capture(.04)}
    result = control.assess(copy.deepcopy(public), public, "CUDA0", "bf16")
    assert not result["passed"]
    assert not result["guards"]["public_api_bf16_cross_backend_numerics"]


def test_cpu_only_result_does_not_invent_cross_backend_measurements():
    result = control.assess({"CPU": capture()}, {"CPU": capture()}, "CPU", "q4_0")
    assert result["passed"] and result["cross_backend"] == {}
    assert "not executed" in result["cross_backend_numerics_role"]


@pytest.mark.parametrize("device", ["CPU", "CUDA0"])
def test_same_backend_logit_error_fails_even_when_tokens_match(device):
    public = {"CPU": capture(), "CUDA0": capture(.04)}
    private = copy.deepcopy(public)
    private[device]["logits"] += .001
    result = control.assess(private, public, "CUDA0", "q4_0")
    assert not result["passed"]
    assert not result["same_backend"][device]["guards"]["max_abs_error"]
    assert result["same_backend"][device]["guards"]["exact_generation"]


def test_cross_backend_token_regression_is_still_a_hard_failure():
    public = {"CPU": capture(), "CUDA0": capture()}
    public["CUDA0"]["logits"][:, 1] = 3.
    public["CUDA0"]["traces"][:] = 1
    result = control.assess(copy.deepcopy(public), public, "CUDA0", "q4_0")
    assert not result["passed"]
    assert all(r["passed"] for r in result["same_backend"].values())
    assert not result["guards"]["public_api_cross_backend_generation"]


@pytest.mark.parametrize("mutation", ["nan", "shape", "trace-shape", "range", "noninteger", "inconsistent"])
def test_malformed_capture_rejected(mutation):
    c = capture()
    if mutation == "nan":
        c["logits"][0, 0] = float("nan")
    elif mutation == "shape":
        c["logits"] = c["logits"][:-1]
    elif mutation == "trace-shape":
        c["traces"] = c["traces"][:-1]
    elif mutation == "range":
        c["traces"][0, 0] = 256
    elif mutation == "noninteger":
        c["traces"] = c["traces"].astype(np.float32)
    else:
        c["traces"][0, 0] = 1
    with pytest.raises(ValueError):
        control.validated_capture(**c)


def test_control_required_on_each_backend():
    with pytest.raises(ValueError, match="missing"):
        control.assess({"CPU": capture(), "CUDA0": capture()}, {"CPU": capture()}, "CUDA0", "q4_0")


def test_binary_capture_round_trip_and_shape_validation(tmp_path):
    c = capture()
    raw = b"RQCTRL01" + struct.pack("<4I", 4, 9, 256, 8) + c["logits"].astype("<f4").tobytes() + c["traces"].astype("<i4").tobytes()
    path = tmp_path / "probe.bin"
    path.write_bytes(raw)
    result = control.read_capture(path)
    np.testing.assert_array_equal(result["logits"], c["logits"])
    for invalid in (raw[:-1], raw + b"x", b"badmagic" + raw[8:], raw[:8] + struct.pack("<4I", 4, 8, 256, 8) + raw[24:]):
        path.write_bytes(invalid)
        with pytest.raises(ValueError):
            control.read_capture(path)


def receipt(tmp_path, monkeypatch):
    binary = tmp_path / "control"
    binary.write_bytes(b"not executed")
    runtime = {"library": "hash"}
    monkeypatch.setattr(control, "runtime_identity", lambda _: runtime)
    record = {"protocol": "rq3-public-api-control-build-v1", "passed": True,
              "executable": str(binary), "executable_sha256": digest(binary),
              "source_sha256": digest(control.SOURCE), "base_revision": control.REVISION,
              "builder_sha256": digest(control.ROOT / "scripts/build_conventional_control.py"),
              "runtime_files": runtime}
    path = tmp_path / "build.json"
    path.write_text(json.dumps(record))
    return path, record


@pytest.mark.parametrize("key", ["protocol", "source_sha256", "base_revision", "builder_sha256", "executable_sha256", "runtime_files", "passed"])
def test_control_receipt_drift_fails_closed(tmp_path, monkeypatch, key):
    path, record = receipt(tmp_path, monkeypatch)
    control.verify_control(path, tmp_path / "library")
    record[key] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        control.verify_control(path, tmp_path / "library")


def test_control_failure_is_saved_and_does_not_call_bridge(tmp_path, monkeypatch):
    path, _ = receipt(tmp_path, monkeypatch)
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"mock fixture")
    monkeypatch.setattr(control, "verify_fixture", lambda *a: None)
    def failure(*a, **kw):
        raise subprocess.CalledProcessError(1, "public API")
    monkeypatch.setattr(control.subprocess, "run", failure)
    monkeypatch.setattr(control, "capture_bridge", lambda *a: pytest.fail("bridge must not run"))
    output = tmp_path / "report.json"
    with pytest.raises(subprocess.CalledProcessError):
        control.check(tmp_path / "library", model, "CUDA0", "q4_0", path, tmp_path, output)
    record = json.loads(output.read_text())
    assert not record["passed"] and "CalledProcessError" in record["error"]
    assert record["gpu_executed"] is False


@pytest.mark.parametrize("mutation", [None, "model", "receipt"])
def test_full_check_preserves_raw_probes_and_detects_identity_changes(tmp_path, monkeypatch, mutation):
    path, _ = receipt(tmp_path, monkeypatch)
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"mock fixture")
    monkeypatch.setattr(control, "verify_fixture", lambda *a: None)
    expected = {"CPU": capture(), "CUDA0": capture(.04)}
    def run(cmd, **kwargs):
        device = cmd[2]
        assert kwargs["env"]["ROTQUANT_REQUIRE_GPU"] == ("0" if device == "CPU" else "1")
        assert kwargs["timeout"] == 90 and kwargs["check"]
    monkeypatch.setattr(control.subprocess, "run", run)
    monkeypatch.setattr(control, "read_capture", lambda file: expected["CUDA0" if "CUDA0" in file.name else "CPU"])
    def private(*args):
        if args[-1] == "CUDA0" and mutation:
            target = model if mutation == "model" else path
            target.write_bytes(target.read_bytes() + b" ")
        return copy.deepcopy(expected[args[-1]])
    monkeypatch.setattr(control, "capture_bridge", private)
    output = tmp_path / "report.json"
    if mutation:
        with pytest.raises(ValueError, match="changed during"):
            control.check(tmp_path / "library", model, "CUDA0", "q4_0", path, tmp_path, output)
    else:
        control.check(tmp_path / "library", model, "CUDA0", "q4_0", path, tmp_path, output)
    result = json.loads(output.read_text())
    assert result["passed"] is (mutation is None)
    assert result["gpu_executed"]  # Explicitly mocked, not a real GPU test.
    assert not result["cross_backend"]["public_api"]["passed"]
    probes_path = tmp_path / "conventional-probes/probes.json"
    assert result["probes_sha256"] == digest(probes_path)
    probes = json.loads(probes_path.read_text())["captures"]
    assert probes["public_api"] == probes["private_bridge"]
    assert len(probes["private_bridge"]["CUDA0"]["logits"]) == 36


@pytest.mark.parametrize("failure", ["custom-op", "fallback"])
def test_private_capture_rejects_custom_ops_and_fallback_and_restores_environment(monkeypatch, failure):
    class Model:
        vocab = 256
        custom_ops = 1
        def __enter__(self):
            assert os.environ["ROTQUANT_REQUIRE_GPU"] == "1"
            if failure == "fallback":
                raise RuntimeError("scheduled on CPU")
            return self
        def __exit__(self, *a):
            pass
        def evaluate(self, *a, **kw):
            return capture()["logits"][0]
    monkeypatch.setattr(control, "NativeTests", lambda _: types.SimpleNamespace(model=lambda *a: Model()))
    monkeypatch.setenv("ROTQUANT_REQUIRE_GPU", "previous")
    with pytest.raises((ValueError, RuntimeError)):
        control.capture_bridge("library", "fixture", "CUDA0")
    assert os.environ["ROTQUANT_REQUIRE_GPU"] == "previous"


def test_cpu_requires_explicit_local_opt_in(tmp_path):
    with pytest.raises(ValueError, match="opt-in"):
        control.check(tmp_path / "lib", tmp_path / "model", "CPU", "q4_0", tmp_path / "control", tmp_path, tmp_path / "out")
