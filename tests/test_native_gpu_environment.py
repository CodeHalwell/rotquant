"""Bootstrap fail-closed unit tests; real installs have a separate Linux check."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import native_gpu_environment as bootstrap


def test_target_inspection_rejects_base_interpreter(tmp_path):
    with pytest.raises(ValueError, match="not the requested virtual environment"):
        bootstrap.inspect_target(tmp_path, "test", tmp_path / "torch.py")


@pytest.mark.parametrize("fault", [None, "version", "origin", "inheritance"])
def test_target_inspection_checks_inherited_torch(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "base_prefix", "/test-base")
    (tmp_path / "pyvenv.cfg").write_text(
        "include-system-site-packages = " + ("false" if fault == "inheritance" else "true") + "\n")
    origin = str(tmp_path.parent / "base-torch.py")
    monkeypatch.setattr(bootstrap.metadata, "version", lambda _: "wrong" if fault == "version" else "test")
    monkeypatch.setattr(bootstrap, "torch_origin", lambda: "shadow-torch.py" if fault == "origin" else origin)
    if fault:
        with pytest.raises(ValueError, match="PyTorch"):
            bootstrap.inspect_target(tmp_path, "test", origin)
    else:
        assert bootstrap.inspect_target(tmp_path, "test", origin)["passed"]


@pytest.mark.parametrize("failed_label", ["create-environment", "verify-target", "check-installer", "dependencies"])
def test_bootstrap_failure_never_reaches_later_install(tmp_path, failed_label):
    calls = []
    def command(words, label):
        calls.append(label)
        if label == failed_label:
            raise subprocess.CalledProcessError(1, words)
    stage = SimpleNamespace(command=command, directory=tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.prepare_environment(stage, tmp_path, tmp_path / "venv", "test")
    assert calls[-1] == failed_label
    assert "install-rotquant" not in calls
