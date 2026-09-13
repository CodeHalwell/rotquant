"""Reproduce copied-SONAME state splitting on Linux without CUDA hardware."""
import ctypes as C
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import native_cuda_diagnostics as diagnostics


def test_real_linux_cache_loader(tmp_path):
    if platform.system() != "Linux":
        pytest.skip("ELF loader regression runs on Linux CI")
    assert shutil.which("cc"), "Linux CI must provide its C compiler"
    subprocess.run([sys.executable, str(Path(__file__).with_name("check_cuda_diagnostic_loader.py")),
                    "--work-dir", str(tmp_path / "fixture")], check=True, timeout=60)


def test_unknown_symbol_owner_fails_closed():
    with pytest.raises(RuntimeError, match="identify"):
        diagnostics.symbol_owner(C.c_void_p())
