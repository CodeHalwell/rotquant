"""Real Linux dynamic-loader regression; stdlib/CPU only, no Torch or CUDA."""
from __future__ import annotations

import argparse
import ctypes as C
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_cuda_diagnostics import CudaDiagnostics
from scripts.native_gpu_cache import ArtifactCache


def run(directory):
    if platform.system() != "Linux" or not shutil.which("cc"):
        raise RuntimeError("Linux and a C compiler are required; no silent skip")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = ROOT / "tests/fixtures/cuda_diagnostics"
    build = directory / "build"
    build.mkdir()
    versioned, alias = build / "libggml-cuda.so.0", build / "libggml-cuda.so"
    bridge = build / "librotquant_ggml_test.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                    "-Wl,-soname,libggml-cuda.so.0", str(source / "counter.c"), "-o", str(versioned)], check=True)
    alias.symlink_to(versioned.name)
    subprocess.run(["cc", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                    str(source / "bridge.c"), "-L", str(build), "-lggml-cuda",
                    "-Wl,-rpath,$ORIGIN", "-o", str(bridge)], check=True)
    cache = ArtifactCache(directory / "private-cache")
    cache.publish("runtime", {"fixture": "ELF only"}, {f"bin/{p.name}": p for p in (versioned, alias, bridge)}, {})
    target, _ = cache.restore("runtime", {"fixture": "ELF only"}, directory / "restored")
    restored = target / "bin"
    assert not (restored / alias.name).is_symlink()
    assert (restored / alias.name).stat().st_ino != (restored / versioned.name).stat().st_ino
    execution = C.CDLL(str(restored / bridge.name))
    execution.fixture_execute.argtypes = []
    execution.fixture_execute.restype = None
    execution.fixture_execute()
    # Reproduce the old bug with the real ELF loader, not a mocked ctypes call.
    wrong = C.CDLL(str(restored / alias.name))
    wrong.ggml_cuda_rq3_decode_dispatches.restype = C.c_uint64
    assert wrong.ggml_cuda_rq3_decode_dispatches() == 0
    repaired = CudaDiagnostics(restored / bridge.name)
    assert repaired.snapshot()["decode_host_dispatches"] == 1
    execution.fixture_execute()
    snapshot = repaired.snapshot()
    assert snapshot["decode_host_dispatches"] == snapshot["tiled_host_dispatches"] == 2
    assert wrong.ggml_cuda_rq3_decode_dispatches() == 0
    assert Path(snapshot["binding"]["provider"]) == restored / versioned.name
    return {"passed": True, "old_alias_counter": 0, "repaired_execution_counter": 2,
            "binding": snapshot["binding"], "boundary": "Real Linux loader/cache regression; CPU fixture, not CUDA validation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().work_dir), indent=2))
