"""Build a tiny public-API probe caller against the already verified libraries.

No CUDA compilation, model downloads, new backend library or modified bridge.
This does not independently validate the shared llama/ggml implementation.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_rq3_runtime import REVISION
from scripts.check_rq3_model import runtime_identity
from scripts.native_gpu_workflow import digest, write_json

SOURCE = ROOT / "scripts/native_controls/conventional_probe.cpp"


def build(library, llama_dir, directory):
    library, llama_dir, directory = (Path(p).resolve() for p in (library, llama_dir, directory))
    if platform.system() not in {"Linux", "Darwin"}:
        raise ValueError("The public-API control currently supports Linux/macOS")
    compiler = shutil.which("c++")
    if compiler is None:
        raise ValueError("C++ compiler unavailable")
    actual = subprocess.check_output(["git", "-C", str(llama_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError("Control headers must come from the pinned llama.cpp revision")
    headers = [llama_dir / "include/llama.h", *sorted((llama_dir / "ggml/include").glob("*.h"))]
    for header in headers:
        relative = header.relative_to(llama_dir).as_posix()
        committed = subprocess.check_output(["git", "-C", str(llama_dir), "show", f"{REVISION}:{relative}"])
        if header.is_symlink() or header.read_bytes() != committed:
            raise ValueError(f"Control public header modified: {relative}")
    directory.mkdir(parents=True, exist_ok=False)
    executable = directory / "conventional-probe"
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-I", str(llama_dir / "include"), "-I", str(llama_dir / "ggml/include"),
               str(SOURCE), "-L", str(library.parent), f"-Wl,-rpath,{library.parent}",
               "-lllama", "-lggml", "-lggml-base", "-o", str(executable)]
    before = runtime_identity(library)
    print("Compiling public-API caller only (reusing native libraries):", " ".join(command), flush=True)
    subprocess.run(command, check=True, timeout=90)
    if runtime_identity(library) != before:
        raise ValueError("Runtime changed while compiling control")
    # Detect missing symbols/linker paths without executing a GPU/model.
    env = dict(os.environ, LD_LIBRARY_PATH=str(library.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    loaded = subprocess.run([str(executable)], env=env, capture_output=True, text=True, timeout=30, check=False)
    if loaded.returncode != 1 or "usage: conventional-probe" not in loaded.stderr:
        raise ValueError(f"Control executable failed its load check: {loaded.stderr[-2000:]}")
    record = {"protocol": "rq3-public-api-control-build-v1", "passed": True,
              "executable": str(executable), "executable_sha256": digest(executable),
              "source_sha256": digest(SOURCE), "builder_sha256": digest(Path(__file__)),
              "base_revision": REVISION, "runtime_files": before,
              "public_headers": {p.relative_to(llama_dir).as_posix(): digest(p) for p in headers},
              "compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
              "command": command, "boundary": "Independent public-API caller; shared llama/ggml backend, no GPU validation yet."}
    write_json(directory / "build.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.library, args.llama_dir, args.output_dir)
