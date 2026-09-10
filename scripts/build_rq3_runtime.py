"""Build the pinned, isolated experimental native-v3 llama.cpp runtime.

Never resets an existing checkout. CUDA compilation is only a build gate, not
GPU conformance: run check_rq3_gpu and check_rq3_model afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/llama.cpp"
REVISION = "17252c769a63c1cb650ce98ae309cf4de0da7778"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(command, *, cwd=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def prepare_source(source):
    patch = INTEGRATION / "rotquant-native-v2.patch"
    contract = json.loads((INTEGRATION / "rotquant-native-v2-files.json").read_text())
    if contract["base_revision"] != REVISION or contract["patch_sha256"] != digest(patch):
        raise ValueError("integration patch identity mismatch")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", source])
        run(["git", "-C", source, "checkout", "--detach", REVISION])
    if git(source, "rev-parse", "HEAD") != REVISION:
        raise ValueError("source checkout is not the pinned base; choose a new --source-dir")
    if not git(source, "status", "--porcelain"):
        run(["git", "-C", source, "apply", "--check", patch])
        run(["git", "-C", source, "apply", patch])
    changed = set(git(source, "diff", "HEAD", "--name-only").splitlines())
    changed.update(git(source, "ls-files", "--others", "--exclude-standard").splitlines())
    expected = contract["files_sha256"]
    if changed != set(expected):
        raise ValueError("source has missing or unrelated changes; it will not be reset")
    for name, sha in expected.items():
        path = source / name
        if path.is_symlink() or digest(path) != sha:
            raise ValueError(f"patched source mismatch: {name}")
    return contract


def build(source, directory, backend, jobs):
    if not 1 <= jobs <= 32:
        raise ValueError("build jobs must be 1..32")
    if backend == "CUDA" and shutil.which("nvcc") is None:
        raise ValueError("CUDA toolkit/nvcc missing; no CPU fallback")
    if backend == "Metal" and platform.system() != "Darwin":
        raise ValueError("Metal requires macOS")
    source, directory = source.resolve(), directory.resolve()
    if directory == source or source in directory.parents:
        raise ValueError("keep build output outside the source checkout")
    contract = prepare_source(source)
    directory.mkdir(parents=True, exist_ok=True)
    run(["cmake", "-S", source, "-B", directory, "-DCMAKE_BUILD_TYPE=Release",
         f"-DROTQUANT_ROOT={ROOT}", f"-DGGML_CUDA={'ON' if backend == 'CUDA' else 'OFF'}",
         f"-DGGML_METAL={'ON' if backend == 'Metal' else 'OFF'}", "-DGGML_METAL_EMBED_LIBRARY=OFF",
         "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=ON",
         "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_UI=OFF", "-DGGML_NATIVE=OFF",
         "-DGGML_CUDA_NCCL=OFF", "-DCMAKE_CUDA_ARCHITECTURES=native"])
    targets = ["rotquant_ggml_test"] + (["ggml-metal-lib"] if backend == "Metal" else [])
    run(["cmake", "--build", directory, "--target", *targets, "--parallel", jobs])
    libraries = list((directory / "bin").glob("*rotquant_ggml_test.*"))
    if len(libraries) != 1:
        raise ValueError("expected one test runtime library")
    # These files live outside the pinned llama.cpp checkout and are compiled in.
    external = sorted([*ROOT.joinpath("integrations/llama.cpp/rq3").glob("*"),
                       *ROOT.joinpath("native").rglob("*.cpp"), *ROOT.joinpath("native").rglob("*.h"),
                       *ROOT.joinpath("native").rglob("*.hpp"), *ROOT.joinpath("native").rglob("CMakeLists.txt")])
    receipt = {"protocol": "rq3-native-build-v1", "backend": backend,
        "base_revision": REVISION, "patch_sha256": contract["patch_sha256"],
        "root_revision": git(ROOT, "rev-parse", "HEAD"), "root_dirty": bool(git(ROOT, "status", "--porcelain")),
        "external_sources": {str(p.relative_to(ROOT)): digest(p) for p in external if p.is_file()},
        "library": str(libraries[0]), "library_sha256": digest(libraries[0]),
        "compiler": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
        "gpu_validated": False, "boundary": "Build only; operator and whole-model conformance are separate gates."}
    (directory / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "Metal", "CUDA"), required=True)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    build(args.source_dir, args.build_dir, args.backend, args.jobs)


if __name__ == "__main__":
    main()
