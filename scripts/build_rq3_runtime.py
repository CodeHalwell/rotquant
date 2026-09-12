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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_hashing import digest

INTEGRATION = ROOT / "integrations/llama.cpp"
REVISION = "17252c769a63c1cb650ce98ae309cf4de0da7778"
LOADER = "src/llama-model-loader.cpp"
OLD_LOADER_SHA = "a0c4e088f8734646d0546ed0ae84073f3c6d4c560247249850fa4330e82526a2"
BOOL_INSTANTIATION = "    template bool llama_model_loader::get_key<bool>       (const std::string & key, bool & result,        bool required);\n"
LOADER_ANCHOR = "    template bool llama_model_loader::get_key<float>      (const std::string & key, float & result,       bool required);\n"


def run(command, *, cwd=None, timeout=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True, timeout=timeout)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_patched_source(source, expected, *, repair_known_loader=False):
    """Validate every patched file before an opt-in, exact-hash loader repair."""
    mismatches = []
    for name, sha in expected.items():
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"patched source mismatch: {name}")
        if digest(path) != sha:
            mismatches.append(name)
    if not mismatches:
        return
    if (mismatches != [LOADER] or not repair_known_loader
            or digest(source / LOADER) != OLD_LOADER_SHA):
        raise ValueError(f"patched source mismatch: {mismatches}; --repair-known-loader only accepts the exact old loader")
    path = source / LOADER
    original = path.read_text()
    repaired = original.replace(LOADER_ANCHOR, BOOL_INSTANTIATION + LOADER_ANCHOR, 1)
    if original.count(LOADER_ANCHOR) != 1 or hashlib.sha256(repaired.encode()).hexdigest() != expected[LOADER]:
        raise ValueError("loader repair does not match the current patch contract")
    path.write_text(repaired)
    print("Repaired exact known loader; preserving CUDA sources/build cache.", flush=True)


def prepare_source(source, *, repair_known_loader=False):
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
    verify_patched_source(source, expected, repair_known_loader=repair_known_loader)
    return contract


def verify_library_load(library):
    """Resolve the actual Python binding in a fresh process; never run a model."""
    run([sys.executable, "-u", "-c",
         ("from pathlib import Path; import sys; "
         "from scripts.rq3_test_runtime import NativeTests; "
         "NativeTests(Path(sys.argv[1])); print('Native library load passed (no GPU conformance yet).')"),
         library], cwd=ROOT, timeout=120)


def build(source, directory, backend, jobs, *, repair_known_loader=False):
    if not 1 <= jobs <= 32:
        raise ValueError("build jobs must be 1..32")
    if backend == "CUDA" and shutil.which("nvcc") is None:
        raise ValueError("CUDA toolkit/nvcc missing; no CPU fallback")
    if backend == "Metal" and platform.system() != "Darwin":
        raise ValueError("Metal requires macOS")
    source, directory = source.resolve(), directory.resolve()
    if directory == source or source in directory.parents:
        raise ValueError("keep build output outside the source checkout")
    contract = prepare_source(source, repair_known_loader=repair_known_loader)
    directory.mkdir(parents=True, exist_ok=True)
    link_flags = ["-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--no-undefined"] if platform.system() == "Linux" else []
    run(["cmake", "-S", source, "-B", directory, "-DCMAKE_BUILD_TYPE=Release",
         f"-DROTQUANT_ROOT={ROOT}", f"-DGGML_CUDA={'ON' if backend == 'CUDA' else 'OFF'}",
         f"-DGGML_METAL={'ON' if backend == 'Metal' else 'OFF'}", "-DGGML_METAL_EMBED_LIBRARY=OFF",
         "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=ON",
         "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_UI=OFF", "-DGGML_NATIVE=OFF",
         "-DGGML_CUDA_NCCL=OFF", "-DCMAKE_CUDA_ARCHITECTURES=native", *link_flags])
    targets = ["rotquant_ggml_test"] + (["ggml-metal-lib"] if backend == "Metal" else [])
    run(["cmake", "--build", directory, "--target", *targets, "--parallel", jobs])
    libraries = list((directory / "bin").glob("*rotquant_ggml_test.*"))
    if len(libraries) != 1:
        raise ValueError("expected one test runtime library")
    verify_library_load(libraries[0])
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
        "load_validated": True, "gpu_validated": False,
        "boundary": "Build and binding load only; operator and whole-model conformance are separate gates."}
    (directory / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "Metal", "CUDA"), required=True)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--repair-known-loader", action="store_true",
                        help="Repair only the exact dd87da2 loader defect, retaining existing compiled CUDA objects")
    args = parser.parse_args()
    build(args.source_dir, args.build_dir, args.backend, args.jobs, repair_known_loader=args.repair_known_loader)


if __name__ == "__main__":
    main()
