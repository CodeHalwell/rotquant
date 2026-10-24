"""Pinned Bonsai 2 4B acquisition and native frozen-token measurement bridge.

Acquisition/build are explicit CLI phases. Importing this module never downloads
weights, builds a runtime, or starts inference. Quality/tasks/timing use the same
NativeBackend and native_speeds interfaces as the RotQuant controls.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_hashing import digest

REPOSITORY = "prism-ml/Ternary-Bonsai-2-4B-gguf"
REVISION = "8d6557168168f91ea6697a4a49ed2ef622b53adb"
RUNTIME_REPOSITORY = "https://github.com/PrismML-Eng/llama.cpp.git"
RUNTIME_REVISION = "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
BRIDGE = Path(__file__).with_name("ternary_bonsai_bridge.cpp")
PREFIX = "Ternary-Bonsai-2-4B-"
PINS = {
    "PTQ1_0": (949549920, "56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60"),
    "PQ2_0": (1146465120, "d406b1e912c0d2a4c124ad10bf38a76ae662fdc4d24b176451403dc7b7f30409"),
    "mmproj-BF16": (675569184, "e039156b6979f47d92597875e6fa2efe70aa2a26d135ed2abec09124e0c1448f"),
}


def write_report(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact_paths(directory, packing):
    if packing not in ("PTQ1_0", "PQ2_0"):
        raise ValueError("packing must be PTQ1_0 or PQ2_0")
    return {kind: Path(directory) / f"{PREFIX}{kind}.gguf"
            for kind in (packing, "mmproj-BF16")}


def verify_file(path, pin):
    size, sha = pin
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != size
            or digest(path) != sha):
        raise ValueError(f"published artifact hash/size mismatch: {path}")
    return {"bytes": size, "sha256": sha}


def verify_assets(directory, packing="PTQ1_0"):
    paths = artifact_paths(directory, packing)
    files = {path.name: verify_file(path, PINS[kind]) for kind, path in paths.items()}
    return {"protocol": "ternary-bonsai-artifacts-v1", "repository": REPOSITORY,
            "revision": REVISION, "packing": packing, "model": str(paths[packing].resolve()),
            "vision": str(paths["mmproj-BF16"].resolve()), "artifact_files": files,
            "language_model_bytes": PINS[packing][0],
            "complete_model_payload_bytes": sum(item["bytes"] for item in files.values()),
            "vision_execution": False,
            "vision_precision": "vendor BF16/F32 sidecar; RotQuant retains source FP16 auxiliary",
            "scope": "text quality/speed; language plus vendor vision payload accounting; no drafter/MTP"}


def acquire(output, packing="PTQ1_0"):
    from huggingface_hub import hf_hub_download

    paths = artifact_paths(output, packing)
    Path(output).mkdir(parents=True, exist_ok=True)
    # Never overwrite a preexisting mismatched artifact, even in a partial run.
    for kind, path in paths.items():
        if path.exists() or path.is_symlink():
            verify_file(path, PINS[kind])
    for kind, path in paths.items():
        if not path.exists():
            downloaded = Path(hf_hub_download(REPOSITORY, path.name, revision=REVISION,
                                              local_dir=str(Path(output).resolve())))
            if downloaded.resolve() != path.resolve():
                raise ValueError("download escaped the requested artifact directory")
            verify_file(path, PINS[kind])
    result = verify_assets(output, packing)
    write_report(Path(output) / "artifacts.json", result)
    return result


def run(command, *, cwd=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def prepare_source(source):
    source = Path(source).resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", "--depth=1", "--filter=blob:none", RUNTIME_REPOSITORY, source])
        run(["git", "-C", source, "fetch", "--depth=1", "origin", RUNTIME_REVISION])
        run(["git", "-C", source, "checkout", "--detach", RUNTIME_REVISION])
    if git(source, "rev-parse", "HEAD") != RUNTIME_REVISION:
        raise ValueError("Prism checkout is not the pinned revision; choose a new directory")
    if git(source, "status", "--porcelain"):
        raise ValueError("Prism checkout must be pristine; existing changes are not reset")
    return source


def cmake_project():
    # Variables are provided separately to CMake, so paths are never interpolated into source.
    return ('cmake_minimum_required(VERSION 3.24)\n'
            'project(bonsai_measure LANGUAGES C CXX)\n'
            'set(CMAKE_CXX_STANDARD 17)\n'
            'add_subdirectory("${BONSAI_SOURCE}" prism)\n'
            'add_library(bonsai_measure SHARED "${BONSAI_BRIDGE}")\n'
            'target_link_libraries(bonsai_measure PRIVATE llama ggml)\n'
            'set_target_properties(bonsai_measure PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n')


def runtime_files(directory):
    paths = [p for p in Path(directory).rglob("*") if p.is_file() and
             (".so" in p.name or p.suffix in (".dylib", ".dll"))]
    return {str(p.relative_to(directory)): digest(p) for p in sorted(paths)}


def build_runtime(source, build, backend="CUDA", jobs=4):
    if backend not in ("CUDA", "CPU", "Metal") or type(jobs) is not int or not 1 <= jobs <= 32:
        raise ValueError("expected CPU/CUDA/Metal backend and 1..32 build jobs")
    if backend == "CUDA" and shutil.which("nvcc") is None:
        raise ValueError("CUDA toolkit/nvcc missing; no CPU fallback")
    if backend == "Metal" and platform.system() != "Darwin":
        raise ValueError("Metal requires macOS")
    source, build = Path(source).resolve(), Path(build).resolve()
    if source == build or source in build.parents or build in source.parents:
        raise ValueError("Prism source and build directories must be separate")
    source = prepare_source(source)
    project = build / "bridge-project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "CMakeLists.txt").write_text(cmake_project())
    run(["cmake", "-S", project, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
         f"-DBONSAI_SOURCE={source}", f"-DBONSAI_BRIDGE={BRIDGE.resolve()}",
         "-DBUILD_SHARED_LIBS=ON", f"-DGGML_CUDA={'ON' if backend == 'CUDA' else 'OFF'}",
         f"-DGGML_METAL={'ON' if backend == 'Metal' else 'OFF'}", "-DGGML_NATIVE=OFF",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
         "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_CURL=OFF", "-DGGML_CUDA_NCCL=OFF",
         "-DCMAKE_CUDA_ARCHITECTURES=native"])
    run(["cmake", "--build", build, "--target", "bonsai_measure", "--parallel", jobs])
    libraries = list((build / "bin").glob("*bonsai_measure.*"))
    if len(libraries) != 1:
        raise ValueError("expected exactly one Bonsai measurement library")
    library = libraries[0].resolve()
    run([sys.executable, "-c", ("from scripts.ternary_research_bonsai import BonsaiRuntime; "
         "import sys; BonsaiRuntime(sys.argv[1], verify=False); print('Bonsai binding load passed')"), library], cwd=ROOT)
    result = {"protocol": "ternary-bonsai-build-v1", "runtime_repository": RUNTIME_REPOSITORY,
              "runtime_revision": RUNTIME_REVISION, "backend": backend,
              "library": str(library), "library_sha256": digest(library),
              "bridge_sha256": digest(BRIDGE), "files_sha256": runtime_files(build),
              "load_validated": True, "model_validated": False, "gpu_validated": False}
    write_report(build / "build-receipt.json", result)
    return result


def verify_runtime(library):
    library = Path(library).resolve(strict=True)
    build = library.parent.parent
    receipt = json.loads((build / "build-receipt.json").read_text())
    if (receipt.get("protocol") != "ternary-bonsai-build-v1"
            or receipt.get("runtime_revision") != RUNTIME_REVISION
            or receipt.get("bridge_sha256") != digest(BRIDGE)
            or receipt.get("library_sha256") != digest(library)
            or receipt.get("files_sha256") != runtime_files(build)):
        raise ValueError("Bonsai runtime differs from its pinned build receipt")
    return receipt


class GGUFMetadata:
    """Bounded metadata-only reader, independent of custom tensor type enums."""
    FORMATS: ClassVar[dict[int, str]] = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I",
                                       5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
                                       12: "d"}

    def __init__(self, handle):
        self.handle = handle

    def read(self, count):
        if count < 0 or self.handle.tell() + count > 64 * 1024 * 1024:
            raise ValueError("GGUF metadata exceeds bounded header limit")
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError("truncated GGUF metadata")
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.read(self.number("Q")).decode("utf-8")

    def value(self, kind, depth=0):
        if kind == 8:
            return self.string()
        if kind == 9 and depth == 0:
            subtype, count = self.number("I"), self.number("Q")
            if count > 2_000_000:
                raise ValueError("GGUF metadata array too large")
            return [self.value(subtype, depth + 1) for _ in range(count)]
        if kind not in self.FORMATS:
            raise ValueError(f"unsupported GGUF metadata value type {kind}")
        return self.number(self.FORMATS[kind])


def tokenizer_fields(path):
    with Path(path).open("rb") as handle:
        reader = GGUFMetadata(handle)
        if reader.read(4) != b"GGUF" or reader.number("I") not in (2, 3):
            raise ValueError("expected GGUF v2/v3 metadata")
        reader.number("Q")  # Tensor count; tensor payload is never read here.
        count = reader.number("Q")
        if count > 10000:
            raise ValueError("too many GGUF metadata fields")
        fields, names = {}, set()
        for _ in range(count):
            name = reader.string()
            if name in names:
                raise ValueError("duplicate GGUF metadata key")
            names.add(name)
            value = reader.value(reader.number("I"))
            if name.startswith("tokenizer.ggml."):
                fields[name] = value
    required = {"tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type"}
    if not required <= fields.keys() or not fields["tokenizer.ggml.tokens"]:
        raise ValueError("missing tokenizer ID mapping/type metadata")
    if len(fields["tokenizer.ggml.tokens"]) != len(fields["tokenizer.ggml.token_type"]):
        raise ValueError("tokenizer token/type lengths disagree")
    return fields


def validate_tokenizer(candidate_gguf, reference_gguf):
    """Validate the shared ID space for the unpadded, direct-decode bridge."""
    candidate, reference = tokenizer_fields(candidate_gguf), tokenizer_fields(reference_gguf)
    # Direct llama_decode consumes frozen unpadded IDs without inserting BOS or padding.
    ignored = {"tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id",
               "tokenizer.ggml.add_bos_token"}
    defaults = {key: {"candidate": candidate.get(key), "reference": reference.get(key)}
                for key in sorted(ignored) if candidate.get(key) != reference.get(key)}
    candidate = {key: value for key, value in candidate.items() if key not in ignored}
    reference = {key: value for key, value in reference.items() if key not in ignored}
    changed = sorted(key for key in candidate.keys() | reference.keys()
                     if candidate.get(key) != reference.get(key))
    if changed:
        raise ValueError(f"Bonsai tokenizer differs from frozen FP16 reference; KL/tasks forbidden: {changed}")
    identity = hashlib.sha256(json.dumps(reference, sort_keys=True, ensure_ascii=True,
                                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"passed": True, "vocabulary_size": len(reference["tokenizer.ggml.tokens"]),
            "tokenizer_metadata_sha256": identity, "compared_keys": sorted(reference),
            "unused_frozen_input_defaults": defaults,
            "reference_gguf": str(Path(reference_gguf).resolve()),
            "boundary": "Exact token-ID, type, merge and active special-token metadata equality; direct decoding of frozen unpadded IDs bypasses vendor templates and BOS/padding defaults; not general tokenizer equivalence"}


class BonsaiRuntime:
    def __init__(self, library, *, verify=True):
        self.library = Path(library).resolve(strict=True)
        self.receipt = verify_runtime(self.library) if verify else None
        self.lib = C.CDLL(str(self.library))
        signatures = {
            "bonsai_bridge_version": ([], C.c_char_p), "bonsai_error": ([], C.c_char_p),
            "bonsai_init": ([], None),
            "bonsai_open": ([C.c_char_p, C.c_char_p, C.c_uint32], C.c_void_p),
            "bonsai_vocab": ([C.c_void_p], C.c_int),
            "bonsai_is_eog": ([C.c_void_p, C.c_int], C.c_bool),
            "bonsai_evaluate": ([C.c_void_p, C.c_void_p, C.c_int, C.c_bool, C.c_int, C.c_void_p], C.c_int),
            "bonsai_close": ([C.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        if self.lib.bonsai_bridge_version() != b"bonsai-measure-v1":
            raise ValueError("unexpected Bonsai bridge version")
        self.lib.bonsai_init()

    def model(self, path, backend="CUDA0", context=512):
        return BonsaiModel(self, path, backend, context)


class BonsaiModel:
    custom_ops = False  # RQ3 operation accounting does not apply to Prism kernels.

    def __init__(self, runtime, path, backend, context):
        if backend not in ("CPU", "CUDA0", "MTL0") or not 32 <= context <= 262144:
            raise ValueError("unsupported backend/context")
        self.runtime, self.backend, self.context = runtime, backend, context
        self.handle = runtime.lib.bonsai_open(os.fsencode(Path(path).resolve(strict=True)),
                                             backend.encode(), context)
        if not self.handle:
            raise RuntimeError(runtime.lib.bonsai_error().decode())
        self.vocab = runtime.lib.bonsai_vocab(self.handle)

    def evaluate(self, ids, *, reset=False, selected=1):
        ids = np.asarray(ids)
        if (not self.handle or ids.ndim != 1 or ids.dtype.kind not in "iu"
                or not 0 < len(ids) <= self.context or (ids < 0).any() or (ids >= self.vocab).any()
                or type(selected) is not int or not 1 <= selected <= min(16, len(ids))):
            raise ValueError("invalid Bonsai token probe or closed model")
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        output = np.empty((selected, self.vocab), dtype=np.float32)
        if self.runtime.lib.bonsai_evaluate(self.handle, ids.ctypes.data, len(ids), reset,
                                           selected, output.ctypes.data):
            raise RuntimeError(self.runtime.lib.bonsai_error().decode())
        if not np.isfinite(output).all():
            raise ValueError("non-finite Bonsai logits")
        return output[0] if selected == 1 else output

    def is_eog(self, token):
        if not self.handle or not 0 <= token < self.vocab:
            raise ValueError("invalid Bonsai token or closed model")
        return bool(self.runtime.lib.bonsai_is_eog(self.handle, int(token)))

    def close(self):
        if self.handle:
            self.runtime.lib.bonsai_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def smoke_model(model, frozen_ids):
    ids = np.asarray(frozen_ids)[:16]
    first = model.evaluate(ids, reset=True)
    second = model.evaluate(ids, reset=True)
    if not np.isfinite(first).all() or not np.array_equal(first, second):
        raise ValueError("Bonsai repeated reset smoke logits differ or are non-finite")
    return {"passed": True, "tokens": len(ids), "vocabulary_size": model.vocab,
            "reset_exact": True, "backend": model.backend, "kv_dtype": "float16",
            "offload": ("CPU explicitly requested" if model.backend == "CPU" else
                        "all layers plus embedding requested; per-operation CPU fallback not audited"),
            "boundary": "Finite deterministic reset smoke, not parity against a separate Bonsai implementation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    download = sub.add_parser("acquire")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--packing", choices=("PTQ1_0", "PQ2_0"), default="PTQ1_0")
    build = sub.add_parser("build")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--build-dir", type=Path, required=True)
    build.add_argument("--backend", choices=("CPU", "CUDA", "Metal"), default="CUDA")
    build.add_argument("--jobs", type=int, default=4)
    for child in (download, build):
        child.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = (acquire(args.output_dir, args.packing) if args.phase == "acquire" else
              build_runtime(args.source_dir, args.build_dir, args.backend, args.jobs))
    write_report(args.report, result)


if __name__ == "__main__":
    main()
