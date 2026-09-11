"""Opt-in Drive persistence for build artifacts and lossless exports.

Cache entries are immutable, hash-verified copies, NOT cached correctness passes.
Restore only to fresh local directories; never execute libraries from Drive.
Only use a private cache created by this workflow: hashes detect corruption,
not a malicious party replacing both a manifest and its payload.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_gpu_workflow import digest, fingerprint, write_json


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or str(path) != name or name == "manifest.json"):
        raise ValueError(f"Unsafe cache path: {name!r}")
    return path


class ArtifactCache:
    def __init__(self, root):
        self.root = Path(root)

    def entry(self, kind, request):
        if kind not in {"runtime", "export"}:
            raise ValueError("Unknown cache kind")
        return self.root / kind / fingerprint(request)

    def verify(self, entry, request):
        manifest_path = entry / "manifest.json"
        if entry.is_symlink() or manifest_path.is_symlink():
            raise ValueError("Cache symlinks are forbidden")
        record = json.loads(manifest_path.read_text())
        if (record.get("protocol") != "rq3-private-artifact-cache-v1"
                or record.get("request") != request or record.get("key") != fingerprint(request)
                or not record.get("files")):
            raise ValueError("Cache identity mismatch")
        actual = set()
        for path in entry.rglob("*"):
            if path.is_symlink():
                raise ValueError("Cache symlinks are forbidden")
            if path.is_file() and path != manifest_path:
                actual.add(path.relative_to(entry).as_posix())
        if actual != set(record["files"]):
            raise ValueError("Cache contains missing or unexpected files")
        for name, value in record["files"].items():
            path = entry.joinpath(*safe_name(name).parts)
            if path.stat().st_size != value["bytes"] or digest(path) != value["sha256"]:
                raise ValueError(f"Cache corruption: {name}; original entry preserved, not executed")
        return record

    def restore(self, kind, request, parent):
        entry = self.entry(kind, request)
        if not entry.exists():
            print(f"CACHE MISS {kind}: {entry.name}", flush=True)
            return None
        print(f"CACHE VERIFY {kind}: {entry}", flush=True)
        record = self.verify(entry, request)
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        total = sum(v["bytes"] for v in record["files"].values())
        if shutil.disk_usage(parent).free < total + 256 * 1024**2:
            raise ValueError("Insufficient local space for cache restore")
        target = Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=parent))
        for name, value in record["files"].items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry / name, destination)
            if digest(destination) != value["sha256"]:
                raise ValueError("Cache changed during restore; no library loaded")
        print(f"CACHE HIT {kind}: restored {total:,} bytes locally", flush=True)
        return target, record

    def publish(self, kind, request, files, metadata):
        entry = self.entry(kind, request)
        identities = {}
        for name, path in files.items():
            safe_name(name)
            path = Path(path)
            if not path.is_file():
                raise ValueError(f"Missing cache source: {path}")
            identities[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if entry.exists():
            existing = self.verify(entry, request)
            # Two valid builds need not be byte-identical. Keep the first;
            # current tests still bind to the actual runtime used this run.
            if kind == "export" and existing["files"] != identities:
                raise ValueError("An immutable export cache entry disagrees with this export")
            return entry
        entry.parent.mkdir(parents=True, exist_ok=True)
        total = sum(v["bytes"] for v in identities.values())
        if shutil.disk_usage(entry.parent).free < total + 256 * 1024**2:
            raise ValueError("Insufficient Drive space for artifact persistence")
        # Incomplete writes never acquire the final key; interrupted staging is
        # preserved for inspection rather than recursively deleting user data.
        staging = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=entry.parent))
        for name, path in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)  # dereference local SONAME aliases
            if digest(destination) != identities[name]["sha256"]:
                raise ValueError("Artifact changed during persistence")
            print(f"CACHE SAVE {kind}: {name} ({identities[name]['bytes']:,} bytes)", flush=True)
        write_json(staging / "manifest.json", {
            "protocol": "rq3-private-artifact-cache-v1", "key": fingerprint(request),
            "request": request, "files": identities, "metadata": metadata,
            "boundary": "Build/export integrity only; fresh numerical gates required after restore.",
        })
        self.verify(staging, request)
        if entry.exists():
            raise FileExistsError("Concurrent cache publisher; both entries preserved")
        staging.rename(entry)
        print(f"CACHE PERSISTED {kind}: {entry}", flush=True)
        return entry


def source_hashes(paths):
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths)) if p.is_file()}


def runtime_request():
    import torch

    from scripts.build_rq3_runtime import INTEGRATION, REVISION

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise ValueError("Persistent native binary reuse currently supports Linux CUDA only")
    def output(*cmd):
        return subprocess.check_output(cmd, text=True, timeout=30).strip()
    return {"version": 1, "base_revision": REVISION,
        "sources": source_hashes([
            ROOT / "scripts/build_rq3_runtime.py", Path(__file__),
            INTEGRATION / "rotquant-native-v2.patch", INTEGRATION / "rotquant-native-v2-files.json",
            *INTEGRATION.joinpath("rq3").glob("*"), *ROOT.joinpath("native").rglob("*.cpp"),
            *ROOT.joinpath("native").rglob("*.h"), *ROOT.joinpath("native").rglob("*.hpp"),
            *ROOT.joinpath("native").rglob("CMakeLists.txt")]),
        "compatibility": {"system": platform.system(), "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "driver": output("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            "nvcc": output("nvcc", "--version"), "cxx": output("c++", "--version"),
            "cmake": output("cmake", "--version"),
            "build_environment": {k: os.environ.get(k) for k in (
                "CC", "CXX", "CFLAGS", "CXXFLAGS", "CUDAFLAGS", "CUDACXX", "CUDAHOSTCXX",
                "CMAKE_PREFIX_PATH", "CUDA_VISIBLE_DEVICES")}}}


def prepare_runtime(args):
    from scripts.build_rq3_runtime import build, prepare_source, verify_library_load
    from scripts.check_rq3_model import runtime_identity

    request = runtime_request()
    cache = ArtifactCache(args.cache_dir)
    # Small pinned source checkout is needed by fixture/export converters even
    # when compilation is skipped. Never restore executable Python from cache.
    prepare_source(args.source_dir, repair_known_loader=True)
    restored = cache.restore("runtime", request, args.build_dir.parent / "restored")
    if restored:
        directory, record = restored
        name = record["metadata"]["library_name"]
        if Path(name).name != name or f"bin/{name}" not in record["files"]:
            raise ValueError("Unsafe cached library name")
        library = directory / "bin" / name
        receipt = json.loads((directory / "build-receipt.json").read_text())
        if receipt.get("library_sha256") != digest(library) or receipt.get("load_validated") is not True:
            raise ValueError("Cached build receipt disagrees with library")
        os.environ["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        verify_library_load(library)
    else:
        receipt = build(args.source_dir, args.build_dir, "CUDA", args.jobs, repair_known_loader=True)
        library = Path(receipt["library"])
        files = {f"bin/{name}": library.parent / name for name in runtime_identity(library)}
        files["build-receipt.json"] = args.build_dir / "build-receipt.json"
        cache.publish("runtime", request, files, {"library_name": library.name})
    return {"library": str(library), "build_receipt": receipt,
            "cache_hit": bool(restored), "cache_key": fingerprint(request),
            "cache_entry": str(cache.entry("runtime", request))}


def prepare_export(args):
    from rotquant.checkpoint import MANIFEST_NAME
    from scripts.run_rq3_retained_gpu import verified_evidence, verified_source_evidence

    verified_source_evidence(args.source_arm)
    request = {"version": 1, "checkpoint_sha256": digest(args.source_arm / "checkpoint" / MANIFEST_NAME),
        "sources": source_hashes([*ROOT.joinpath("rotquant").rglob("*.py"),
            ROOT / "scripts/export_rotquant_gguf_v2.py", Path(__file__),
            ROOT / "integrations/llama.cpp/rotquant-native-v2-files.json"])}
    cache = ArtifactCache(args.cache_dir)
    restored = cache.restore("export", request, args.destination.parent / "restored")
    if restored:
        directory, _ = restored
    else:
        directory = args.destination
        subprocess.run([sys.executable, "-u", ROOT / "scripts/export_rotquant_gguf_v2.py",
                        args.source_arm / "checkpoint", directory, "--llama-cpp-dir", args.source_dir], check=True)
    _, receipt = verified_evidence(args.source_arm, directory)
    if not restored:
        cache.publish("export", request, {name: directory / name for name in ["export.json", *receipt["artifact_files"]]}, {})
    return {"directory": str(directory), "receipt": receipt, "cache_hit": bool(restored),
            "cache_key": fingerprint(request), "cache_entry": str(cache.entry("export", request))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("runtime", "export"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--source-arm", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = prepare_runtime(args) if args.kind == "runtime" else prepare_export(args)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
