"""Bundle the current reviewed source for Colab without requiring a Git push."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_hashing import digest

DIRECTORIES = ("rotquant", "scripts", "configs", "native", "integrations", "requirements",
               "research/eval_suites", "notebooks", "docs")
ROOT_FILES = ("pyproject.toml", "uv.lock", "README.md", "LICENSE")
SUFFIXES = {".py", ".pyi", ".typed", ".yaml", ".yml", ".json", ".patch", ".txt", ".md",
            ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".in", ".ipynb"}


def source_paths(root):
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in DIRECTORIES:
        paths.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and not path.is_symlink() and path.suffix in SUFFIXES
                     and "__pycache__" not in path.parts and ".ipynb_checkpoints" not in path.parts)
    return sorted(set(paths))


def build_bundle(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("write the source ZIP outside the project checkout")
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    paths = source_paths(root)
    manifest = {"protocol": "rotquant-source-bundle-v1", "revision": revision,
                "files_sha256": {str(path.relative_to(root)): digest(path) for path in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, str(path.relative_to(root)))
            archive.writestr("SOURCE_BUNDLE.json", json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest(output),
            "files": len(paths), "base_revision": revision, "includes_working_tree_changes": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output), indent=2))


if __name__ == "__main__":
    main()
