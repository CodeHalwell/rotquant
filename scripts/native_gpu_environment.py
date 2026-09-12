"""Managed native-test dependencies without Colab's optional ensurepip module.

The notebook interpreter supplies pip; pip's --python option targets the venv.
Never install into the notebook interpreter or replace its CUDA Torch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path


def torch_origin():
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise ValueError("The notebook interpreter must already provide PyTorch")
    return str(Path(spec.origin).resolve())


def inspect_target(prefix, expected_torch, expected_torch_file):
    prefix = Path(prefix).resolve()
    if Path(sys.prefix).resolve() != prefix or sys.prefix == sys.base_prefix:
        raise ValueError("Dependency target is not the requested virtual environment")
    config = (prefix / "pyvenv.cfg").read_text()
    settings = dict(line.split(" = ", 1) for line in config.splitlines() if " = " in line)
    if settings.get("include-system-site-packages") != "true":
        raise ValueError("Managed environment must inherit the notebook's CUDA PyTorch")
    version, origin = metadata.version("torch"), torch_origin()
    if version != expected_torch or origin != str(Path(expected_torch_file).resolve()):
        raise ValueError("Managed environment changed/shadowed the notebook's PyTorch")
    return {"passed": True, "python": sys.version, "prefix": str(prefix),
            "base_prefix": sys.base_prefix, "torch": version, "torch_file": origin,
            "include_system_site_packages": True}


def prepare_environment(stage, repo, directory, expected_torch):
    """Run the real, logged installation path; safe to retry a partial venv.

    Reapply venv configuration on every attempt without clearing its contents.
    An executable left behind by a failed ensurepip step is not a health check.
    Base pip >=22.3 is required; no apt/global-pip upgrade or download script.
    """
    repo, directory = Path(repo), Path(directory)
    python = directory / "bin/python"
    origin = torch_origin()
    stage.command([sys.executable, "-m", "venv", "--without-pip",
                   "--system-site-packages", "--copies", directory], "create-environment")
    check = [python, "-u", Path(__file__).resolve(), "--prefix", directory,
             "--expected-torch", expected_torch, "--expected-torch-file", origin]
    stage.command([*check, "--output", stage.directory / "bootstrap-before.json"], "verify-target")
    # --isolated ignores user pip settings/PIP_* overrides. --require-virtualenv
    # is evaluated by the selected interpreter, not by the notebook's Python.
    pip = [sys.executable, "-m", "pip", "--isolated", "--python", python,
           "--require-virtualenv", "--disable-pip-version-check"]
    stage.command([*pip, "--version"], "check-installer")
    constraints = stage.directory / "torch-constraint.txt"
    constraints.write_text(f"torch=={expected_torch}\n")
    stage.command([*pip, "install", "-r", repo / "requirements/native-gpu.txt", "-c", constraints,
                   "--report", stage.directory / "dependency-install.json"], "dependencies")
    stage.command([*pip, "install", "-e", repo, "--no-deps",
                   "--report", stage.directory / "rotquant-install.json"], "install-rotquant")
    stage.command([*check, "--output", stage.directory / "bootstrap-after.json"], "verify-installed-target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--expected-torch", required=True)
    parser.add_argument("--expected-torch-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_target(args.prefix, args.expected_torch, args.expected_torch_file)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
