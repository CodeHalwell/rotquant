"""Real CPU-only bootstrap regression for a disposable Linux test container.

Requires preinstalled CPU Torch/base pip and absent ensurepip. Runs the same
managed installs as Colab (network required), never a native build or GPU job.
Does not disable/remove ensurepip itself or modify the base Python environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_gpu_environment import prepare_environment, torch_origin
from scripts.native_gpu_workflow import Workflow, digest, write_json


def base_snapshot():
    return {"packages": sorted((d.metadata["Name"], d.version, str(d.locate_file("")))
                               for d in metadata.distributions()
                               if Path(d.locate_file("")).resolve().is_relative_to(Path(sys.base_prefix).resolve())),
            "torch_file": torch_origin(), "torch_sha256": digest(torch_origin())}


IMPORT_CHECK = '''
import json, sys
from pathlib import Path
from importlib import metadata
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
import rotquant
from scripts.preflight_native_gpu import PACKAGES
versions = {name: metadata.version(name) for name in PACKAGES}
assert versions == PACKAGES, versions
assert torch.equal(torch.ones(2, 2) @ torch.ones(2, 2), torch.full((2, 2), 2.))
report = {"passed": True, "packages": versions, "torch": torch.__version__,
          "torch_file": torch.__file__, "rotquant_file": rotquant.__file__,
          "qwen_import": Qwen3_5ForCausalLM.__name__, "prefix": sys.prefix,
          "boundary": "CPU bootstrap/import check only; no GPU or model validation."}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\\n")
print(json.dumps(report), flush=True)
'''


def check(work_dir, output_dir):
    if platform.system() != "Linux" or importlib.util.find_spec("ensurepip") is not None:
        raise ValueError("Run only in a disposable Linux container with ensurepip already absent")
    work_dir.mkdir(parents=True, exist_ok=False)
    before = base_snapshot()
    expected_torch = metadata.version("torch")
    controls = {"protocol": "native-gpu-pipless-bootstrap-regression-v1",
                "python": sys.version, "platform": platform.platform(),
                "torch": expected_torch, "base_pip": metadata.version("pip"),
                "ensurepip_available": False,
                "sources": {str(p.relative_to(ROOT)): digest(p) for p in (
                    Path(__file__), ROOT / "scripts/native_gpu_environment.py",
                    ROOT / "scripts/run_native_gpu_validation.py", ROOT / "requirements/native-gpu.txt")}}
    results = []
    with Workflow(output_dir, ROOT, controls, 12) as workflow:
        partial = work_dir / "partial"
        # Reproduce the user's actual failing command, not a mocked exception.
        def reproduce(stage):
            command = [sys.executable, "-m", "venv", "--system-site-packages", "--copies", partial]
            try:
                stage.command(command, "old-ensurepip-command")
            except subprocess.CalledProcessError as error:
                result = {"passed": error.returncode != 0 and (partial / "bin/python").exists(),
                          "returncode": error.returncode, "partial_python_exists": (partial / "bin/python").exists()}
                if not result["passed"]:
                    raise ValueError("Did not reproduce the original partial-venv failure") from error
                path = stage.directory / "reproduction.json"
                write_json(path, result)
                return result, [path]
            raise ValueError("Old ensurepip command unexpectedly succeeded")
        workflow.stage("reproduce-original-failure", reproduce, reuse=False)
        for case, target in (("fresh", work_dir / "fresh"), ("partial-recovery", partial), ("repeat", partial)):
            def install(stage, target=target):
                prepare_environment(stage, ROOT, target, expected_torch)
                report = stage.directory / "imports.json"
                stage.command([target / "bin/python", "-u", "-c", IMPORT_CHECK, report], "real-pinned-imports")
                if base_snapshot() != before:
                    raise ValueError("Managed installation modified base package metadata or Torch origin")
                result = json.loads(report.read_text())
                result["base_packages_unchanged"] = True
                write_json(report, result)
                return result, [report]
            results.append(workflow.stage(case, install, minutes=8, reuse=False))
        write_json(output_dir / "bootstrap-regression.json", {
            **controls, "passed": all(r["passed"] and r["base_packages_unchanged"] for r in results),
            "cases": ["fresh", "partial-recovery", "repeat"], "results": results,
            "base_before": before, "base_after": base_snapshot(),
            "boundary": "Real Linux CPU dependency install, not Colab/CUDA/full-4B execution."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="new disposable directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check(args.work_dir, args.output_dir)
