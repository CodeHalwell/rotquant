"""Bind allocation measurements to actual weights, code and execution."""
from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from pathlib import Path

import torch

from rotquant.utils import gpu_info


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(str((name, tuple(tensor.shape), tensor.dtype)).encode())
        for chunk in tensor.detach().reshape(-1).split(262144):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def score_cache_identity(model, config):
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "rotquant").rglob("*.py")), Path(__file__),
             root / "scripts/run_experiment.py"]
    tensors = list(model.parameters()) + list(model.buffers())
    return {
        "model_sha256": model_digest(model),
        "dtypes": sorted({str(t.dtype) for t in tensors if t.is_floating_point()}),
        "devices": sorted({str(t.device) for t in tensors}),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "gpu": gpu_info(), "cuda": torch.version.cuda,
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "threads": torch.get_num_threads()},
        "libraries": {name: metadata.version(name) for name in ("torch", "numpy", "scipy", "transformers")},
        "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "stage": {key: config.get(key) for key in ("dtype", "device", "stage_code_revision",
                   "stage_source_fingerprint", "stage_runtime_fingerprint")},
    }
