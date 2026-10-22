"""Content-bound, atomic local resume records for ternary block training."""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path

import torch

from .codebooks import ScalarCodebook, VectorCodebook
from .linear import QuantLinear
from .pack import PackedTensor


def _hash_update(digest, value):
    if torch.is_tensor(value):
        data = value.detach().cpu().contiguous()
        digest.update(str((str(data.dtype), tuple(data.shape))).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(str(("dict", len(value))).encode())
        for key in sorted(value):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(str((type(value).__name__, len(value))).encode())
        for item in value:
            _hash_update(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
        digest.update(repr((type(value).__name__, value)).encode())
    else:
        raise TypeError(f"unsupported resume identity value: {type(value).__name__}")


def content_digest(value) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _quantized_source_state(weight):
    values = {}
    for name, value in vars(weight).items():
        if name == "scale_diagnostics":
            continue
        if isinstance(value, PackedTensor):
            values[name] = {"data": value.data, "shape": value.shape,
                            "bits": value.bits, "numel": value.numel}
        elif isinstance(value, (ScalarCodebook, VectorCodebook)):
            values[name] = {"type": type(value).__name__, "name": value.name,
                            "centroids": value.centroids}
        else:
            values[name] = value
    return values


def source_state_digest(source):
    # QuantLinear's packed arrays are dataclass fields, not state_dict tensors.
    # MLP QAT can start with an already-quantised W4 attention remainder.
    packed = {name: _quantized_source_state(module.qweight)
              for name, module in source.named_modules() if isinstance(module, QuantLinear)}
    return content_digest({"ordinary": source.state_dict(), "packed": packed})


def checkpoint_contract(source, name, config, train_calls, validation_calls):
    settings = asdict(config)
    for key in ("checkpoint_dir", "checkpoint_interval", "resume", "max_blocks"):
        settings.pop(key, None)
    def calls_digest(records):
        return content_digest([(record.args, record.kwargs, record.output) for record in records])
    return {
        "version": 4, "name": name, "config": content_digest(settings),
        "source": source_state_digest(source),
        "train": calls_digest(train_calls), "validation": calls_digest(validation_calls),
    }


def save_training_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_state(path: Path | None, contract: dict, resume: bool):
    if path is None or not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"ternary checkpoint exists; enable resume or choose a new directory: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("contract") != contract:
        raise ValueError("ternary resume source/data/config identity mismatch")
    return state


def parameter_snapshot(module):
    return {name: value.detach().cpu().clone()
            for name, value in module.named_parameters() if value.requires_grad}


@torch.no_grad()
def restore_parameters(module, state):
    parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("ternary resume trainable parameter names differ")
    for name, value in parameters.items():
        if value.shape != state[name].shape or not torch.isfinite(state[name]).all():
            raise ValueError("ternary resume parameters are invalid")
        value.copy_(state[name].to(value.device))
