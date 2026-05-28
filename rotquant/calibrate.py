"""Calibration: forward-hook activation capture and incremental layer Hessians.

For each target ``nn.Linear`` we accumulate the (input) Hessian ``H = sum x^T x``
incrementally over a calibration set -- we never store the activations themselves.
A damping term (default 1% of the mean diagonal) is added, auto-increasing on
Cholesky failure, which is the instability the spec warns about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .utils import get_logger

logger = get_logger()


class HessianAccumulator:
    """Incrementally accumulates ``H = sum_t x_t^T x_t`` for one linear layer."""

    def __init__(self, in_features: int, device=None, dtype=torch.float32):
        self.in_features = in_features
        self.H = torch.zeros(in_features, in_features, device=device, dtype=dtype)
        self.n_samples = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # x: (..., in_features) -> flatten leading dims to tokens
        x = x.reshape(-1, self.in_features).to(self.H.dtype)
        n = x.shape[0]
        if n == 0:
            return
        # Running mean of x^T x keeps the scale stable across batches.
        self.H *= self.n_samples / (self.n_samples + n)
        self.H += (x.transpose(0, 1) @ x) / (self.n_samples + n)
        self.n_samples += n

    def finalize(self, damp_frac: float = 0.01) -> torch.Tensor:
        H = self.H.clone()
        mean_diag = torch.diag(H).mean().clamp_min(1e-8)
        H[range(self.in_features), range(self.in_features)] += damp_frac * mean_diag
        return H


@dataclass
class CalibrationResult:
    hessians: Dict[str, torch.Tensor] = field(default_factory=dict)
    n_samples: Dict[str, int] = field(default_factory=dict)


def _iter_linears(model: nn.Module, include: Optional[Iterable[str]] = None):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if include is None or any(k in name for k in include):
                yield name, mod


@torch.no_grad()
def collect_hessians(model: nn.Module, dataloader: Iterable, device,
                     include: Optional[Iterable[str]] = None,
                     max_batches: Optional[int] = None,
                     damp_frac: float = 0.01) -> CalibrationResult:
    """Run the model over calibration batches, capturing per-linear input Hessians.

    ``dataloader`` yields tensors / dicts suitable for ``model(**batch)`` or
    ``model(batch)``. Use 128-512 sequences of the model's context length.
    """
    accums: Dict[str, HessianAccumulator] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module, inputs, _output):
            x = inputs[0]
            if name not in accums:
                accums[name] = HessianAccumulator(in_features, device=x.device)
            accums[name].update(x)
        return hook

    for name, mod in _iter_linears(model, include):
        handles.append(mod.register_forward_hook(make_hook(name, mod.in_features)))

    model.eval()
    try:
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            if isinstance(batch, dict):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                model(**batch)
            elif isinstance(batch, (list, tuple)):
                model(*[b.to(device) if torch.is_tensor(b) else b for b in batch])
            else:
                model(batch.to(device))
    finally:
        for h in handles:
            h.remove()

    result = CalibrationResult()
    for name, acc in accums.items():
        result.hessians[name] = acc.finalize(damp_frac=damp_frac)
        result.n_samples[name] = acc.n_samples
    logger.info("Collected Hessians for %d linear layers", len(result.hessians))
    return result
