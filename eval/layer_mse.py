"""Per-layer normalised output-MSE and cosine drift diagnostics.

``layer_output_mse`` computes ``||xW - xW_hat||^2 / ||xW||^2`` per linear, captured
with hooks on the original vs quantised model over the same inputs. This is the
diagnostic that connects single-layer behaviour to end-to-end degradation and is
the primary signal for the E7 consistency trap (cosine drift across depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from rotquant.linear import QuantLinear


@dataclass
class LayerMSEResult:
    mse: Dict[str, float] = field(default_factory=dict)
    cosine: Dict[str, float] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)


def _linear_names(model: nn.Module) -> List[str]:
    return [n for n, m in model.named_modules()
            if isinstance(m, (nn.Linear, QuantLinear))]


@torch.no_grad()
def _capture_io(model: nn.Module, batch, device) -> Dict[str, tuple]:
    captured: Dict[str, tuple] = {}
    handles = []

    def mk(name):
        def hook(_m, inp, out):
            captured[name] = (inp[0].detach(), out.detach())
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, QuantLinear)):
            handles.append(mod.register_forward_hook(mk(name)))
    try:
        if isinstance(batch, dict):
            model(**{k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()})
        else:
            model(batch.to(device))
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def layer_output_mse(fp_model: nn.Module, quant_model: nn.Module, batch,
                     device=None) -> LayerMSEResult:
    """Normalised per-linear output MSE + cosine between fp and quantised outputs.

    Both models must share architecture/layer names (quant_model is the patched
    copy). Pass the *same* calibration batch to both.
    """
    device = device or next(fp_model.parameters()).device
    fp_io = _capture_io(fp_model, batch, device)
    q_io = _capture_io(quant_model, batch, device)

    result = LayerMSEResult()
    for name in fp_io:
        if name not in q_io:
            continue
        y = fp_io[name][1].reshape(-1).float()
        yq = q_io[name][1].reshape(-1).float()
        denom = (y.pow(2).sum() + 1e-12)
        result.mse[name] = float(((y - yq).pow(2).sum() / denom).item())
        result.cosine[name] = float(
            torch.nn.functional.cosine_similarity(y, yq, dim=0).item())
        result.order.append(name)
    return result
