"""Per-layer normalised output-MSE and cosine drift diagnostics.

``layer_output_mse`` computes ``||xW - xW_hat||^2 / ||xW||^2`` per linear, captured
with hooks on the original vs quantised model over the same inputs. This is the
diagnostic that connects single-layer behaviour to end-to-end degradation and is
the primary signal for the E7 consistency trap (cosine drift across depth).

Because ``patch_model`` mutates the model in place, the runner uses the two-step
API: :func:`capture_outputs` on the fp model *before* patching, again after, then
:func:`drift_between` on the two captures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from rotquant.linear import QuantLinear


@dataclass
class LayerMSEResult:
    mse: dict[str, float] = field(default_factory=dict)
    cosine: dict[str, float] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)


@torch.no_grad()
def capture_outputs(model: nn.Module, batch, device,
                    to_cpu: bool = True) -> dict[str, torch.Tensor]:
    """Forward ``batch`` once, returning each linear's output keyed by module name.

    Outputs are detached (and moved to CPU by default so a capture of every layer
    of a large model does not sit in VRAM between the fp and quantised passes).
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def mk(name):
        def hook(_m, _inp, out):
            o = out.detach()
            captured[name] = o.cpu() if to_cpu else o
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
def drift_between(fp_out: dict[str, torch.Tensor],
                  q_out: dict[str, torch.Tensor]) -> LayerMSEResult:
    """Normalised per-layer output MSE + cosine between two captures.

    Layer names must match (patching replaces modules at the same paths, so the
    fp and quantised captures share keys). Iteration order follows the fp capture,
    which is module (depth) order.
    """
    result = LayerMSEResult()
    for name, fp_val in fp_out.items():
        if name not in q_out:
            continue
        y = fp_val.reshape(-1).float()
        yq = q_out[name].reshape(-1).to(y.device).float()
        denom = (y.pow(2).sum() + 1e-12)
        result.mse[name] = float(((y - yq).pow(2).sum() / denom).item())
        result.cosine[name] = float(
            torch.nn.functional.cosine_similarity(y, yq, dim=0).item())
        result.order.append(name)
    return result


@torch.no_grad()
def layer_output_mse(fp_model: nn.Module, quant_model: nn.Module, batch,
                     device=None) -> LayerMSEResult:
    """Normalised per-linear output MSE + cosine between fp and quantised outputs.

    Both models must share architecture/layer names (quant_model is the patched
    copy). Pass the *same* calibration batch to both.
    """
    device = device or next(fp_model.parameters()).device
    fp_out = capture_outputs(fp_model, batch, device)
    q_out = capture_outputs(quant_model, batch, device)
    return drift_between(fp_out, q_out)
