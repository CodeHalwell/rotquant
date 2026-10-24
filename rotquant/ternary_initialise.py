"""Optional legal ternary fitting against captured rotated activation covariance."""
from __future__ import annotations

import torch

from ._internal import expand_scales
from .block_train import _tree_to


@torch.no_grad()
def rotated_covariance(source, site, rotation, records, device):
    covariance = torch.zeros(rotation.dim, rotation.dim, device=device, dtype=torch.float32)
    count = 0
    def capture(_module, args):
        nonlocal count
        values = rotation.rotate_activation(args[0]).reshape(-1, rotation.dim).float()
        covariance.add_(values.T @ values)
        count += values.shape[0]
    handle = source.get_submodule(site).register_forward_pre_hook(capture)
    try:
        for record in records:
            source(*_tree_to(record.args, device, None), **_tree_to(record.kwargs, device, None))
    finally:
        handle.remove()
    if not count or not torch.isfinite(covariance).all():
        raise ValueError("activation-aware ternary fitting requires finite captured activations")
    return covariance / count


def weighted_error(weight, quantized, covariance):
    residual = quantized - weight
    return ((residual @ covariance) * residual).sum(-1)


def legal_codes(weight, scales, group_size):
    expanded = expand_scales(scales, group_size, weight.shape[1])
    return torch.where(weight.abs() > expanded / 2, weight.sign(), 0)


def fit_scales(weight, codes, scales, covariance, group_size):
    quantized = codes * expand_scales(scales, group_size, weight.shape[1])
    for group, start in enumerate(range(0, weight.shape[1], group_size)):
        end = min(start + group_size, weight.shape[1])
        selected = codes[:, start:end]
        direction = selected @ covariance[start:end]
        denominator = (direction[:, start:end] * selected).sum(-1)
        residual = weight - quantized
        numerator = (direction * residual).sum(-1) + scales[:, group] * denominator
        proposal = (numerator / denominator.clamp_min(1e-30)).clamp(2 ** -24, 65504)
        proposal = proposal.half().float()
        candidate = quantized.clone()
        candidate[:, start:end] = selected * proposal[:, None]
        accept = weighted_error(weight, candidate, covariance) <= weighted_error(weight, quantized, covariance)
        scales[:, group] = torch.where(accept, proposal, scales[:, group])
        quantized = torch.where(accept[:, None], candidate, quantized)
    return scales


@torch.no_grad()
def activation_aware_initialise(module, covariance, *, iterations=4, row_batch=64):
    if covariance.shape != (module.in_features, module.in_features):
        raise ValueError("rotated covariance shape differs from ternary input width")
    covariance = covariance.to(module.latent_weight).float()
    if not torch.isfinite(covariance).all() or not torch.allclose(covariance, covariance.T):
        raise ValueError("rotated covariance must be finite and symmetric")
    lipschitz = covariance.abs().sum(-1).max().clamp_min(1e-12)
    before, after = 0., 0.
    initial_scales, initial_codes = module.stored_scales().detach(), module.hard_codes()
    for start in range(0, module.out_features, row_batch):
        end = min(start + row_batch, module.out_features)
        weight = module.latent_weight[start:end].detach()
        scales = initial_scales[start:end].clone()
        codes = initial_codes[start:end]
        quantized = codes * expand_scales(scales, module.config.group_size, module.in_features)
        before += float(weighted_error(weight, quantized, covariance).sum())
        for _ in range(iterations):
            proposed = legal_codes(quantized - ((quantized - weight) @ covariance) / lipschitz,
                                   scales, module.config.group_size)
            candidate = proposed * expand_scales(scales, module.config.group_size, module.in_features)
            accept = weighted_error(weight, candidate, covariance) < weighted_error(weight, quantized, covariance)
            codes = torch.where(accept[:, None], proposed, codes)
            scales = fit_scales(weight, codes, scales, covariance, module.config.group_size)
            quantized = codes * expand_scales(scales, module.config.group_size, module.in_features)
        after += float(weighted_error(weight, quantized, covariance).sum())
        module.latent_weight[start:end].copy_(quantized)
        module.log_scales[start:end].copy_(scales.log())
    return {"covariance_basis": "captured_rotated_activations", "weighted_error_before": before,
            "weighted_error_after": after, "iterations": iterations}
