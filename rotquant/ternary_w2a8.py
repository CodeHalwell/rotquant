"""Grouped W2A8 numerical oracle; this is an explicit new arithmetic lane."""
from __future__ import annotations

import types
from contextlib import contextmanager

import torch

from .ternary_latents import ternary_sites, validate_ternary_module
from .ternary_w2a8_reference import native_w2a8_reference

__all__ = ["grouped_q8", "grouped_w2a8_model", "native_w2a8_reference", "ternary_w2a8_forward"]


def grouped_q8(values, group_size=128):
    if values.shape[-1] % group_size or not torch.isfinite(values).all():
        raise ValueError("W2A8 requires finite complete activation groups")
    groups = values.float().reshape(*values.shape[:-1], -1, group_size)
    maxima = groups.abs().amax(-1, keepdim=True)
    scales = torch.where(maxima > 0, maxima / 127, torch.ones_like(maxima))
    codes = torch.round(groups / scales).clamp(-127, 127).to(torch.int8)
    return codes, scales.squeeze(-1)


def rounded_signed_fwht(values, signs):
    work = values.float().half().float() * signs.float()
    original = work.shape
    work = work.reshape(-1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = work.reshape(-1, 128 // (2 * stride), 2, stride)
        lo, hi = pairs[:, :, 0], pairs[:, :, 1]
        work = torch.stack((lo + hi, lo - hi), dim=2).reshape(-1, 128)
    return (work * 0.08838834764831845).half().float().reshape(original)


def ternary_group_codes(module, first, stop):
    words_per_row = module.in_features // 16
    rows = torch.arange(first, stop, device=module.qweight.packed.data.device)
    columns = torch.arange(8, device=rows.device)
    groups = torch.arange(module.in_features // 128, device=rows.device)
    indices = rows[None, :, None] * words_per_row + groups[:, None, None] * 8 + columns[None, None]
    words = module.qweight.packed.data[indices].long()
    shifts = torch.arange(16, device=rows.device) * 2
    codes = ((words[..., None] >> shifts) & 3).reshape(len(groups), stop - first, 128)
    return codes.float() - 1


def ternary_w2a8_forward(module, values, *, validated=False, row_batch=64):
    if not validated:
        validate_ternary_module(module)
    if module.qweight.group_size != 128 or module.act_rotation.block != 128:
        raise ValueError("W2A8 requires group128 and rotation128")
    rotated = rounded_signed_fwht(values, module.act_rotation.signs)
    activation, activation_scales = grouped_q8(rotated)
    shape = values.shape[:-1]
    activation = activation.reshape(-1, module.in_features // 128, 128).float()
    activation_scales = activation_scales.reshape(-1, module.in_features // 128)
    output = torch.zeros(activation.shape[0], module.out_features, device=values.device)
    with exact_integer_matmul():
        for first in range(0, module.out_features, row_batch):
            stop = min(first + row_batch, module.out_features)
            codes = ternary_group_codes(module, first, stop)
            integer_dot = torch.bmm(activation.transpose(0, 1), codes.transpose(1, 2))
            weight_scales = module.qweight.scales[first:stop].float().T[:, None]
            scales = activation_scales.T[:, :, None] * weight_scales
            products = integer_dot * scales
            sums = output[:, first:stop]
            for group in range(module.in_features // 128):
                sums = sums + products[group]
            output[:, first:stop] = sums
    output = output.half().to(values.dtype).reshape(*shape, module.out_features)
    return output if module.bias is None else output + module.bias.to(output.dtype)


@contextmanager
def exact_integer_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@contextmanager
def grouped_w2a8_model(model):
    restored = []
    try:
        for module in ternary_sites(model).values():
            validate_ternary_module(module)
            if module.qweight.group_size != 128 or module.act_rotation.block != 128:
                raise ValueError("W2A8 requires every ternary site to use group128/rotation128")
            restored.append((module, "forward" in vars(module), module.forward))
            def forward(this, value):
                return ternary_w2a8_forward(this, value, validated=True)
            module.forward = types.MethodType(forward, module)
        yield {"numerical_lane": "group128_signed_fwht_q8_dp4a", "sites": list(ternary_sites(model)),
               "weight_row_staging": 64, "dense_weights_retained": False}
    finally:
        for module, had_override, original in reversed(restored):
            if had_override:
                module.forward = original
            else:
                del module.forward
