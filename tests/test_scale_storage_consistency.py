"""Packed codes must be assigned against exactly the scales the artifact stores.

With 8-bit double-quantised scales the affine grid of a block depends on the
block's extreme values, so encoding scales twice (once for assignment, once for
storage) can place them on two different grids.  Every path that assigns codes
must therefore encode once and retain that triple verbatim.
"""
from __future__ import annotations

import pytest
import torch

from rotquant.pack import unpack_indices
from rotquant.quantize import (
    QuantConfig,
    Quantizer,
    _encoded_storage_scales,
    _quantize_groups,
)


def _weight(out: int = 96, d: int = 512, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    row_scale = torch.linspace(0.3, 3.0, out).unsqueeze(1)
    return torch.randn(out, d, generator=generator) * row_scale


def _hessian(d: int, n: int = 2048, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator) * torch.linspace(0.2, 2.0, d)
    return x.T @ x / n


@pytest.mark.parametrize("scale_bits", [8.0, 16.0])
def test_plain_rounding_codes_match_stored_scales(scale_bits):
    w = _weight()
    quantizer = Quantizer(QuantConfig(
        bits=4, codebook="gaussian", scale="mse_search", group_size=128,
        scale_bits=scale_bits, scale_quant_group_size=256))
    qw = quantizer.quantize_weight(w)
    stored_scales = qw.main_scales()
    q, idx = _quantize_groups(w, stored_scales, qw.codebook, 128)
    assert torch.equal(idx, unpack_indices(qw.packed).reshape_as(w))
    assert torch.equal(q, qw.dequantize())


@pytest.mark.parametrize("scale_bits", [8.0, 16.0])
def test_gptq_codes_match_stored_scales_with_lazy_refit(scale_bits):
    w = _weight()
    hessian = _hessian(w.shape[1])
    quantizer = Quantizer(QuantConfig(
        bits=4, codebook="gaussian", scale="mse_search", group_size=128,
        error_comp="gptq", gptq_actorder=True, gptq_recompute_scales=True,
        scale_bits=scale_bits, scale_quant_group_size=256))
    selected = quantizer.select_scales(w)
    decoded, _stored, offsets, steps = _encoded_storage_scales(
        selected, scale_bits, 256)
    grid = (offsets, steps) if scale_bits == 8.0 else None
    q_assigned, idx, _decoded, _encoded = quantizer._gptq_with_scale_grid(
        w.clone(), decoded, hessian.clone(), grid)
    qw = quantizer.quantize_weight(w, H=hessian)
    assert torch.equal(idx, unpack_indices(qw.packed).reshape_as(w))
    # The deployed reconstruction is bit-identical to what GPTQ assigned
    # codes against; the previous per-group/row-major double encoding left a
    # ~1e-3 relative discrepancy on the 8-bit path.
    assert torch.equal(qw.dequantize(), q_assigned)


def test_gptq_rejects_scale_grid_for_wide_scales():
    w = _weight(out=8, d=128)
    quantizer = Quantizer(QuantConfig(
        bits=3, group_size=32, error_comp="gptq", scale_bits=16.0))
    decoded, _stored, offsets, steps = _encoded_storage_scales(
        quantizer.select_scales(w), 8.0, 256)
    with pytest.raises(ValueError, match="only meaningful for 8-bit"):
        quantizer._gptq_with_scale_grid(
            w.clone(), decoded, torch.eye(128), (offsets, steps))


@pytest.mark.parametrize("scale_bits", [8.0, 16.0])
def test_residual_codes_match_stored_residual_scales(scale_bits):
    w = _weight(out=48, d=384)
    quantizer = Quantizer(QuantConfig(
        bits=3, codebook="gaussian", scale="rms", group_size=128,
        error_comp="residual", residual_bits=2,
        scale_bits=scale_bits, scale_quant_group_size=256))
    qw = quantizer.quantize_weight(w)
    primary, _ = _quantize_groups(w, qw.main_scales(), qw.codebook, 128)
    residual = w - primary
    _rq, ridx = _quantize_groups(
        residual, qw.residual_scale_values(), qw.residual_codebook, 128)
    assert torch.equal(ridx, unpack_indices(qw.residual_packed).reshape_as(w))


def test_encoded_scales_are_retained_verbatim_rather_than_re_encoded():
    scales = torch.rand(300, 40) * 3 + 0.01
    decoded, stored, offsets, steps = _encoded_storage_scales(scales, 8.0, 256)
    assert stored.dtype == torch.uint8
    assert offsets.dtype == torch.float16 and steps.dtype == torch.float16
    assert torch.equal(
        decoded,
        offsets.float()[torch.arange(scales.numel()) // 256].reshape_as(scales)
        + stored.float()
        * steps.float()[torch.arange(scales.numel()) // 256].reshape_as(scales),
    )
