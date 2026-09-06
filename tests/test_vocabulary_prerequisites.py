"""Numerical gates required before evaluating high-precision vocabulary codes."""
import pytest
import torch
from torch import nn

from rotquant.linear import QuantLinear
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import RandomizedHadamard, fwht


@pytest.mark.parametrize("scale_bits", [8, 16, 32])
@pytest.mark.parametrize("error_comp", ["none", "residual"])
def test_execution_dtype_does_not_round_packed_metadata(scale_bits, error_comp):
    torch.manual_seed(12)
    layer = QuantLinear.from_linear(nn.Linear(128, 16), QuantConfig(
        bits=4, scale_bits=scale_bits, error_comp=error_comp))
    before = layer.qweight.dequantize().clone()
    before_bytes = layer.packed_state_bytes()
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        layer.to(dtype=dtype)
        assert layer.packed_state_bytes() == before_bytes
        torch.testing.assert_close(layer.qweight.dequantize(), before, rtol=0, atol=0)


def test_normalized_half_fwht_does_not_overflow_intermediate_sums():
    x = torch.zeros(2, 128, dtype=torch.float16)
    x[:, :2] = 40000
    result = fwht(x)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, fwht(x.float()).half(), rtol=0, atol=0)
    assert result.dtype == x.dtype


def test_weight_rotation_is_in_fp32_before_quantizing():
    torch.manual_seed(10)
    layer = nn.Linear(128, 16, bias=False).half()
    rotation = RandomizedHadamard(128, block=128, seed=5)
    config = QuantConfig(bits=6, scale="mse_search")
    actual = QuantLinear.from_linear(layer, config, weight_rotation=rotation)
    expected = Quantizer(config).quantize_weight(
        rotation.rotate_weight(layer.weight.float()))
    assert torch.equal(actual.qweight.packed.data, expected.packed.data)


def test_uniform_control_uses_a_valid_scale_range():
    torch.manual_seed(1)
    w = torch.randn(256, 128)
    corrected = Quantizer(QuantConfig(bits=4, codebook="uniform", scale="mse_search"))
    old = Quantizer(QuantConfig(bits=4, codebook="uniform", scale="mse_search",
                               mse_search_lo=0.5, mse_search_hi=1.5))
    absmax = Quantizer(QuantConfig(bits=4, codebook="uniform", scale="absmax"))
    error = lambda q: (q.quantize_weight(w).dequantize() - w).square().mean()
    assert error(corrected) < error(old) * 0.4
    assert error(absmax) < error(old) * 0.5
    assert QuantConfig().mse_search_lo == 0.5
    assert QuantConfig(codebook="uniform").mse_search_hi == 4.0


def test_absmax_gptq_identity_matches_rounding():
    torch.manual_seed(19)
    w = torch.randn(8, 128)
    cfg = {"bits": 4, "codebook": "uniform", "scale": "absmax", "group_size": 32}
    plain = Quantizer(QuantConfig(**cfg)).quantize_weight(w)
    gptq = Quantizer(QuantConfig(**cfg, error_comp="gptq")).quantize_weight(w, H=torch.eye(128))
    torch.testing.assert_close(plain.dequantize(), gptq.dequantize(), rtol=0, atol=0)


@pytest.mark.parametrize("scale_bits", [8, 16, 32])
def test_scale_telemetry_measures_decoded_metadata(scale_bits):
    torch.manual_seed(31)
    weight = torch.randn(8, 128)
    quantizer = Quantizer(QuantConfig(bits=5, group_size=32, scale_bits=scale_bits,
                                     collect_scale_diagnostics=True))
    selected = quantizer.select_scales(weight)
    packed = quantizer.quantize_weight(weight)
    relative = (packed.main_scales().float() - selected).abs() / selected
    assert packed.scale_diagnostics["count"] == selected.numel()
    assert packed.scale_diagnostics["relative_rounding_error_max"] == pytest.approx(float(relative.max()))
