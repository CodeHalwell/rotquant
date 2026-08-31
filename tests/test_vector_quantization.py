"""Finite-rate vector codebooks must be exact-rate research comparators."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from rotquant.checkpoint import _quantized_weight_spec
from rotquant.codebooks import (
    FiniteE8Codebook,
    VectorCodebook,
    build_finite_e8_codebook,
    fit_vector_codebook,
)
from rotquant.linear import QuantLinear
from rotquant.quantize import QuantConfig, Quantizer


def _config(codebook: str) -> QuantConfig:
    return QuantConfig(
        bits=2,
        codebook=codebook,
        scale="rms",
        group_size=32,
        vector_dim=2,
        vector_samples=4096,
        vector_iters=15,
        seed=7,
    )


def test_vector_codebook_fit_is_deterministic_and_fixed_rate() -> None:
    generator = torch.Generator().manual_seed(11)
    samples = torch.randn(2048, 2, generator=generator)
    first = fit_vector_codebook(samples, 16, seed=3, iters=10)
    second = fit_vector_codebook(samples, 16, seed=3, iters=10)

    assert torch.equal(first.centroids, second.centroids)
    assert first.code_bits == 4
    assert first.bits_per_weight == 2


def test_finite_e8p_is_a_real_two_bit_packed_codebook() -> None:
    codebook = build_finite_e8_codebook(2)
    assert isinstance(codebook, FiniteE8Codebook)
    assert codebook.levels == 65_536
    assert codebook.dim == 8
    assert codebook.bits_per_weight == 2

    torch.manual_seed(29)
    weight = torch.randn(8, 64)
    qweight = Quantizer(QuantConfig(
        bits=2,
        codebook="e8p",
        scale="rms",
        group_size=32,
    )).quantize_weight(weight)
    assert qweight.packed.bits == 16
    assert qweight.packed.shape == (8, 8)
    assert qweight.bit_budget().code_bits == 16
    assert torch.isfinite(qweight.dequantize()).all()

    samples = torch.randn(8, 8) * 2.5
    encoded = codebook.encode(samples)
    brute = (
        (samples[:, None, :] - codebook.centroids[None, :, :])
        .square()
        .sum(dim=-1)
        .argmin(dim=1)
    )
    assert torch.equal(encoded, brute)


def test_vector_quantizer_matches_scalar_storage_rate_and_improves_nmse() -> None:
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(64, 128, generator=generator)
    scalar = Quantizer(_config("gaussian")).quantize_weight(weight)
    vector = Quantizer(_config("vector")).quantize_weight(weight)

    assert vector.packed.bits == 4
    assert vector.packed.shape == (64, 64)
    assert vector.packed.data.numel() == scalar.packed.data.numel()
    assert vector.bit_budget().bits_per_weight == scalar.bit_budget().bits_per_weight
    scalar_nmse = (scalar.dequantize() - weight).square().mean()
    vector_nmse = (vector.dequantize() - weight).square().mean()
    assert vector_nmse < scalar_nmse


def test_vector_quantlinear_runs_without_a_dense_persistent_weight() -> None:
    torch.manual_seed(13)
    source = nn.Linear(16, 8, bias=True)
    module = QuantLinear.from_linear(source, _config("vector"), fallback=False)
    output = module(torch.randn(3, 16))

    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()
    assert module._fp_cache is None


def test_vector_profiles_fail_closed_on_unsupported_shapes_and_gptq() -> None:
    with pytest.raises(ValueError, match="divide group_size"):
        QuantConfig(bits=2, codebook="vector", group_size=3, vector_dim=2)
    with pytest.raises(ValueError, match="error_comp='none'"):
        QuantConfig(
            bits=2,
            codebook="vector",
            group_size=32,
            vector_dim=2,
            error_comp="gptq",
        )
    with pytest.raises(ValueError, match="divide in_features"):
        Quantizer(_config("vector")).quantize_weight(torch.randn(4, 15))
    wrong_dimension = VectorCodebook(torch.randn(16, 4))
    with pytest.raises(ValueError, match="dimension does not match"):
        Quantizer(_config("vector"), codebook=wrong_dimension)


def test_vector_checkpoint_contract_and_scale_training_boundary() -> None:
    source = nn.Linear(16, 8, bias=False)
    module = QuantLinear.from_linear(source, _config("vector"), fallback=False)

    spec = _quantized_weight_spec(0, module.qweight, {})
    assert spec["codebook"]["kind"] == "vector"
    assert spec["codebook"]["dimension"] == 2
    with pytest.raises(TypeError, match="scale fine-tuning"):
        module.enable_scale_finetuning()


def test_weight_calibrated_scalar_codebook_is_deterministic_and_deployable() -> None:
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(16, 64, generator=generator).pow(3)
    config = QuantConfig(
        bits=3,
        codebook="calibrated",
        scale="mse_search",
        group_size=32,
        calibrated_samples=512,
        calibrated_iters=20,
    )
    first = Quantizer(config).quantize_weight(weight)
    second = Quantizer(config).quantize_weight(weight)

    assert first.codebook.name == "calibrated_w3"
    assert first.codebook.levels == 8
    assert torch.equal(first.codebook.centroids, second.codebook.centroids)
    assert torch.equal(first.packed.data, second.packed.data)
    assert torch.isfinite(first.dequantize()).all()
