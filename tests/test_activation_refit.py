"""Stored scalar refinement must improve calibrated reconstruction, row by row."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rotquant.activation_refit import ActivationRefitter, StoredScaleGrid
from rotquant.pack import unpack_indices
from rotquant.quantize import QuantConfig, Quantizer


def _problem():
    generator = torch.Generator().manual_seed(61)
    weight = torch.randn(8, 256, generator=generator) * torch.linspace(0.2, 2, 8)[:, None]
    inputs = torch.randn(96, 256, generator=generator) * torch.logspace(-1, 1, 256)
    return weight, inputs, inputs.T @ inputs / inputs.shape[0]


def _config(group_size=64, **options):
    return QuantConfig(
        bits=4, codebook="gaussian", scale="mse_search", error_comp="gptq",
        group_size=group_size, scale_bits=8, scale_quant_group_size=16, **options,
    )


def _output_error(weight, reconstruction, inputs):
    return ((weight.double() - reconstruction.double()) @ inputs.double().T).square().mean(1)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_scale_refit_reduces_anisotropic_output_error_on_stored_scale8(group_size):
    weight, inputs, hessian = _problem()
    config = _config(group_size)
    base = Quantizer(config).quantize_weight(weight, H=hessian)
    fitted = Quantizer(replace(config, scale_refit="hessian")).quantize_weight(weight, H=hessian)
    old_error = _output_error(weight, base.dequantize(), inputs)
    new_error = _output_error(weight, fitted.dequantize(), inputs)
    assert new_error.sum() < old_error.sum() * 0.99
    assert (new_error <= old_error + 1e-9).all()
    assert torch.equal(fitted.packed.data, base.packed.data)
    assert torch.equal(fitted.scale_offsets, base.scale_offsets)
    assert torch.equal(fitted.scale_steps, base.scale_steps)
    assert fitted.scales.dtype == torch.uint8
    assert fitted.bit_budget() == base.bit_budget()
    stats = fitted.scale_diagnostics["activation_refit"]
    assert stats["objective_after"] == pytest.approx(float(new_error.sum()), rel=2e-5)
    assert stats["objective_before"] == pytest.approx(float(old_error.sum()), rel=2e-5)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_code_refit_reduces_output_error_without_changing_scale_storage(group_size):
    weight, inputs, hessian = _problem()
    config = _config(group_size)
    base = Quantizer(config).quantize_weight(weight, H=hessian)
    fitted = Quantizer(replace(config, code_refit_passes=1)).quantize_weight(weight, H=hessian)
    old_error = _output_error(weight, base.dequantize(), inputs)
    new_error = _output_error(weight, fitted.dequantize(), inputs)
    assert new_error.sum() < old_error.sum() * 0.9
    assert (new_error <= old_error + 1e-9).all()
    assert torch.equal(fitted.scales, base.scales)
    assert torch.equal(fitted.scale_offsets, base.scale_offsets)
    assert torch.equal(fitted.scale_steps, base.scale_steps)
    movement = unpack_indices(fitted.packed) - unpack_indices(base.packed)
    assert movement.abs().max() == 1
    assert fitted.bit_budget() == base.bit_budget()


def test_combined_refit_is_deterministic_and_preserves_reused_inputs():
    weight, inputs, hessian = _problem()
    quantizer = Quantizer(_config(scale_refit="hessian", scale_refit_passes=2, code_refit_passes=2))
    scales = quantizer.select_scales(weight)
    originals = [value.clone() for value in (weight, hessian, scales, quantizer.codebook.centroids)]
    first = quantizer.quantize_weight(weight, H=hessian, scales_override=scales)
    second = quantizer.quantize_weight(weight, H=hessian, scales_override=scales)
    for actual, original in zip((weight, hessian, scales, quantizer.codebook.centroids), originals):
        assert torch.equal(actual, original)
    assert torch.equal(first.packed.data, second.packed.data)
    assert torch.equal(first.scales, second.scales)
    assert torch.equal(first.dequantize(), second.dequantize())
    assert first.scale_diagnostics == second.scale_diagnostics
    stats = first.scale_diagnostics["activation_refit"]
    error = _output_error(weight, first.dequantize(), inputs).sum()
    assert stats["objective_after"] == pytest.approx(float(error), rel=2e-5)
    assert stats["objective_after"] < stats["objective_before"]


@pytest.mark.parametrize("scale_passes,code_passes,expected", [
    (2, 2, ["scale", "code", "scale", "code"]),
    (2, 1, ["scale", "code", "scale"]),
    (1, 3, ["scale", "code", "code", "code"]),
])
def test_joint_sweeps_alternate_and_scales_use_the_latest_codes(
        monkeypatch, scale_passes, code_passes, expected):
    events, coefficients, after_codes = [], [], []
    original_scales, original_codes = ActivationRefitter.refine_scales, ActivationRefitter.refine_codes

    def scale_sweep(self, *args):
        events.append("scale")
        coefficients.append(args[2].clone())
        return original_scales(self, *args)

    def code_sweep(self, *args):
        events.append("code")
        result = original_codes(self, *args)
        after_codes.append(args[4][result[1]].clone())
        return result

    monkeypatch.setattr(ActivationRefitter, "refine_scales", scale_sweep)
    monkeypatch.setattr(ActivationRefitter, "refine_codes", code_sweep)
    weight, inputs, hessian = _problem()
    config = _config(scale_refit="hessian", scale_refit_passes=scale_passes, code_refit_passes=code_passes)
    fitted = Quantizer(config).quantize_weight(weight, H=hessian)
    assert events == expected
    if scale_passes > 1:
        assert torch.equal(coefficients[1], after_codes[0])
        assert not torch.equal(coefficients[0], coefficients[1])
    stats = fitted.scale_diagnostics["activation_refit"]
    error = _output_error(weight, fitted.dequantize(), inputs).sum()
    assert error < stats["objective_before"]
    assert float(error) == pytest.approx(stats["objective_after"], rel=2e-5)


@pytest.mark.parametrize("zero_hessian", [False, True])
def test_singular_second_moments_are_finite_and_zero_second_moment_is_noop(zero_hessian):
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(3, 70, generator=generator)
    inputs = torch.randn(1, 70, generator=generator)
    if zero_hessian:
        inputs.zero_()
    hessian = inputs.T @ inputs
    config = QuantConfig(bits=4, group_size=32, scale_bits=8, scale_quant_group_size=7)
    base = Quantizer(config).quantize_weight(weight)
    fitted = Quantizer(replace(config, scale_refit="hessian", code_refit_passes=1)).quantize_weight(
        weight, H=hessian)
    assert torch.isfinite(fitted.dequantize()).all()
    assert (_output_error(weight, fitted.dequantize(), inputs)
            <= _output_error(weight, base.dequantize(), inputs) + 1e-8).all()
    if zero_hessian:
        assert torch.equal(fitted.packed.data, base.packed.data)
        assert torch.equal(fitted.scales, base.scales)
        assert fitted.scale_diagnostics["activation_refit"]["accepted_rows"] == 0


def test_scale_refit_accounts_for_error_in_other_groups():
    inputs = torch.tensor([[1.0, 0.0, 0.8, 0.0], [0.0, 1.0, 0.0, 0.8],
                           [0.0, 0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 0.6]])
    hessian = inputs.T @ inputs
    weight = torch.tensor([[1.2, 1.2, 1.0, 1.0]])
    base = torch.ones_like(weight)
    indices = torch.zeros_like(weight, dtype=torch.int64)
    scales = torch.ones(1, 2)
    fitted, _, _ = ActivationRefitter(hessian, 2).refit(
        weight, base.clone(), indices.clone(), torch.ones(1),
        StoredScaleGrid(scales, scales, 32, 256))
    diagonal, _, _ = ActivationRefitter(torch.diag(hessian.diag()), 2).refit(
        weight, base.clone(), indices.clone(), torch.ones(1),
        StoredScaleGrid(scales, scales, 32, 256))
    assert fitted[0, 1] > 1
    assert diagonal[0, 1] == 1
    reconstruction = fitted.repeat_interleave(2, dim=1)
    assert _output_error(weight, reconstruction, inputs) < _output_error(weight, base, inputs)


def test_code_refit_rejects_joint_moves_with_harmful_cross_terms():
    hessian = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    weight = torch.tensor([[0.6, 0.1]])
    reconstruction = torch.zeros_like(weight)
    indices = torch.zeros_like(weight, dtype=torch.int64)
    scales = torch.ones_like(weight)
    _, _, stats = ActivationRefitter(hessian, 1, scale_passes=0, code_passes=1).refit(
        weight, reconstruction, indices, torch.tensor([0.0, 1.0]),
        StoredScaleGrid(scales, scales, 32, 256))
    assert torch.equal(indices, torch.tensor([[1, 0]]))
    assert torch.equal(reconstruction, torch.tensor([[1.0, 0.0]]))
    assert stats["objective_after"] < stats["objective_before"]


def test_scale8_grid_spanning_row_batches_produces_identical_results():
    weight, _, hessian = _problem()
    base = Quantizer(_config()).quantize_weight(weight, H=hessian)
    grid = StoredScaleGrid(base.main_scales(), base.scales, 8, 16,
                           base.scale_offsets, base.scale_steps)
    results = []
    for batch in (2, 128):
        reconstruction = base.dequantize()
        indices = unpack_indices(base.packed).reshape_as(weight)
        _, stored, _ = ActivationRefitter(
            hessian, 64, scale_passes=2, code_passes=1, row_batch=batch,
        ).refit(weight, reconstruction, indices, base.codebook.centroids, grid)
        results.append((reconstruction, indices, stored))
    for small, large in zip(*results):
        assert torch.equal(small, large)


@pytest.mark.parametrize("scale_bits", [8, 16, 32])
def test_refined_packed_weight_roundtrips_exactly(tmp_path, scale_bits):
    safetensors = pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from rotquant.checkpoint import _quantized_weight_spec, _read_quantized_weight

    weight, _, hessian = _problem()
    config = replace(_config(), scale_refit="hessian", code_refit_passes=1, scale_bits=scale_bits)
    fitted = Quantizer(config).quantize_weight(weight, H=hessian)
    tensors = {}
    spec = _quantized_weight_spec(0, fitted, tensors)
    path = tmp_path / "refined.safetensors"
    save_file(tensors, path)
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        restored = _read_quantized_weight(handle, spec)
    assert torch.equal(fitted.dequantize(), restored.dequantize())
    assert torch.equal(fitted.packed.data, restored.packed.data)
    assert torch.equal(fitted.scales, restored.scales)
    assert fitted.bit_budget() == restored.bit_budget()


@pytest.mark.parametrize("options", [
    {"scale_refit": "diagonal"}, {"scale_refit_passes": 0}, {"scale_refit_passes": True},
    {"code_refit_passes": -1}, {"code_refit_passes": 0.5}, {"code_refit_passes": True},
    {"scale_refit": "hessian", "codebook": "vector"},
    {"scale_refit": "hessian", "error_comp": "residual"},
    {"code_refit_passes": 1, "scale": "turboquant"},
])
def test_invalid_refit_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        QuantConfig(**options)


def test_activation_refit_requires_calibrated_second_moment():
    quantizer = Quantizer(QuantConfig(bits=4, scale_refit="hessian"))
    with pytest.raises(ValueError, match="calibrated Hessian"):
        quantizer.quantize_weight(torch.ones(2, 4))


@pytest.mark.parametrize("hessian, message", [
    (torch.eye(3), "input-sized"),
    (torch.eye(4) * float("nan"), "finite"),
    (torch.triu(torch.ones(4, 4)), "symmetric"),
    (-torch.eye(4), "nonnegative"),
])
def test_invalid_second_moment_is_rejected(hessian, message):
    quantizer = Quantizer(QuantConfig(bits=4, group_size=2, scale_refit="hessian"))
    with pytest.raises(ValueError, match=message):
        quantizer.quantize_weight(torch.ones(2, 4), H=hessian)
