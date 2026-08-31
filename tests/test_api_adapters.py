"""Public optimisation API and architecture registry behaviour."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import rotquant.patch as patch_module
from rotquant import (
    AdapterRegistry,
    ModelAdapter,
    QuantLinear,
    RotQuantConfig,
    inspect_model,
    list_model_adapters,
    optimize_model,
    resolve_model_adapter,
)


class TinyModel(nn.Module):
    def __init__(self, model_type: str = "llama") -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.proj = nn.Linear(8, 6, bias=False)
        self.lm_head = nn.Linear(6, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.proj(value))


class Conv1DLike(nn.Module):
    """Minimal Transformers-Conv1D-style transposed weight projection."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight + self.bias


class Conv1DLikeAdapter(ModelAdapter):
    def iter_quantizable_modules(self, model):
        for name, module in model.named_modules():
            if isinstance(module, Conv1DLike):
                yield name, module

    def to_linear(self, module):
        linear = nn.Linear(
            module.weight.shape[0],
            module.weight.shape[1],
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(module.weight.T)
            linear.bias.copy_(module.bias)
        return linear


def test_adapter_resolution_is_architecture_aware_with_safe_fallback() -> None:
    assert resolve_model_adapter(TinyModel("llama")).name == "dense-decoder"
    assert resolve_model_adapter(TinyModel("mixtral")).name == "moe-decoder"
    assert resolve_model_adapter(TinyModel("unknown-new-model")).name == (
        "generic-linear"
    )
    assert "hybrid-decoder" in list_model_adapters()


def test_adapter_registry_is_extensible_and_rejects_collisions() -> None:
    registry = AdapterRegistry()
    registry.register(ModelAdapter("custom", frozenset({"custom_type"})))
    registry.register(ModelAdapter("fallback", fallback=True))
    assert registry.resolve(TinyModel("custom_type")).name == "custom"
    assert registry.resolve(TinyModel("other")).name == "fallback"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ModelAdapter("custom"))
    with pytest.raises(ValueError, match="fallback"):
        registry.register(ModelAdapter("fallback"), replace=True)
    assert registry.resolve(TinyModel("other")).name == "fallback"


def test_model_inspection_is_non_mutating() -> None:
    model = TinyModel()
    support = inspect_model(model)
    assert support.adapter == "dense-decoder"
    assert support.quantizable_modules == 2
    assert support.quantizable_parameters == 72
    assert isinstance(model.proj, nn.Linear)


def test_custom_adapter_converts_non_linear_projection(monkeypatch) -> None:
    torch.manual_seed(5)
    model = nn.Sequential(Conv1DLike(8, 6))
    adapter = Conv1DLikeAdapter("conv1d-like")
    monkeypatch.setattr(
        patch_module, "resolve_model_adapter", lambda model, name: adapter
    )
    inputs = torch.randn(3, 8)
    quant_config = patch_module.QuantConfig(
        bits=8, group_size=4, codebook="uniform", scale="rms"
    )
    expected = QuantLinear.from_linear(
        adapter.to_linear(model[0]), quant_config
    )(inputs)

    patch_module.patch_model(
        model,
        patch_module.PatchConfig(
            quant=quant_config,
            rotation="none",
            exclude=(),
        ),
    )

    assert isinstance(model[0], QuantLinear)
    actual = model(inputs)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bits", range(1, 9))
def test_public_api_optimizes_every_profile_bit_width(bits: int) -> None:
    torch.manual_seed(0)
    model = TinyModel()
    report: dict = {}
    returned = optimize_model(
        model,
        RotQuantConfig(
            bits=bits,
            group_size=4,
            scale="rms",
            rotation="none",
        ),
        report=report,
    )
    assert returned is model
    assert isinstance(model.proj, QuantLinear)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.proj.qweight.packed.bits == bits
    assert model.proj.qweight.codebook.centroids.numel() == 1 << bits
    assert report["profile"]["bits"] == bits
    assert report["model_support"]["adapter"] == "dense-decoder"


@pytest.mark.parametrize("bits", [0, 4.0, 9, True])
def test_public_profiles_fail_closed_outside_one_to_eight(bits) -> None:
    with pytest.raises(ValueError, match="profile bits"):
        RotQuantConfig(bits=bits)


def test_public_api_rejects_models_without_quantizable_modules() -> None:
    model = nn.Sequential(nn.ReLU())
    with pytest.raises(ValueError, match="found no quantizable modules"):
        optimize_model(model, RotQuantConfig(rotation="none"))


@pytest.mark.parametrize(
    "selection",
    [
        {"include": ("does-not-exist",)},
        {"exclude": ("proj", "lm_head")},
    ],
)
def test_public_api_rejects_empty_filtered_selection(selection) -> None:
    model = TinyModel()

    with pytest.raises(ValueError, match="selected no quantizable modules"):
        optimize_model(
            model,
            RotQuantConfig(rotation="none", **selection),
        )

    assert isinstance(model.proj, nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
