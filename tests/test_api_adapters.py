"""Public optimisation API and architecture registry behaviour."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
