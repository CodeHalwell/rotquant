"""Architecture discovery and extension points for model optimisation.

All built-in adapters currently target ordinary ``torch.nn.Linear`` modules.
Keeping discovery behind a registry makes that limitation explicit and gives
Conv1D, expert, recurrent, and multimodal paths a stable place to specialize
without growing model-name conditionals inside the quantizer.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class ModelSupport:
    """Read-only architecture discovery result for one model instance."""

    adapter: str
    model_type: str | None
    quantizable_modules: int
    quantizable_parameters: int
    capabilities: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.quantizable_modules > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "model_type": self.model_type,
            "quantizable_modules": self.quantizable_modules,
            "quantizable_parameters": self.quantizable_parameters,
            "capabilities": list(self.capabilities),
            "supported": self.supported,
        }


@dataclass(frozen=True, slots=True)
class ModelAdapter:
    """A discoverable family adapter.

    Subclasses may override :meth:`iter_quantizable_modules` when a model uses
    projections other than ``nn.Linear`` or requires architecture-specific
    traversal.  Merely resolving an adapter means the module layout can be
    discovered; quality and kernel support remain separate validation gates.
    """

    name: str
    model_types: frozenset[str] = frozenset()
    capabilities: tuple[str, ...] = ("linear_weight_quantization",)
    fallback: bool = False

    def matches(self, model: nn.Module) -> bool:
        return self.model_type(model) in self.model_types

    def model_type(self, model: nn.Module) -> str | None:
        value = getattr(getattr(model, "config", None), "model_type", None)
        return value if isinstance(value, str) else None

    def iter_quantizable_modules(
        self, model: nn.Module
    ) -> Iterator[tuple[str, nn.Linear]]:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                yield name, module

    def inspect(self, model: nn.Module) -> ModelSupport:
        modules = list(self.iter_quantizable_modules(model))
        parameters = sum(module.weight.numel() for _, module in modules)
        return ModelSupport(
            adapter=self.name,
            model_type=self.model_type(model),
            quantizable_modules=len(modules),
            quantizable_parameters=parameters,
            capabilities=self.capabilities,
        )


class AdapterRegistry:
    """Ordered registry with explicit override and a required fallback."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter, *, replace: bool = False) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter must be a ModelAdapter")
        if not adapter.name:
            raise ValueError("adapter name must not be empty")
        if adapter.name in self._adapters and not replace:
            raise ValueError(f"adapter is already registered: {adapter.name}")
        if adapter.fallback and any(
            current.fallback and current.name != adapter.name
            for current in self._adapters.values()
        ):
            raise ValueError("adapter registry already has a fallback")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> ModelAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            choices = ", ".join(self.names())
            raise ValueError(
                f"unknown model adapter {name!r}; available: {choices}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def resolve(self, model: nn.Module, name: str | None = None) -> ModelAdapter:
        if name is not None:
            return self.get(name)
        for adapter in self._adapters.values():
            if not adapter.fallback and adapter.matches(model):
                return adapter
        fallbacks = [adapter for adapter in self._adapters.values() if adapter.fallback]
        if len(fallbacks) != 1:
            raise RuntimeError("adapter registry must contain exactly one fallback")
        return fallbacks[0]


DENSE_DECODER_MODEL_TYPES = frozenset(
    {
        "gemma",
        "gemma2",
        "gemma3_text",
        "llama",
        "mistral",
        "phi",
        "phi3",
        "qwen2",
        "qwen3",
    }
)
MOE_DECODER_MODEL_TYPES = frozenset(
    {
        "deepseek_v2",
        "deepseek_v3",
        "mixtral",
        "qwen2_moe",
        "qwen3_moe",
    }
)
HYBRID_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
    }
)
MULTIMODAL_MODEL_TYPES = frozenset(
    {
        "gemma3",
        "llava",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3_vl",
    }
)
ENCODER_DECODER_MODEL_TYPES = frozenset({"bart", "mbart", "mt5", "t5"})
ENCODER_MODEL_TYPES = frozenset(
    {"bert", "deberta", "deberta-v2", "roberta"}
)


ADAPTERS = AdapterRegistry()
ADAPTERS.register(
    ModelAdapter(
        "dense-decoder",
        DENSE_DECODER_MODEL_TYPES,
        ("linear_weight_quantization", "causal_generation"),
    )
)
ADAPTERS.register(
    ModelAdapter(
        "moe-decoder",
        MOE_DECODER_MODEL_TYPES,
        ("linear_weight_quantization", "expert_projections", "causal_generation"),
    )
)
ADAPTERS.register(
    ModelAdapter(
        "hybrid-decoder",
        HYBRID_MODEL_TYPES,
        ("linear_weight_quantization", "hybrid_recurrent_attention"),
    )
)
ADAPTERS.register(
    ModelAdapter(
        "multimodal",
        MULTIMODAL_MODEL_TYPES,
        ("linear_weight_quantization", "multimodal_module_discovery"),
    )
)
ADAPTERS.register(
    ModelAdapter(
        "encoder-decoder",
        ENCODER_DECODER_MODEL_TYPES,
        ("linear_weight_quantization", "encoder_decoder"),
    )
)
ADAPTERS.register(
    ModelAdapter(
        "encoder",
        ENCODER_MODEL_TYPES,
        ("linear_weight_quantization", "encoder"),
    )
)
ADAPTERS.register(ModelAdapter("generic-linear", fallback=True))


def register_model_adapter(adapter: ModelAdapter, *, replace: bool = False) -> None:
    """Register a custom architecture adapter in the process-wide registry."""

    ADAPTERS.register(adapter, replace=replace)


def resolve_model_adapter(
    model: nn.Module, name: str | None = None
) -> ModelAdapter:
    """Resolve an explicit adapter or auto-detect one from ``config.model_type``."""

    return ADAPTERS.resolve(model, name)


def inspect_model_support(
    model: nn.Module, adapter: str | None = None
) -> ModelSupport:
    """Inspect a model without mutating it or allocating quantized weights."""

    return resolve_model_adapter(model, adapter).inspect(model)


def list_model_adapters() -> tuple[str, ...]:
    return ADAPTERS.names()


__all__ = [
    "ADAPTERS",
    "AdapterRegistry",
    "ModelAdapter",
    "ModelSupport",
    "inspect_model_support",
    "list_model_adapters",
    "register_model_adapter",
    "resolve_model_adapter",
]
