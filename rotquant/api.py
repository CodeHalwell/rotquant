"""Stable, model-oriented public API for RotQuant optimisation.

The lower-level ``QuantConfig`` and ``PatchConfig`` remain available for
research experiments.  This module provides the narrower production surface:
validated 1--8 bit profiles, model inspection, in-place optimisation, and
checkpoint save/load functions.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .adapters import ModelSupport, inspect_model_support
from .checkpoint import load_packed_model, save_packed_checkpoint
from .format import validate_profile_bits
from .patch import PATCH_MODES, PatchConfig, patch_model
from .quantize import QuantConfig


@dataclass(frozen=True, slots=True)
class RotQuantConfig:
    """Validated general-purpose optimisation profile.

    Bit widths outside 1--8 remain available through the research-level
    ``QuantConfig`` packer, but are intentionally excluded here until they have
    dedicated kernel profiles.
    """

    bits: int = 4
    group_size: int = 128
    codebook: str = "gaussian"
    scale: str = "mse_search"
    scale_bits: float = 16.0
    error_comp: str = "none"
    residual_bits: int = 1
    residual_codebook: str = "gaussian"
    rotation: str = "fwht"
    rotation_block: int = 128
    mode: str = "consistent"
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ("lm_head", "embed_out")
    fallback: bool = False
    seed: int = 0
    adapter: str | None = None
    codebook_dim: int | None = None
    bias_correction: str = "none"

    def __post_init__(self) -> None:
        validate_profile_bits(self.bits)
        validate_profile_bits(self.residual_bits)
        if self.rotation_block < 1:
            raise ValueError("rotation_block must be >= 1")
        if self.mode not in PATCH_MODES:
            raise ValueError(f"mode must be one of {PATCH_MODES}")
        # Reuse the research config's complete codebook/scale validation.
        self.to_quant_config()

    def to_quant_config(self) -> QuantConfig:
        spherical = self.codebook.lower() in {
            "sphere", "spherical", "beta", "finite_beta"}
        return QuantConfig(
            bits=self.bits,
            codebook=self.codebook,
            codebook_dim=(
                self.codebook_dim or self.rotation_block
            ) if spherical else None,
            scale=self.scale,
            group_size=self.group_size,
            error_comp=self.error_comp,
            bias_correction=self.bias_correction,
            residual_bits=self.residual_bits,
            residual_codebook=self.residual_codebook,
            seed=self.seed,
            scale_bits=self.scale_bits,
        )

    def to_patch_config(self) -> PatchConfig:
        return PatchConfig(
            quant=self.to_quant_config(),
            rotation=self.rotation,
            block=self.rotation_block,
            mode=self.mode,
            include=self.include,
            exclude=self.exclude,
            fallback=self.fallback,
            seed=self.seed,
            adapter=self.adapter,
        )


def inspect_model(model: nn.Module, *, adapter: str | None = None) -> ModelSupport:
    """Return non-mutating architecture and quantizable-parameter metadata."""

    return inspect_model_support(model, adapter)


def optimize_model(
    model: nn.Module,
    config: RotQuantConfig | None = None,
    *,
    hessians: Mapping[str, torch.Tensor] | None = None,
    activations: Mapping[str, torch.Tensor] | None = None,
    report: MutableMapping[str, Any] | None = None,
) -> nn.Module:
    """Optimise ``model`` in place and return the same model instance.

    In-place operation is explicit because copying a multi-billion-parameter
    model can temporarily double host or accelerator memory.  Use
    :func:`inspect_model` first when integrating an unfamiliar architecture.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    resolved = config or RotQuantConfig()
    support = inspect_model_support(model, resolved.adapter)
    if not support.supported:
        raise ValueError(
            f"adapter {support.adapter!r} found no quantizable modules for "
            f"model_type={support.model_type!r}"
        )
    stats: dict[str, Any] = {}
    patch_model(
        model,
        resolved.to_patch_config(),
        hessians=dict(hessians or {}),
        activations=dict(activations or {}),
        stats_out=stats,
    )
    if not stats.get("patched_modules"):
        raise ValueError(
            f"adapter {support.adapter!r} selected no quantizable modules after "
            f"include={resolved.include!r} and exclude={resolved.exclude!r}"
        )
    if report is not None:
        report.update(
            {
                "profile": {
                    "bits": resolved.bits,
                    "group_size": resolved.group_size,
                    "codebook": resolved.codebook,
                    "scale": resolved.scale,
                    "rotation": resolved.rotation,
                },
                "model_support": support.to_dict(),
                "patch": stats,
            }
        )
    return model


def save_pretrained(
    model: nn.Module,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Save an optimised model in the versioned, pickle-free RotQuant format."""

    return save_packed_checkpoint(model, output_dir, **kwargs)


def from_pretrained(
    checkpoint_dir: str | Path,
    **kwargs: Any,
) -> nn.Module:
    """Load a versioned RotQuant checkpoint as a normal model instance."""

    return load_packed_model(checkpoint_dir, **kwargs)


__all__ = [
    "RotQuantConfig",
    "from_pretrained",
    "inspect_model",
    "optimize_model",
    "save_pretrained",
]
