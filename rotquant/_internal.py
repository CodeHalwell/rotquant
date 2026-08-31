"""Shared internal helpers with a stable intra-package contract.

Several modules (``linear``, ``patch``, ``dynamic``, ``block_train``,
``kv_cache``, ``train_rotation``) need the exact scale-expansion,
group-quantization, and module-surgery primitives that ``quantize`` defines, so
that a packed weight produced anywhere in the package is bit-identical to one
produced by :class:`~rotquant.quantize.Quantizer`.

Import them from here rather than reaching into another module's underscored
names: renaming or reshaping one of them breaks this module immediately (and
CI's import of the package with it) instead of silently breaking four
consumers. This module must never import ``patch`` or ``linear`` — it sits
below them in the import graph.

Everything here is internal API: no compatibility promise outside this
repository.
"""
from __future__ import annotations

import torch
from torch import nn

from .quantize import _encode_storage_scales as encode_storage_scales
from .quantize import _expand_scales as expand_scales
from .quantize import _generate_sketch_matrix as generate_sketch_matrix
from .quantize import _group_scales_rms as group_scales_rms
from .quantize import _quantize_groups as quantize_groups
from .quantize import _storage_scales as storage_scales

__all__ = [
    "cpu_staging_linear",
    "encode_storage_scales",
    "expand_scales",
    "generate_sketch_matrix",
    "get_parent",
    "group_scales_rms",
    "quantize_groups",
    "storage_scales",
]


def get_parent(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    """Resolve a dotted module path to its parent module and leaf attribute name."""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def cpu_staging_linear(linear: nn.Linear) -> nn.Linear:
    """Copy one accelerator layer to fp32 CPU for one-time quantization.

    MPS is excellent for dense fp16 inference but extremely slow for this
    project's branchy 41-candidate scale search. Staging one layer at a time
    avoids retaining a second full model while making patching orders of
    magnitude faster.
    """
    staged = nn.Linear(linear.in_features, linear.out_features,
                       bias=linear.bias is not None, device="cpu",
                       dtype=torch.float32)
    with torch.no_grad():
        staged.weight.copy_(
            linear.weight.detach().to(device="cpu", dtype=torch.float32)
        )
        if linear.bias is not None:
            staged.bias.copy_(
                linear.bias.detach().to(device="cpu", dtype=torch.float32)
            )
    return staged
