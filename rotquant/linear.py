"""``QuantLinear`` -- the runtime module that keeps weights packed and rotates the
activation in the forward pass.

Two modes:

* **packed** (default): only the packed code buffer + per-group scales are stored.
  The fp16 weight is *never* persisted (avoiding the 22 GB-vs-4 GB trap). Without a
  fused kernel the matmul transiently dequantises, which is the "slower without a
  real fused kernel" footnote in E8 -- storage stays small either way.
* **fallback**: the fp16 weight is materialised and cached once. Only for quick
  quality checks on small models; it is flagged loudly because it OOMs on 7B+.

The activation rotation and the weight rotation are kept as *separate* handles so
the patcher can build the deliberately-broken "mismatched" mode for E7.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import QuantConfig, Quantizer, QuantizedWeight
from .rotate import Identity, Rotation
from .utils import get_logger

logger = get_logger()


class QuantLinear(nn.Module):
    def __init__(self, qweight: QuantizedWeight, act_rotation: Rotation,
                 bias: Optional[torch.Tensor] = None, fallback: bool = False):
        super().__init__()
        self.qweight = qweight
        self.act_rotation = act_rotation
        self.in_features = qweight.in_features
        self.out_features = qweight.out_features
        self.fallback = fallback
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self._fp_cache: Optional[torch.Tensor] = None
        if fallback:
            logger.warning(
                "QuantLinear in FALLBACK mode: materialising fp16 weight "
                "(%d x %d). Use only for small-model quality checks -- this OOMs "
                "on 7B+ and must NOT be used for footprint numbers.",
                self.out_features, self.in_features,
            )
            self._fp_cache = self.qweight.dequantize()

    def _weight(self) -> torch.Tensor:
        if self.fallback:
            return self._fp_cache
        return self.qweight.dequantize()  # transient in packed mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.act_rotation.rotate_activation(x)
        return F.linear(xr, self._weight().to(xr.dtype), self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuantConfig,
                    weight_rotation: Optional[Rotation] = None,
                    act_rotation: Optional[Rotation] = None,
                    H: Optional[torch.Tensor] = None,
                    fallback: bool = False) -> "QuantLinear":
        """Quantise an ``nn.Linear``.

        ``weight_rotation`` rotates the weight before quantisation; ``act_rotation``
        is applied to activations at runtime. In the consistent case they are the
        same object; passing different ones yields the mismatched (E7) mode.
        """
        weight_rotation = weight_rotation or Identity(linear.in_features)
        act_rotation = act_rotation or weight_rotation
        w = weight_rotation.rotate_weight(linear.weight.data)
        qw = Quantizer(config).quantize_weight(w, H=H)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(qw, act_rotation=act_rotation, bias=bias, fallback=fallback)

    def packed_state_bytes(self) -> int:
        """Persistent storage in packed mode (codes + scales), in bytes."""
        from .pack import packed_bytes
        b = packed_bytes(self.qweight.packed)
        b += self.qweight.scales.numel() * 2  # fp16 scales
        if self.qweight.residual_packed is not None:
            b += packed_bytes(self.qweight.residual_packed)
            b += self.qweight.residual_scales.numel() * 2
        return b
