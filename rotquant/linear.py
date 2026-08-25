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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pack import packed_bytes, unpack_indices
from .quantize import QuantConfig, Quantizer, QuantizedWeight, _generate_sketch_matrix
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
        # Non-persistent caches for QJL sketch inference.  Populated lazily on the first
        # forward pass; invalidated automatically when device or dtype changes.
        self.register_buffer("_sketch_G", None, persistent=False)
        self.register_buffer("_sketch_mat", None, persistent=False)
        self.register_buffer("_sketch_norms", None, persistent=False)
        if fallback:
            logger.warning(
                "QuantLinear in FALLBACK mode: materialising fp16 weight "
                "(%d x %d). Use only for small-model quality checks -- this OOMs "
                "on 7B+ and must NOT be used for footprint numbers.",
                self.out_features, self.in_features,
            )
            self._fp_cache = self.qweight.dequantize()

    def _ensure_sketch_cache(self, xr: torch.Tensor) -> None:
        """Materialise G, ±1 sketch matrix, and row norms; rebuild when device/dtype changes."""
        qw = self.qweight
        if (self._sketch_G is None
                or self._sketch_G.device != xr.device
                or self._sketch_G.dtype != xr.dtype):
            k = qw.sketch_k
            self._sketch_G = _generate_sketch_matrix(
                self.in_features, k, qw.sketch_seed, xr.device,
            ).to(xr.dtype)
            self._sketch_mat = (
                unpack_indices(qw.sketch)
                .reshape(self.out_features, k)
                .to(device=xr.device, dtype=xr.dtype)
            ) * 2 - 1   # {0, 1} -> {-1, +1}
            self._sketch_norms = qw.sketch_row_norms.to(device=xr.device, dtype=xr.dtype)

    def _apply(self, fn, recurse: bool = True):
        """Keep the packed dataclass tensors in step with ``.to()``/``.cuda()``/
        ``.half()`` -- they are not registered buffers (state_dict round-tripping
        of packed weights is unsupported), so ``nn.Module`` would otherwise leave
        them behind and the first forward after a device move would fail.

        ``fn`` is the standard ``Module.to`` convert closure: it only changes the
        dtype of floating-point tensors, so the int32 code buffers are device-moved
        untouched. Sketch caches are rebuilt lazily via their device/dtype check.
        """
        super()._apply(fn, recurse)
        qw = self.qweight
        qw.packed.data = fn(qw.packed.data)
        if qw.scales is not None:
            qw.scales = fn(qw.scales)
        if qw.residual_packed is not None:
            qw.residual_packed.data = fn(qw.residual_packed.data)
            qw.residual_scales = fn(qw.residual_scales)
        if qw.sketch is not None:
            qw.sketch.data = fn(qw.sketch.data)
            qw.sketch_row_norms = fn(qw.sketch_row_norms)
        if self._fp_cache is not None:
            self._fp_cache = fn(self._fp_cache)
        return self

    def _weight(self) -> torch.Tensor:
        if self.fallback:
            return self._fp_cache
        return self.qweight.dequantize()  # transient in packed mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.act_rotation.rotate_activation(x)
        base_out = F.linear(xr, self._weight().to(xr.dtype), self.bias)
        if self.qweight.sketch is not None:
            self._ensure_sketch_cache(xr)
            # Asymmetric QJL inner-product estimator (only the stored residual is
            # sign-quantised; the activation keeps its full projection). Exactly
            # unbiased for Gaussian G at any angle, and linear in xr:
            #   E[sqrt(pi/2)/sqrt(k) * ||r_i|| * (xr@G) . sign(r_i@G)] = xr . r_i
            # (G's columns already carry the 1/sqrt(k) JL normalisation, which
            # the sqrt(k) in the constant folds against.)
            xr_proj = xr @ self._sketch_G                          # [..., k]
            k = self.qweight.sketch_k
            correction = (
                (xr_proj @ self._sketch_mat.T)                     # [..., out]
                * (self._sketch_norms * (math.sqrt(math.pi / 2)
                                         / math.sqrt(k)))          # [out]
            )
            return base_out + correction
        return base_out

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
        """Persistent storage in packed mode (codes + scales + sketch), in bytes.

        Scale/norm tensors are charged at their actual element size -- with the
        default 16-bit scale budget they are stored fp16, so this matches the
        bits/weight accounting instead of assuming it.
        """
        qw = self.qweight
        b = packed_bytes(qw.packed)
        if qw.scales is not None:
            b += qw.scales.numel() * qw.scales.element_size()
        if qw.residual_packed is not None:
            b += packed_bytes(qw.residual_packed)
            b += qw.residual_scales.numel() * qw.residual_scales.element_size()
        if qw.sketch is not None:
            b += packed_bytes(qw.sketch)
            b += qw.sketch_row_norms.numel() * qw.sketch_row_norms.element_size()
        return b
