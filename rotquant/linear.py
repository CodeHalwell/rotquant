"""``QuantLinear`` -- the runtime module that keeps weights packed and rotates the
activation in the forward pass.

Two modes:

* **packed** (default): only the packed code buffer + per-group scales are stored.
  The fp16 weight is *never* persisted (avoiding the 22 GB-vs-4 GB trap). Without a
  fused kernel the matmul transiently dequantises, which is the "slower without a
  real fused kernel" footnote in E8 -- storage stays small either way.
* **fallback**: the weight is materialised and cached once in the source layer's
  dtype (normally fp16 on GPU). Only for quick quality checks on small models; it
  is flagged loudly because it OOMs on 7B+.

The activation rotation and the weight rotation are kept as *separate* handles so
the patcher can build the deliberately-broken "mismatched" mode for E7.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._internal import (
    encode_storage_scales,
    expand_scales,
    generate_sketch_matrix,
    storage_scales,
)
from .codebooks import VectorCodebook
from .pack import packed_bytes, unpack_indices
from .quantize import QuantConfig, QuantizedWeight, Quantizer
from .rotate import Identity, Rotation
from .utils import get_logger

logger = get_logger(__name__)
_fallback_warning_emitted = False


class QuantLinear(nn.Module):
    def __init__(self, qweight: QuantizedWeight, act_rotation: Rotation,
                 bias: torch.Tensor | None = None, fallback: bool = False,
                 fallback_dtype: torch.dtype | None = None,
                 activation_bits: int | None = None):
        super().__init__()
        self.qweight = qweight
        self.act_rotation = act_rotation
        self.in_features = qweight.in_features
        self.out_features = qweight.out_features
        self.fallback = fallback
        if activation_bits is not None and (
            isinstance(activation_bits, bool)
            or not isinstance(activation_bits, int)
            or not 2 <= activation_bits <= 16
        ):
            raise ValueError("activation_bits must be None or an integer in [2, 16]")
        self.activation_bits = activation_bits
        self._fallback_dtype = fallback_dtype
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None
        )
        self._fp_cache: torch.Tensor | None = None
        # Non-persistent caches for QJL sketch inference.  Populated lazily on the first
        # forward pass; invalidated automatically when device or dtype changes.
        self.register_buffer("_sketch_G", None, persistent=False)
        self.register_buffer("_sketch_mat", None, persistent=False)
        self.register_buffer("_sketch_norms", None, persistent=False)
        # Optional calibration-only state for model-level fine-tuning. Codes
        # remain fixed; only existing fp16 group scales are adjusted.
        self.register_parameter("_log_scale_multiplier", None)
        self.register_buffer("_scale_training_base", None, persistent=False)
        self.register_buffer("_scale_training_codes", None, persistent=False)
        self._scale_multiplier_min = 0.5
        self._scale_multiplier_max = 1.5
        self.register_parameter("lora_A", None)
        self.register_parameter("lora_B", None)
        self.lora_rank = 0
        self.lora_alpha = 0.0
        # Recovery-only metadata. Neither item is a parameter/buffer, so it is
        # excluded from checkpoints and ``.to()`` does not duplicate a CPU
        # source copy onto the accelerator.
        self._quant_config: QuantConfig | None = None
        self._recovery_source_weight: torch.Tensor | None = None
        if fallback:
            global _fallback_warning_emitted
            if not _fallback_warning_emitted:
                logger.warning(
                    "QuantLinear FALLBACK mode: materialising source-dtype weights. "
                    "Use for quality checks only; do not use for packed footprint "
                    "or throughput measurements.")
                _fallback_warning_emitted = True
            self._fp_cache = self.qweight.dequantize()
            if fallback_dtype is not None:
                self._fp_cache = self._fp_cache.to(dtype=fallback_dtype)

    def _ensure_sketch_cache(self, xr: torch.Tensor) -> None:
        """Materialise G, ±1 sketch matrix, and row norms; rebuild when device/dtype changes."""
        qw = self.qweight
        if (self._sketch_G is None
                or self._sketch_G.device != xr.device
                or self._sketch_G.dtype != xr.dtype):
            k = qw.sketch_k
            self._sketch_G = generate_sketch_matrix(
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
            if qw.scale_offsets is not None:
                qw.scale_offsets = fn(qw.scale_offsets)
                qw.scale_steps = fn(qw.scale_steps)
        if qw.residual_packed is not None:
            qw.residual_packed.data = fn(qw.residual_packed.data)
            qw.residual_scales = fn(qw.residual_scales)
            if qw.residual_scale_offsets is not None:
                qw.residual_scale_offsets = fn(qw.residual_scale_offsets)
                qw.residual_scale_steps = fn(qw.residual_scale_steps)
        if qw.sketch is not None:
            qw.sketch.data = fn(qw.sketch.data)
            qw.sketch_row_norms = fn(qw.sketch_row_norms)
        if self._fp_cache is not None:
            self._fp_cache = fn(self._fp_cache)
        return self

    def _weight(self) -> torch.Tensor:
        if self._log_scale_multiplier is not None:
            factor = self._log_scale_multiplier.exp().clamp(
                self._scale_multiplier_min, self._scale_multiplier_max)
            scales = self._scale_training_base * factor
            stored = storage_scales(
                scales,
                self.qweight.scale_bits_main,
                self.qweight.scale_quant_group_size,
            )
            # Exact storage-rounded forward values with an identity STE.
            scales = scales + (stored.to(scales.dtype) - scales).detach()
            scale_group_size = (
                self.qweight.scale_group_size
                if self.qweight.scale_group_size is not None
                else self.qweight.group_size)
            return self._scale_training_codes * expand_scales(
                scales, scale_group_size, self.in_features)
        if self.fallback:
            return self._fp_cache
        return self.qweight.dequantize()  # transient in packed mode

    def enable_scale_finetuning(self, multiplier_min: float = 0.5,
                                multiplier_max: float = 1.5) -> nn.Parameter:
        """Expose fixed-code group scales as bounded calibration parameters."""
        if self._log_scale_multiplier is not None:
            return self._log_scale_multiplier
        qw = self.qweight
        if isinstance(qw.codebook, VectorCodebook):
            raise TypeError(
                "scale fine-tuning is not yet implemented for vector codebooks"
            )
        if qw.scales is None:
            raise ValueError("scale fine-tuning requires stored group scales")
        if qw.residual_packed is not None or qw.sketch is not None:
            raise ValueError(
                "scale fine-tuning is not implemented for residual/sketch weights")
        if multiplier_min <= 0 or multiplier_max < multiplier_min:
            raise ValueError("scale multiplier bounds require 0 < min <= max")
        indices = unpack_indices(qw.packed).reshape(
            self.out_features, self.in_features)
        centroids = qw.codebook.centroids.to(indices.device)
        self._scale_training_codes = centroids[indices].float()
        self._scale_training_base = qw.main_scales().detach().float().clone()
        self._scale_multiplier_min = multiplier_min
        self._scale_multiplier_max = multiplier_max
        self._log_scale_multiplier = nn.Parameter(
            torch.zeros_like(self._scale_training_base))
        self._fp_cache = None
        return self._log_scale_multiplier

    def scale_finetuning_multiplier(self) -> torch.Tensor | None:
        if self._log_scale_multiplier is None:
            return None
        return self._log_scale_multiplier.exp().clamp(
            self._scale_multiplier_min, self._scale_multiplier_max)

    @torch.no_grad()
    def commit_scale_finetuning(self) -> None:
        """Store trained scales at their claimed precision and drop train state."""
        if self._log_scale_multiplier is None:
            return
        scales = self._scale_training_base * self.scale_finetuning_multiplier()
        stored = storage_scales(
            scales,
            self.qweight.scale_bits_main,
            self.qweight.scale_quant_group_size,
        )
        (self.qweight.scales,
         self.qweight.scale_offsets,
         self.qweight.scale_steps) = encode_storage_scales(
            stored,
            self.qweight.scale_bits_main,
            self.qweight.scale_quant_group_size,
        )
        self._log_scale_multiplier = None
        self._scale_training_base = None
        self._scale_training_codes = None
        if self.fallback:
            self._fp_cache = self.qweight.dequantize()
            if self._fallback_dtype is not None:
                self._fp_cache = self._fp_cache.to(self._fallback_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.act_rotation.cached_rotate_activation(x)
        xr = self._quantize_activation(xr)
        base_out = F.linear(xr, self._weight().to(xr.dtype), self.bias)
        if self.lora_A is not None:
            adapter_hidden = F.linear(xr, self.lora_A.to(xr.dtype))
            adapter_out = F.linear(
                adapter_hidden, self.lora_B.to(xr.dtype))
            base_out = base_out + adapter_out * (
                self.lora_alpha / self.lora_rank)
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

    def _quantize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """Symmetric per-token activation quantization reference semantics.

        A fused runtime can keep the signed integer values and scale through the
        matrix multiply. This portable path immediately dequantizes them, which
        is slower but makes W4A8 quality and integration independently testable.
        During recovery training an identity STE preserves gradients.
        """

        if self.activation_bits is None:
            return activation
        qmax = (1 << (self.activation_bits - 1)) - 1
        work = activation.float()
        scale = work.detach().abs().amax(dim=-1, keepdim=True)
        scale = (scale / qmax).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = torch.round(work / scale).clamp(-qmax, qmax)
        dequantized = (quantized * scale).to(activation.dtype)
        if torch.is_grad_enabled() and activation.requires_grad:
            return activation + (dequantized - activation).detach()
        return dequantized

    @torch.no_grad()
    def apply_mean_bias_correction(
        self, source_weight: torch.Tensor, activation_mean: torch.Tensor
    ) -> torch.Tensor:
        """Fold calibration-mean quantization error into the output bias."""

        source = source_weight.detach().to(
            device=self.qweight.packed.data.device, dtype=torch.float32
        )
        mean = activation_mean.detach().reshape(1, self.in_features).to(
            device=source.device, dtype=torch.float32
        )
        rotated_source = self.act_rotation.rotate_weight(source)
        rotated_mean = self.act_rotation.rotate_activation(mean)
        residual = rotated_source - self.qweight.dequantize().float()
        correction = (rotated_mean @ residual.T).reshape(-1)
        base = (
            self.bias.detach().to(device=source.device, dtype=torch.float32)
            if self.bias is not None
            else torch.zeros(self.out_features, device=source.device)
        )
        corrected = (base + correction).to(
            dtype=self._fallback_dtype or source_weight.dtype
        )
        self.bias = corrected
        return correction

    def enable_lora(self, rank: int, alpha: float, *, init: str = "zero",
                    residual: torch.Tensor | None = None,
                    oversample: int = 4, niter: int = 1,
                    ) -> tuple[nn.Parameter, nn.Parameter]:
        """Attach a low-rank adapter in the deployed rotated basis.

        ``residual_svd`` initializes the adapter from a randomized low-rank
        approximation of ``W_rotated - W_quantized``. This gives optimization a
        useful nonzero starting point while preserving the exact packed base.
        """
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA requires rank >= 1 and alpha > 0")
        if init not in ("zero", "residual_svd"):
            raise ValueError("LoRA init must be 'zero' or 'residual_svd'")
        if rank > min(self.out_features, self.in_features):
            raise ValueError("LoRA rank exceeds the matrix's smaller dimension")
        if oversample < 0 or niter < 0:
            raise ValueError("LoRA SVD oversample and iterations must be nonnegative")
        if self.lora_A is not None:
            if self.lora_rank != rank or self.lora_alpha != alpha:
                raise ValueError("LoRA is already enabled with different settings")
            return self.lora_A, self.lora_B
        device = self.qweight.packed.data.device
        if init == "zero":
            matrix_a = torch.empty(
                rank, self.in_features, device=device, dtype=torch.float32)
            matrix_b = torch.zeros(
                self.out_features, rank, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(matrix_a, a=math.sqrt(5))
        else:
            if residual is None or tuple(residual.shape) != (
                    self.out_features, self.in_features):
                raise ValueError(
                    "residual_svd LoRA requires an [out_features, in_features] "
                    "residual")
            work = residual.detach().float()
            if work.device.type == "mps":
                work = work.cpu()
            q = min(rank + oversample, min(work.shape))
            u, singular, v = torch.svd_lowrank(work, q=q, niter=niter)
            u = u[:, :rank]
            singular = singular[:rank].clamp_min(0).sqrt()
            v = v[:, :rank]
            # The runtime multiplies B@A by alpha/rank. Split the inverse
            # coefficient evenly so the initial adapter equals the rank-r SVD.
            coefficient = math.sqrt(rank / float(alpha))
            matrix_b = (u * singular.unsqueeze(0) * coefficient).to(device)
            matrix_a = (
                singular.unsqueeze(1) * v.T * coefficient).to(device)
        self.lora_A = nn.Parameter(matrix_a)
        self.lora_B = nn.Parameter(matrix_b)
        self.lora_rank = rank
        self.lora_alpha = float(alpha)
        return self.lora_A, self.lora_B

    @torch.no_grad()
    def commit_lora(self, storage_dtype: torch.dtype = torch.float16) -> None:
        if self.lora_A is None:
            return
        self.lora_A.data = self.lora_A.detach().to(storage_dtype)
        self.lora_B.data = self.lora_B.detach().to(storage_dtype)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def disable_lora(self) -> None:
        self.lora_A = None
        self.lora_B = None
        self.lora_rank = 0
        self.lora_alpha = 0.0

    def adapter_state_bytes(self) -> int:
        if self.lora_A is None:
            return 0
        return (self.lora_A.numel() * self.lora_A.element_size()
                + self.lora_B.numel() * self.lora_B.element_size())

    def retain_recovery_source(self, weight: torch.Tensor) -> None:
        """Keep an fp16 CPU source solely for alternating code refresh."""
        self._recovery_source_weight = weight.detach().to(
            device="cpu", dtype=torch.float16).clone()

    @torch.no_grad()
    def refresh_quantization(self) -> None:
        """Reassign packed codes for the current learned rotation and scales."""
        if self._recovery_source_weight is None or self._quant_config is None:
            raise ValueError("quantization refresh requires retained source weights")
        device = self.qweight.packed.data.device
        source = self._recovery_source_weight.to(device=device, dtype=torch.float32)
        rotated = self.act_rotation.rotate_weight(source)
        scales_override = None
        if self._log_scale_multiplier is not None:
            scales_override = (
                self._scale_training_base
                * self.scale_finetuning_multiplier()).detach()
        self.qweight = Quantizer(self._quant_config).quantize_weight(
            rotated, scales_override=scales_override)
        if self._log_scale_multiplier is not None:
            indices = unpack_indices(self.qweight.packed).reshape(
                self.out_features, self.in_features)
            centroids = self.qweight.codebook.centroids.to(indices.device)
            self._scale_training_codes = centroids[indices].float()
            self._scale_training_base = (
                self.qweight.main_scales().detach().float().clone())
            self._log_scale_multiplier.zero_()
        self._fp_cache = None
        if self.fallback:
            self._fp_cache = self.qweight.dequantize()
            if self._fallback_dtype is not None:
                self._fp_cache = self._fp_cache.to(self._fallback_dtype)

    def drop_recovery_source(self) -> None:
        self._recovery_source_weight = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuantConfig,
                    weight_rotation: Rotation | None = None,
                    act_rotation: Rotation | None = None,
                    H: torch.Tensor | None = None,
                    scales_override: torch.Tensor | None = None,
                    fallback: bool = False,
                    fallback_dtype: torch.dtype | None = None,
                    activation_bits: int | None = None) -> QuantLinear:
        """Quantise an ``nn.Linear``.

        ``weight_rotation`` rotates the weight before quantisation; ``act_rotation``
        is applied to activations at runtime. In the consistent case they are the
        same object; passing different ones yields the mismatched (E7) mode.
        """
        weight_rotation = weight_rotation or Identity(linear.in_features)
        act_rotation = act_rotation or weight_rotation
        w = weight_rotation.rotate_weight(linear.weight.data)
        qw = Quantizer(config).quantize_weight(
            w, H=H, scales_override=scales_override)
        bias = linear.bias.data if linear.bias is not None else None
        module = cls(
            qw, act_rotation=act_rotation, bias=bias, fallback=fallback,
            fallback_dtype=fallback_dtype or linear.weight.dtype,
            activation_bits=activation_bits)
        module._quant_config = config
        return module

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
            if qw.scale_offsets is not None:
                b += qw.scale_offsets.numel() * qw.scale_offsets.element_size()
                b += qw.scale_steps.numel() * qw.scale_steps.element_size()
        if qw.residual_packed is not None:
            b += packed_bytes(qw.residual_packed)
            b += qw.residual_scales.numel() * qw.residual_scales.element_size()
            if qw.residual_scale_offsets is not None:
                b += (qw.residual_scale_offsets.numel()
                      * qw.residual_scale_offsets.element_size())
                b += (qw.residual_scale_steps.numel()
                      * qw.residual_scale_steps.element_size())
        if qw.sketch is not None:
            b += packed_bytes(qw.sketch)
            b += qw.sketch_row_norms.numel() * qw.sketch_row_norms.element_size()
        return b
