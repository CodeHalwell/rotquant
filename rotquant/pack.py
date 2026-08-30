"""Bit-packing and bits/weight accounting.

Packs integer code indices (``0 <= idx < 2**bits``) into a dense ``uint8`` /
``int32`` buffer so quantised weights never materialise as fp16, and exposes the
true bits/weight via :class:`~rotquant.utils.BitBudget`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .format import packed_word_count, validate_storage_bits
from .utils import BitBudget


@dataclass
class PackedTensor:
    data: torch.Tensor          # packed buffer of int32 words (uint32 bit patterns)
    shape: tuple[int, ...]      # logical shape of the index tensor
    bits: int                   # bits per code
    numel: int                  # number of codes

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the in-memory representation against the v1 bitstream."""

        if not isinstance(self.data, torch.Tensor):
            raise TypeError("packed data must be a torch.Tensor")
        if self.data.dtype != torch.int32:
            raise TypeError("packed data must use int32 storage words")
        if self.data.ndim != 1:
            raise ValueError("packed data must be a one-dimensional word buffer")
        validate_storage_bits(self.bits)
        if not isinstance(self.shape, tuple):
            raise TypeError("packed logical shape must be a tuple")
        if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in self.shape
        ):
            raise ValueError("packed logical shape must contain non-negative integers")
        if isinstance(self.numel, bool) or not isinstance(self.numel, int):
            raise TypeError("packed numel must be an integer")
        logical_numel = math.prod(self.shape)
        if self.numel != logical_numel:
            raise ValueError(
                f"packed numel {self.numel} does not match logical shape "
                f"({logical_numel})"
            )
        expected_words = packed_word_count(self.numel, self.bits)
        if self.data.numel() != expected_words:
            raise ValueError(
                f"packed buffer has {self.data.numel()} words; expected "
                f"{expected_words}"
            )

    def bit_budget(self, group_size: int, scale_bits: float = 16.0,
                   sign_bits: float = 0.0) -> BitBudget:
        return BitBudget(levels=2 ** self.bits, group_size=group_size,
                         scale_bits=scale_bits, sign_bits=sign_bits)


def pack_indices(idx: torch.Tensor, bits: int) -> PackedTensor:
    """Pack an integer index tensor into a uint32 bitstream (LSB-first).

    Lossless for any ``bits`` in ``[1, 16]``. Used so the packed path never holds
    a dequantised fp16 copy.
    """
    validate_storage_bits(bits)
    if not isinstance(idx, torch.Tensor):
        raise TypeError("idx must be a torch.Tensor")
    try:
        torch.iinfo(idx.dtype)
    except TypeError as exc:
        raise TypeError("idx must use an integer dtype") from exc
    output_device = idx.device
    # MPS executes int64 shifts/scatter_add pathologically slowly for model-sized
    # tensors. Packing is a one-time serialization step, so do its integer work on
    # CPU and copy only the compact int32 result back to the source device.
    work_device = torch.device("cpu") if output_device.type == "mps" else output_device
    flat = idx.reshape(-1).to(device=work_device, dtype=torch.int64)
    device = flat.device
    if flat.numel() and (flat.min() < 0 or flat.max() >= (1 << bits)):
        raise ValueError("index out of range for given bits")
    n = flat.numel()
    words = packed_word_count(n, bits)
    out = torch.zeros(words, dtype=torch.int64, device=device)
    bit_positions = torch.arange(n, dtype=torch.int64, device=device) * bits
    word_idx = bit_positions // 32
    offset = bit_positions % 32
    # Low part of each code in its starting word.
    out.scatter_add_(0, word_idx, (flat << offset) & 0xFFFFFFFF)
    # Spill into the next word when a code straddles a 32-bit boundary.
    spill = offset + bits > 32
    if spill.any():
        si = word_idx[spill] + 1
        sval = (flat[spill] >> (32 - offset[spill])) & 0xFFFFFFFF
        out.scatter_add_(0, si, sval)
    # Store as genuine 32-bit words so the in-memory footprint (4 bytes/word)
    # matches the bits/weight accounting. Values >= 2^31 wrap to negative int32,
    # which is fine: unpack reads each word back as unsigned.
    data = out.to(torch.int32)
    if data.device != output_device:
        data = data.to(output_device)
    return PackedTensor(data=data, shape=tuple(idx.shape),
                        bits=bits, numel=n)


def unpack_indices(packed: PackedTensor) -> torch.Tensor:
    """Inverse of :func:`pack_indices`."""
    n, bits = packed.numel, packed.bits
    output_device = packed.data.device
    source = packed.data.cpu() if output_device.type == "mps" else packed.data
    # Read the stored int32 words back as unsigned 32-bit values.
    data = source.to(torch.int64) & 0xFFFFFFFF
    device = data.device
    bit_positions = torch.arange(n, dtype=torch.int64, device=device) * bits
    word_idx = bit_positions // 32
    offset = bit_positions % 32
    mask = (1 << bits) - 1
    low = (data[word_idx] >> offset) & mask
    spill = offset + bits > 32
    if spill.any():
        hi = (data[word_idx[spill] + 1] << (32 - offset[spill])) & mask
        low[spill] = (low[spill] | hi) & mask
    result = low.reshape(packed.shape)
    return result.to(output_device) if result.device != output_device else result


def packed_bytes(packed: PackedTensor) -> int:
    # The buffer is stored as int32, so this is the true in-memory footprint
    # (4 bytes per 32-bit word) -- it matches what a real uint32 kernel would use.
    return packed.data.numel() * packed.data.element_size()
