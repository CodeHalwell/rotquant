"""Bit-packing and bits/weight accounting.

Packs integer code indices (``0 <= idx < 2**bits``) into a dense ``uint8`` /
``int32`` buffer so quantised weights never materialise as fp16, and exposes the
true bits/weight via :class:`~rotquant.utils.BitBudget`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .utils import BitBudget


@dataclass
class PackedTensor:
    data: torch.Tensor          # packed uint32 buffer
    shape: Tuple[int, ...]      # logical shape of the index tensor
    bits: int                   # bits per code
    numel: int                  # number of codes

    def bit_budget(self, group_size: int, scale_bits: float = 16.0,
                   sign_bits: float = 0.0) -> BitBudget:
        return BitBudget(levels=2 ** self.bits, group_size=group_size,
                         scale_bits=scale_bits, sign_bits=sign_bits)


def pack_indices(idx: torch.Tensor, bits: int) -> PackedTensor:
    """Pack an integer index tensor into a uint32 bitstream (LSB-first).

    Lossless for any ``bits`` in ``[1, 16]``. Used so the packed path never holds
    a dequantised fp16 copy.
    """
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    flat = idx.reshape(-1).to(torch.int64)
    if flat.numel() and (flat.min() < 0 or flat.max() >= (1 << bits)):
        raise ValueError("index out of range for given bits")
    n = flat.numel()
    total_bits = n * bits
    words = (total_bits + 31) // 32
    out = torch.zeros(words, dtype=torch.int64)
    bit_positions = torch.arange(n, dtype=torch.int64) * bits
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
    return PackedTensor(data=out.to(torch.int64), shape=tuple(idx.shape),
                        bits=bits, numel=n)


def unpack_indices(packed: PackedTensor) -> torch.Tensor:
    """Inverse of :func:`pack_indices`."""
    n, bits = packed.numel, packed.bits
    out = torch.empty(n, dtype=torch.int64)
    data = packed.data
    bit_positions = torch.arange(n, dtype=torch.int64) * bits
    word_idx = bit_positions // 32
    offset = bit_positions % 32
    mask = (1 << bits) - 1
    low = (data[word_idx] >> offset) & mask
    spill = offset + bits > 32
    if spill.any():
        hi = (data[word_idx[spill] + 1] << (32 - offset[spill])) & mask
        low[spill] = (low[spill] | hi) & mask
    return low.reshape(packed.shape)


def packed_bytes(packed: PackedTensor) -> int:
    # Each element of the buffer holds one logical 32-bit word; real kernels
    # store it as uint32 (4 bytes), which is what the footprint accounting uses.
    return packed.data.numel() * 4
