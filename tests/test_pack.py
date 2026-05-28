"""Bit-packing round-trip + footprint accounting.

The packed code buffer is the storage primitive the whole footprint argument
rests on; if pack/unpack isn't lossless (or silently allocates on the wrong
device) every GPU run is wrong, so this is a guard test.
"""
import torch

from rotquant.pack import pack_indices, unpack_indices, packed_bytes


def _roundtrip(idx, bits):
    packed = pack_indices(idx, bits)
    out = unpack_indices(packed)
    assert out.shape == idx.shape
    assert torch.equal(out, idx.to(torch.int64)), f"bits={bits} not lossless"
    return packed


def test_pack_roundtrip_all_bitwidths():
    torch.manual_seed(0)
    for bits in range(1, 17):
        idx = torch.randint(0, 1 << bits, (37, 53), dtype=torch.int64)
        _roundtrip(idx, bits)


def test_pack_roundtrip_boundary_straddle():
    # 3-bit codes don't divide 32 evenly, so many codes straddle word boundaries.
    idx = torch.arange(8, dtype=torch.int64).repeat(100) % 8
    _roundtrip(idx, 3)


def test_pack_high_bit_values_survive_int32_storage():
    # All-ones 16-bit codes set the top bit of each word; the int32 store wraps
    # to negative but unpack must still recover the unsigned value.
    idx = torch.full((64,), (1 << 16) - 1, dtype=torch.int64)
    packed = _roundtrip(idx, 16)
    assert packed.data.dtype == torch.int32


def test_pack_rejects_out_of_range():
    bad = torch.tensor([0, 1, 8], dtype=torch.int64)  # 8 needs >3 bits
    try:
        pack_indices(bad, 3)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_packed_bytes_is_true_footprint():
    idx = torch.randint(0, 8, (128, 128), dtype=torch.int64)
    packed = pack_indices(idx, 3)
    # int32 buffer -> 4 bytes per word, and that is the actual tensor memory.
    assert packed_bytes(packed) == packed.data.numel() * 4
    assert packed_bytes(packed) == packed.data.element_size() * packed.data.numel()
    # 3-bit codes must pack to under fp16 (2 bytes/code).
    assert packed_bytes(packed) < idx.numel() * 2


def test_pack_on_cuda_stays_on_device():
    if not torch.cuda.is_available():
        return
    idx = torch.randint(0, 8, (64, 64), dtype=torch.int64, device="cuda")
    packed = pack_indices(idx, 3)
    assert packed.data.is_cuda
    out = unpack_indices(packed)
    assert out.is_cuda
    assert torch.equal(out, idx.to(torch.int64))
