"""Bit-packing round-trip + footprint accounting.

The packed code buffer is the storage primitive the whole footprint argument
rests on; if pack/unpack isn't lossless (or silently allocates on the wrong
device) every GPU run is wrong, so this is a guard test.
"""
import pytest
import torch

from rotquant.format import packed_word_count
from rotquant.pack import PackedTensor, pack_indices, packed_bytes, unpack_indices


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


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("numel", [0, 1, 7, 31, 32, 33, 65, 257])
def test_optimized_profile_packing_has_exact_word_count(bits, numel):
    torch.manual_seed(bits * 1000 + numel)
    idx = torch.randint(0, 1 << bits, (numel,), dtype=torch.int64)
    packed = _roundtrip(idx, bits)
    assert packed.data.numel() == packed_word_count(numel, bits)
    assert packed_bytes(packed) == packed_word_count(numel, bits) * 4


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bool])
def test_pack_rejects_non_integer_indices(dtype):
    values = torch.zeros(4, dtype=dtype)
    with pytest.raises(TypeError, match="integer dtype"):
        pack_indices(values, 2)


def test_packed_tensor_validates_storage_contract():
    with pytest.raises(TypeError, match="int32"):
        PackedTensor(
            data=torch.zeros(1, dtype=torch.int64),
            shape=(1,),
            bits=4,
            numel=1,
        )
    with pytest.raises(ValueError, match="expected 2"):
        PackedTensor(
            data=torch.zeros(1, dtype=torch.int32),
            shape=(9,),
            bits=4,
            numel=9,
        )


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
