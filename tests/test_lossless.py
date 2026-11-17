"""Exactness contracts for the lossless stages around the quantiser."""
import math

import numpy as np
import pytest
import torch

from rotquant.lossless import (
    code_entropy,
    compress_codes,
    decompress_codes,
    fixed_point_to_floats,
    floats_to_fixed_point,
    integer_butterfly,
    integer_butterfly_inverse,
    rans_decode_rows,
    rans_encode_rows,
    symbol_entropy,
)
from rotquant.pack import pack_indices
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import ButterflyRotation, RandomizedHadamard


def _signs(dim, seed):
    return RandomizedHadamard(dim, block=min(dim, 128), seed=seed).signs


@pytest.mark.parametrize("block", [2, 8, 128])
def test_integer_hadamard_is_a_bijection(block):
    generator = torch.Generator().manual_seed(block)
    dim = 2 * block
    values = torch.randint(-(1 << 26), 1 << 26, (37, dim), generator=generator)
    signs = _signs(dim, 3) if block == 128 else torch.ones(dim, dtype=torch.int8)
    rotated = integer_butterfly(values, signs, block)
    assert rotated.dtype == torch.int64
    assert not torch.equal(rotated, values)
    assert torch.equal(integer_butterfly_inverse(rotated, signs, block), values)


def test_integer_butterfly_inverts_arbitrary_angles_and_tracks_the_real_rotation():
    dim, block = 256, 128
    rotation = ButterflyRotation(dim, block=block, seed=5)
    with torch.no_grad():
        rotation.theta.add_(0.4 * torch.randn(rotation.theta.shape,
                                              generator=torch.Generator().manual_seed(1)))
    generator = torch.Generator().manual_seed(2)
    values = torch.randint(-(1 << 24), 1 << 24, (16, dim), generator=generator)
    rotated = integer_butterfly(values, rotation.signs, block, rotation.theta)
    restored = integer_butterfly_inverse(rotated, rotation.signs, block, rotation.theta)
    assert torch.equal(restored, values)
    # The integer transform is the real rotation plus bounded rounding noise:
    # three shears per stage, each off by at most half a unit before growth.
    reference = rotation.rotate_activation(values.to(torch.float64))
    assert (rotated.to(torch.float64) - reference).abs().max() < 64
    # ...which is why it preserves energy to rounding, like the real one.
    ratio = rotated.to(torch.float64).square().sum() / values.to(torch.float64).square().sum()
    assert abs(float(ratio) - 1.0) < 1e-6


def test_floating_point_rotation_does_not_round_trip_but_the_integer_one_does():
    torch.manual_seed(0)
    weight = (torch.randn(32, 512) * 0.02).to(torch.float16)
    rotation = RandomizedHadamard(512, block=128, seed=1)
    floating = rotation.inverse_activation(rotation.rotate_weight(weight))
    assert not torch.equal(floating, weight)

    fixed = floats_to_fixed_point(weight)
    rotated = integer_butterfly(fixed.integers, rotation.signs, 128)
    restored = integer_butterfly_inverse(rotated, rotation.signs, 128)
    recovered = fixed_point_to_floats(
        type(fixed)(restored, fixed.exponent, fixed.dtype, fixed.negative_zero))
    assert recovered.dtype == torch.float16
    assert torch.equal(recovered.view(torch.int16), weight.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fixed_point_view_is_exact_including_subnormals_and_negative_zero(dtype):
    torch.manual_seed(3)
    values = (torch.randn(4, 64) * 0.05).to(dtype)
    values[0, 0] = 0.0
    values[0, 1] = -0.0
    if dtype == torch.float16:
        values[0, 2] = 2.0 ** -24          # smallest fp16 subnormal
        values[0, 3] = -3 * 2.0 ** -24
    fixed = floats_to_fixed_point(values)
    recovered = fixed_point_to_floats(fixed)
    integer_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(recovered.view(integer_dtype), values.view(integer_dtype))
    assert torch.equal(
        torch.ldexp(fixed.integers.double(), torch.tensor(fixed.exponent)), values.double())


def test_integer_rotation_rejects_inputs_that_could_overflow():
    values = torch.full((1, 128), 1 << 40, dtype=torch.int64)
    with pytest.raises(ValueError, match="must stay below"):
        integer_butterfly(values, torch.ones(128, dtype=torch.int8), 128)
    with pytest.raises(TypeError):
        integer_butterfly(values.float(), torch.ones(128, dtype=torch.int8), 128)


@pytest.mark.parametrize("alphabet,rows,columns", [(2, 5, 1), (4, 1, 300), (16, 9, 257),
                                                  (32, 64, 128), (256, 7, 1000)])
def test_rans_round_trips_and_approaches_the_entropy(alphabet, rows, columns):
    generator = np.random.default_rng(alphabet + rows)
    # A discretised Gaussian over the alphabet, like Lloyd-Max codes.
    centred = generator.normal(alphabet / 2, alphabet / 6, size=(rows, columns))
    symbols = np.clip(np.rint(centred), 0, alphabet - 1).astype(np.int64)
    coded = rans_encode_rows(symbols, alphabet)
    assert np.array_equal(rans_decode_rows(coded), symbols)
    if rows * columns >= 4096:
        ideal = symbol_entropy(symbols, alphabet) * symbols.size / 8
        # Table precision plus four state bytes per row.
        assert coded.words.size * 2 <= ideal * 1.02 + 4 * rows


def test_rans_handles_degenerate_streams_and_detects_corruption():
    constant = np.full((3, 50), 5, dtype=np.int64)
    coded = rans_encode_rows(constant, 8)
    assert coded.words.size == 3 * 2          # nothing but the flushed states
    assert np.array_equal(rans_decode_rows(coded), constant)
    assert rans_decode_rows(rans_encode_rows(np.zeros((0, 4), dtype=np.int64), 4)).shape == (0, 4)

    symbols = np.random.default_rng(0).integers(0, 16, size=(4, 400))
    coded = rans_encode_rows(symbols, 16)
    broken = type(coded)(coded.frequencies, coded.row_words, coded.words.copy(), coded.shape)
    broken.words[5] ^= 0x1234
    try:
        decoded = rans_decode_rows(broken)
    except ValueError:
        return
    assert not np.array_equal(decoded, symbols)
    with pytest.raises(ValueError):
        rans_encode_rows(np.array([[16]]), 16)


@pytest.mark.parametrize("bits", [3, 4, 5, 6])
def test_entropy_coding_packed_codes_is_bit_exact_and_smaller(bits):
    torch.manual_seed(bits)
    weight = RandomizedHadamard(512, block=128, seed=0).rotate_weight(torch.randn(96, 512))
    qweight = Quantizer(QuantConfig(bits=bits, group_size=128, scale="mse_search")
                        ).quantize_weight(weight)
    report = code_entropy(qweight.packed)
    assert 0.0 < report["saving_fraction"] < 0.15
    assert report["entropy_bits_per_symbol"] < bits

    coded = compress_codes(qweight.packed)
    restored = decompress_codes(coded, bits, qweight.packed.shape)
    assert torch.equal(restored.data, qweight.packed.data)
    assert restored.shape == qweight.packed.shape
    # Real coded bytes sit within 2% of the entropy plus the per-row state.
    assert coded.nbytes * 8 <= report["entropy_bits"] * 1.02 + (32 + 32) * 96 + 16 * 2**bits
    assert coded.nbytes < qweight.packed.data.numel() * 4


def test_code_entropy_matches_a_known_distribution():
    indices = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])
    report = code_entropy(pack_indices(indices, 2))
    assert math.isclose(report["entropy_bits_per_symbol"], 1.75)
    assert math.isclose(report["saving_fraction"], 1 - 1.75 / 2)
