"""Storage fidelity is independent of model/rotation execution parity."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from rotquant.native import encode_quantized_weight
from rotquant.native_v3 import (
    NATIVE_V3_HEADER,
    NativeV3Layout,
    NativeV3Matrix,
    decode_native_v3_rows,
    encode_native_v3,
)
from rotquant.quantize import QuantConfig, Quantizer


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("scale_bits", [8, 16])
def test_v3_retains_canonical_bytes_and_weights(bits, scale_bits):
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(7, 19, generator=generator) * torch.arange(1, 8)[:, None]
    q = Quantizer(QuantConfig(bits=bits, scale_bits=scale_bits, scale="rms",
                             group_size=7, scale_quant_group_size=4)).quantize_weight(weight)
    matrix = encode_native_v3(q)
    assert matrix.words.tobytes() == q.packed.data.numpy().tobytes()
    assert matrix.scales.tobytes() == q.scales.numpy().tobytes()
    assert matrix.codebook.tobytes() == q.codebook.centroids.numpy().tobytes()
    if scale_bits == 8:
        assert matrix.offsets.tobytes() == q.scale_offsets.numpy().tobytes()
        assert matrix.steps.tobytes() == q.scale_steps.numpy().tobytes()
    assert matrix.to_bytes() == NativeV3Matrix.from_bytes(matrix.to_bytes()).to_bytes()
    assert matrix.to_manifest()["total_bytes"] == matrix.persistent_bytes
    assert not matrix.words.flags.writeable
    np.testing.assert_array_equal(decode_native_v3_rows(matrix), q.dequantize().numpy())
    np.testing.assert_array_equal(decode_native_v3_rows(matrix, 1, 2), q.dequantize().numpy()[1:3])
    before = matrix.to_bytes()
    q.packed.data.zero_()
    assert matrix.to_bytes() == before


def tiny_matrix():
    return NativeV3Matrix(NativeV3Layout(5, 2, 1, 3, 8, 2),
                          np.array([0], dtype="<i4"), np.array([[1, 255]], dtype="u1"),
                          np.array([1], dtype="<f2"), np.array([2**-24], dtype="<f2"),
                          np.arange(32, dtype="<f4"))


@pytest.mark.parametrize("field,value", [(0, b"BADMAGIC"), (8, (4).to_bytes(4, "little")),
                                         (48, (1).to_bytes(4, "little")),
                                         (52, (1).to_bytes(4, "little"))])
def test_v3_rejects_unknown_header(field, value):
    raw = tiny_matrix().to_bytes()
    changed = raw[:field] + value + raw[field + len(value):]
    with pytest.raises(ValueError, match="header"):
        NativeV3Matrix.from_bytes(changed)


def test_v3_rejects_invalid_storage_and_lengths():
    matrix = tiny_matrix()
    for raw in (b"", matrix.to_bytes()[:-1], matrix.to_bytes() + b"\0"):
        with pytest.raises(ValueError):
            NativeV3Matrix.from_bytes(raw)
    with pytest.raises(ValueError, match="unused bits"):
        replace(matrix, words=np.array([-1], dtype="<i4"))
    with pytest.raises(ValueError, match="non-finite"):
        replace(matrix, steps=np.array([np.inf], dtype="<f2"))
    with pytest.raises(ValueError, match="non-negative"):
        replace(matrix, offsets=np.array([-1], dtype="<f2"))
    with pytest.raises(TypeError, match="storage dtype"):
        replace(matrix, codebook=matrix.codebook.astype(np.float16))
    for start, count in ((-1, 1), (0, 0), (True, 1), (0, 2)):
        with pytest.raises(ValueError):
            decode_native_v3_rows(matrix, start, count)
    assert NATIVE_V3_HEADER.size == 64


def test_v3_oracle_does_not_allocate_group_padding():
    matrix = NativeV3Matrix(NativeV3Layout(1, 2**32 - 1, 1, 1, 16),
                            np.array([1], dtype="<i4"), np.array([[2]], dtype="<f2"),
                            np.empty(0, dtype="<f2"), np.empty(0, dtype="<f2"),
                            np.array([-1, 1], dtype="<f4"))
    np.testing.assert_array_equal(decode_native_v3_rows(matrix), [[2.]])
    with pytest.raises(ValueError, match="scale slice"):
        matrix.decoded_scales(-1, 1)


def test_v3_rejects_float32_scales_and_residuals():
    q = Quantizer(QuantConfig(bits=5, scale="rms", group_size=8)).quantize_weight(torch.ones(2, 9))
    with pytest.raises(ValueError, match="8/16-bit"):
        encode_native_v3(replace(q, scales=q.scales.float(), scale_bits_main=32))
    with pytest.raises(ValueError, match="residual"):
        encode_native_v3(replace(q, residual_packed=q.packed))
    with pytest.raises(ValueError, match="matching dimensions"):
        encode_native_v3(replace(q, out_features=1))


@pytest.mark.parametrize("scale_bits", [8, 32])
def test_native_v2_rejects_lossy_scale_conversion(scale_bits):
    q = Quantizer(QuantConfig(bits=5, scale_bits=scale_bits, scale="rms",
                             group_size=128)).quantize_weight(torch.randn(3, 256))
    with pytest.raises(ValueError, match="requires stored 16-bit scales"):
        encode_quantized_weight(q)


def test_scale8_is_not_equivalent_to_a_float16_scale_export():
    torch.manual_seed(7)
    weight = torch.randn(64, 256) * torch.linspace(.3, 3., 64)[:, None]
    q = Quantizer(QuantConfig(bits=5, scale_bits=8, group_size=128,
                             scale="rms")).quantize_weight(weight)
    exact = decode_native_v3_rows(encode_native_v3(q))
    lossy = replace(q, scales=q.main_scales().half(), scale_bits_main=16,
                    scale_offsets=None, scale_steps=None)
    assert np.max(np.abs(exact - lossy.dequantize().numpy())) > .001
