"""Conformance tests for the backend-independent native v2 block layout."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rotquant.gguf import pack_qdata, unpack_qdata
from rotquant.native import (
    NATIVE_FORMAT_VERSION,
    NativeEncodedMatrix,
    NativeLayout,
    encode_quantized_weight,
    pack_native_blocks,
    reference_dequantize,
    reference_matmul,
    reference_streaming_matmul,
    unpack_native_blocks,
)
from rotquant.quantize import QuantConfig, Quantizer
from scripts.benchmark_native_reference import benchmark_case


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("group_size", [7, 32, 128])
def test_native_v2_round_trip_all_profile_widths(bits, group_size):
    rng = np.random.default_rng(bits * 100 + group_size)
    in_features = group_size + 3
    layout = NativeLayout(bits=bits, group_size=group_size)
    indices = rng.integers(
        0, 1 << bits, size=(3, in_features), dtype=np.uint16
    )
    scales = rng.uniform(0.01, 0.5, size=(3, 2)).astype(np.float16)
    qdata = pack_native_blocks(indices, scales, layout)
    actual_indices, actual_scales = unpack_native_blocks(
        qdata, in_features=in_features, layout=layout
    )
    np.testing.assert_array_equal(actual_indices, indices.astype(np.uint8))
    np.testing.assert_array_equal(actual_scales, scales)
    assert qdata.shape == (3, layout.row_bytes(in_features))
    assert qdata.dtype == np.int8


def test_native_v2_manifest_fixes_binary_interpretation():
    layout = NativeLayout(bits=3, group_size=64)
    manifest = layout.to_manifest()
    assert manifest == {
        "format": "rotquant-native-blocks",
        "format_version": NATIVE_FORMAT_VERSION,
        "bits": 3,
        "group_size": 64,
        "scale_dtype": "float16-le",
        "bit_order": "lsb_first",
        "code_bytes_per_group": 24,
        "bytes_per_group": 26,
    }


def test_native_v2_four_bit_blocks_are_byte_exact_with_gguf_v1():
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 16, size=(5, 256), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.5, size=(5, 2)).astype(np.float16)
    legacy = pack_qdata(indices, scales)
    generic = pack_native_blocks(indices, scales, NativeLayout(4, 128))
    np.testing.assert_array_equal(generic, legacy)
    legacy_indices, legacy_scales = unpack_qdata(legacy, in_features=256)
    actual_indices, actual_scales = unpack_native_blocks(
        generic, in_features=256, layout=NativeLayout(4, 128)
    )
    np.testing.assert_array_equal(actual_indices, legacy_indices)
    np.testing.assert_array_equal(actual_scales, legacy_scales)


@pytest.mark.parametrize("bits", range(1, 9))
def test_native_v2_encodes_quantized_weight_without_requantizing(bits):
    torch.manual_seed(bits)
    weight = torch.randn(5, 19)
    qweight = Quantizer(
        QuantConfig(
            bits=bits,
            codebook="gaussian",
            scale="rms",
            group_size=8,
        )
    ).quantize_weight(weight)
    encoded = encode_quantized_weight(qweight)
    expected = qweight.dequantize().numpy()
    actual = reference_dequantize(encoded)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert encoded.layout.bits == bits
    assert encoded.codebook.shape == (1 << bits,)
    assert encoded.persistent_bytes == encoded.qdata.nbytes + encoded.codebook.nbytes


def test_native_v2_reference_matmul_uses_encoded_matrix():
    torch.manual_seed(9)
    qweight = Quantizer(
        QuantConfig(bits=3, codebook="uniform", scale="rms", group_size=8)
    ).quantize_weight(torch.randn(7, 17))
    encoded = encode_quantized_weight(qweight)
    values = np.random.default_rng(3).normal(size=(2, 4, 17)).astype(np.float32)
    expected = values @ qweight.dequantize().numpy().T
    actual = reference_matmul(values, encoded)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    streaming = reference_streaming_matmul(values, encoded)
    np.testing.assert_allclose(streaming, expected, rtol=1e-6, atol=1e-5)


def test_native_v2_accepts_spherical_codebook_and_folded_length_correction():
    torch.manual_seed(31)
    qweight = Quantizer(QuantConfig(
        bits=2,
        codebook="spherical",
        codebook_dim=64,
        scale="rms",
        group_size=64,
        bias_correction="length",
    )).quantize_weight(torch.randn(6, 128))
    encoded = encode_quantized_weight(qweight)
    np.testing.assert_allclose(
        reference_dequantize(encoded), qweight.dequantize().numpy(), rtol=0, atol=0)
    assert encoded.codebook.shape == (4,)


def test_native_encoded_matrix_rejects_wrong_codebook_size():
    layout = NativeLayout(2, 8)
    qdata = pack_native_blocks(
        np.zeros((1, 8), dtype=np.uint8),
        np.ones((1, 1), dtype=np.float16),
        layout,
    )
    with pytest.raises(ValueError, match=r"2\*\*bits"):
        NativeEncodedMatrix(
            qdata=qdata,
            codebook=np.zeros(3, dtype=np.float32),
            layout=layout,
            in_features=8,
            out_features=1,
        )


def test_native_reference_benchmark_smoke_is_self_checking():
    result = benchmark_case(
        bits=3,
        out_features=8,
        in_features=17,
        group_size=8,
        batch=2,
        iterations=1,
        warmup=0,
        seed=0,
    )
    assert result["max_abs_error"] < 1e-5
    assert result["qdata_bytes"] > 0
    assert result["persistent_bits_per_weight"] > result["qdata_bits_per_weight"]
