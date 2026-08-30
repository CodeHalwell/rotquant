"""Conformance tests for the native RotQuant-GGUF byte layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rotquant.gguf import (
    GROUP_SIZE,
    native_tensor,
    native_tied_tensor,
    pack_qdata,
    reference_embedding,
    reference_linear,
    unpack_qdata,
)
from rotquant.linear import QuantLinear
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import ButterflyRotation
from scripts.export_rotquant_gguf import _kv_cache_metadata, _with_kv_cache_config

FROZEN_MIXED_RECIPE = [
    {"layer": 3, "key_bits": 2, "value_bits": 2},
    {"layer": 7, "key_bits": 4, "value_bits": 4},
    {"layer": 11, "key_bits": 3, "value_bits": 3},
    {"layer": 15, "key_bits": 4, "value_bits": 2},
    {"layer": 19, "key_bits": 3, "value_bits": 3},
    {"layer": 23, "key_bits": 3, "value_bits": 2},
    {"layer": 27, "key_bits": 4, "value_bits": 3},
    {"layer": 31, "key_bits": 2, "value_bits": 4},
]


def _manifest_with_frozen_kv(**overrides):
    cache = {
        "codebook": "gaussian",
        "group_size": 64,
        "effective_bpv": 3.25,
        "frozen_recipe": FROZEN_MIXED_RECIPE,
    }
    cache.update(overrides)
    return {"deployment": {"kv_cache": cache}}


def test_frozen_mixed_kv_metadata_is_exact_release_contract():
    metadata = _kv_cache_metadata(_manifest_with_frozen_kv())
    assert metadata is not None
    assert metadata["effective_bpv"] == 3.25
    assert metadata["key_bits"] == [
        0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4,
        0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 2,
    ]
    assert metadata["value_bits"] == [
        0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 2,
        0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4,
    ]


def test_explicit_kv_config_preserves_other_deployment_metadata():
    manifest = {"deployment": {"recipe": "uniform_w4"}}
    config = _manifest_with_frozen_kv()["deployment"]["kv_cache"]
    updated = _with_kv_cache_config(manifest, config)
    assert updated["deployment"]["recipe"] == "uniform_w4"
    assert updated["deployment"]["kv_cache"] == config
    assert "kv_cache" not in manifest["deployment"]


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"group_size": 128}, "group_size=64"),
        ({"codebook": "uniform"}, "Gaussian"),
        ({"frozen_recipe": FROZEN_MIXED_RECIPE[:-1]}, "cover.*exactly"),
        ({"effective_bpv": 3.0}, "does not match"),
        ({
            "effective_bpv": 4.25,
            "frozen_recipe": [
                {"layer": layer, "key_bits": 4, "value_bits": 4}
                for layer in range(3, 32, 4)
            ],
        }, "requires effective_bpv=3.25"),
    ],
)
def test_frozen_mixed_kv_metadata_rejects_contract_drift(override, match):
    with pytest.raises(ValueError, match=match):
        _kv_cache_metadata(_manifest_with_frozen_kv(**override))


def _layer(out_features: int = 5, in_features: int = 256):
    torch.manual_seed(17)
    weight = torch.randn(out_features, in_features)
    rotation = ButterflyRotation(in_features, block=128, seed=9)
    with torch.no_grad():
        rotation.theta.add_(torch.randn_like(rotation.theta) * 0.015)
    qweight = Quantizer(
        QuantConfig(
            bits=4,
            codebook="gaussian",
            scale="mse_search",
            group_size=GROUP_SIZE,
        )
    ).quantize_weight(rotation.rotate_weight(weight))
    return QuantLinear(qweight, act_rotation=rotation).eval()


def test_native_qdata_round_trip():
    rng = np.random.default_rng(4)
    indices = rng.integers(0, 16, size=(3, 256), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.2, size=(3, 2)).astype(np.float16)
    packed = pack_qdata(indices, scales)
    actual_indices, actual_scales = unpack_qdata(packed, in_features=256)
    np.testing.assert_array_equal(actual_indices, indices)
    np.testing.assert_array_equal(actual_scales, scales)


def test_native_reference_matches_quant_linear():
    layer = _layer()
    x = torch.randn(2, 3, layer.in_features)
    with torch.no_grad():
        expected = layer(x).numpy()
    actual = reference_linear(
        x.numpy(), native_tensor(layer.qweight, layer.act_rotation)
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_complete_block_column_permutation_preserves_rebased_function():
    layer = _layer(out_features=7)
    block_permutation = np.array([1, 0])
    column_permutation = np.concatenate(
        [np.arange(block * 128, (block + 1) * 128) for block in block_permutation]
    )
    converted = native_tensor(
        layer.qweight,
        layer.act_rotation,
        column_permutation=column_permutation,
    )
    x = np.random.default_rng(5).normal(size=(4, layer.in_features)).astype(np.float32)
    expected = reference_linear(
        x[:, np.argsort(column_permutation)],
        native_tensor(layer.qweight, layer.act_rotation),
    )
    actual = reference_linear(x, converted)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)


def test_row_permutation_reorders_outputs():
    layer = _layer(out_features=7)
    permutation = np.array([6, 0, 5, 1, 4, 2, 3])
    original = native_tensor(layer.qweight, layer.act_rotation)
    converted = native_tensor(
        layer.qweight, layer.act_rotation, row_permutation=permutation
    )
    x = np.random.default_rng(8).normal(size=(2, layer.in_features)).astype(np.float32)
    expected = reference_linear(x, original)[:, permutation]
    actual = reference_linear(x, converted)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)


def test_tied_native_embedding_and_output_share_one_approximate_weight():
    torch.manual_seed(21)
    source = torch.randn(19, 256)
    native = native_tied_tensor(
        source, seed=7, scale="rms", chunk_rows=5)
    token_ids = np.array([[0, 3, 18], [7, 2, 11]])
    embedding = reference_embedding(token_ids, native)
    hidden = np.random.default_rng(12).normal(size=(4, 256)).astype(np.float32)
    logits = reference_linear(hidden, native)
    # Reconstructing every token row through the embedding lookup produces the
    # same tied approximate matrix consumed by the output projection.
    approximate_weight = reference_embedding(np.arange(19), native)
    np.testing.assert_allclose(
        embedding, approximate_weight[token_ids], rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(
        logits, hidden @ approximate_weight.T, rtol=3e-5, atol=2e-5)
