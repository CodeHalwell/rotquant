"""Rotation-consistent packed KV-cache simulation and calibration."""
import pytest
import torch

from rotquant.kv_cache import (
    KVQuantConfig,
    build_kv_rotation,
    kv_fidelity_metrics,
    kv_retrieval_metrics,
    oracle_value_retrieval_curve,
    quantize_kv,
    reference_attention,
    retrieval_rotquant_decode,
    rotquant_attention,
    train_kv_rotations,
)


def _triplet(seed: int, length: int = 8):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 2, length, 128)
    return tuple(torch.randn(shape, generator=generator) for _ in range(3))


def test_non_spherical_small_groups_ignore_codebook_dimension():
    config = KVQuantConfig(codebook="gaussian", group_size=1).quant_config()
    assert config.group_size == 1
    assert config.codebook_dim is None


def test_spherical_codebook_dimension_defaults_to_group_size():
    config = KVQuantConfig(codebook="spherical", group_size=3).quant_config()
    assert config.codebook_dim == 3


def test_packed_kv_round_trip_and_attention_metrics():
    queries, keys, values = _triplet(0)
    config = KVQuantConfig(bits=4, group_size=64, seed=3)
    key_rotation = build_kv_rotation(128, config)
    packed = quantize_kv(keys, key_rotation, config)
    assert packed.dequantize().shape == keys.shape
    assert packed.dequantize(original_basis=True).shape == keys.shape
    assert packed.packed_state_bytes() < keys.numel() * 2

    output, _, _ = rotquant_attention(
        queries, keys, values, key_rotation,
        build_kv_rotation(128, config, value=True), config)
    assert output.shape == reference_attention(queries, keys, values).shape
    metrics = kv_fidelity_metrics(queries, keys, values, config)
    assert 0 < metrics["relative_attention_mse"] < 1
    assert metrics["cosine"] > 0.9
    assert metrics["compression_ratio"] > 2
    assert metrics["relative_attention_logit_mse"] > 0
    assert metrics["attention_logit_mae"] > 0
    assert metrics["key_nmse"] > 0
    assert 0 < metrics["key_self_dot_ratio"] < 2


def test_kv_length_correction_targets_self_dot_at_same_rate():
    queries, keys, values = _triplet(9)
    plain = kv_fidelity_metrics(
        queries, keys, values,
        KVQuantConfig(bits=2, group_size=128, bias_correction="none", seed=2),
    )
    corrected = kv_fidelity_metrics(
        queries, keys, values,
        KVQuantConfig(bits=2, group_size=128, bias_correction="length", seed=2),
    )
    assert corrected["effective_bpv"] == plain["effective_bpv"]
    assert abs(corrected["key_self_dot_ratio"] - 1) < (
        abs(plain["key_self_dot_ratio"] - 1))


def test_full_candidate_retrieval_matches_dense_rotquant_decode():
    queries, keys, values = _triplet(17, length=12)
    queries = queries[..., -1:, :]
    config = KVQuantConfig(bits=3, group_size=64, seed=8)
    key_rotation = build_kv_rotation(128, config)
    value_rotation = build_kv_rotation(128, config, value=True)
    dense, _, _ = rotquant_attention(
        queries, keys, values, key_rotation, value_rotation, config)
    retrieved, selected, _, _ = retrieval_rotquant_decode(
        queries,
        keys,
        values,
        key_rotation,
        value_rotation,
        config,
        retrieval_k=keys.shape[-2],
    )
    assert selected.shape[-1] == keys.shape[-2]
    assert torch.allclose(retrieved, dense, atol=1e-5)


def test_selective_kv_retrieval_reports_mass_and_value_read_reduction():
    queries, keys, values = _triplet(19, length=16)
    metrics = kv_retrieval_metrics(
        queries[..., -1:, :],
        keys,
        values,
        KVQuantConfig(bits=4, group_size=64, seed=3),
        retrieval_k=8,
        recent_window=3,
        sink_tokens=1,
    )
    assert metrics["selected_fraction"] == 0.5
    assert metrics["value_vector_read_fraction"] == 0.5
    assert 0 < metrics["reference_attention_mass_coverage"] <= 1
    assert metrics["relative_attention_mse"] >= 0


def test_retrieval_budget_must_cover_mandatory_tokens():
    queries, keys, values = _triplet(23, length=8)
    config = KVQuantConfig(bits=4, group_size=64)
    with torch.no_grad(), pytest.raises(ValueError, match="mandatory"):
        retrieval_rotquant_decode(
            queries[..., -1:, :],
            keys,
            values,
            build_kv_rotation(128, config),
            build_kv_rotation(128, config, value=True),
            config,
            retrieval_k=2,
            recent_window=2,
            sink_tokens=2,
        )


def test_oracle_value_retrieval_curve_reports_adaptive_dense_fallback() -> None:
    torch.manual_seed(31)
    values = torch.randn(1, 2, 16, 8)
    concentrated = torch.full((1, 4, 1, 16), 0.01 / 15)
    concentrated[..., 5] = 0.99
    diffuse = torch.full((1, 4, 1, 16), 1 / 16)
    weights = torch.cat([concentrated, diffuse], dim=2)

    curve = oracle_value_retrieval_curve(
        weights,
        values,
        [1, 4, 16],
        mass_threshold=0.9,
    )

    assert curve[0]["mean_attention_mass"] > 0.5
    assert curve[0]["dense_fallback_fraction"] == 0.5
    assert curve[0]["effective_value_read_fraction"] == pytest.approx(0.53125)
    assert curve[-1]["mean_attention_mass"] == pytest.approx(1.0)
    assert curve[-1]["gated_relative_attention_mse"] < 1e-12


def test_oracle_value_retrieval_uses_source_values_as_common_reference() -> None:
    weights = torch.tensor([[[[0.7, 0.2, 0.1]]]])
    source = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    quantized = source + 0.25

    source_full = oracle_value_retrieval_curve(weights, source, [3])[0]
    quantized_full = oracle_value_retrieval_curve(
        weights,
        quantized,
        [3],
        reference_values=source,
    )[0]

    assert source_full["relative_attention_mse"] == pytest.approx(0.0, abs=1e-7)
    assert quantized_full["relative_attention_mse"] > 0
    assert quantized_full["relative_attention_mse"] == pytest.approx(
        quantized_full["dense_value_relative_attention_mse"]
    )

    gated = oracle_value_retrieval_curve(
        weights,
        quantized,
        [1],
        reference_values=source,
        mass_threshold=0.95,
    )[0]
    assert gated["dense_fallback_fraction"] == 1.0
    assert gated["gated_relative_attention_mse"] == pytest.approx(
        gated["dense_value_relative_attention_mse"]
    )


def test_learned_kv_rotation_uses_held_out_best_checkpoint():
    config = KVQuantConfig(bits=3, group_size=64, seed=5)
    key_rotation, value_rotation, stats = train_kv_rotations(
        _triplet(1), _triplet(2), config, steps=3, lr=1e-3)
    assert 0 <= stats["best_step"] <= 3
    assert stats["best_validation_mse"] <= stats["initial_validation_mse"]
    assert not key_rotation.theta.requires_grad
    assert not value_rotation.theta.requires_grad
