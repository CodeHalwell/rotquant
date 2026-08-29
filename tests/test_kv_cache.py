"""Rotation-consistent packed KV-cache simulation and calibration."""
import torch

from rotquant.kv_cache import (
    KVQuantConfig,
    build_kv_rotation,
    kv_fidelity_metrics,
    quantize_kv,
    reference_attention,
    rotquant_attention,
    train_kv_rotations,
)


def _triplet(seed: int, length: int = 8):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 2, length, 128)
    return tuple(torch.randn(shape, generator=generator) for _ in range(3))


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


def test_learned_kv_rotation_uses_held_out_best_checkpoint():
    config = KVQuantConfig(bits=3, group_size=64, seed=5)
    key_rotation, value_rotation, stats = train_kv_rotations(
        _triplet(1), _triplet(2), config, steps=3, lr=1e-3)
    assert 0 <= stats["best_step"] <= 3
    assert stats["best_validation_mse"] <= stats["initial_validation_mse"]
    assert not key_rotation.theta.requires_grad
    assert not value_rotation.theta.requires_grad
