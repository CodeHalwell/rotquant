"""End-to-end tests for Transformers-style cache simulation."""
from types import SimpleNamespace

import pytest
import torch

from rotquant.eval.kv_cache import (
    KVCacheEvalConfig,
    KVDynamicConfig,
    evaluate_kv_cache,
    simulate_packed_kv_cache,
)
from rotquant.kv_cache import KVQuantConfig


class _CacheLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values
        self.is_initialized = True

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = torch.cat((self.keys, keys), dim=-2)
        self.values = torch.cat((self.values, values), dim=-2)
        return self.keys, self.values


class _RecurrentLayer:
    def __init__(self):
        self.recurrent_states = torch.ones(1, 4, 4)


class _Cache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.layers = [_CacheLayer(keys, values), _RecurrentLayer()]

    def update(self, keys, values, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(keys, values)


def _states(ids: torch.Tensor, dim: int = 8):
    basis = torch.linspace(0.25, 1.75, dim, device=ids.device)
    state = torch.sin(ids.float().unsqueeze(1).unsqueeze(-1) * basis)
    return state, torch.cos(state * 1.7)


class _CacheModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=True):
        keys, values = _states(input_ids)
        if past_key_values is None:
            cache = _Cache(keys, values)
        else:
            cache = past_key_values
            keys, values = cache.update(keys, values, 0)
        summary = (keys.mean(dim=(-3, -2, -1))
                   + 2 * values.mean(dim=(-3, -2, -1)))
        vocab = torch.linspace(-1.0, 1.0, 32, device=input_ids.device)
        logits = summary[:, None, None] * vocab[None, None, :]
        logits = logits.expand(-1, input_ids.shape[1], -1).contiguous()
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_simulated_cache_packs_only_kv_and_quantizes_new_writes():
    ids = torch.arange(1, 7).reshape(1, -1)
    keys, values = _states(ids)
    source = _Cache(keys, values)
    simulated, metrics = simulate_packed_kv_cache(
        source, KVQuantConfig(bits=4, group_size=4, rotation_block=8))

    assert metrics["kv_layers"] == 1
    assert metrics["packed_kv_bytes"] < metrics["source_kv_bytes"]
    assert metrics["prefill_key_nmse"] > 0
    assert metrics["prefill_value_nmse"] > 0
    assert metrics["prefill_kv_nmse"] > 0
    assert metrics["non_kv_state_bytes"] > 0
    assert simulated.layers[0].keys.shape == source.layers[0].keys.shape
    new_keys, new_values = _states(torch.tensor([[7]]))
    simulated.update(new_keys, new_values, 0)
    assert simulated.layers[0].keys.shape[-2] == 7
    assert source.layers[0].keys.shape[-2] == 6


def test_prefill_reconstruction_nmse_improves_with_precision():
    ids = torch.arange(1, 33).reshape(1, -1)
    keys, values = _states(ids)
    four_bit = simulate_packed_kv_cache(
        _Cache(keys, values),
        KVQuantConfig(bits=4, group_size=4, rotation_block=8),
    )[1]
    eight_bit = simulate_packed_kv_cache(
        _Cache(keys, values),
        KVQuantConfig(bits=8, group_size=4, rotation_block=8),
    )[1]

    assert eight_bit["prefill_key_nmse"] < four_bit["prefill_key_nmse"]
    assert eight_bit["prefill_value_nmse"] < four_bit["prefill_value_nmse"]


def test_end_to_end_cache_eval_reports_global_quality_and_bytes():
    ids = torch.arange(1, 25).reshape(1, -1) % 31
    config = KVCacheEvalConfig(
        bits=4,
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=4,
        skip=0,
    )
    metrics = evaluate_kv_cache(
        _CacheModel(), [{"input_ids": ids}], config, "cpu")

    assert metrics["evaluated_tokens"] == 4
    assert metrics["kv_layers"] == 1
    assert metrics["mean_teacher_kl"] >= 0
    assert 0 <= metrics["top1_agreement"] <= 1
    assert metrics["source_total_cache_bytes"] > metrics["deployed_total_cache_bytes"]


def test_cache_eval_supplies_absolute_decode_positions_when_supported():
    class PositionAwareCacheModel(_CacheModel):
        def __init__(self):
            super().__init__()
            self.decode_positions = []

        def forward(self, input_ids, attention_mask=None, past_key_values=None,
                    position_ids=None, use_cache=True):
            if past_key_values is not None:
                expected = past_key_values.layers[0].keys.shape[-2]
                assert position_ids is not None
                assert position_ids.shape == (input_ids.shape[0], 1)
                assert position_ids.eq(expected).all()
                self.decode_positions.append(expected)
            return super().forward(
                input_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=use_cache)

    ids = torch.arange(1, 25).reshape(1, -1) % 31
    model = PositionAwareCacheModel()
    config = KVCacheEvalConfig(
        bits=4,
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=3,
        skip=0,
    )

    evaluate_kv_cache(model, [{"input_ids": ids}], config, "cpu")

    assert model.decode_positions == [8, 8, 9, 9, 10, 10]


def test_qwen35_multimodal_stale_rope_state_accepts_cached_decode():
    qwen35 = pytest.importorskip("transformers.models.qwen3_5")
    text_config = qwen35.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        eos_token_id=63,
        pad_token_id=63,
    )
    vision_config = qwen35.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    config = qwen35.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = qwen35.Qwen3_5ForConditionalGeneration(config).eval()
    # ``generate`` leaves this multimodal RoPE state populated. Without an
    # explicit one-token position the wrapper expands positions to the full
    # attention-mask width during the next independent cache evaluation.
    model.model.rope_deltas = torch.zeros((1, 1), dtype=torch.long)
    ids = torch.arange(1, 25).reshape(1, -1) % 50
    eval_config = KVCacheEvalConfig(
        bits=4,
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=2,
        skip=0,
    )

    metrics = evaluate_kv_cache(
        model, [{"input_ids": ids}], eval_config, "cpu")

    assert metrics["evaluated_tokens"] == 2
    assert metrics["kv_layers"] == 1


def test_dynamic_cache_allocator_uses_disjoint_selection_and_exact_budget():
    batches = [
        {"input_ids": (torch.arange(offset, offset + 24).reshape(1, -1) % 31)}
        for offset in (1, 7)
    ]
    config = KVCacheEvalConfig(
        bits=4,
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=2,
        skip=0,
        dynamic={
            "candidate_bits": [2, 4],
            "target_bpv": 7.0,
            "selection_batches": 1,
        },
    )
    metrics = evaluate_kv_cache(_CacheModel(), batches, config, "cpu")

    dynamic = metrics["dynamic"]
    assert dynamic["target_reached"]
    assert dynamic["deployed_bytes"] <= dynamic["target_bytes"]
    assert dynamic["recipe"] == [{"layer": 0, "key_bits": 2, "value_bits": 4}]
    assert metrics["key_bits"] == 2
    assert metrics["value_bits"] == 4
    assert metrics["evaluated_tokens"] == 2


def test_uniform_and_dynamic_use_same_explicit_held_out_batches():
    batches = [
        {"input_ids": (torch.arange(offset, offset + 24).reshape(1, -1) % 31)}
        for offset in (1, 7)
    ]
    common = {
        "bits": 4,
        "group_size": 4,
        "rotation_block": 8,
        "batches": 1,
        "eval_offset_batches": 1,
        "prompt_len": 8,
        "continuation_len": 2,
        "skip": 0,
    }
    uniform = evaluate_kv_cache(
        _CacheModel(), batches, KVCacheEvalConfig(**common), "cpu")
    dynamic = evaluate_kv_cache(
        _CacheModel(), batches,
        KVCacheEvalConfig(**common, dynamic={
            "candidate_bits": [4],
            "target_bpv": 8.0,
            "selection_batches": 1,
        }),
        "cpu",
    )

    assert dynamic["dynamic"]["recipe"] == [
        {"layer": 0, "key_bits": 4, "value_bits": 4}]
    assert dynamic["mean_teacher_kl"] == uniform["mean_teacher_kl"]
    assert dynamic["nll_delta"] == uniform["nll_delta"]
    assert dynamic["top1_agreement"] == uniform["top1_agreement"]


def test_frozen_recipe_matches_explicit_uniform_profile():
    batches = [
        {"input_ids": (torch.arange(offset, offset + 24).reshape(1, -1) % 31)}
        for offset in (1, 7)
    ]
    common = {
        "bits": 4,
        "group_size": 4,
        "rotation_block": 8,
        "batches": 1,
        "eval_offset_batches": 1,
        "prompt_len": 8,
        "continuation_len": 2,
        "skip": 0,
    }
    uniform = evaluate_kv_cache(
        _CacheModel(), batches,
        KVCacheEvalConfig(**common, key_bits=2, value_bits=4), "cpu")
    frozen = evaluate_kv_cache(
        _CacheModel(), batches,
        KVCacheEvalConfig(**common, frozen_recipe=[{
            "layer": 0, "key_bits": 2, "value_bits": 4,
        }]),
        "cpu",
    )

    assert frozen["frozen_recipe"]["validated_layers"] == 1
    assert frozen["key_bits"] == 2
    assert frozen["value_bits"] == 4
    assert frozen["mean_teacher_kl"] == uniform["mean_teacher_kl"]
    assert frozen["nll_delta"] == uniform["nll_delta"]
    assert frozen["top1_agreement"] == uniform["top1_agreement"]


def test_frozen_recipe_requires_exact_cache_layers():
    batch = {"input_ids": torch.arange(1, 25).reshape(1, -1) % 31}
    config = KVCacheEvalConfig(
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=2,
        skip=0,
        frozen_recipe=[{"layer": 1, "key_bits": 3, "value_bits": 3}],
    )
    try:
        evaluate_kv_cache(_CacheModel(), [batch], config, "cpu")
    except ValueError as error:
        assert "missing layers [0]" in str(error)
        assert "unknown layers [1]" in str(error)
    else:
        raise AssertionError("mismatched frozen recipe was accepted")


def test_frozen_recipe_and_dynamic_are_mutually_exclusive():
    try:
        KVCacheEvalConfig(
            dynamic={"candidate_bits": [4], "target_bpv": 4.25},
            frozen_recipe=[{"layer": 0, "key_bits": 4, "value_bits": 4}],
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("dynamic and frozen recipes were accepted together")


def test_dynamic_cache_config_rejects_invalid_candidate_subset():
    try:
        KVDynamicConfig(candidate_bits=(3, 4), key_candidate_bits=(8,))
    except ValueError as error:
        assert "subset" in str(error)
    else:
        raise AssertionError("invalid K/V candidate subset was accepted")


def test_cache_eval_seed_is_forwarded_to_quant_config():
    config = KVCacheEvalConfig(
        seed=17,
        codebook="spherical",
        codebook_dim=64,
        bias_correction="length",
    )
    assert config.quant_config().seed == 17
    assert config.quant_config(seed=23).seed == 23
    assert config.quant_config().codebook == "spherical"
    assert config.quant_config().codebook_dim == 64
    assert config.quant_config().bias_correction == "length"
