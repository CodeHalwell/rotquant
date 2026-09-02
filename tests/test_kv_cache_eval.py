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

    # The mandatory 8-bit endpoint pass runs first on the same calls, then
    # the candidate; both supply the absolute one-token position.
    assert model.decode_positions == [8, 8, 9, 9, 10, 10] * 2


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


def test_endpoint_check_runs_first_and_passes_on_a_faithful_simulator():
    ids = torch.arange(1, 25).reshape(1, -1) % 31
    config = KVCacheEvalConfig(
        bits=2,
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=4,
        skip=0,
    )
    metrics = evaluate_kv_cache(
        _CacheModel(), [{"input_ids": ids}], config, "cpu")

    endpoint = metrics["endpoint_check"]
    assert endpoint["bits"] == 8
    assert endpoint["passed"]
    assert endpoint["mean_teacher_kl"] <= config.endpoint_max_kl
    assert endpoint["prefill_kv_nmse"] < metrics["prefill_kv_nmse"]


def test_endpoint_check_fails_closed_on_a_bit_independent_floor(monkeypatch):
    import rotquant.eval.kv_cache as module

    faithful = module._evaluate_kv_cache

    def corrupted(model, batches, config, device, layer_configs=None):
        metrics = faithful(model, batches, config, device, layer_configs)
        metrics["mean_teacher_kl"] = 0.5  # the same floor at every bit width
        return metrics

    monkeypatch.setattr(module, "_evaluate_kv_cache", corrupted)
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
    with pytest.raises(RuntimeError, match="endpoint check failed"):
        evaluate_kv_cache(_CacheModel(), [{"input_ids": ids}], config, "cpu")


def test_endpoint_check_can_be_disabled_explicitly():
    ids = torch.arange(1, 25).reshape(1, -1) % 31
    config = KVCacheEvalConfig(
        bits=4,
        group_size=4,
        rotation_block=8,
        batches=1,
        prompt_len=8,
        continuation_len=4,
        skip=0,
        endpoint_check_bits=None,
    )
    metrics = evaluate_kv_cache(
        _CacheModel(), [{"input_ids": ids}], config, "cpu")
    assert "endpoint_check" not in metrics


def test_tiered_cache_requantizes_rows_that_age_out_of_the_recent_window():
    ids = torch.arange(1, 9).reshape(1, -1)
    keys, values = (state.half() for state in _states(ids))
    source_keys = keys.clone()
    simulated, _metrics = simulate_packed_kv_cache(
        _Cache(keys, values),
        KVQuantConfig(bits=2, group_size=4, rotation_block=8,
                      sink_tokens=1, recent_window=2),
    )
    layer = simulated.layers[0]

    def exact(position: int) -> bool:
        # fp16 tiers are stored in the rotated basis, so an "exact" row is
        # exact up to one fp16 rounding of the rotation round trip (~1e-3);
        # a 2-bit packed row differs by ~0.3.
        gap = (layer.keys[..., position, :].float()
               - source_keys[..., position, :].float()).abs().max().item()
        return gap < 5e-3

    # After prefill: the sink and the two newest rows are exact, the middle is packed.
    assert exact(0) and exact(6) and exact(7)
    assert not exact(3)

    first_keys, first_values = (state.half() for state in _states(torch.tensor([[9]])))
    simulated.update(first_keys, first_values, 0)
    assert layer.keys.shape[-2] == 9
    # Row 6 has left the two-row window and is packed exactly once; row 7 and
    # the new row 8 remain exact; the sink is never packed.
    assert not exact(6)
    assert exact(7)
    assert (layer.keys[..., 8, :].float() - first_keys[..., 0, :].float()).abs().max() < 5e-3
    assert exact(0)

    second_keys, second_values = (state.half() for state in _states(torch.tensor([[10]])))
    simulated.update(second_keys, second_values, 0)
    assert not exact(7)
    assert (layer.keys[..., 8, :].float() - first_keys[..., 0, :].float()).abs().max() < 5e-3
    assert (layer.keys[..., 9, :].float() - second_keys[..., 0, :].float()).abs().max() < 5e-3
    assert exact(0)


def test_decode_writes_without_tiers_are_packed_immediately():
    ids = torch.arange(1, 9).reshape(1, -1)
    keys, values = (state.half() for state in _states(ids))
    simulated, _metrics = simulate_packed_kv_cache(
        _Cache(keys, values),
        KVQuantConfig(bits=2, group_size=4, rotation_block=8),
    )
    new_keys, new_values = (state.half() for state in _states(torch.tensor([[9]])))
    simulated.update(new_keys, new_values, 0)
    gap = (simulated.layers[0].keys[..., 8, :].float()
           - new_keys[..., 0, :].float()).abs().max().item()
    assert gap > 5e-2


def _chunk_and_stepwise(prefill_ids, new_ids, quant):
    keys, values = (state.half() for state in _states(prefill_ids))
    chunked, _metrics = simulate_packed_kv_cache(
        _Cache(keys.clone(), values.clone()), quant)
    stepwise, _metrics = simulate_packed_kv_cache(
        _Cache(keys.clone(), values.clone()), quant)
    new_keys, new_values = (state.half() for state in _states(new_ids))
    chunked.update(new_keys, new_values, 0)
    for column in range(new_ids.shape[1]):
        stepwise.update(new_keys[..., column:column + 1, :],
                        new_values[..., column:column + 1, :], 0)
    return chunked, stepwise, new_keys


def _row_gap(cache, position: int, reference: torch.Tensor) -> float:
    return (cache.layers[0].keys[..., position, :].float()
            - reference.float()).abs().max().item()


def test_chunked_tiered_writes_pack_each_row_exactly_once():
    quant = KVQuantConfig(bits=2, group_size=4, rotation_block=8,
                          sink_tokens=1, recent_window=2)
    new_ids = torch.tensor([[9, 10, 11, 12, 13]])
    chunked, stepwise, new_keys = _chunk_and_stepwise(
        torch.arange(1, 9).reshape(1, -1), new_ids, quant)

    assert chunked.layers[0].keys.shape[-2] == 13
    # A five-row write whose first three rows leave the two-row window at
    # once must produce the same cache as the same rows written one at a time
    # (each row rotated, held in fp16 and packed exactly once).
    assert torch.equal(chunked.layers[0].keys, stepwise.layers[0].keys)
    assert torch.equal(chunked.layers[0].values, stepwise.layers[0].values)
    # Rows that left the window are packed; rows inside it are exact up to
    # the fp16 rotation round trip.
    assert _row_gap(chunked, 8, new_keys[..., 0, :]) > 5e-2
    assert _row_gap(chunked, 10, new_keys[..., 2, :]) > 5e-2
    assert _row_gap(chunked, 11, new_keys[..., 3, :]) < 5e-3
    assert _row_gap(chunked, 12, new_keys[..., 4, :]) < 5e-3


def test_chunked_tiered_writes_are_invariant_to_artifact_wide_quantizer_state():
    """Ageing must not group rows by how the decode happened to be chunked.

    8-bit affine scale blocks and a calibrated codebook are fitted across the
    matrix being packed, so quantizing a whole aged slice at once would make
    the cache depend on chunk size. The repository's cache configurations use
    ``scale_bits: 8``, which the 16-bit case above cannot detect.
    """
    base = torch.arange(1, 9).reshape(1, -1)
    for new_ids, overrides in (
        (torch.tensor([[9, 10, 11, 12, 13]]),
         {"scale_bits": 8, "scale_quant_group_size": 4}),
        (torch.tensor([[9, 10, 11, 12, 13, 14, 15, 16, 17]]),
         {"scale_bits": 8, "scale_quant_group_size": 4}),
        (torch.tensor([[9, 10, 11, 12, 13]]),
         {"scale_bits": 8, "scale_quant_group_size": 4,
          "codebook": "calibrated"}),
    ):
        quant = KVQuantConfig(bits=2, group_size=4, rotation_block=8,
                              sink_tokens=1, recent_window=2, **overrides)
        chunked, stepwise, _ = _chunk_and_stepwise(base, new_ids, quant)
        assert torch.equal(chunked.layers[0].keys, stepwise.layers[0].keys), overrides
        assert torch.equal(
            chunked.layers[0].values, stepwise.layers[0].values), overrides

def test_chunked_tiered_writes_decide_sinks_by_absolute_position():
    quant = KVQuantConfig(bits=2, group_size=4, rotation_block=8,
                          sink_tokens=3, recent_window=2)
    new_ids = torch.tensor([[3, 4, 5, 6]])
    # The two-row prefill is shorter than the sink prefix, so the write's
    # first row (absolute position 2) is a sink and stays exact; position 3
    # ages out of the window immediately; positions 4 and 5 stay exact.
    chunked, stepwise, new_keys = _chunk_and_stepwise(
        torch.arange(1, 3).reshape(1, -1), new_ids, quant)

    assert chunked.layers[0].keys.shape[-2] == 6
    assert torch.equal(chunked.layers[0].keys, stepwise.layers[0].keys)
    assert torch.equal(chunked.layers[0].values, stepwise.layers[0].values)
    assert _row_gap(chunked, 2, new_keys[..., 0, :]) < 5e-3
    assert _row_gap(chunked, 3, new_keys[..., 1, :]) > 5e-2
    assert _row_gap(chunked, 4, new_keys[..., 2, :]) < 5e-3
    assert _row_gap(chunked, 5, new_keys[..., 3, :]) < 5e-3


def test_cache_level_state_is_counted_in_the_byte_accounting():
    """State stored on the cache, not in a layer, is still an unpacked cost.

    ``_clone_cache`` clones it and the shared-storage check sees it, so the
    byte accounting must see it too; otherwise it vanishes from
    ``non_kv_state_bytes`` and the whole-cache ratio reads too favourably.
    """
    ids = torch.arange(1, 9).reshape(1, -1)
    keys, values = _states(ids)

    baseline = _Cache(keys.clone(), values.clone())
    _plain, plain_metrics = simulate_packed_kv_cache(
        baseline, KVQuantConfig(bits=4, group_size=4, rotation_block=8))

    source = _Cache(keys.clone(), values.clone())
    # Some Transformers caches hold conv/recurrent state on the cache object.
    source.global_state = torch.ones(2, 64)
    source.grouped_state = {0: torch.ones(3, 32)}
    extra = (source.global_state.numel() * source.global_state.element_size()
             + 3 * 32 * source.grouped_state[0].element_size())

    packed, metrics = simulate_packed_kv_cache(
        source, KVQuantConfig(bits=4, group_size=4, rotation_block=8))

    assert metrics["non_kv_state_bytes"] == plain_metrics["non_kv_state_bytes"] + extra
    assert (metrics["source_total_cache_bytes"]
            == plain_metrics["source_total_cache_bytes"] + extra)
    # Unpacked state is added to both sides, so the ratio can only fall.
    assert (metrics["total_cache_compression_ratio"]
            < plain_metrics["total_cache_compression_ratio"])

    # And the clone still isolates it.
    assert packed.global_state is not source.global_state
    assert packed.grouped_state[0] is not source.grouped_state[0]
