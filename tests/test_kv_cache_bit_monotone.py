"""The KV-cache simulator must not contain a bit-independent error floor.

A rotated 8-bit Gaussian cache reconstructs K/V to ~1e-4 relative error, so the
next-token KL between the full-precision cache and the simulated packed cache
must be tiny and must fall monotonically as the code width grows.  A floor that
does not move between 2 and 8 bits means the two decode paths differ by
something other than quantization (shared or corrupted state, positions, dtype),
which invalidates every recipe comparison built on the metric.  The hybrid
Qwen3.5 layout (linear-attention recurrent state plus full-attention K/V) is
used because that is the model family whose cache results are reported.
"""
from __future__ import annotations

import pytest
import torch

from rotquant.eval.kv_cache import KVCacheEvalConfig, evaluate_kv_cache


def _tiny_hybrid_qwen35():
    qwen35 = pytest.importorskip("transformers.models.qwen3_5")
    text_config = qwen35.Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        eos_token_id=255,
        pad_token_id=255,
    )
    vision_config = qwen35.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=64,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    config = qwen35.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )
    torch.manual_seed(0)
    model = qwen35.Qwen3_5ForConditionalGeneration(config).eval()
    with torch.no_grad():
        # Sharpen the random model so attention is not uniformly flat.
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and "embed" not in name:
                parameter.mul_(2.5)
    return model


@pytest.mark.parametrize("sink_tokens,recent_window", [(0, 0), (4, 16)])
def test_hybrid_kv_simulator_has_no_bit_independent_floor(sink_tokens, recent_window):
    model = _tiny_hybrid_qwen35()
    generator = torch.Generator().manual_seed(11)
    batches = [
        {"input_ids": torch.randint(1, 240, (1, 96), generator=generator)}
        for _ in range(2)
    ]
    kl_by_bits = {}
    for bits in (8, 4, 2):
        config = KVCacheEvalConfig(
            bits=bits,
            group_size=16,
            rotation_block=32,
            batches=2,
            prompt_len=64,
            continuation_len=16,
            skip=0,
            sink_tokens=sink_tokens,
            recent_window=recent_window,
            bootstrap_draws=20,
        )
        metrics = evaluate_kv_cache(model, batches, config, "cpu")
        assert metrics["kv_layers"] == 2
        # Recurrent/conv state of the linear-attention layers must be visible
        # to the simulator, otherwise it cannot have been cloned per cache.
        assert metrics["non_kv_state_bytes"] > 0
        kl_by_bits[bits] = metrics["mean_teacher_kl"]
    assert kl_by_bits[8] < 1e-4, kl_by_bits
    assert kl_by_bits[8] < kl_by_bits[4] < kl_by_bits[2], kl_by_bits
