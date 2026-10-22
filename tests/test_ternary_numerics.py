from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rotquant.ternary_numerics import qwen_fp32_training, stable_l2norm, stable_qwen_fallbacks


def _original(value, dim=-1, eps=1e-6):
    return value * torch.rsqrt((value * value).sum(dim=dim, keepdim=True) + eps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_stable_normalisation_forward_is_bit_exact(dtype):
    torch.manual_seed(41)
    value = torch.randn(2, 5, 8, dtype=dtype) * 0.001
    assert torch.equal(stable_l2norm(value), _original(value))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytical_backward_matches_full_precision_autograd(dtype):
    torch.manual_seed(12)
    original = (torch.randn(2, 3, 8, dtype=dtype) * 0.003).requires_grad_()
    candidate = original.detach().clone().requires_grad_()
    upstream = torch.randn_like(candidate)
    _original(original).backward(upstream)
    stable_l2norm(candidate).backward(upstream)
    torch.testing.assert_close(candidate.grad, original.grad, rtol=1e-6, atol=1e-5)


def test_fp16_rsqrt_overflow_is_avoided_without_changing_forward():
    original = torch.full((2, 4), 0.0001, dtype=torch.float16, requires_grad=True)
    candidate = original.detach().clone().requires_grad_()
    upstream = torch.tensor([[1, -1, 1, -1], [1, -1, 1, -1]], dtype=torch.float16)
    old_output = _original(original)
    output = stable_l2norm(candidate)
    assert torch.equal(old_output, output)
    old_output.backward(upstream)
    output.backward(upstream)
    assert not torch.isfinite(original.grad).all()
    assert torch.isfinite(candidate.grad).all()
    # This zero projection removes the second term, leaving inv_norm*g.
    expected = upstream * torch.rsqrt((candidate.detach().square()).sum(-1, keepdim=True) + 1e-6)
    assert torch.equal(candidate.grad, expected)


def _model():
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    model = nn.Sequential(nn.Identity())
    model[0].chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
    model[0].recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
    return model, qwen


def test_context_changes_only_model_local_functions_and_restores_copies():
    model, qwen = _model()
    other, _ = _model()
    original = model[0].chunk_gated_delta_rule
    original_norm = qwen.l2norm
    with stable_qwen_fallbacks(model) as info:
        replacement = model[0].chunk_gated_delta_rule
        assert replacement is not original
        assert replacement.__globals__["l2norm"] is stable_l2norm
        assert other[0].chunk_gated_delta_rule is original
        assert qwen.l2norm is original_norm
        assert len(info["patched_fallbacks"]) == 2
        with stable_qwen_fallbacks(model) as nested:
            assert nested["patched_fallbacks"] == []
            model[0] = copy.deepcopy(model[0])
        assert model[0].chunk_gated_delta_rule is replacement
    assert model[0].chunk_gated_delta_rule is original
    assert qwen.l2norm is original_norm


def test_context_restores_on_exception_and_leaves_fused_functions_untouched():
    model, _ = _model()
    original = model[0].chunk_gated_delta_rule
    def fused(*args):
        return None
    model[0].recurrent_gated_delta_rule = fused
    with pytest.raises(RuntimeError, match="interrupted"), stable_qwen_fallbacks(model) as info:
        assert len(info["patched_fallbacks"]) == 1
        assert model[0].recurrent_gated_delta_rule is fused
        raise RuntimeError("interrupted")
    assert model[0].chunk_gated_delta_rule is original


def test_fp32_training_forces_reference_kernels_and_restores_originals():
    from scripts.preflight_packed_validation import tiny_model
    model = tiny_model().float()
    gdn = model.model.language_model.layers[0].linear_attn
    attention = model.model.language_model.layers[1].self_attn
    original_norm = gdn.norm
    original_attention = attention.config._attn_implementation
    def fp16_only(*args, **kwargs):
        raise AssertionError("FP32 training reached an optional reduced-precision kernel")
    gdn.chunk_gated_delta_rule = fp16_only
    gdn.recurrent_gated_delta_rule = fp16_only
    gdn.causal_conv1d_fn = fp16_only
    gdn.causal_conv1d_update = fp16_only
    with qwen_fp32_training(model):
        assert gdn.norm is not original_norm
        assert gdn.norm.weight is original_norm.weight
        assert attention.config._attn_implementation == "eager"
        loss = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits.square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert gdn.chunk_gated_delta_rule is fp16_only
    assert gdn.causal_conv1d_fn is fp16_only
    assert gdn.norm is original_norm
    assert attention.config._attn_implementation == original_attention
