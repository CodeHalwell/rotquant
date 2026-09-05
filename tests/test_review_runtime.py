"""Regression coverage for the September project review's runtime failures."""
from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

from rotquant import RotQuantConfig, inspect_model, optimize_model
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.rotate import _activation_cache_frames


class SharedSite(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.fail = False

    def forward(self, x):
        q = self.q_proj(x)
        if self.fail:
            raise RuntimeError("injected shared-site failure")
        return q + self.k_proj(x) + self.v_proj(x)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_shared_cache_is_invocation_scoped(inference, fail, monkeypatch):
    model = nn.Sequential(SharedSite()).eval()
    patch_model(model, PatchConfig(
        quant=QuantConfig(bits=4, group_size=8), block=8, share_rotations=True,
    ))
    rotation = model[0].q_proj.act_rotation
    assert rotation is model[0].k_proj.act_rotation
    model[0].fail = fail
    references = []
    original = rotation.rotate_activation

    def observed(x):
        result = original(x)
        references.extend([weakref.ref(x), weakref.ref(result)])
        return result

    monkeypatch.setattr(rotation, "rotate_activation", observed)
    with torch.inference_mode() if inference else torch.no_grad():
        x = torch.randn(2, 32, 16)
        if fail:
            with pytest.raises(RuntimeError, match="injected"):
                model(x)
        else:
            result = model(x)
            assert torch.isfinite(result).all()
            # Reusing ordinary tensors within a site still avoids two FWHTs.
            assert len(references) == (6 if inference else 2)
            del result
        assert not _activation_cache_frames.get()
        assert rotation._cached_activation_input is None
        assert rotation._cached_activation_output is None
        del x
    gc.collect()
    assert all(reference() is None for reference in references)


def test_public_api_llama_inference_and_generation():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        eos_token_id=63, pad_token_id=0,
    )).eval()
    optimized = optimize_model(model, RotQuantConfig(group_size=8, rotation_block=8))
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3, 4]])
        logits = optimized(ids).logits
        generated = optimized.generate(ids, max_new_tokens=2, do_sample=False)
    assert logits.shape == (1, 4, 64)
    assert generated.shape[1] >= 5
    assert not _activation_cache_frames.get()


@pytest.mark.parametrize("parent", ["attention", "encoder", "decoder"])
def test_parent_weight_readers_are_excluded(parent):
    if parent == "attention":
        unsafe = nn.MultiheadAttention(16, 2, batch_first=True)
    elif parent == "encoder":
        unsafe = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)
    else:
        unsafe = nn.TransformerDecoderLayer(16, 2, 32, batch_first=True)
    support = inspect_model(unsafe)
    assert not support.supported
    assert support.excluded_modules
    with pytest.raises(ValueError):
        optimize_model(unsafe)
    container = nn.ModuleDict({"unsafe": unsafe, "safe": nn.Linear(16, 16)}).eval()
    optimize_model(container, RotQuantConfig(group_size=8, rotation_block=8))
    with torch.inference_mode():
        x = container["safe"](torch.randn(1, 4, 16))
        if parent == "attention":
            out = unsafe(x, x, x)[0]
        elif parent == "decoder":
            out = unsafe(x, x)
        else:
            out = unsafe(x)
    assert torch.isfinite(out).all()
