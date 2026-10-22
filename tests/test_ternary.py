from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rotquant.pack import pack_indices, unpack_indices
from rotquant.ternary import (
    TernaryConfig,
    TernaryLinear,
    packed_ternary_state,
    restore_packed_ternary,
)


def _linear():
    torch.manual_seed(42)
    return nn.Linear(8, 5)


def test_hard_ternary_has_three_symbols_and_trainable_latents():
    module = TernaryLinear(_linear(), TernaryConfig(4, 4))
    normalized = module.quantized_weight() / module.stored_scales().repeat_interleave(4, 1)
    assert set(normalized.detach().flatten().tolist()) == {-1.0, 0.0, 1.0}
    module(torch.randn(3, 8)).square().sum().backward()
    assert module.latent_weight.grad.abs().sum() > 0
    assert module.log_scales.grad.abs().sum() > 0
    assert module.act_rotation.signs.dtype == torch.int8
    assert not list(module.act_rotation.parameters())


def test_latent_update_can_change_zero_support_and_sign():
    module = TernaryLinear(_linear(), TernaryConfig(4, 4, learn_scales=False))
    with torch.no_grad():
        module.log_scales.fill_(0)
        module.latent_weight.zero_()
    target = torch.ones_like(module.latent_weight)
    target[:, ::2] *= -1
    optimizer = torch.optim.SGD([module.latent_weight], lr=1)
    for _ in range(2):
        optimizer.zero_grad()
        (module.quantized_weight() - target).square().sum().backward()
        optimizer.step()
    assert torch.equal(module.hard_codes(), target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_forward_and_compact_restore_are_exact(dtype):
    source = _linear().to(dtype)
    module = TernaryLinear(source, TernaryConfig(3, 4))
    x = torch.randn(2, 3, 8, dtype=dtype)
    packed = module.to_quant_linear()
    restored = restore_packed_ternary(packed_ternary_state(packed), source)
    assert packed.qweight.packed.bits == 2
    assert packed.qweight.codebook.levels == 3
    assert packed.qweight.codebook.name == "ternary"
    assert packed.qweight.scales.dtype == torch.float16
    assert unpack_indices(packed.qweight.packed).max() <= 2
    assert torch.equal(module(x), packed(x))
    assert torch.equal(module(x), restored(x))


def test_existing_scale_recovery_preserves_ternary_codes_and_exact_commit():
    module = TernaryLinear(_linear(), TernaryConfig(4, 4)).to_quant_linear()
    codes = module.qweight.packed.data.clone()
    parameter = module.enable_scale_finetuning()
    with torch.no_grad():
        parameter.fill_(0.08)
    x = torch.randn(3, 8)
    before = module(x)
    before.square().mean().backward()
    assert parameter.grad.abs().sum() > 0
    module.commit_scale_finetuning()
    assert torch.equal(module(x), before)
    assert torch.equal(module.qweight.packed.data, codes)
    assert module.qweight.codebook.levels == 3


def test_reserved_code_rejected_on_compact_restore():
    source = _linear()
    state = packed_ternary_state(TernaryLinear(source, TernaryConfig(4, 4)).to_quant_linear())
    state["packed"] = pack_indices(torch.full_like(source.weight, 3, dtype=torch.int64), 2).data
    with pytest.raises(ValueError, match="reserved code 3"):
        restore_packed_ternary(state, source)


@pytest.mark.parametrize("change", ["nan_scale", "zero_scale", "bad_signs", "wrong_shape"])
def test_corrupt_scale_rotation_and_shape_rejected(change):
    source = _linear()
    state = packed_ternary_state(TernaryLinear(source, TernaryConfig(4, 4)).to_quant_linear())
    if change == "nan_scale":
        state["scales"][0, 0] = float("nan")
    elif change == "zero_scale":
        state["scales"][0, 0] = 0
    elif change == "bad_signs":
        state["signs"][0] = 0
    else:
        state["in_features"] = 9
    with pytest.raises(ValueError):
        restore_packed_ternary(state, source)


def test_zero_group_and_scale_extremes_are_finite_positive_fp16():
    source = nn.Linear(8, 4, bias=False)
    source.weight.data.zero_()
    module = TernaryLinear(source, TernaryConfig(4, 4))
    assert torch.equal(module.quantized_weight(), torch.zeros_like(source.weight))
    with torch.no_grad():
        module.log_scales[0].fill_(1000)
        module.log_scales[1].fill_(-1000)
    scales = module.to_quant_linear().qweight.scales
    assert scales.dtype == torch.float16
    assert torch.isfinite(scales).all() and (scales > 0).all()


def test_bad_rotation_width_fails_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="must divide"):
        TernaryLinear(nn.Linear(10, 4), TernaryConfig(4, 4))


def test_existing_checkpoint_reload_keeps_three_centroids(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from rotquant.checkpoint import load_packed_model, save_packed_checkpoint

    config = transformers.LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                                      num_hidden_layers=1, num_attention_heads=2,
                                      num_key_value_heads=2, max_position_embeddings=32)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    model.model.layers[0].mlp.up_proj = TernaryLinear(
        model.model.layers[0].mlp.up_proj, TernaryConfig(4, 4)).to_quant_linear()
    batch = torch.tensor([[2, 3, 5, 7]])
    expected = model(batch).logits.detach()
    save_packed_checkpoint(model, tmp_path / "model")
    restored = load_packed_model(tmp_path / "model", dtype=torch.float32).eval()
    assert restored.model.layers[0].mlp.up_proj.qweight.codebook.levels == 3
    assert torch.equal(expected, restored(batch).logits)
    # Deep-copying an exported layer also preserves its complete packed state.
    cloned = copy.deepcopy(restored.model.layers[0].mlp.up_proj)
    assert torch.equal(cloned.qweight.packed.data,
                       restored.model.layers[0].mlp.up_proj.qweight.packed.data)
