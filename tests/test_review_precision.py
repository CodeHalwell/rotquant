from __future__ import annotations

import pytest
import torch
from torch import nn

from rotquant.checkpoint import load_packed_model, save_packed_checkpoint
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.rotate import ButterflyRotation, DenseOrthogonal, LearnedRotation
from tests.test_checkpoint import _tiny_packed_llama


@pytest.mark.parametrize("storage", [torch.float16, torch.bfloat16, torch.float32])
def test_butterfly_storage_precision_preserves_orthogonality(storage):
    rotation = ButterflyRotation(128, block=128, dtype=storage).eval()
    matrix = rotation.as_matrix(dtype=torch.float64)
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(128, dtype=torch.float64),
                               atol=1e-6, rtol=1e-6)
    inputs = torch.randn(3, 128).half()
    torch.testing.assert_close(rotation.inverse_activation(rotation.rotate_activation(inputs)),
                               inputs, atol=.002, rtol=.002)


@pytest.mark.parametrize("kind", [ButterflyRotation, DenseOrthogonal, LearnedRotation])
def test_model_dtype_moves_preserve_rotation_storage(kind):
    rotation = kind(16)
    before = {key: tensor.clone() for key, tensor in rotation.state_dict().items()}
    rotation.half().bfloat16().float()
    for key, tensor in rotation.state_dict().items():
        assert torch.equal(tensor, before[key])
        assert tensor.dtype == before[key].dtype


@pytest.mark.parametrize("execution", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotation", ["butterfly", "dense", "learned"])
def test_checkpoint_preserves_rotation_and_adapter_storage(tmp_path, execution, rotation):
    model = _tiny_packed_llama(rotation, dtype=execution)
    layer = next(module for module in model.modules() if isinstance(module, QuantLinear))
    layer.enable_lora(2, 2)
    with torch.no_grad():
        layer.lora_B.normal_()
    layer.commit_lora(torch.float32)
    ids = torch.tensor([[1, 5, 8, 13]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()
              if '.act_rotation.' in key or '.lora_' in key}
    save_packed_checkpoint(model, tmp_path / "model", model_loader="causal_lm")
    loaded = load_packed_model(tmp_path / "model", dtype=execution, fallback=True)
    for key, tensor in before.items():
        actual = loaded.state_dict()[key]
        assert actual.dtype == tensor.dtype
        assert torch.equal(actual, tensor)
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, rtol=0, atol=0)


def test_selection_scores_committed_storage(monkeypatch):
    from rotquant import train_rotation

    select = train_rotation.select_butterfly_checkpoint
    observed = []

    def check(rotation, reference, *args, **kwargs):
        observed.append((rotation.theta.dtype, reference.theta.dtype))
        return select(rotation, reference, *args, **kwargs)

    monkeypatch.setattr(train_rotation, "select_butterfly_checkpoint", check)
    model = nn.Sequential(nn.Linear(8, 8))
    patch_model(model, PatchConfig(quant=QuantConfig(bits=4, group_size=8), block=8,
                                  rotation="butterfly", rotation_storage_dtype="float16",
                                  train_rotation={"objective": "activation", "steps": 1, "max_tokens": 4}),
                activations={"0": torch.randn(8, 8)})
    assert observed == [(torch.float16, torch.float16)]
    assert model[0].act_rotation.theta.dtype == torch.float16


def test_adapter_dtype_moves_preserve_gradients():
    layer = QuantLinear.from_linear(nn.Linear(8, 8), QuantConfig(bits=4, group_size=8))
    layer.enable_lora(2, 2)
    layer.lora_A.grad = torch.randn_like(layer.lora_A)
    expected = layer.lora_A.grad.clone()
    layer.half()
    assert layer.lora_A.grad.dtype == torch.float32
    assert torch.equal(layer.lora_A.grad, expected)


def test_dense_checkpoint_preserves_fp64_storage(tmp_path):
    model = _tiny_packed_llama("dense", dtype=torch.float16)
    for module in model.modules():
        if isinstance(module, DenseOrthogonal):
            module.R = module.R.double()
    save_packed_checkpoint(model, tmp_path / "model", model_loader="causal_lm")
    loaded = load_packed_model(tmp_path / "model", dtype=torch.float16)
    for name, tensor in model.state_dict().items():
        if ".act_rotation.R" in name:
            actual = loaded.state_dict()[name]
            assert actual.dtype == torch.float64
            assert torch.equal(actual, tensor)
