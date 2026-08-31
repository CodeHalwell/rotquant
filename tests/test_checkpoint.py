"""Packed checkpoint serialization must reproduce the deployed model exactly."""
from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import rotquant.checkpoint as checkpoint_module
from rotquant.adapters import ADAPTERS, ModelAdapter
from rotquant.checkpoint import (
    MANIFEST_NAME,
    MODEL_STATE_NAME,
    PACKED_STATE_NAME,
    checkpoint_manifest,
    load_packed_model,
    save_packed_checkpoint,
)
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig


class _Conv1DProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight + self.bias


class _Conv1DCheckpointAdapter(ModelAdapter):
    def iter_quantizable_modules(self, model):
        for name, module in model.named_modules():
            if isinstance(module, _Conv1DProjection):
                yield name, module

    def to_linear(self, module):
        linear = nn.Linear(
            module.weight.shape[0],
            module.weight.shape[1],
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(module.weight.T)
            linear.bias.copy_(module.bias)
        return linear


class _CustomCheckpointModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.proj = _Conv1DProjection(8, 6)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


def _tiny_packed_llama(rotation="butterfly"):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    patch_model(
        model,
        PatchConfig(
            quant=QuantConfig(
                bits=4,
                codebook="gaussian",
                scale="mse_search",
                group_size=8,
            ),
            rotation=rotation,
            block=8,
            exclude=("lm_head",),
            # Match CUDA quality training: prove the cache affects neither the
            # artifact nor the reloaded packed result.
            fallback=True,
            seed=7,
        ),
    )
    return model


@pytest.mark.parametrize("rotation", ["fwht", "butterfly"])
def test_packed_checkpoint_round_trip_reproduces_logits(tmp_path, rotation):
    torch.manual_seed(13)
    model = _tiny_packed_llama(rotation)
    input_ids = torch.tensor([[1, 7, 9, 3, 2]])
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits

    export_dir = tmp_path / "packed"
    info = save_packed_checkpoint(
        model,
        export_dir,
        base_model="local/tiny-llama",
        model_loader="causal_lm",
    )
    assert info["quantized_modules"] > 0
    assert not info["fallback_cache_serialized"]
    assert (export_dir / MANIFEST_NAME).is_file()
    assert (export_dir / MODEL_STATE_NAME).is_file()
    assert (export_dir / PACKED_STATE_NAME).is_file()

    restored = load_packed_model(export_dir, dtype=torch.float32)
    with torch.no_grad():
        actual = restored(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    original_layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, QuantLinear)
    }
    restored_layers = {
        name: module for name, module in restored.named_modules()
        if isinstance(module, QuantLinear)
    }
    assert restored_layers.keys() == original_layers.keys()
    assert all(module._fp_cache is not None for module in original_layers.values())
    assert all(module._fp_cache is None for module in restored_layers.values())
    for name, original in original_layers.items():
        reloaded = restored_layers[name]
        assert torch.equal(reloaded.qweight.packed.data, original.qweight.packed.data)
        assert torch.equal(reloaded.qweight.scales, original.qweight.scales)
        if hasattr(original.act_rotation, "theta"):
            assert torch.equal(reloaded.act_rotation.theta, original.act_rotation.theta)
        else:
            assert torch.equal(reloaded.act_rotation.signs, original.act_rotation.signs)


def test_checkpoint_refuses_nonempty_directory_without_overwrite(tmp_path):
    model = _tiny_packed_llama()
    export_dir = tmp_path / "occupied"
    export_dir.mkdir()
    (export_dir / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError, match="not empty"):
        save_packed_checkpoint(model, export_dir)
    assert (export_dir / "keep.txt").read_text() == "user data"


def test_manifest_is_plain_json_and_records_loader(tmp_path):
    model = _tiny_packed_llama()
    export_dir = tmp_path / "packed"
    save_packed_checkpoint(model, export_dir, model_loader="causal_lm")
    manifest = checkpoint_manifest(export_dir)
    assert manifest["format"] == "rotquant-packed"
    assert manifest["format_version"] == 1
    assert manifest["packing"]["word_bits"] == 32
    assert manifest["packing"]["bit_order"] == "lsb_first"
    assert manifest["packing"]["optimized_profile_bits"] == list(range(1, 9))
    assert manifest["architecture"]["adapter"] == "dense-decoder"
    assert manifest["architecture"]["model_type"] == "llama"
    assert manifest["model_loader"] == "causal_lm"
    assert manifest["base_model_revision"] is None
    assert manifest["quantized_modules"]
    assert json.loads((export_dir / MANIFEST_NAME).read_text()) == manifest


def test_manifest_preserves_json_deployment_metadata(tmp_path):
    model = _tiny_packed_llama()
    export_dir = tmp_path / "packed"
    deployment = {
        "recipe": "uniform_w4__frozen_mixed_3.25",
        "kv_cache": {
            "codebook": "gaussian",
            "group_size": 64,
            "frozen_recipe": [
                {"layer": 3, "key_bits": 2, "value_bits": 2},
            ],
        },
    }
    save_packed_checkpoint(
        model,
        export_dir,
        base_model_revision="abc123",
        model_loader="causal_lm",
        deployment_metadata=deployment,
    )
    deployment["kv_cache"]["group_size"] = 128

    manifest = checkpoint_manifest(export_dir)
    assert manifest["deployment"]["recipe"] == (
        "uniform_w4__frozen_mixed_3.25"
    )
    assert manifest["base_model_revision"] == "abc123"
    assert manifest["deployment"]["kv_cache"]["group_size"] == 64


def test_checkpoint_rejects_non_json_deployment_metadata(tmp_path):
    model = _tiny_packed_llama()
    with pytest.raises(TypeError, match="JSON serializable"):
        save_packed_checkpoint(
            model,
            tmp_path / "packed",
            deployment_metadata={"tensor": torch.tensor(1)},
        )
    assert not (tmp_path / "packed").exists()


@pytest.mark.parametrize("family", ["encoder", "encoder-decoder"])
def test_checkpoint_auto_loader_uses_saved_adapter_family(tmp_path, family):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    if family == "encoder":
        config = transformers.BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
        model = transformers.BertModel(config).eval()
        inputs = {"input_ids": torch.tensor([[1, 7, 3, 2]])}
        include = None

        def output(result):
            return result.last_hidden_state
    else:
        config = transformers.T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
        model = transformers.T5ForConditionalGeneration(config).eval()
        inputs = {
            "input_ids": torch.tensor([[1, 7, 3, 2]]),
            "decoder_input_ids": torch.tensor([[0, 5, 4]]),
        }
        # T5's feed-forward wrapper directly introspects ``wo.weight``. That
        # module needs an architecture-specific wrapper before the whole family
        # can claim fused runtime support; attention projections exercise the
        # checkpoint loader contract without making that broader claim.
        include = ("SelfAttention.q",)

        def output(result):
            return result.logits

    patch_model(
        model,
        PatchConfig(
            quant=QuantConfig(
                bits=4,
                codebook="gaussian",
                scale="rms",
                group_size=8,
            ),
            rotation="none",
            include=include,
        ),
    )
    with torch.no_grad():
        expected = output(model(**inputs))

    export_dir = tmp_path / family
    save_packed_checkpoint(model, export_dir, model_loader="auto")
    manifest = checkpoint_manifest(export_dir)
    assert manifest["architecture"]["adapter"] == family

    restored = load_packed_model(export_dir, dtype=torch.float32)
    with torch.no_grad():
        actual = output(restored(**inputs))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_uses_saved_custom_adapter_hooks_before_model_build(
    tmp_path, monkeypatch
):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    adapter = _Conv1DCheckpointAdapter("checkpoint-conv1d")
    monkeypatch.setitem(ADAPTERS._adapters, adapter.name, adapter)
    model = _CustomCheckpointModel(transformers.BertConfig()).eval()
    inputs = torch.randn(3, 8)
    patch_model(
        model,
        PatchConfig(
            quant=QuantConfig(
                bits=4,
                codebook="gaussian",
                scale="rms",
                group_size=4,
            ),
            rotation="none",
            exclude=(),
            adapter=adapter.name,
        ),
    )
    with torch.no_grad():
        expected = model(inputs)

    export_dir = tmp_path / "custom-adapter"
    save_packed_checkpoint(model, export_dir, model_loader="auto")

    class CustomAutoModel:
        @classmethod
        def from_config(cls, config, **kwargs):
            del kwargs
            return _CustomCheckpointModel(config)

    def resolve_model_class(config, model_loader, saved_adapter):
        del config, model_loader
        assert saved_adapter is adapter
        return CustomAutoModel

    monkeypatch.setattr(
        checkpoint_module, "_resolve_model_class", resolve_model_class
    )
    restored = load_packed_model(export_dir, dtype=torch.float32)
    with torch.no_grad():
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
