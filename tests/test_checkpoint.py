"""Packed checkpoint serialization must reproduce the deployed model exactly."""
from __future__ import annotations

import json

import pytest
import torch

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
