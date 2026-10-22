"""Retained checkpoint export: stored bytes, shared vocabulary and fail-closed recipes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from rotquant.native_v3 import NativeV3Matrix, encode_native_v3
from rotquant.quantize import QuantConfig, Quantizer
from scripts.export_rotquant_gguf_v2 import (
    make_converter,
    recurrent_layer_map,
    stored_signs,
    validate_recipe,
)


def test_preserve_explicit_hybrid_topology_instead_of_default_interval():
    assert recurrent_layer_map({"num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"]}) == [True, False]
    assert recurrent_layer_map({"text_config": {"num_hidden_layers": 4}}) == [True, True, True, False]
    with pytest.raises(ValueError):
        recurrent_layer_map({"num_hidden_layers": 2, "layer_types": ["linear_attention", "unknown"]})


@pytest.fixture(params=[(5, 128), (4, 128), (4, 64), (4, 32)])
def retained(tmp_path, request):
    packed = {}

    def weight(name, rows, bits, scale_bits, group_size=128):
        q = Quantizer(QuantConfig(bits=bits, group_size=group_size, scale="rms", scale_bits=scale_bits,
                                 scale_quant_group_size=4)).quantize_weight(
            torch.randn(rows, 128, generator=torch.Generator().manual_seed(rows)))
        native = encode_native_v3(q)
        for suffix, array in {"codes": native.words, "scales": native.scales,
                              "centroids": native.codebook}.items():
            packed[name + "." + suffix] = torch.from_numpy(array.copy())
        spec = {"packed": {"tensor": name + ".codes", "shape": [rows * 128],
                           "numel": rows * 128, "bits": bits},
                "scales": name + ".scales", "scale_bits_main": scale_bits,
                "scale_quant_group_size": 4, "group_size": group_size,
                "in_features": 128, "out_features": rows,
                "codebook": {"kind": "scalar", "dimension": 1, "centroids": name + ".centroids"}}
        if scale_bits == 8:
            for key, array in (("scale_offsets", native.offsets), ("scale_steps", native.steps)):
                spec[key] = name + "." + key
                packed[spec[key]] = torch.from_numpy(array.copy())
        return spec, native

    bits, group_size = request.param
    backbone, original = weight("backbone", 8, bits, 8, group_size)
    chunks = [weight("vocabulary0", 3, 6, 16), weight("vocabulary1", 2, 6, 16)]
    module = {"name": "model.language_model.layers.0.mlp.down_proj", "has_bias": False,
              "in_features": 128, "out_features": 8, "rotation_id": "r0", "lora_rank": 0,
              "activation_bits": None, "rotation": {"kind": "fwht", "block": 128, "dim": 128},
              "qweight": backbone}
    vocabulary = {"embedding": "model.language_model.embed_tokens", "head": "lm_head",
                  "config": {"bits": 6, "block": 128, "group_size": 128}, "shape": [5, 128],
                  "execution_dtype": "float16", "projection_mode": "dense_equivalent",
                  "chunks": [chunk[0] for chunk in chunks]}
    manifest = {"torch_dtype": "float16", "quantized_modules": [module], "tied_vocabulary": vocabulary,
                "model_state": "model.safetensors", "packed_state": "packed.safetensors"}
    save_file(packed, tmp_path / manifest["packed_state"])
    sign_key = module["name"] + ".act_rotation.signs"
    state = {sign_key: torch.ones(128, dtype=torch.int8),
             "lm_head.owner.rotation.signs": -torch.ones(128, dtype=torch.int8),
             "model.language_model.norm.weight": torch.ones(128, dtype=torch.float16),
             "model.visual.patch_embed.weight": torch.ones(2, 2, dtype=torch.float16)}
    save_file(state, tmp_path / manifest["model_state"])
    return tmp_path, manifest, original, chunks


class Writer:
    def __init__(self):
        self.tensors = {}

    def add_tensor(self, name, value):
        assert name not in self.tensors
        self.tensors[name] = value.copy()


class Converter:
    model_arch = "qwen35"

    @classmethod
    def filter_tensors(cls, item):
        return None if ".visual." in item[0] else item

    def prepare_tensors(self):
        for name, loader in self.index_tensors().items():
            for target, value in self.modify_tensors(loader(), name, None):
                self.gguf_writer.add_tensor(target, value.numpy())

    def modify_tensors(self, value, name, bid):
        # The pinned upstream converter computes these before choosing F32 storage.
        if name.endswith(".A_log"):
            value = -value.exp()
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            value = value + 1
        yield name, value

    def map_tensor_name(self, name):
        return name.replace("model.layers.0.mlp.down_proj", "blk.0.ffn_down")


def test_export_preserves_payload_and_has_one_shared_vocabulary(retained):
    path, manifest, original, chunks = retained
    audit = {}
    cls = make_converter(Converter, SimpleNamespace(), path, manifest, audit)
    converter = cls()
    converter.hparams = {"tie_word_embeddings": True, "num_hidden_layers": 1}
    converter.gguf_writer = Writer()
    converter.prepare_tensors()
    tensors = converter.gguf_writer.tensors
    assert tensors["blk.0.ffn_down.rqv3"].tobytes() == original.to_bytes()
    vocabulary = NativeV3Matrix.from_bytes(tensors["token_embd.rqv3"].tobytes())
    assert vocabulary.words.tobytes() == b"".join(c[1].words.tobytes() for c in chunks)
    assert set(tensors) == {"blk.0.ffn_down.rqv3", "blk.0.ffn_down.rqsign",
                            "token_embd.rqv3", "token_embd.rqsign", "model.language_model.norm.weight"}
    assert audit["auxiliary_source_keys"] == ["model.visual.patch_embed.weight"]
    assert [m["vocabulary"] for m in audit["matrices"]] == [False, True]
    assert audit["matrices"][0]["group_size"] == original.layout.group_size
    assert np.all(tensors["token_embd.rqsign"] == -1)


def test_retained_export_transforms_dense_parameters_at_hf_precision(retained):
    path, manifest, _, _ = retained
    norm = torch.tensor([.1, -.1], dtype=torch.float16)
    decay = torch.tensor([.7, -.7], dtype=torch.float16)
    values = {
        "model.language_model.norm.weight": norm,
        "model.language_model.layers.0.linear_attn.A_log": decay,
        "model.language_model.layers.0.linear_attn.norm.weight": norm,
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": norm,
    }
    state = load_file(path / manifest["model_state"])
    state.update({key: value.clone() for key, value in values.items()})
    save_file(state, path / manifest["model_state"])
    cls = make_converter(Converter, SimpleNamespace(), path, manifest, {})
    converter = cls()
    converter.hparams = {"tie_word_embeddings": True, "num_hidden_layers": 1}
    converter.gguf_writer = Writer()
    converter.prepare_tensors()
    tensors = converter.gguf_writer.tensors
    expected_norm, expected_decay = 1 + norm.float(), -decay.float().exp()
    assert not torch.equal(expected_norm, (1 + norm).float())
    assert not torch.equal(expected_decay, (-decay.exp()).float())
    for name, expected in zip(list(values)[:2], (expected_norm, expected_decay)):
        assert tensors[name].dtype == np.float32
        np.testing.assert_array_equal(tensors[name], expected.numpy())
    for name in list(values)[2:]:
        assert tensors[name].dtype == np.float16
        np.testing.assert_array_equal(tensors[name], values[name].numpy())


@pytest.mark.parametrize("change", ["bias", "lora", "rotation", "activation", "dtype", "vocabulary"])
def test_export_rejects_unimplemented_semantics(retained, change):
    _, saved, _, _ = retained
    manifest = copy.deepcopy(saved)
    module = manifest["quantized_modules"][0]
    if change == "bias":
        module["has_bias"] = True
    elif change == "lora":
        module["lora_rank"] = 8
    elif change == "rotation":
        module["rotation"]["kind"] = "butterfly"
    elif change == "activation":
        module["activation_bits"] = 8
    elif change == "dtype":
        manifest["torch_dtype"] = "bfloat16"
    else:
        manifest["tied_vocabulary"]["projection_mode"] = "rotated"
    with pytest.raises(ValueError):
        validate_recipe(manifest)


@pytest.mark.parametrize("bits,group_size,scale_bits", [(3, 128, 8), (5, 32, 8),
                                                        (4, 16, 8), (4, 256, 8), (4, 32, 16)])
def test_export_rejects_backbone_formats_without_full_graph_support(retained, bits, group_size, scale_bits):
    _, saved, _, _ = retained
    manifest = copy.deepcopy(saved)
    weight = manifest["quantized_modules"][0]["qweight"]
    weight["packed"]["bits"] = bits
    weight["group_size"] = group_size
    weight["scale_bits_main"] = scale_bits
    with pytest.raises(ValueError, match="backbone recipe"):
        validate_recipe(manifest)


def test_shared_signs_are_loaded_not_regenerated():
    data = {"a": torch.ones(128, dtype=torch.int8), "b": -torch.ones(128, dtype=torch.int8)}
    handle = SimpleNamespace(keys=lambda: data.keys(), get_tensor=lambda key: data[key])
    with pytest.raises(ValueError, match="missing"):
        stored_signs(handle, ["not_there"], 128)
    with pytest.raises(ValueError, match="inconsistent"):
        stored_signs(handle, ["a", "b"], 128)
    np.testing.assert_array_equal(stored_signs(handle, ["not_there", "b"], 128), data["b"].numpy())
