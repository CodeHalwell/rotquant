import pytest
import torch
from torch import nn

from rotquant.vocabulary import (
    PackedEmbedding,
    PackedOutputHead,
    VocabularyConfig,
    quantize_vocabulary,
    tensor_digest,
    tied_modules,
    vocabulary_prototype,
)


class TinyTied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 64)
        self.head = nn.Linear(64, 17, bias=False)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


@pytest.mark.parametrize("bits", [6, 8])
def test_chunked_prototype_restores_source_and_tie(tmp_path, bits):
    torch.manual_seed(5)
    model = TinyTied()
    before = model.embedding.weight.detach().clone()
    config = VocabularyConfig(bits=bits, block=32, group_size=32, chunk_rows=4)
    messages = []
    owner = quantize_vocabulary(model.embedding.weight, config, cache_dir=tmp_path,
                                progress=messages.append)
    assert torch.equal(model.embedding.weight, before)
    assert messages[-1]["rows"] == 17
    with (pytest.raises(RuntimeError, match="evaluation failed"),
          vocabulary_prototype(model, owner) as record):
        assert record["artifact_bytes"] is None
        assert not record["packed_artifact_verified"]
        assert model.embedding.weight is model.head.weight
        assert not torch.equal(model.embedding.weight, before)
        raise RuntimeError("evaluation failed")
    assert torch.equal(model.embedding.weight, before)
    resumed = []
    again = quantize_vocabulary(model.embedding.weight, config, cache_dir=tmp_path,
                                progress=resumed.append)
    assert all(row["resumed"] for row in resumed)
    assert again.fingerprint == owner.fingerprint


def test_chunk_invariance_and_packed_two_use_reference():
    torch.manual_seed(6)
    model = TinyTied()
    small = quantize_vocabulary(model.embedding.weight, VocabularyConfig(
        bits=6, block=32, group_size=32, chunk_rows=4))
    large = quantize_vocabulary(model.embedding.weight, VocabularyConfig(
        bits=6, block=32, group_size=32, chunk_rows=32))
    dense = torch.cat([x for _, x in small.reconstructed_chunks()])
    torch.testing.assert_close(dense, torch.cat([x for _, x in large.reconstructed_chunks()]),
                               rtol=0, atol=0)
    assert small.packed_bytes() == large.packed_bytes()
    embedding, head = PackedEmbedding(small), PackedOutputHead(small)
    assert embedding.owner is head.owner
    ids = torch.tensor([[0, 0, 1, 16, 9]])
    torch.testing.assert_close(embedding(ids), dense[ids], rtol=0, atol=0)
    x = torch.randn(2, 3, 64)
    torch.testing.assert_close(head(x), torch.nn.functional.linear(x, dense), rtol=2e-5, atol=2e-5)
    scales = [q.scales.clone() for q in small.chunks]
    small.bfloat16()
    assert all(torch.equal(q.scales, scale) for q, scale in zip(small.chunks, scales))
    assert embedding(ids).dtype == torch.bfloat16
    with pytest.raises(IndexError):
        embedding(torch.tensor([17]))


def test_untied_models_and_changed_source_fail_closed():
    model = TinyTied()
    owner = quantize_vocabulary(model.embedding.weight, VocabularyConfig(
        group_size=32, block=32))
    original = tensor_digest(model.embedding.weight)
    with torch.no_grad():
        model.embedding.weight.add_(1)
    with pytest.raises(ValueError, match="unchanged source"), vocabulary_prototype(model, owner):
        pass
    assert original != tensor_digest(model.embedding.weight)
    model.head.weight = nn.Parameter(model.head.weight.detach().clone())
    with pytest.raises(ValueError, match="share one"):
        tied_modules(model)


def test_corrupt_vocabulary_chunk_is_not_resumed(tmp_path):
    model = TinyTied()
    config = VocabularyConfig(group_size=32, block=32)
    quantize_vocabulary(model.embedding.weight, config, cache_dir=tmp_path)
    path = next(tmp_path.rglob("*.safetensors"))
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        quantize_vocabulary(model.embedding.weight, config, cache_dir=tmp_path)


def test_shared_vocabulary_checkpoint_reload_and_generation(tmp_path):
    import copy
    import json

    transformers = pytest.importorskip("transformers")
    from safetensors import safe_open

    from rotquant.checkpoint import load_packed_model, save_packed_checkpoint
    from rotquant.format import FormatValidationError, validate_checkpoint_manifest
    from rotquant.patch import PatchConfig, patch_model
    from rotquant.quantize import QuantConfig
    from rotquant.vocabulary import install_packed_vocabulary

    torch.manual_seed(34)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=65, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
        eos_token_id=64, pad_token_id=0,
    )).eval()
    owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                VocabularyConfig(bits=6, block=32, group_size=32, chunk_rows=16))
    patch_model(model, PatchConfig(quant=QuantConfig(bits=5, group_size=32), block=32))
    install_packed_vocabulary(model, owner)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = model(ids).logits
        generated = model.generate(ids, max_new_tokens=2, do_sample=False)
    report = save_packed_checkpoint(model, tmp_path / "packed", model_loader="causal_lm")
    assert report["format_version"] == 3
    manifest = json.loads((tmp_path / "packed/rotquant_config.json").read_text())
    for field, value in (("shape", [66, 64]), ("execution_dtype", "int32"),
                         ("source_digest", "missing"), ("head", "model.embed_tokens")):
        invalid = copy.deepcopy(manifest)
        invalid["tied_vocabulary"][field] = value
        with pytest.raises(FormatValidationError):
            validate_checkpoint_manifest(invalid)
    reloaded = load_packed_model(tmp_path / "packed", dtype="float32")
    assert reloaded.get_input_embeddings().owner is reloaded.get_output_embeddings().owner
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, expected, rtol=0, atol=0)
        assert torch.equal(reloaded.generate(ids, max_new_tokens=2, do_sample=False), generated)
    with safe_open(tmp_path / "packed" / "rotquant_model.safetensors", framework="pt") as handle:
        keys = handle.keys()
        assert not any(key.endswith("embed_tokens.weight") or key == "lm_head.weight"
                       for key in keys)
    with pytest.raises(AttributeError):
        reloaded.resize_token_embeddings(70)
    bf16 = load_packed_model(tmp_path / "packed", dtype="bfloat16")
    for expected_chunk, actual_chunk in zip(owner.chunks, bf16.get_input_embeddings().owner.chunks):
        assert actual_chunk.scales.dtype == torch.float16
        torch.testing.assert_close(actual_chunk.scales, expected_chunk.scales, rtol=0, atol=0)
    from rotquant.linear import QuantLinear
    original_layers = {name: module for name, module in model.named_modules()
                       if isinstance(module, QuantLinear)}
    for name, module in bf16.named_modules():
        if isinstance(module, QuantLinear):
            torch.testing.assert_close(module.qweight.dequantize(),
                                       original_layers[name].qweight.dequantize(), rtol=0, atol=0)


def test_qwen_hybrid_generation_with_prototype_and_packed_vocabulary():
    pytest.importorskip("transformers")
    from scripts.preflight_vocabulary_budget import qwen_hybrid_smoke

    assert qwen_hybrid_smoke()["prototype_and_packed_generation"]


def test_dynamic_allocator_rejects_unaccounted_packed_vocabulary():
    from rotquant.dynamic import select_dynamic_quantization
    from rotquant.patch import PatchConfig
    from rotquant.quantize import QuantConfig
    from rotquant.vocabulary import install_packed_vocabulary

    model = TinyTied()
    owner = quantize_vocabulary(model.embedding.weight, VocabularyConfig(group_size=32, block=32))
    install_packed_vocabulary(model, owner)
    with pytest.raises(ValueError, match="vocabulary-conditioned"):
        select_dynamic_quantization(model, PatchConfig(quant=QuantConfig()))
