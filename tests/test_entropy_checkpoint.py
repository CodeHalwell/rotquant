"""Opt-in entropy-coded code streams must reload as the identical packed model.

``save_packed_checkpoint(..., code_encoding="rans-rows-v1")`` stores primary
scalar code streams as row-wise rANS (format v4). These tests pin the contract:
bit-identical reload, smaller files with exact byte accounting, an unchanged
default artifact, and fail-closed loading of damaged or unknown encodings.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct

import numpy as np
import pytest
import torch

import rotquant.checkpoint as checkpoint_module
import rotquant.format as format_module
from rotquant.checkpoint import (
    MANIFEST_NAME,
    PACKED_STATE_NAME,
    checkpoint_manifest,
    load_packed_model,
    save_packed_checkpoint,
    verify_checkpoint,
)
from rotquant.format import (
    RANS_ROWS_V1,
    FormatValidationError,
    validate_checkpoint_manifest,
)
from rotquant.linear import QuantLinear
from rotquant.lossless import compress_codes, decompress_codes
from rotquant.pack import pack_indices
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from tests.test_checkpoint import _tiny_packed_llama

ENCODING = "rans-rows-v1"
INPUT_IDS = torch.tensor([[1, 7, 9, 3, 2]])


def _wide_packed_llama(quant_config, hidden=512, intermediate=1024):
    """A one-layer Llama whose rows are wide enough for rANS to pay off.

    Each coded row carries eight bytes of overhead (flushed state and row
    length), so a row needs a few hundred codes; real projections have 896 or
    more. Rotated Gaussian-initialised weights give Lloyd-Max code statistics
    close to those of real rotated weights (about 3.8 coded bits per W4 code).
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    torch.manual_seed(29)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    patch_model(
        model,
        PatchConfig(quant=quant_config, rotation="fwht", block=128, exclude=("lm_head",),
                    fallback=False, seed=7),
    )
    return model


@pytest.fixture(scope="module")
def w4_model():
    return _wide_packed_llama(
        QuantConfig(bits=4, codebook="gaussian", scale="rms", group_size=128))


@pytest.fixture(scope="module")
def encoded_dir(w4_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("encoded") / "checkpoint"
    result = save_packed_checkpoint(
        w4_model, path, model_loader="causal_lm", code_encoding=ENCODING)
    return path, result


@pytest.fixture
def encoded_copy(encoded_dir, tmp_path):
    path = tmp_path / "checkpoint"
    shutil.copytree(encoded_dir[0], path)
    return path


def _quant_layers(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, QuantLinear)}


def _logits(model):
    with torch.no_grad():
        return model(input_ids=INPUT_IDS, use_cache=False).logits


def _canonical_files(path):
    """Content digests of every artifact file.

    safetensors serialises its ``__metadata__`` map in an unspecified order, so
    raw digests of two identical saves can differ; compare the parsed header
    and the data section instead. The manifest drops the per-save generation id
    and the raw file hashes, which inherit the metadata-order noise.
    """
    digests = {}
    for file in sorted(path.iterdir()):
        raw = file.read_bytes()
        if file.suffix == ".safetensors":
            size = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + size])
            raw = json.dumps(header, sort_keys=True).encode() + raw[8 + size:]
        elif file.name == MANIFEST_NAME:
            manifest = json.loads(raw)
            manifest.pop("generation_id")
            manifest["files_sha256"] = sorted(manifest["files_sha256"])
            raw = json.dumps(manifest).encode()  # key order is part of the bytes
        digests[file.name] = hashlib.sha256(raw).hexdigest()
    return digests


def _stored_tensors(path):
    from safetensors import safe_open

    with safe_open(path / PACKED_STATE_NAME, framework="pt") as handle:
        keys = handle.keys()
        return {key: handle.get_tensor(key).clone() for key in keys}, handle.metadata()


def _rewrite(path, mutate):
    """Apply ``mutate(tensors, manifest)`` and re-record the integrity hashes.

    Re-hashing models a producer that wrote a bad stream consistently, so the
    loader's own stream checks, not the file hashes, must catch the damage.
    """
    from safetensors.torch import save_file

    tensors, metadata = _stored_tensors(path)
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    mutate(tensors, manifest)
    save_file(tensors, str(path / PACKED_STATE_NAME), metadata=metadata)
    manifest["files_sha256"][PACKED_STATE_NAME] = checkpoint_module._sha256(
        path / PACKED_STATE_NAME)
    (path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")


def _last_spec(manifest):
    return manifest["quantized_modules"][-1]["qweight"]["packed"]


# --------------------------------------------------------------------------- #
# Exact round trip and size
# --------------------------------------------------------------------------- #
def test_encoded_round_trip_is_bit_identical_to_fixed_rate(w4_model, encoded_dir, tmp_path):
    path, result = encoded_dir
    plain = tmp_path / "plain"
    save_packed_checkpoint(w4_model, plain, model_loader="causal_lm")
    expected = _logits(w4_model)
    original = _quant_layers(w4_model)
    for restored in (load_packed_model(plain, dtype=torch.float32),
                     load_packed_model(path, dtype=torch.float32)):
        layers = _quant_layers(restored)
        assert layers.keys() == original.keys()
        for name, module in original.items():
            packed, reloaded = module.qweight.packed, layers[name].qweight.packed
            assert reloaded.data.dtype == torch.int32
            assert torch.equal(reloaded.data, packed.data)
            assert (reloaded.shape, reloaded.bits, reloaded.numel) == (
                packed.shape, packed.bits, packed.numel)
            assert torch.equal(layers[name].qweight.scales, module.qweight.scales)
            # Resident storage is the fixed-rate stream: only the file shrinks.
            assert layers[name].packed_state_bytes() == module.packed_state_bytes()
        torch.testing.assert_close(_logits(restored), expected, rtol=0, atol=0)

    manifest = verify_checkpoint(path)
    assert result["format_version"] == manifest["format_version"] == 4
    assert manifest["code_encoding"] == RANS_ROWS_V1.to_manifest()
    tensors, metadata = _stored_tensors(path)
    assert metadata["rotquant_format"] == "4"
    for index, module_spec in enumerate(manifest["quantized_modules"]):
        spec = module_spec["qweight"]["packed"]
        assert spec["encoding"] == ENCODING and "tensor" not in spec
        assert (spec["rows"], spec["columns"]) == (
            module_spec["out_features"], module_spec["in_features"])
        assert f"layer_{index:05d}.packed" not in tensors
        frequencies = tensors[spec["frequencies"]]
        row_words = tensors[spec["row_words"]]
        words = tensors[spec["words"]]
        assert frequencies.dtype == row_words.dtype == torch.int32
        assert words.dtype == torch.int16
        assert frequencies.shape == (1 << spec["bits"],)
        assert int(frequencies.sum()) == 1 << RANS_ROWS_V1.probability_bits
        assert row_words.shape == (spec["rows"],)
        assert words.numel() == spec["word_count"] == int(row_words.sum())


def test_encoded_checkpoint_is_smaller_and_counts_stored_bytes(w4_model, encoded_dir, tmp_path):
    path, result = encoded_dir
    plain = save_packed_checkpoint(w4_model, tmp_path / "plain", model_loader="causal_lm")
    layers = list(_quant_layers(w4_model).values())
    fixed = sum(module.qweight.packed.data.numel() * 4 for module in layers)
    tensors, _ = _stored_tensors(path)
    stored = sum(
        tensors[spec[field]].numel() * tensors[spec[field]].element_size()
        for spec in (item["qweight"]["packed"]
                     for item in checkpoint_manifest(path)["quantized_modules"])
        for field in ("frequencies", "row_words", "words")
    )
    streams = result["code_streams"]
    assert streams["encoding"] == ENCODING
    assert streams["streams"] == streams["encoded_streams"] == len(layers)
    assert streams["not_encoded"] == {}
    assert streams["fixed_rate_bytes"] == fixed
    assert streams["stored_bytes"] == stored
    assert streams["saved_bytes"] == fixed - stored
    # About 3.4% of the W4 code bytes, after eight bytes of overhead per row.
    assert 0.02 < streams["saved_bytes"] / fixed < 0.08
    assert result["artifact_bytes"] == sum(
        file.stat().st_size for file in path.iterdir() if file.is_file())
    assert result["artifact_bytes"] < plain["artifact_bytes"]
    packed_file_saving = ((tmp_path / "plain" / PACKED_STATE_NAME).stat().st_size
                          - (path / PACKED_STATE_NAME).stat().st_size)
    # Three tensor entries per layer lengthen the header a little.
    assert 0.9 * streams["saved_bytes"] < packed_file_saving <= streams["saved_bytes"]


# --------------------------------------------------------------------------- #
# The default artifact does not change
# --------------------------------------------------------------------------- #
def test_default_save_is_unchanged_by_the_option(w4_model, tmp_path):
    default = save_packed_checkpoint(w4_model, tmp_path / "default", model_loader="causal_lm")
    explicit = save_packed_checkpoint(
        w4_model, tmp_path / "none", model_loader="causal_lm", code_encoding=None)
    assert _canonical_files(tmp_path / "default") == _canonical_files(tmp_path / "none")
    assert "code_streams" not in default and "code_streams" not in explicit
    assert default["format_version"] == 2
    text = (tmp_path / "default" / MANIFEST_NAME).read_text()
    assert "encoding" not in text and "rans" not in text
    tensors, metadata = _stored_tensors(tmp_path / "default")
    assert metadata["rotquant_format"] == "2"
    manifest = json.loads(text)
    for module_spec, module in zip(manifest["quantized_modules"],
                                   _quant_layers(w4_model).values(), strict=True):
        spec = module_spec["qweight"]["packed"]
        assert set(spec) == {"tensor", "shape", "bits", "numel"}
        assert torch.equal(tensors[spec["tensor"]], module.qweight.packed.data)
    assert not any("rans" in key for key in tensors)


def test_encoding_that_saves_nothing_writes_the_default_artifact(tmp_path):
    # Sixteen-code rows cannot amortise the per-row state, so every stream
    # stays fixed-rate and the artifact keeps the version older readers know.
    model = _tiny_packed_llama("fwht")
    default = save_packed_checkpoint(model, tmp_path / "default", model_loader="causal_lm")
    tried = save_packed_checkpoint(
        model, tmp_path / "tried", model_loader="causal_lm", code_encoding=ENCODING)
    streams = tried["code_streams"]
    assert streams["encoded_streams"] == 0
    assert streams["not_encoded"] == {"no_saving": streams["streams"]}
    assert streams["stored_bytes"] == streams["fixed_rate_bytes"]
    assert tried["format_version"] == default["format_version"] == 2
    assert _canonical_files(tmp_path / "tried") == _canonical_files(tmp_path / "default")


# --------------------------------------------------------------------------- #
# Damaged streams never load
# --------------------------------------------------------------------------- #
def _flip_word(tensors, manifest):
    words = tensors[_last_spec(manifest)["words"]]
    words[words.numel() // 2] ^= 0x2A5


def _drop_final_word(tensors, manifest):
    key = _last_spec(manifest)["words"]
    tensors[key] = tensors[key][:-1].clone()


def _truncate_last_row(tensors, manifest):
    spec = _last_spec(manifest)
    tensors[spec["words"]] = tensors[spec["words"]][:-3].clone()
    tensors[spec["row_words"]][-1] -= 3
    spec["word_count"] -= 3


def _pad_last_row(tensors, manifest):
    spec = _last_spec(manifest)
    words = tensors[spec["words"]]
    tensors[spec["words"]] = torch.cat([words, words[-1:]])
    tensors[spec["row_words"]][-1] += 1
    spec["word_count"] += 1


def _drop_last_row(tensors, manifest):
    spec = _last_spec(manifest)
    row_words = tensors[spec["row_words"]]
    tensors[spec["words"]] = tensors[spec["words"]][:-int(row_words[-1])].clone()
    spec["word_count"] -= int(row_words[-1])
    tensors[spec["row_words"]] = row_words[:-1].clone()


def _shift_row_boundary(tensors, manifest):
    row_words = tensors[_last_spec(manifest)["row_words"]]
    row_words[0] += 1
    row_words[1] -= 1


def _swap_frequencies(tensors, manifest):
    frequencies = tensors[_last_spec(manifest)["frequencies"]]
    order = torch.argsort(frequencies)
    low, high = int(order[0]), int(order[-1])
    assert frequencies[low] != frequencies[high]
    frequencies[[low, high]] = frequencies[[high, low]]


def _unnormalised_frequencies(tensors, manifest):
    tensors[_last_spec(manifest)["frequencies"]][0] += 1


def _zero_initial_state(tensors, manifest):
    tensors[_last_spec(manifest)["words"]][0] = 0


def _wrong_digest(tensors, manifest):
    spec = _last_spec(manifest)
    spec["decoded_sha256"] = hashlib.sha256(spec["decoded_sha256"].encode()).hexdigest()


def _widened_words(tensors, manifest):
    key = _last_spec(manifest)["words"]
    tensors[key] = tensors[key].to(torch.int32)


@pytest.mark.parametrize("damage", [
    _flip_word, _drop_final_word, _truncate_last_row, _pad_last_row, _drop_last_row,
    _shift_row_boundary, _swap_frequencies, _unnormalised_frequencies,
    _zero_initial_state, _wrong_digest, _widened_words,
], ids=lambda damage: damage.__name__.strip("_"))
def test_damaged_stream_fails_closed(encoded_copy, damage):
    _rewrite(encoded_copy, damage)
    verify_checkpoint(encoded_copy)  # consistent hashes: only stream checks remain
    with pytest.raises(ValueError, match="corrupt entropy-coded stream"):
        load_packed_model(encoded_copy, dtype=torch.float32)


def test_unrecorded_corruption_is_caught_by_the_file_hashes(encoded_copy):
    packed = encoded_copy / PACKED_STATE_NAME
    raw = bytearray(packed.read_bytes())
    raw[-7] ^= 0x10
    packed.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_packed_model(encoded_copy, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Unknown or misplaced encodings fail closed before any tensor is read
# --------------------------------------------------------------------------- #
def _set(path, value):
    def mutate(manifest):
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


_PRIMARY = ("quantized_modules", -1, "qweight", "packed")


@pytest.mark.parametrize(("mutation", "message"), [
    (_set(("code_encoding", "name"), "rans-rows-v9"), "unsupported code encoding"),
    (_set((*_PRIMARY, "encoding"), "zstd-v1"), "unsupported code encoding"),
    (_set(("code_encoding", "probability_bits"), 15), "does not match the rans-rows-v1"),
    (lambda manifest: manifest.pop("code_encoding"), "code_encoding: must be an object"),
    (_set(("format_version",), 2), "requires format v4"),
    (_set(("format_version",), 3), "requires format v4"),
    (_set((*_PRIMARY, "rows"), 256), "rows \\* columns must equal numel"),
    (_set((*_PRIMARY, "bits"), 15), "codes at most 14 bits"),
    (_set((*_PRIMARY, "decoded_sha256"), "0" * 63), "SHA-256 hex digest"),
    (_set((*_PRIMARY, "tensor"), "layer_00006.packed"), "must be absent"),
    (_set((*_PRIMARY, "words"), "layer_00006.packed.rans_row_words"), "distinct keys"),
])
def test_manifest_with_unknown_or_invalid_encoding_fails_closed(
    encoded_copy, mutation, message
):
    manifest_path = encoded_copy / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FormatValidationError, match=message):
        checkpoint_manifest(encoded_copy)
    with pytest.raises(FormatValidationError, match=message):
        load_packed_model(encoded_copy, dtype=torch.float32)


def test_only_primary_scalar_row_streams_may_be_encoded(encoded_dir):
    manifest = checkpoint_manifest(encoded_dir[0])
    module = manifest["quantized_modules"][-1]
    encoded = dict(module["qweight"]["packed"])
    for field, message in (("residual_packed", "only primary"), ("sketch", "only primary")):
        invalid = json.loads(json.dumps(manifest))
        invalid["quantized_modules"][-1]["qweight"][field] = encoded
        with pytest.raises(FormatValidationError, match=message):
            validate_checkpoint_manifest(invalid)
    vector = json.loads(json.dumps(manifest))
    vector["quantized_modules"][-1]["qweight"]["codebook"]["kind"] = "vector"
    with pytest.raises(FormatValidationError, match="only scalar-codebook"):
        validate_checkpoint_manifest(vector)
    transposed = json.loads(json.dumps(manifest))
    packed = transposed["quantized_modules"][-1]["qweight"]["packed"]
    packed["rows"], packed["columns"] = packed["columns"], packed["rows"]
    with pytest.raises(FormatValidationError, match="one row per output feature"):
        validate_checkpoint_manifest(transposed)


def test_reader_that_predates_v4_rejects_the_artifact(encoded_dir, monkeypatch):
    # An older loader's only gate is the version list; it must stop there
    # rather than look for fixed-rate words that are not in the file.
    monkeypatch.setattr(format_module, "SUPPORTED_FORMAT_VERSIONS", (1, 2, 3))
    with pytest.raises(FormatValidationError, match="unsupported version 4"):
        load_packed_model(encoded_dir[0], dtype=torch.float32)


def test_unknown_requested_encoding_writes_nothing(w4_model, tmp_path):
    with pytest.raises(ValueError, match="unsupported code_encoding"):
        save_packed_checkpoint(w4_model, tmp_path / "out", code_encoding="zstd")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".*staging*"))


def test_coder_that_drifted_from_the_contract_is_refused(
    w4_model, encoded_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(checkpoint_module, "RANS_PROBABILITY_BITS", 15)
    with pytest.raises(RuntimeError, match="new encoding name"):
        save_packed_checkpoint(w4_model, tmp_path / "out", code_encoding=ENCODING)
    with pytest.raises(RuntimeError, match="new encoding name"):
        load_packed_model(encoded_dir[0], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Mixed models: only primary scalar streams are coded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("error_comp", ["residual", "turboquant"])
def test_residual_and_sketch_streams_stay_fixed_rate(tmp_path, error_comp):
    quant = QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=128,
                        error_comp=error_comp, residual_bits=2, sketch_k=16)
    # 256-wide rows do not pay at W3, 1024-wide ones do: a mixed artifact.
    model = _wide_packed_llama(quant, hidden=256, intermediate=1024)
    expected = _logits(model)
    result = save_packed_checkpoint(model, tmp_path / "mixed", model_loader="causal_lm",
                                    code_encoding=ENCODING)
    manifest = checkpoint_manifest(tmp_path / "mixed")
    assert result["format_version"] == 4
    assert result["code_streams"]["encoded_streams"] == 1
    assert result["code_streams"]["not_encoded"] == {"no_saving": 6}
    secondary = "residual_packed" if error_comp == "residual" else "sketch"
    for module_spec in manifest["quantized_modules"]:
        qweight = module_spec["qweight"]
        assert ("encoding" in qweight["packed"]) == module_spec["name"].endswith("down_proj")
        assert set(qweight[secondary]) == {"tensor", "shape", "bits", "numel"}
    restored = load_packed_model(tmp_path / "mixed", dtype=torch.float32)
    layers = _quant_layers(restored)
    for name, module in _quant_layers(model).items():
        assert torch.equal(layers[name].qweight.packed.data, module.qweight.packed.data)
        original_secondary = getattr(module.qweight, secondary)
        assert torch.equal(getattr(layers[name].qweight, secondary).data,
                           original_secondary.data)
    torch.testing.assert_close(_logits(restored), expected, rtol=0, atol=0)


def test_vector_codebooks_stay_fixed_rate(tmp_path):
    model = _tiny_packed_llama("fwht", QuantConfig(
        bits=2, codebook="vector", vector_dim=2, vector_samples=512, vector_iters=5,
        scale="rms", group_size=8))
    result = save_packed_checkpoint(model, tmp_path / "vector", code_encoding=ENCODING)
    streams = result["code_streams"]
    assert streams["not_encoded"] == {"vector_codebook": streams["streams"]}
    assert result["format_version"] == 2


def test_shared_vocabulary_stays_fixed_rate_beside_a_coded_backbone(tmp_path):
    transformers = pytest.importorskip("transformers")
    from rotquant.vocabulary import (
        VocabularyConfig,
        install_packed_vocabulary,
        quantize_vocabulary,
    )

    torch.manual_seed(34)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=65, hidden_size=512, intermediate_size=1024, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True,
        eos_token_id=64, pad_token_id=0,
    )).eval()
    owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                VocabularyConfig(bits=6, block=128, group_size=128,
                                                 chunk_rows=32))
    patch_model(model, PatchConfig(quant=QuantConfig(bits=4, scale="rms", group_size=128),
                                   block=128))
    install_packed_vocabulary(model, owner)
    expected = _logits(model)
    result = save_packed_checkpoint(model, tmp_path / "vocab", model_loader="causal_lm",
                                    code_encoding=ENCODING)
    manifest = checkpoint_manifest(tmp_path / "vocab")
    assert result["format_version"] == 4 and "tied_vocabulary" in manifest
    assert result["code_streams"]["encoded_streams"] > 0
    assert all("encoding" not in chunk["packed"]
               for chunk in manifest["tied_vocabulary"]["chunks"])
    reloaded = load_packed_model(tmp_path / "vocab", dtype="float32")
    torch.testing.assert_close(_logits(reloaded), expected, rtol=0, atol=0)
    invalid = json.loads(json.dumps(manifest))
    invalid["tied_vocabulary"]["chunks"][0]["packed"] = _last_spec(manifest)
    with pytest.raises(FormatValidationError, match="only primary"):
        validate_checkpoint_manifest(invalid)


# --------------------------------------------------------------------------- #
# The chunked decoder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", range(1, 15))
def test_chunked_decode_repacks_exactly_like_pack_indices(bits):
    rows, columns = 13, 37  # rows do not end on word boundaries for most widths
    generator = np.random.default_rng(bits)
    symbols = torch.from_numpy(generator.integers(0, 1 << bits, size=(rows, columns)))
    packed = pack_indices(symbols.reshape(-1), bits)
    coded = compress_codes(packed, rows=rows)
    for chunk in (1, 100, 1 << 23):
        restored = decompress_codes(coded, bits, packed.shape, max_chunk_symbols=chunk)
        assert torch.equal(restored.data, packed.data)
        assert restored.shape == packed.shape
    with pytest.raises(ValueError, match="code width"):
        decompress_codes(coded, bits + 1, packed.shape)
