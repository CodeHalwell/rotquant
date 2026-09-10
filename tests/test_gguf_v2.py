"""Whole-model export primitives must preserve the canonical stored representation."""
import numpy as np
import pytest
import torch

from rotquant.gguf_v2 import (
    NativeProjection,
    checkpoint_matrix,
    join_vocabulary_chunks,
    qwen35_permutations,
)
from rotquant.native_v3 import NativeV3Matrix, decode_native_v3_rows, encode_native_v3
from rotquant.quantize import QuantConfig, Quantizer


def matrix(bits=5, rows=7, cols=256, scale_bits=8):
    q = Quantizer(QuantConfig(bits=bits, group_size=128, scale="rms", scale_bits=scale_bits,
                             scale_quant_group_size=4)).quantize_weight(
        torch.randn(rows, cols, generator=torch.Generator().manual_seed(10)))
    return encode_native_v3(q)


def test_projection_permutations_do_not_reencode_scale8_blocks():
    original = matrix()
    rows = np.arange(6, -1, -1, dtype=np.int32)
    columns = np.arange(255, -1, -1, dtype=np.int32)
    projection = NativeProjection(original, np.ones(256, dtype=np.int8), rows, columns)
    tensors = projection.tensors("blk.0.attn_output")
    loaded = NativeV3Matrix.from_bytes(tensors["blk.0.attn_output.rqv3"].tobytes())
    assert loaded.to_bytes() == original.to_bytes()
    np.testing.assert_array_equal(tensors["blk.0.attn_output.rqcol"], columns)


@pytest.mark.parametrize("bits", [6, 8])
def test_joined_vocabulary_preserves_all_codes_and_fp16_scales(bits):
    chunks = [matrix(bits, rows=rows, scale_bits=16) for rows in (2, 3, 1)]
    joined = join_vocabulary_chunks(chunks)
    assert joined.words.tobytes() == b"".join(c.words.tobytes() for c in chunks)
    assert joined.scales.tobytes() == b"".join(c.scales.tobytes() for c in chunks)
    np.testing.assert_array_equal(decode_native_v3_rows(joined),
                                  np.concatenate([decode_native_v3_rows(c) for c in chunks]))
    tensors = NativeProjection(joined, np.ones(256, dtype=np.int8), vocabulary=True).tensors("token_embd")
    assert set(tensors) == {"token_embd.rqv3", "token_embd.rqsign"}


def test_qwen_gdn_permutation_is_a_runtime_map_not_a_scale_reorder():
    hp = {"linear_num_key_heads": 2, "linear_num_value_heads": 4,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128}
    rows, _ = qwen35_permutations("model.layers.0.linear_attn.in_proj_z", hp)
    _, cols = qwen35_permutations("model.layers.0.linear_attn.out_proj", hp)
    assert rows.shape == cols.shape == (512,)
    np.testing.assert_array_equal(rows[cols], np.arange(512))
    qkv, _ = qwen35_permutations("model.layers.0.linear_attn.in_proj_qkv", hp)
    np.testing.assert_array_equal(qkv[:512], np.arange(512))
    np.testing.assert_array_equal(qkv[512:], rows + 512)


def test_checkpoint_matrix_reads_original_storage_without_quantizer():
    original = matrix()
    values = {"w": original.words, "s": original.scales, "o": original.offsets,
              "t": original.steps, "c": original.codebook}
    class Handle:
        def get_tensor(self, key):
            return torch.from_numpy(values[key].copy())
    spec = {"packed": {"tensor": "w", "shape": [1792], "numel": 1792, "bits": 5},
            "scales": "s", "scale_offsets": "o", "scale_steps": "t", "scale_bits_main": 8,
            "scale_quant_group_size": 4, "codebook": {"kind": "scalar", "centroids": "c"},
            "out_features": 7, "in_features": 256, "group_size": 128}
    assert checkpoint_matrix(Handle(), spec).to_bytes() == original.to_bytes()
    with pytest.raises(ValueError, match="8/16-bit"):
        checkpoint_matrix(Handle(), {**spec, "scale_bits_main": 32})
    with pytest.raises(ValueError, match="permutation"):
        NativeProjection(original, np.ones(256, dtype=np.int8), rows=np.zeros(7, dtype=np.int32))
