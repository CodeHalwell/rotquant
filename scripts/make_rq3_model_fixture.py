"""Small random two-layer Qwen3.5 GGUF for whole-graph plumbing, never quality."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rotquant.gguf_v2 import NativeProjection, qwen35_permutations
from rotquant.native_v3 import encode_native_v3
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import RandomizedHadamard


def make_fixture(output, llama_dir, vocabulary_bits=6):
    if Path(output).exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(Path(llama_dir) / "gguf-py"))
    import gguf
    writer = gguf.GGUFWriter(str(output), "qwen35")
    parameters = {"block_count": 2, "context_length": 512, "embedding_length": 256,
        "feed_forward_length": 512, "attention.head_count": 2, "attention.head_count_kv": 2,
        "attention.key_length": 128, "attention.value_length": 128,
        "ssm.conv_kernel": 4, "ssm.state_size": 128, "ssm.group_count": 2,
        "ssm.time_step_rank": 4, "ssm.inner_size": 512, "full_attention_interval": 2,
        "rope.dimension_count": 64}
    for key, value in parameters.items():
        writer.add_uint32("qwen35." + key, value)
    writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32("qwen35.rope.freq_base", 10000000.0)
    writer.add_array("qwen35.rope.dimension_sections", [11, 11, 10, 0])
    writer.add_uint32("rotquant.version", 2)
    writer.add_uint32("rotquant.matrix_version", 3)
    writer.add_uint32("rotquant.rotation_block_size", 128)
    writer.add_bool("rotquant.tied_embedding", True)
    writer.add_string("rotquant.vocabulary_mode", "dense_equivalent_fp16")
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list([f"t{i}" for i in range(256)])
    writer.add_token_types([1] * 256)
    writer.add_token_merges(["t0 t1"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)
    rng = np.random.default_rng(3702)
    hp = {"linear_num_key_heads": 2, "linear_num_value_heads": 4,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128}

    def dense(name, shape, *, ones=False, fp32=False):
        values = np.ones(shape, dtype=np.float32) if ones else (rng.standard_normal(shape) * 0.02).astype(np.float32 if fp32 else np.float16)
        writer.add_tensor(name, values)

    def packed(name, rows, cols, source_name="", vocabulary=False):
        bits = vocabulary_bits if vocabulary else 5
        q = Quantizer(QuantConfig(bits=bits, group_size=128, scale="rms", scale_bits=16 if vocabulary else 8)).quantize_weight(
            torch.from_numpy((rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)))
        rotation = RandomizedHadamard(cols, block=128, seed=3703)
        rp, cp = qwen35_permutations(source_name, hp)
        projection = NativeProjection(encode_native_v3(q), rotation.signs.numpy().astype(np.int8), rp, cp, vocabulary)
        for key, value in projection.tensors(name).items():
            writer.add_tensor(key, value)

    packed("token_embd", 256, 256, vocabulary=True)
    dense("output_norm.weight", (256,), ones=True)
    for layer in range(2):
        stem = f"blk.{layer}."
        dense(stem + "attn_norm.weight", (256,), ones=True)
        dense(stem + "post_attention_norm.weight", (256,), ones=True)
        for name, rows, cols in (("ffn_gate", 512, 256), ("ffn_up", 512, 256), ("ffn_down", 256, 512)):
            packed(stem + name, rows, cols)
        if layer == 0:
            packed(stem + "attn_qkv", 1024, 256, "model.layers.0.linear_attn.in_proj_qkv")
            packed(stem + "attn_gate", 512, 256, "model.layers.0.linear_attn.in_proj_z")
            packed(stem + "ssm_out", 256, 512, "model.layers.0.linear_attn.out_proj")
            dense(stem + "ssm_conv1d.weight", (1024, 4), fp32=True)
            dense(stem + "ssm_alpha.weight", (4, 256))
            dense(stem + "ssm_beta.weight", (4, 256))
            dense(stem + "ssm_norm.weight", (128,), ones=True)
            writer.add_tensor(stem + "ssm_a", -np.ones(4, dtype=np.float32))
            writer.add_tensor(stem + "ssm_dt.bias", -np.ones(4, dtype=np.float32))
        else:
            for name, rows in (("attn_q", 512), ("attn_k", 256), ("attn_v", 256), ("attn_output", 256)):
                packed(stem + name, rows, 256)
            dense(stem + "attn_q_norm.weight", (128,), ones=True)
            dense(stem + "attn_k_norm.weight", (128,), ones=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--vocabulary-bits", type=int, choices=(6, 8), default=6)
    args = parser.parse_args()
    make_fixture(args.output, args.llama_dir, args.vocabulary_bits)
