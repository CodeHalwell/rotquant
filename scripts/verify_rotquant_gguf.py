#!/usr/bin/env python
"""Verify that a native GGUF exactly preserves a packed RotQuant checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotquant.checkpoint import (
    MANIFEST_NAME,
    _build_rotation,
    _read_quantized_weight,
)
from rotquant.gguf import native_tensor, native_tied_tensor
from scripts.export_rotquant_gguf import _qwen_permutations, _resolve_checkpoint

_QWEN_STEMS = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def _gguf_stem(module_name: str) -> str:
    parts = module_name.split(".")
    if parts[:3] != ["model", "language_model", "layers"] or len(parts) < 6:
        raise ValueError(f"unsupported Qwen module name: {module_name}")
    layer = int(parts[3])
    tail = ".".join(parts[4:])
    try:
        suffix = _QWEN_STEMS[tail]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen projection: {tail}") from exc
    return f"blk.{layer}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="local checkpoint path or Hub model ID")
    parser.add_argument("gguf", type=Path, help="native RotQuant GGUF")
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path(os.environ.get("LLAMA_CPP_DIR", "third_party/llama.cpp")),
        help="pinned llama.cpp checkout providing gguf-py",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tied-seed", type=int, default=17001)
    parser.add_argument(
        "--tied-scale", choices=("rms", "mse_search"), default="rms")
    parser.add_argument("--tied-chunk-rows", type=int, default=1024)
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint, args.revision)
    with (checkpoint / MANIFEST_NAME).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (checkpoint / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    hparams = {**config, **config.get("text_config", {})}

    sys.path.insert(0, str(args.llama_cpp_dir.resolve() / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(args.gguf.resolve())
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    specs = manifest["quantized_modules"]

    with (
        safe_open(
            checkpoint / manifest["packed_state"], framework="pt", device="cpu"
        ) as packed,
        safe_open(
            checkpoint / manifest["model_state"], framework="pt", device="cpu"
        ) as model_state,
    ):
        for index, spec in enumerate(specs, 1):
            qweight = _read_quantized_weight(packed, spec["qweight"])
            rotation = _build_rotation(spec["rotation"])
            prefix = spec["name"] + ".act_rotation."
            state = {
                key: model_state.get_tensor(prefix + key)
                for key in rotation.state_dict()
            }
            rotation.load_state_dict(state, strict=True)
            row_perm, column_perm = _qwen_permutations(spec["name"], hparams)
            expected = native_tensor(
                qweight,
                rotation,
                row_permutation=row_perm,
                column_permutation=column_perm,
            )

            stem = _gguf_stem(spec["name"])
            qname = stem + ".rqweight"
            rname = stem + ".rqrotation"
            if qname not in tensors or rname not in tensors:
                raise ValueError(f"missing native tensors for {spec['name']}")
            if not np.array_equal(tensors[qname].data, expected.qdata):
                raise ValueError(f"packed code/scale mismatch: {qname}")
            if not np.array_equal(tensors[rname].data.reshape(-1), expected.rotation):
                raise ValueError(f"rotation mismatch: {rname}")
            if index % 25 == 0 or index == len(specs):
                print(f"verified {index}/{len(specs)} projections", flush=True)

    native_qweights = {name for name in tensors if name.endswith(".rqweight")}
    tied_qname = "token_embd.rqweight"
    tied_rname = "token_embd.rqrotation"
    has_tied = tied_qname in tensors
    if has_tied:
        with safe_open(
            checkpoint / manifest["model_state"], framework="pt", device="cpu"
        ) as model_state:
            keys = list(model_state.keys())
            tied_key = (
                "lm_head.weight" if "lm_head.weight" in keys
                else next(name for name in keys
                          if name.endswith("embed_tokens.weight")))
            expected_tied = native_tied_tensor(
                model_state.get_tensor(tied_key), seed=args.tied_seed,
                scale=args.tied_scale, chunk_rows=args.tied_chunk_rows)
        if not np.array_equal(tensors[tied_qname].data, expected_tied.qdata):
            raise ValueError("packed tied embedding mismatch")
        if not np.array_equal(
                tensors[tied_rname].data.reshape(-1), expected_tied.rotation):
            raise ValueError("tied embedding rotation mismatch")
        print("verified tied token embedding/output projection", flush=True)

    expected_count = len(specs) + int(has_tied)
    if len(native_qweights) != expected_count:
        raise ValueError(
            f"GGUF has {len(native_qweights)} native weights; expected {expected_count}"
        )
    print("PASS: every packed code, fp16 scale, and rotation value is exact")


if __name__ == "__main__":
    main()
