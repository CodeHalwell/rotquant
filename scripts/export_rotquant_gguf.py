#!/usr/bin/env python
"""Export a packed Qwen3.5 checkpoint to native RotQuant-GGUF v1.

The output intentionally requires the RotQuant llama.cpp patch.  It contains
raw ``*.rqweight`` and ``*.rqrotation`` tensors and never reconstructs dense
weights or runs a second quantizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotquant.checkpoint import (
    MANIFEST_NAME,
    _build_rotation,
    _read_quantized_weight,
)
from rotquant.gguf import (
    FORMAT_NAME,
    FORMAT_VERSION,
    GROUP_SIZE,
    ROTATION_BLOCK,
    native_tensor,
    native_tied_tensor,
)

PINNED_LLAMA_CPP_COMMIT = "17252c769a63c1cb650ce98ae309cf4de0da7778"


def _resolve_checkpoint(value: str, revision: str | None) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hub model IDs require huggingface-hub; install rotquant[eval]"
        ) from exc
    return Path(snapshot_download(value, revision=revision))


def _load_llama_modules(llama_cpp_dir: Path):
    if not (llama_cpp_dir / "gguf-py" / "gguf").is_dir():
        raise FileNotFoundError(
            f"not a llama.cpp checkout (missing gguf-py): {llama_cpp_dir}"
        )
    sys.path.insert(0, str(llama_cpp_dir))
    sys.path.insert(0, str(llama_cpp_dir / "gguf-py"))
    import gguf
    from conversion.qwen import Qwen3_5TextModel

    return gguf, Qwen3_5TextModel


def _v_head_permutation(
    num_k_heads: int, num_v_heads: int, head_dim: int
) -> np.ndarray:
    if num_v_heads % num_k_heads:
        raise ValueError("linear value heads must be divisible by key heads")
    per_key = num_v_heads // num_k_heads
    return (
        np.arange(num_v_heads * head_dim, dtype=np.int64)
        .reshape(num_k_heads, per_key, head_dim)
        .transpose(1, 0, 2)
        .reshape(-1)
    )


def _qwen_permutations(
    module_name: str, hparams: dict
) -> tuple[np.ndarray | None, np.ndarray | None]:
    num_k = int(hparams.get("linear_num_key_heads", 0))
    num_v = int(hparams.get("linear_num_value_heads", 0))
    if not num_k or not num_v or num_k == num_v or ".linear_attn." not in module_name:
        return None, None
    key_dim = int(hparams["linear_key_head_dim"])
    value_dim = int(hparams["linear_value_head_dim"])
    value_perm = _v_head_permutation(num_k, num_v, value_dim)

    if module_name.endswith(".linear_attn.in_proj_qkv"):
        q_dim = key_dim * num_k
        k_dim = key_dim * num_k
        row = np.concatenate(
            (
                np.arange(q_dim + k_dim, dtype=np.int64),
                value_perm + q_dim + k_dim,
            )
        )
        return row, None
    if module_name.endswith(".linear_attn.in_proj_z"):
        return value_perm, None
    if module_name.endswith(".linear_attn.out_proj"):
        return None, value_perm
    return None, None


def _rotation_from_state(module_spec: dict, model_state_path: Path):
    from safetensors import safe_open

    rotation = _build_rotation(module_spec["rotation"])
    prefix = module_spec["name"] + ".act_rotation."
    state = {}
    with safe_open(model_state_path, framework="pt", device="cpu") as handle:
        for key in rotation.state_dict():
            state[key] = handle.get_tensor(prefix + key)
    rotation.load_state_dict(state, strict=True)
    return rotation


def _make_converter(base_class, gguf, checkpoint: Path, manifest: dict,
                    *, quantize_tied_embedding: bool = False,
                    tied_seed: int = 17001, tied_scale: str = "rms",
                    tied_chunk_rows: int = 1024):
    from safetensors import safe_open

    model_state_path = checkpoint / manifest["model_state"]
    packed_state_path = checkpoint / manifest["packed_state"]
    quantized_names = {item["name"] for item in manifest["quantized_modules"]}
    base_model_arch = base_class.model_arch

    class RotQuantQwen35Model(base_class):
        # llama.cpp's converter validates that concrete subclasses declare this
        # attribute themselves rather than only inheriting it.
        model_arch = base_model_arch
        # Packed RotQuant artifacts contain the deployed 32-layer language
        # trunk only.  Qwen's config advertises an optional MTP layer even
        # when its tensors were not saved, so explicitly export no MTP head.
        no_mtp = True

        def index_tensors(self, remote_hf_model_id=None):
            del remote_hf_model_id
            hparams = {**self.hparams, **self.hparams.get("text_config", {})}
            block_key = next(
                key
                for key in (
                    "n_layers",
                    "num_hidden_layers",
                    "n_layer",
                    "num_layers",
                )
                if key in hparams
            )
            type(self)._original_block_count = hparams[block_key]
            type(self).opt_num_mtp_layers = 0
            tensors: dict[str, Callable[[], torch.Tensor]] = {}
            with safe_open(model_state_path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())

            # save_model() keeps only one side of tied parameters.  This
            # checkpoint retained lm_head.weight, while llama.cpp can derive
            # its missing output head from a canonical token embedding.  Map
            # the retained data to that name so it is stored only once.
            remap_tied_head = (
                self.hparams.get("text_config", {}).get(
                    "tie_word_embeddings",
                    self.hparams.get("tie_word_embeddings", False),
                )
                and "lm_head.weight" in keys
                and not any(name.endswith("embed_tokens.weight") for name in keys)
            )
            tied_source_key = None
            if quantize_tied_embedding:
                if not remap_tied_head:
                    tied_source_key = next(
                        (name for name in keys
                         if name.endswith("embed_tokens.weight")), None)
                else:
                    tied_source_key = "lm_head.weight"
                if tied_source_key is None:
                    raise ValueError(
                        "--quantize-tied-embedding requires a tied vocabulary weight")
            type(self)._rotquant_tied_source_key = tied_source_key

            for name in keys:
                if name == tied_source_key:
                    continue
                if any(
                    name.startswith(prefix + ".act_rotation.")
                    or name in {prefix + ".lora_A", prefix + ".lora_B"}
                    for prefix in quantized_names
                ):
                    continue

                def load_tensor(key=name):
                    with safe_open(
                        model_state_path, framework="pt", device="cpu"
                    ) as state_handle:
                        return state_handle.get_tensor(key)

                source_name = (
                    "model.language_model.embed_tokens.weight"
                    if remap_tied_head and name == "lm_head.weight"
                    else name
                )
                item = self.filter_tensors((source_name, load_tensor))
                if item is not None:
                    tensors[item[0]] = item[1]
            return tensors

        def prepare_tensors(self):
            super().prepare_tensors()
            with safe_open(packed_state_path, framework="pt", device="cpu") as handle:
                for index, module_spec in enumerate(manifest["quantized_modules"], 1):
                    qweight = _read_quantized_weight(handle, module_spec["qweight"])
                    rotation = _rotation_from_state(module_spec, model_state_path)
                    row_perm, column_perm = _qwen_permutations(
                        module_spec["name"], self.hparams
                    )
                    native = native_tensor(
                        qweight,
                        rotation,
                        row_permutation=row_perm,
                        column_permutation=column_perm,
                    )

                    source_name = (
                        module_spec["name"].replace("model.language_model.", "model.")
                        + ".weight"
                    )
                    mapped = self.map_tensor_name(source_name)
                    stem = mapped.removesuffix(".weight")
                    qname = stem + ".rqweight"
                    rname = stem + ".rqrotation"
                    self.gguf_writer.add_tensor(qname, native.qdata)
                    self.gguf_writer.add_tensor(rname, native.rotation)
                    logging.getLogger("hf-to-gguf").info(
                        "RotQuant %d/%d: %s [%d, %d] -> %s + %s",
                        index,
                        len(manifest["quantized_modules"]),
                        module_spec["name"],
                        native.out_features,
                        native.in_features,
                        qname,
                        rname,
                    )

            tied_source_key = getattr(
                type(self), "_rotquant_tied_source_key", None)
            if tied_source_key is not None:
                with safe_open(
                    model_state_path, framework="pt", device="cpu"
                ) as handle:
                    weight = handle.get_tensor(tied_source_key)
                logging.getLogger("hf-to-gguf").info(
                    "RotQuant tied embedding: %s [%d, %d]",
                    tied_source_key, weight.shape[0], weight.shape[1])
                native = native_tied_tensor(
                    weight, seed=tied_seed, scale=tied_scale,
                    chunk_rows=tied_chunk_rows)
                mapped = self.map_tensor_name(
                    "model.language_model.embed_tokens.weight")
                stem = mapped.removesuffix(".weight")
                self.gguf_writer.add_tensor(stem + ".rqweight", native.qdata)
                self.gguf_writer.add_tensor(stem + ".rqrotation", native.rotation)
                del weight

        def prepare_metadata(self, vocab_only: bool):
            super().prepare_metadata(vocab_only=vocab_only)
            self.gguf_writer.add_string("rotquant.format", FORMAT_NAME)
            self.gguf_writer.add_uint32("rotquant.version", FORMAT_VERSION)
            self.gguf_writer.add_uint32("rotquant.bits", 4)
            self.gguf_writer.add_uint32("rotquant.group_size", GROUP_SIZE)
            self.gguf_writer.add_uint32("rotquant.rotation_block_size", ROTATION_BLOCK)
            self.gguf_writer.add_string("rotquant.codebook", "lloyd_max_gaussian")
            self.gguf_writer.add_bool(
                "rotquant.tied_embedding", quantize_tied_embedding)
            if quantize_tied_embedding:
                self.gguf_writer.add_uint32(
                    "rotquant.tied_embedding_seed", tied_seed)
                self.gguf_writer.add_string(
                    "rotquant.tied_embedding_scale", tied_scale)
            self.gguf_writer.add_string(
                "rotquant.required_llama_cpp_commit", PINNED_LLAMA_CPP_COMMIT
            )

    return RotQuantQwen35Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="local checkpoint path or Hub model ID")
    parser.add_argument("output", type=Path, help="output .gguf path")
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path(os.environ.get("LLAMA_CPP_DIR", "third_party/llama.cpp")),
        help="checkout providing conversion and gguf-py modules",
    )
    parser.add_argument("--revision", default=None, help="optional Hub revision")
    parser.add_argument("--model-name", default="Qwen3.5-4B-RotQuant")
    parser.add_argument("--use-temp-file", action="store_true")
    parser.add_argument(
        "--quantize-tied-embedding", action="store_true",
        help="store the tied token embedding/output head as native RotQuant")
    parser.add_argument("--tied-seed", type=int, default=17001)
    parser.add_argument(
        "--tied-scale", choices=("rms", "mse_search"), default="rms")
    parser.add_argument("--tied-chunk-rows", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    checkpoint = _resolve_checkpoint(args.checkpoint, args.revision)
    with (checkpoint / MANIFEST_NAME).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "rotquant-packed":
        raise ValueError("checkpoint is not a RotQuant packed artifact")
    if manifest.get("model_loader") != "multimodal_lm":
        raise ValueError(
            "native GGUF v1 currently targets Qwen3.5 multimodal artifacts"
        )
    if any(int(item["lora_rank"]) for item in manifest["quantized_modules"]):
        raise ValueError("native GGUF v1 does not yet support retained LoRA tensors")

    gguf, base_class = _load_llama_modules(args.llama_cpp_dir.resolve())
    converter_class = _make_converter(
        base_class, gguf, checkpoint, manifest,
        quantize_tied_embedding=args.quantize_tied_embedding,
        tied_seed=args.tied_seed,
        tied_scale=args.tied_scale,
        tied_chunk_rows=args.tied_chunk_rows)
    converter = converter_class(
        checkpoint,
        gguf.LlamaFileType.MOSTLY_F16,
        args.output.resolve(),
        eager=True,
        model_name=args.model_name,
        use_temp_file=args.use_temp_file,
    )
    if converter.hf_arch not in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
    }:
        raise ValueError(f"unsupported model architecture: {converter.hf_arch}")
    converter.write()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
