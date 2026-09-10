"""Losslessly package a retained W5/scale8 + W6/W8 vocabulary checkpoint.

This is an experimental GGUF-v2 exporter, not a claim of runtime readiness.
It never re-quantizes a weight or regenerates a rotation. Non-text tensors are
retained in a counted sidecar; the GGUF is a text-inference model only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rotquant.checkpoint import MANIFEST_NAME, resolve_checkpoint_directory, verify_checkpoint
from rotquant.gguf_v2 import (
    GGUF_VERSION,
    NativeProjection,
    checkpoint_matrix,
    join_vocabulary_chunks,
    qwen35_permutations,
)
from scripts.native_hashing import digest as file_digest

LLAMA_REVISION = "17252c769a63c1cb650ce98ae309cf4de0da7778"
LOG = logging.getLogger("rotquant.export.gguf-v2")


def validate_recipe(manifest: dict) -> None:
    if manifest.get("torch_dtype") != "float16":
        raise ValueError("GGUF v2 currently requires FP16 execution boundaries")
    modules = manifest["quantized_modules"]
    if not modules or len({m["name"] for m in modules}) != len(modules):
        raise ValueError("missing or duplicated quantized modules")
    for module in modules:
        rotation = module["rotation"]
        qweight = module["qweight"]
        if (module.get("has_bias") or module.get("lora_rank", 0)
                or module.get("activation_bits") is not None
                or rotation != {"kind": "fwht", "block": 128, "dim": module["in_features"]}
                or qweight["in_features"] != module["in_features"]
                or qweight["out_features"] != module["out_features"]
                or qweight["packed"]["bits"] != 5 or qweight.get("scale_bits_main", 16) != 8):
            raise ValueError(f"unsupported retained backbone recipe: {module['name']}")
    vocabulary = manifest.get("tied_vocabulary") or {}
    if (vocabulary.get("execution_dtype") != "float16"
            or vocabulary.get("projection_mode") != "dense_equivalent"
            or vocabulary.get("config", {}).get("bits") not in (6, 8)
            or vocabulary.get("config", {}).get("block") != 128
            or vocabulary.get("config", {}).get("group_size") != 128):
        raise ValueError("GGUF v2 requires the saved dense-equivalent W6/W8 vocabulary")


def stored_signs(handle, candidates: list[str], dimension: int) -> np.ndarray:
    keys = set(handle.keys())
    present = [name for name in candidates if name in keys]
    if not present:
        raise ValueError(f"saved rotation signs missing: {candidates}")
    values = [handle.get_tensor(key).numpy() for key in present]
    signs = values[0]
    if (signs.dtype != np.int8 or signs.shape != (dimension,)
            or not np.all((signs == 1) | (signs == -1))
            or any(not np.array_equal(signs, other) for other in values[1:])):
        raise ValueError("invalid or inconsistent shared saved rotation signs")
    return signs.copy()


def recurrent_layer_map(hparams):
    """Preserve explicit hybrid topology; the pinned converter assumes interval 4."""
    hp = {**hparams, **hparams.get("text_config", {})}
    layers = hp.get("layer_types")
    count = hp["num_hidden_layers"]
    if layers is None:
        interval = hp.get("full_attention_interval", 4)
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("invalid full-attention interval")
        return [(i + 1) % interval != 0 for i in range(count)]
    if len(layers) != count or any(kind not in ("linear_attention", "full_attention") for kind in layers):
        raise ValueError("unsupported Qwen hybrid layer topology")
    return [kind == "linear_attention" for kind in layers]


def make_converter(base_class, gguf, checkpoint: Path, manifest: dict, audit: dict):
    """Reuse the pinned Qwen converter for dense tensors, metadata and tokenizer."""
    validate_recipe(manifest)
    model_path, packed_path = (checkpoint / manifest[k] for k in ("model_state", "packed_state"))
    modules = manifest["quantized_modules"]
    vocabulary = manifest["tied_vocabulary"]
    sign_groups = {}
    for module in modules:
        key = module.get("rotation_id") or module["name"]
        sign_groups.setdefault(key, []).append(module["name"] + ".act_rotation.signs")
    vocab_sign_keys = [vocabulary[k] + ".owner.rotation.signs" for k in ("embedding", "head")]
    sign_keys = {key for values in sign_groups.values() for key in values} | set(vocab_sign_keys)
    model_architecture = base_class.model_arch

    class RetainedQwen35Model(base_class):
        model_arch = model_architecture
        no_mtp = True

        def index_tensors(self, remote_hf_model_id=None):
            if remote_hf_model_id is not None:
                raise ValueError("export requires a verified local checkpoint")
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            if not hp.get("tie_word_embeddings", False):
                raise ValueError("GGUF v2 requires a tied vocabulary")
            type(self)._original_block_count = hp["num_hidden_layers"]
            type(self).opt_num_mtp_layers = 0
            tensors = {}
            audit["dense_source_keys"], audit["auxiliary_source_keys"] = [], []
            with safe_open(model_path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
            for key in keys:
                if key in sign_keys:
                    continue
                if ".owner." in key or ".act_rotation." in key or key in {
                    vocabulary["embedding"] + ".weight", vocabulary["head"] + ".weight"
                }:
                    raise ValueError(f"unexpected extra vocabulary/rotation state: {key}")

                def load_tensor(name=key):
                    with safe_open(model_path, framework="pt", device="cpu") as handle:
                        return handle.get_tensor(name)

                item = self.filter_tensors((key, load_tensor))
                if item is None:
                    audit["auxiliary_source_keys"].append(key)
                else:
                    if item[0] in tensors:
                        raise ValueError(f"duplicate converted dense tensor: {item[0]}")
                    tensors[item[0]] = item[1]
                    audit["dense_source_keys"].append(key)
            audit["stored_rotation_keys"] = sorted(set(keys) & sign_keys)
            return tensors

        def prepare_tensors(self):
            super().prepare_tensors()
            hp = {**self.hparams, **self.hparams.get("text_config", {})}
            audit["matrices"] = []
            with (safe_open(packed_path, framework="pt", device="cpu") as packed,
                  safe_open(model_path, framework="pt", device="cpu") as state):
                for i, module in enumerate(modules, 1):
                    native = checkpoint_matrix(packed, module["qweight"])
                    group = module.get("rotation_id") or module["name"]
                    signs = stored_signs(state, sign_groups[group], native.layout.in_features)
                    rows, columns = qwen35_permutations(module["name"], hp)
                    projection = NativeProjection(native, signs, rows, columns)
                    name = module["name"].replace("model.language_model.", "model.") + ".weight"
                    stem = self.map_tensor_name(name).removesuffix(".weight")
                    self.add_native(stem, projection)
                    LOG.info("backbone %d/%d: %s", i, len(modules), stem)
                LOG.info("assembling saved vocabulary chunks (no dense reconstruction)")
                native = join_vocabulary_chunks([
                    checkpoint_matrix(packed, chunk) for chunk in vocabulary["chunks"]])
                if [native.layout.out_features, native.layout.in_features] != vocabulary["shape"]:
                    raise ValueError("vocabulary chunk dimensions disagree with manifest")
                if native.layout.bits != vocabulary["config"]["bits"]:
                    raise ValueError("vocabulary bit width disagrees with manifest")
                signs = stored_signs(state, vocab_sign_keys, native.layout.in_features)
                self.add_native("token_embd", NativeProjection(native, signs, vocabulary=True))

        def add_native(self, stem, projection):
            tensors = projection.tensors(stem)
            for key, value in tensors.items():
                self.gguf_writer.add_tensor(key, value)
            audit["matrices"].append({
                "stem": stem, "bits": projection.matrix.layout.bits,
                "scale_bits": projection.matrix.layout.scale_bits,
                "rows": projection.matrix.layout.out_features,
                "columns": projection.matrix.layout.in_features,
                "vocabulary": projection.vocabulary,
                "tensor_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in tensors.items()},
                "tensor_bytes": {k: v.nbytes for k, v in tensors.items()},
            })

        def prepare_metadata(self, vocab_only):
            super().prepare_metadata(vocab_only=vocab_only)
            writer = self.gguf_writer
            writer.add_array("qwen35.attention.recurrent_layers", recurrent_layer_map(self.hparams))
            writer.add_string("rotquant.format", "rotquant-native")
            writer.add_uint32("rotquant.version", GGUF_VERSION)
            writer.add_uint32("rotquant.matrix_version", 3)
            writer.add_uint32("rotquant.rotation_block_size", 128)
            writer.add_bool("rotquant.tied_embedding", True)
            writer.add_string("rotquant.vocabulary_mode", "dense_equivalent_fp16")
            writer.add_string("rotquant.required_llama_cpp_commit", LLAMA_REVISION)
            writer.add_string("rotquant.checkpoint_sha256", audit["checkpoint_manifest_sha256"])

    return RetainedQwen35Model


def export_checkpoint(checkpoint: Path, output: Path, llama_cpp: Path) -> dict:
    checkpoint = resolve_checkpoint_directory(checkpoint).resolve()
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"choose a new export directory: {output}")
    LOG.info("verifying all checkpoint file hashes")
    manifest = verify_checkpoint(checkpoint)
    validate_recipe(manifest)
    revision = subprocess.check_output(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], text=True).strip()
    if revision != LLAMA_REVISION:
        raise ValueError(f"converter must use pinned llama.cpp {LLAMA_REVISION}")
    sys.path.insert(0, str(llama_cpp))
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf
    from conversion.qwen import Qwen3_5TextModel

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    audit = {"protocol": "rotquant-gguf-v2-export-v1", "runtime_validated": False,
             "checkpoint_manifest_sha256": file_digest(checkpoint / MANIFEST_NAME),
             "source_files_sha256": manifest["files_sha256"], "converter_revision": revision,
             "scope": "text runtime; original non-text tensors retained in auxiliary.safetensors"}
    try:
        converter_type = make_converter(Qwen3_5TextModel, gguf, checkpoint, manifest, audit)
        converter = converter_type(checkpoint, gguf.LlamaFileType.MOSTLY_F16,
                                   staging / "model.gguf", eager=True, use_temp_file=True,
                                   model_name="Qwen3.5-RotQuant-Retained")
        if converter.hf_arch not in {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"}:
            raise ValueError(f"unsupported model architecture: {converter.hf_arch}")
        converter.write()
        if audit["auxiliary_source_keys"]:
            with safe_open(checkpoint / manifest["model_state"], framework="pt", device="cpu") as handle:
                save_file({k: handle.get_tensor(k).contiguous() for k in audit["auxiliary_source_keys"]},
                          staging / "auxiliary.safetensors")
        audit["artifact_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": file_digest(p)}
                                   for p in sorted(staging.iterdir()) if p.is_file()}
        audit["complete_model_payload_bytes"] = sum(p["bytes"] for p in audit["artifact_files"].values())
        (staging / "export.json").write_text(json.dumps(audit, indent=2) + "\n")
        # Publish only a complete export; failed staging is retained for inspection.
        os.rename(staging, output)
        return audit
    except BaseException:
        LOG.error("export incomplete; staging retained at %s", staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="new directory (never overwrites an artifact)")
    parser.add_argument("--llama-cpp-dir", type=Path, default=Path("third_party/llama.cpp"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = export_checkpoint(args.checkpoint, args.output, args.llama_cpp_dir.resolve())
    print(json.dumps({"output": str(args.output), "bytes": result["complete_model_payload_bytes"],
                      "runtime_validated": False}))


if __name__ == "__main__":
    main()
