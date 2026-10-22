"""Matched FP16 text GGUF plus the verified, unchanged auxiliary payload."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.export_rotquant_gguf_v2 import (
    LLAMA_REVISION,
    Qwen35FP32Transforms,
    recurrent_layer_map,
)
from scripts.native_hashing import digest


def source_tensors(source: Path) -> dict[str, Path]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        names = set(mapping.values())
    else:
        mapping = None
        names = {"model.safetensors"}
    tensors = {}
    for name in sorted(names):
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source):
            raise ValueError("source shard escapes model directory")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for key in keys:
                if key in tensors or (mapping is not None and mapping.get(key) != name):
                    raise ValueError(f"inconsistent source tensor index: {key}")
                tensors[key] = path
    if mapping is not None and set(tensors) != set(mapping):
        raise ValueError("source tensor index is incomplete")
    return tensors


def verified_export_artifacts(quant_export: Path, receipt: dict) -> dict:
    if receipt.get("converter_revision") != LLAMA_REVISION:
        raise ValueError("quantized export uses a different converter revision")
    artifacts = receipt.get("artifact_files", {})
    if "model.gguf" not in artifacts:
        raise ValueError("quantized export is missing its model receipt")
    for name, expected in artifacts.items():
        path = (quant_export / name).resolve(strict=True)
        if not path.is_relative_to(quant_export) or not path.is_file():
            raise ValueError("invalid quantized export artifact path")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"quantized export artifact changed: {name}")
    return artifacts


def compare_auxiliary_tensors(source: Path, auxiliary: Path, expected_keys: set[str]) -> None:
    tensors = source_tensors(source)
    with safe_open(auxiliary, framework="pt", device="cpu") as retained:
        if set(retained.keys()) != expected_keys:
            raise ValueError("auxiliary tensor names disagree with export receipt")
        for key in sorted(expected_keys):
            if key not in tensors:
                raise ValueError(f"FP16 source is missing auxiliary tensor: {key}")
            with safe_open(tensors[key], framework="pt", device="cpu") as original:
                left, right = original.get_tensor(key), retained.get_tensor(key)
                if left.dtype != right.dtype or not torch.equal(left, right):
                    raise ValueError(f"FP16 source auxiliary tensor differs: {key}")


def verified_auxiliary(source: Path, quant_export: Path, receipt: dict) -> Path | None:
    artifacts = verified_export_artifacts(quant_export, receipt)
    expected_keys = set(receipt.get("auxiliary_source_keys", []))
    if not expected_keys:
        if "auxiliary.safetensors" in artifacts:
            raise ValueError("unaccounted auxiliary tensors in quantized export")
        return None
    if "auxiliary.safetensors" not in artifacts:
        raise ValueError("quantized export is missing its auxiliary receipt")
    auxiliary = quant_export / "auxiliary.safetensors"
    compare_auxiliary_tensors(source, auxiliary, expected_keys)
    return auxiliary


def converter_modules(llama_cpp: Path):
    sys.path.insert(0, str(llama_cpp))
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf
    from conversion import qwen
    if not Path(qwen.__file__).resolve().is_relative_to(llama_cpp):
        raise ValueError("a converter from a different source tree is already imported")
    return gguf, qwen.Qwen3_5TextModel


def verify_export_sources(source: Path, llama_cpp: Path, quant_export: Path):
    revision = subprocess.check_output(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], text=True).strip()
    if revision != LLAMA_REVISION:
        raise ValueError(f"converter must use pinned llama.cpp {LLAMA_REVISION}")
    config = json.loads((source / "config.json").read_text())
    hp = {**config, **config.get("text_config", {})}
    # save_pretrained records the actual model dtype on the outer config;
    # nested text configs can retain the source checkpoint's original dtype.
    declared_dtype = (config.get("dtype", config.get("torch_dtype"))
                      or hp.get("dtype", hp.get("torch_dtype")))
    if declared_dtype not in ("float16", "torch.float16"):
        raise ValueError("saved FP16 source config must declare float16 execution")
    recurrent = recurrent_layer_map(config)
    receipt_path = quant_export / "export.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("protocol") != "rotquant-gguf-v2-export-v1":
        raise ValueError("expected a verified retained-model GGUF export")
    source_tensors(source)
    auxiliary = verified_auxiliary(source, quant_export, receipt)
    source_files = {str(path.relative_to(source)): digest(path)
                    for path in sorted(source.rglob("*")) if path.is_file()}
    audit = {"protocol": "rotquant-w4-fp16-export-v1", "runtime_validated": False,
             "converter_revision": revision, "source_config_sha256": source_files["config.json"],
             "source_files_sha256": source_files, "quantized_export_sha256": digest(receipt_path),
             "quantized_source_files_sha256": receipt["source_files_sha256"],
             "auxiliary_source_keys": receipt.get("auxiliary_source_keys", []),
             "auxiliary_exact_source_match": True,
             "recurrent_layers": recurrent, "execution_dtype": "float16",
             "scope": "text runtime; unchanged source auxiliary tensors retained and counted"}
    return auxiliary, audit


def export_fp16(source: Path, output: Path, llama_cpp: Path, quant_export: Path) -> dict:
    """Export the saved teacher using the same converter and counted auxiliaries."""
    source, llama_cpp, quant_export = (Path(path).resolve(strict=True)
                                     for path in (source, llama_cpp, quant_export))
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    auxiliary, audit = verify_export_sources(source, llama_cpp, quant_export)
    gguf, base_class = converter_modules(llama_cpp)
    omitted_keys = set()

    class MatchedFP16Model(Qwen35FP32Transforms, base_class):
        model_arch = base_class.model_arch
        no_mtp = True

        @classmethod
        def filter_tensors(cls, item):
            result = super().filter_tensors(item)
            if result is None:
                omitted_keys.add(item[0])
            return result

        def prepare_metadata(self, vocab_only):
            super().prepare_metadata(vocab_only=vocab_only)
            self.gguf_writer.add_array("qwen35.attention.recurrent_layers", audit["recurrent_layers"])

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    with torch.inference_mode():
        converter = MatchedFP16Model(source, gguf.LlamaFileType.MOSTLY_F16,
                                    staging / "model.gguf", eager=True, use_temp_file=True,
                                    model_name="Qwen3.5-FP16-Research-Control")
        if converter.hf_arch not in {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"}:
            raise ValueError(f"unsupported model architecture: {converter.hf_arch}")
        converter.write()
    if omitted_keys != set(audit["auxiliary_source_keys"]):
        raise ValueError("FP16 converter omitted tensors differ from the counted auxiliary payload")
    if auxiliary is not None:
        shutil.copyfile(auxiliary, staging / "auxiliary.safetensors")
    audit["artifact_files"] = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                               for path in sorted(staging.iterdir()) if path.is_file()}
    if "model.gguf" not in audit["artifact_files"]:
        raise ValueError("converter did not produce the FP16 model")
    audit["complete_model_payload_bytes"] = sum(item["bytes"] for item in audit["artifact_files"].values())
    (staging / "export.json").write_text(json.dumps(audit, indent=2) + "\n")
    os.rename(staging, output)
    return audit
