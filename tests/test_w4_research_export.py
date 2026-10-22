"""FP16 export preserves topology and accounts for the same source auxiliaries."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts import w4_research_export as export


@pytest.fixture
def export_inputs(tmp_path, monkeypatch):
    source, quant, llama = (tmp_path / name for name in ("source", "quant", "llama"))
    for path in (source, quant, llama):
        path.mkdir()
    config = {"dtype": "float16", "num_hidden_layers": 2,
              "layer_types": ["full_attention", "linear_attention"]}
    (source / "config.json").write_text(json.dumps(config))
    auxiliary = {"model.visual.weight": torch.ones(2, 3, dtype=torch.float16)}
    save_file({**auxiliary, "model.language_model.weight": torch.zeros(3, 3, dtype=torch.float16)},
              source / "model.safetensors")
    save_file(auxiliary, quant / "auxiliary.safetensors")
    (quant / "model.gguf").write_bytes(b"packed model")
    receipt = {"protocol": "rotquant-gguf-v2-export-v1", "converter_revision": export.LLAMA_REVISION,
               "auxiliary_source_keys": list(auxiliary), "source_files_sha256": {"config.json": "source-receipt"},
               "artifact_files": {path.name: {"bytes": path.stat().st_size, "sha256": export.digest(path)}
                                  for path in quant.iterdir()}}
    (quant / "export.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(export.subprocess, "check_output", lambda *a, **kw: export.LLAMA_REVISION)

    class Converter:
        model_arch = "qwen35"
        hf_arch = "Qwen3_5ForConditionalGeneration"

        def __init__(self, source, outtype, output, **kwargs):
            assert outtype == "f16"
            self.source, self.output = source, output
            self.arrays = {}
            self.gguf_writer = SimpleNamespace(add_array=self.arrays.__setitem__)

        @classmethod
        def filter_tensors(cls, item):
            return None if ".visual." in item[0] else item

        def prepare_metadata(self, vocab_only):
            assert vocab_only is False

        def write(self):
            for key in export.source_tensors(self.source):
                self.filter_tensors((key, None))
            self.prepare_metadata(False)
            assert self.arrays["qwen35.attention.recurrent_layers"] == [False, True]
            self.output.write_bytes(b"fp16 model")

    gguf = SimpleNamespace(LlamaFileType=SimpleNamespace(MOSTLY_F16="f16"))
    monkeypatch.setattr(export, "converter_modules", lambda path: (gguf, Converter))
    return source, tmp_path / "output", llama, quant


@pytest.mark.parametrize("nested_dtype", [None, "bfloat16"])
def test_fp16_export_preserves_topology_auxiliary_bytes_and_receipts(export_inputs, nested_dtype):
    source, output, llama, quant = export_inputs
    if nested_dtype:
        config = json.loads((source / "config.json").read_text())
        config["text_config"] = {"dtype": nested_dtype}
        (source / "config.json").write_text(json.dumps(config))
    audit = export.export_fp16(source, output, llama, quant)
    assert audit["recurrent_layers"] == [False, True]
    assert audit["runtime_validated"] is False
    assert audit["auxiliary_exact_source_match"] is True
    assert (output / "auxiliary.safetensors").read_bytes() == (quant / "auxiliary.safetensors").read_bytes()
    assert audit["complete_model_payload_bytes"] == sum(
        (output / name).stat().st_size for name in ("model.gguf", "auxiliary.safetensors"))
    assert audit["source_config_sha256"] == export.digest(source / "config.json")
    assert audit["quantized_export_sha256"] == export.digest(quant / "export.json")
    with pytest.raises(FileExistsError):
        export.export_fp16(source, output, llama, quant)


@pytest.mark.parametrize("change,match", [("auxiliary", "auxiliary tensor differs"),
                                         ("receipt", "artifact changed"),
                                         ("dtype", "float16 execution"),
                                         ("extra", "omitted tensors differ")])
def test_fp16_export_rejects_mismatched_source_or_export(export_inputs, change, match):
    source, output, llama, quant = export_inputs
    if change == "auxiliary":
        save_file({"model.visual.weight": torch.zeros(2, 3, dtype=torch.float16)}, source / "model.safetensors")
    elif change == "receipt":
        (quant / "model.gguf").write_bytes(b"changed")
    elif change == "dtype":
        config = json.loads((source / "config.json").read_text())
        config["dtype"] = "bfloat16"
        (source / "config.json").write_text(json.dumps(config))
    else:
        save_file({"model.visual.weight": torch.ones(2, 3, dtype=torch.float16),
                   "model.visual.extra": torch.zeros(1, dtype=torch.float16)}, source / "model.safetensors")
    with pytest.raises(ValueError, match=match):
        export.export_fp16(source, output, llama, quant)
    assert not output.exists()


def test_source_tensor_index_rejects_unindexed_or_missing_keys(tmp_path):
    save_file({"a": torch.ones(1)}, tmp_path / "model-00001-of-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"b": "model-00001-of-00001.safetensors"}}))
    with pytest.raises(ValueError, match="inconsistent source tensor index"):
        export.source_tensors(tmp_path)
