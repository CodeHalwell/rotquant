"""Early source validation without downloading or executing a model."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from scripts import preflight_native_gpu as preflight


@pytest.mark.parametrize("mutation", [None, "mask", "ids", "logits", "extra", "trace"])
def test_source_probe_schema_checked_before_native_build(tmp_path, monkeypatch, mutation):
    import transformers

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / preflight.MANIFEST_NAME).write_text("{}")
    (checkpoint / "weights.bin").write_bytes(b"test")
    (tmp_path / "prepared.json").write_text("{}")
    probes = tmp_path / "packed_probes.safetensors"
    tensors = {"p0.input.input_ids": torch.tensor([[3, 4, 5, 6]]),
               "p0.input.attention_mask": torch.ones((1, 4), dtype=torch.int64),
               "p0.logits": torch.zeros((1, 2, 16)),
               "p0.generated": torch.tensor([[3, 4, 5, 6, 7, 8]])}
    if mutation == "mask":
        tensors["p0.input.attention_mask"][0, 0] = 0
    elif mutation == "ids":
        tensors["p0.input.input_ids"][0, 0] = -1
    elif mutation == "logits":
        tensors["p0.logits"][0, 0, 0] = float("nan")
    elif mutation == "extra":
        tensors["p0.input.pixel_values"] = torch.zeros(1)
    elif mutation == "trace":
        tensors["p0.generated"][0, 0] = 12
    save_file(tensors, probes)
    manifest = {"tied_vocabulary": {"config": {"bits": 6}, "shape": [16, 128]},
                "files_sha256": {"weights.bin": "unused in this schema-only test"}}
    monkeypatch.setattr(preflight, "verified_source_evidence", lambda _: (probes, manifest))
    monkeypatch.setattr(preflight, "validate_recipe", lambda _: None)
    def tokenizer(path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    if mutation:
        with pytest.raises(ValueError):
            preflight.source(tmp_path, 6)
    else:
        assert preflight.source(tmp_path, 6)["probes"] == 1
        with pytest.raises(ValueError, match="bit width"):
            preflight.source(tmp_path, 8)
