import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rotquant.gguf_v2 import checkpoint_matrix
from rotquant.native_v3 import decode_native_v3_rows, encode_native_v3
from rotquant.ternary import TernaryConfig, TernaryLinear
from scripts import ternary_research_plan as plan
from scripts.ternary_research_results import comparison


def config():
    return yaml.safe_load(plan.DEFAULT_CONFIG.read_text())


def test_default_study_has_exact_controls_and_recovery_dependencies():
    value = plan.validate_config(config())
    arms = plan.arms()
    assert {a.name for a in arms} | {"fp16", "bonsai2_4b"} == set(plan.LABELS)
    assert next(a for a in arms if a.name == "ternary_recovered").parent == "ternary_blockqat"
    assert next(a for a in arms if a.name == "w4_v6_recovered").parent == "w4_v6"
    assert value["recovery"]["distill_ce_weight"] > 0
    for arm in arms:
        if arm.method == "scalar":
            assert arm.config(value)["quant"]["bits"] == arm.bits


@pytest.mark.parametrize("section,key,value", [
    ("ternary", "max_blocks", 2), ("ternary", "steps", 0),
    ("ternary", "train_batches", 3), ("recovery", "distill_train_rotations", True),
    ("recovery", "distill_train_scales", False), ("bonsai", "packing", "BF16"),
])
def test_study_rejects_partial_or_incompatible_configs(section, key, value):
    settings = config()
    settings[section][key] = value
    with pytest.raises(ValueError):
        plan.validate_config(settings)


def test_run_identity_rejects_changed_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "source_files", lambda: {"train.py": "abc"})
    value = config()
    first = plan.bind_run(tmp_path, value, {"torch": "test"})
    assert first == plan.bind_run(tmp_path, value, {"torch": "test"})
    value["ternary"]["lr"] *= 2
    with pytest.raises(ValueError, match="changed"):
        plan.bind_run(tmp_path, value, {"torch": "test"})


def test_pending_and_failed_comparisons_never_appear_as_zero_measurements():
    result = comparison({}, {"bonsai2_4b": {"artifact_bytes": 1625119104,
                                            "language_payload_bytes": 949549920}}, {}, None,
                        {"ternary_blockqat": "interrupted"})
    assert not result["complete"]
    rows = {r["label"]: r for r in result["rows"]}
    assert rows["ternary_blockqat"]["status"] == "failed"
    assert rows["ternary_blockqat"]["quality"] is None
    assert rows["bonsai2_4b"]["status"] == "pending"
    assert rows["bonsai2_4b"]["native_payload_bytes"] == 1625119104


def test_ternary_native_padding_does_not_add_a_quantization_level():
    torch.manual_seed(7)
    source = torch.nn.Linear(256, 8, bias=False)
    packed = TernaryLinear.from_linear(source, TernaryConfig()).to_quant_linear().qweight
    wire = encode_native_v3(packed)
    np.testing.assert_array_equal(wire.codebook, [-1., 0., 1., 0.])
    np.testing.assert_array_equal(decode_native_v3_rows(wire), packed.dequantize().numpy())
    packed.packed.data[0] |= 3
    with pytest.raises(ValueError, match="reserved"):
        encode_native_v3(packed)


def test_saved_three_centroid_checkpoint_exports_without_requantizing(tmp_path):
    from safetensors import safe_open
    from transformers import LlamaConfig, LlamaForCausalLM

    from rotquant.checkpoint import MANIFEST_NAME, save_packed_checkpoint

    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=128, intermediate_size=128,
                                        num_hidden_layers=1, num_attention_heads=1)).half()
    module = TernaryLinear.from_linear(model.model.layers[0].mlp.up_proj).to_quant_linear()
    model.model.layers[0].mlp.up_proj = module
    path = tmp_path / "checkpoint"
    save_packed_checkpoint(model, path)
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    spec = manifest["quantized_modules"][0]["qweight"]
    with safe_open(path / manifest["packed_state"], framework="pt", device="cpu") as handle:
        actual = checkpoint_matrix(handle, spec)
    assert actual.to_bytes() == encode_native_v3(module.qweight).to_bytes()


def test_driver_passes_saved_parent_to_recovery(tmp_path):
    from scripts.run_qwen35_ternary_research import TernaryStudy

    study = object.__new__(TernaryStudy)
    study.root = tmp_path
    study.teacher = {"teacher_dir": "teacher"}
    study.runtime = {"source": "llama"}
    study.prepared = {"ternary_blockqat": {"checkpoint": "original-packed-parent"}}
    study.exported = {}
    calls = []
    def worker(name, phase, **kwargs):
        calls.append((name, phase, kwargs))
        return {"checkpoint": "recovered", "probes": "probes"}
    study.worker = worker
    study.summarize = lambda: None
    study.prepare(next(a for a in plan.arms() if a.name == "ternary_recovered"))
    assert calls[0][2]["parent"] == "original-packed-parent"
    assert [phase for _, phase, _ in calls] == ["prepare", "reload", "export"]
    assert Path(tmp_path / "arms/ternary_recovered.json").is_file()
