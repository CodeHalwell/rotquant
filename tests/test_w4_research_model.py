import copy

import pytest
import torch
import yaml

from rotquant.checkpoint import load_packed_model, verify_checkpoint
from rotquant.linear import QuantLinear
from scripts import run_experiment as experiment
from scripts import w4_research_model as model_work
from scripts.preflight_packed_validation import tiny_model
from scripts.w4_research_data import batches
from scripts.w4_research_plan import DEFAULT_CONFIG, Arm


def tiny_inputs():
    parts = ("calibration", "recovery_train", "recovery_validation", "recovery_selection", "selection")
    return {"documents": {
        part: [{"id": part, "source_row": row, "input_ids": list(range(1 + row, 18 + row))}]
        for row, part in enumerate(parts)}}


def tiny_config():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config.update(device="cpu", n_calib=1, calib_seq_len=17, hessian_collection="joint")
    config["research"].update(scale_refit_passes=1, code_refit_passes=1)
    config["vocabulary"]["chunk_rows"] = 64
    config["recovery"].update(distill_steps=2, distill_train_batches=1,
                              distill_validation_batches=1, distill_selection_batches=1,
                              distill_validation_interval=1, distill_teacher_topk=8,
                              distill_checkpoint_interval=1)
    return config


def test_tiny_qwen_research_arm_recovers_exports_and_reloads(tmp_path, monkeypatch):
    torch.manual_seed(73)
    config, data = tiny_config(), tiny_inputs()
    source = tiny_model(dtype=torch.float16)
    monkeypatch.setattr(model_work, "load_source", lambda cfg: (source, None, None, "multimodal_lm"))
    monkeypatch.setattr(experiment, "build_calib_loader",
                        lambda *args, **kwargs: batches(data["documents"]["calibration"]))
    expected = []
    def evaluate(model, tokenizer, device, destination):
        with torch.inference_mode():
            ids = batches(data["documents"]["selection"])[0]["input_ids"]
            expected.append(model(input_ids=ids, use_cache=False).logits.clone())
        return {"mean_teacher_kl": .001}
    arm = Arm("tiny-combined", group_size=32, code_refit_passes=1, recovery=True)
    report = model_work.prepare_arm(arm, config, data, tmp_path, tmp_path / "arm", "identity", evaluate)
    path = tmp_path / "arm/checkpoint"
    verify_checkpoint(path)
    restored = load_packed_model(path, device="cpu", dtype=torch.float16, fallback=True)
    with torch.inference_mode():
        ids = batches(data["documents"]["selection"])[0]["input_ids"]
        actual = restored(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
    assert report["artifact_bytes"] == sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    assert report["metrics"]["recovery"]["steps"] == 2
    numerics = report["metrics"]["recovery"]["numerics"]
    assert numerics["policy"] == "source_forward_fp32_l2_backward_v1"
    assert numerics["patched_fallbacks"]
    for module in source.modules():
        for attribute in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
            function = getattr(module, attribute, None)
            assert not hasattr(function, "_rotquant_numerics_token")
    layers = [m for m in restored.modules() if isinstance(m, QuantLinear)]
    assert layers and all(m.qweight.packed.bits == 4 and m.qweight.group_size == 32 for m in layers)
    assert restored.get_input_embeddings().owner.config.bits == 6
    assert (tmp_path / "recovery_state/tiny-combined/recovery.pt").is_file()


def test_backbone_refuses_same_source_rows_with_different_tokens(tmp_path, monkeypatch):
    config, frozen = tiny_config(), tiny_inputs()
    changed = copy.deepcopy(frozen["documents"]["calibration"])
    changed[0]["input_ids"][0] = 80
    manifest = experiment.token_batch_manifest(
        batches(changed), dataset="allenai/c4", split="train", revision=config["calib_dataset_revision"],
        skip=0, seq_len=17)
    def prepare(cfg, model, tokenizer, device, qcfg, pcfg, metrics):
        metrics["data_manifest"]["hessian_calibration"] = manifest
    monkeypatch.setattr(experiment, "_prepare_calibration", prepare)
    monkeypatch.setattr(experiment, "_apply_quantization", lambda *args: pytest.fail("must fail before quantisation"))
    with pytest.raises(ValueError, match="frozen inputs"):
        model_work.quantize_backbone(None, None, config, frozen, tmp_path)
