"""Tiny real Qwen hybrid integration; no downloads or long training runs."""
from __future__ import annotations

import copy

import pytest
import torch
import yaml
from torch import nn

from rotquant.checkpoint import load_packed_model, verify_checkpoint
from rotquant.linear import QuantLinear
from rotquant.pack import unpack_indices
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.ternary_state import source_state_digest
from scripts import ternary_research_model as model_work
from scripts.preflight_packed_validation import tiny_model
from scripts.ternary_research_plan import DEFAULT_CONFIG, Arm
from scripts.w4_research_data import batches


def _inputs():
    splits = ("calibration", "recovery_train", "recovery_validation", "recovery_selection", "selection")
    return {"fingerprint": "tiny-frozen-documents-v1", "documents": {
        name: [{"id": name, "source_row": row, "input_ids": list(range(row + 1, row + 18))}]
        for row, name in enumerate(splits)}}


def _config():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config.update(device="cpu", n_calib=1, calib_seq_len=17, hessian_collection="joint")
    config["vocabulary"]["chunk_rows"] = 64
    config["ternary"].update(scope="all", steps=1, train_batches=1, validation_batches=1,
                              validation_interval=1, checkpoint_interval=1)
    config["recovery"].update(distill_steps=2, distill_train_batches=1,
                               distill_validation_batches=1, distill_selection_batches=1,
                               distill_validation_interval=1, distill_checkpoint_interval=1,
                               distill_teacher_topk=8)
    return config


def _logits(model, data):
    with torch.inference_mode():
        return model(**batches(data["documents"]["selection"])[0], use_cache=False).logits.clone()


def _fixture(monkeypatch):
    torch.manual_seed(97)
    source = tiny_model(dtype=torch.float16)
    config, data = _config(), _inputs()
    monkeypatch.setattr(model_work, "load_source",
                        lambda cfg: (copy.deepcopy(source), None, None, "multimodal_lm"))
    expected = []
    def evaluate(model, tokenizer, device, destination):
        expected.append(_logits(model, data))
        return {"mean_teacher_kl": 0.001}
    return source, config, data, expected, evaluate


@pytest.mark.parametrize("method,steps", [("ternary_ptq", 0), ("ternary_qat", 1)])
def test_tiny_qwen_ternary_arm_exports_reloads_and_preserves_sensitive_paths(
        tmp_path, monkeypatch, method, steps):
    source, config, data, expected, evaluate = _fixture(monkeypatch)
    arm = Arm(method, 2, method)
    report = model_work.prepare_arm(arm, config, data, tmp_path, tmp_path / "arm", "identity", evaluate)
    checkpoint = tmp_path / "arm/checkpoint"
    verify_checkpoint(checkpoint)
    restored = load_packed_model(checkpoint, device="cpu", dtype=torch.float16, fallback=False)
    assert torch.equal(_logits(restored, data), expected[0])
    linears = [(name, module) for name, module in restored.named_modules() if isinstance(module, QuantLinear)]
    assert linears and all(module.qweight.codebook.levels == 3 for _, module in linears)
    for name, module in linears:
        assert "language_model.layers" in name
        assert module.qweight.packed.bits == 2 and module.qweight.scales.dtype == torch.float16
        assert unpack_indices(module.qweight.packed).max() <= 2
    protected = {name: module for name, module in source.named_modules()
                 if isinstance(module, nn.Linear) and ("in_proj_a" in name or "in_proj_b" in name)}
    assert protected
    for name, original in protected.items():
        actual = restored.get_submodule(name)
        assert isinstance(actual, nn.Linear)
        assert torch.equal(actual.weight, original.weight)
    packed_names = [name + "." for name, _ in linears]
    for name, parameter in source.named_parameters():
        if "language_model.layers" in name and not any(name.startswith(prefix) for prefix in packed_names):
            actual = restored.get_parameter(name)
            assert actual.dtype == parameter.dtype and torch.equal(actual, parameter)
    for name, buffer in source.named_buffers():
        if "inv_freq" in name:
            assert torch.equal(restored.get_buffer(name), buffer)
    assert restored.get_input_embeddings().owner.config.bits == 6
    assert report["artifact_bytes"] == sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    stats = report["metrics"]["block_qat"]["ternary_train"]["blocks"]
    assert len(stats) == 2 and all(row["steps_completed"] == steps for row in stats.values())
    assert all(row["packed_validation_loss"] == row["validation_loss"] for row in stats.values())
    assert all(row["optimiser_seen_tokens"] == steps * 17 for row in stats.values())
    assert all(row["training_compute_dtype"] == "float32" for row in stats.values())
    assert all(row["source_compute_dtype"] == "float16" for row in stats.values())
    assert all(row["wall_seconds_this_attempt"] >= 0 for row in stats.values())
    assert all(row["peak_cuda_allocated_bytes"] is None for row in stats.values())


def test_recovery_loads_exact_saved_parent_and_does_not_retrain(tmp_path, monkeypatch):
    _, config, data, expected, evaluate = _fixture(monkeypatch)
    parent_arm = Arm("ternary_blockqat", 2, "ternary_qat")
    model_work.prepare_arm(parent_arm, config, data, tmp_path, tmp_path / "parent", "identity", evaluate)
    parent_path = tmp_path / "parent/checkpoint"
    parent = load_packed_model(parent_path, device="cpu", dtype=torch.float16, fallback=True)
    parent_logits = _logits(parent, data)
    original_capture, original_apply = model_work.capture_recovery, model_work.apply_recovery
    captured = []
    gradient_norms = []
    original_step = torch.optim.Adam.step
    def record_step(optimizer, *args, **kwargs):
        gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"]
                     if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        gradient_norms.append(sum(float(gradient.abs().sum()) for gradient in gradients))
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    def capture(source, cfg, frozen):
        assert not any(isinstance(module, QuantLinear) for module in source.modules())
        calls = original_capture(source, cfg, frozen)
        captured.append(calls)
        return calls
    def apply(candidate, cfg, calls, destination):
        assert len(captured) == 1 and calls is captured[0]
        assert torch.equal(_logits(candidate, data), parent_logits)
        for name, module in candidate.named_modules():
            if isinstance(module, QuantLinear):
                original = parent.get_submodule(name)
                assert torch.equal(module.qweight.packed.data, original.qweight.packed.data)
                assert torch.equal(module.qweight.scales, original.qweight.scales)
        return original_apply(candidate, cfg, calls, destination)
    def forbidden(*args, **kwargs):
        pytest.fail("recovered arm must not repeat QAT, backbone quantisation or vocabulary quantisation")
    monkeypatch.setattr(model_work, "capture_recovery", capture)
    monkeypatch.setattr(model_work, "apply_recovery", apply)
    monkeypatch.setattr(model_work, "train_ternary", forbidden)
    monkeypatch.setattr(model_work, "quantize_backbone", forbidden)
    monkeypatch.setattr(model_work, "quantize_vocabulary_in_place", forbidden)
    arm = Arm("ternary_recovered", 2, "ternary_qat", "ternary_blockqat")
    result = model_work.prepare_arm(arm, config, data, tmp_path, tmp_path / "recovered",
                                    "identity", evaluate, parent=parent_path)
    assert result["metrics"]["recovery"]["steps"] == 2
    assert len(gradient_norms) == 2 and all(norm > 0 for norm in gradient_norms)
    restored = load_packed_model(tmp_path / "recovered/checkpoint", device="cpu", dtype=torch.float16)
    assert torch.equal(_logits(restored, data), expected[-1])
    for name, module in restored.named_modules():
        if isinstance(module, QuantLinear):
            assert module.qweight.codebook.levels == 3
            assert torch.equal(module.qweight.packed.data, parent.get_submodule(name).qweight.packed.data)
    assert (tmp_path / "recovery_state/ternary_recovered/recovery.pt").exists()


def test_recovery_binding_rejects_changed_parent_identity(tmp_path, monkeypatch):
    _, config, data, _, evaluate = _fixture(monkeypatch)
    parent_arm = Arm("ternary_ptq", 2, "ternary_ptq")
    model_work.prepare_arm(parent_arm, config, data, tmp_path, tmp_path / "parent", "identity", evaluate)
    arm = Arm("ternary_recovered", 2, "ternary_qat", "ternary_ptq")
    parent = tmp_path / "parent/checkpoint"
    model_work.bind_recovery(tmp_path, arm, "identity", data, parent)
    with pytest.raises(ValueError, match="parent/data changed"):
        model_work.bind_recovery(tmp_path, arm, "changed-identity", data, parent)


def test_mlp_arms_share_w4_remainder_before_qat_and_preserve_it(tmp_path, monkeypatch):
    _, config, data, _, evaluate = _fixture(monkeypatch)
    config["ternary"]["scope"] = "mlp"
    prepared, seen = [], []
    def quantize_remainder(model, tokenizer, cfg, frozen, root):
        assert cfg["quant"]["bits"] == 4
        assert ".mlp." in cfg["patch"]["exclude"]
        patch_model(model, PatchConfig(
            quant=QuantConfig(bits=4, group_size=128, scale_bits=8),
            rotation="fwht", block=128, fallback=False,
            include=cfg["patch"]["include"], exclude=cfg["patch"]["exclude"], seed=cfg["seed"]))
        packed = {name: copy.deepcopy(module) for name, module in model.named_modules()
                  if isinstance(module, QuantLinear)}
        assert packed and all(".mlp." not in name for name in packed)
        prepared.append(packed)
        return {"tiny_real_w4_remainder": True}
    original_train = model_work.train_ternary
    def train(model, arm, cfg, frozen, root, identity):
        expected = prepared[-1]
        for name, module in expected.items():
            assert source_state_digest(model.get_submodule(name)) == source_state_digest(module)
        result = original_train(model, arm, cfg, frozen, root, identity)
        for name, module in expected.items():
            assert source_state_digest(model.get_submodule(name)) == source_state_digest(module)
        seen.append(arm.method)
        return result
    monkeypatch.setattr(model_work, "quantize_backbone", quantize_remainder)
    monkeypatch.setattr(model_work, "train_ternary", train)
    for method in ("ternary_ptq", "ternary_qat"):
        arm = Arm(method, 2, method)
        model_work.prepare_arm(arm, config, data, tmp_path, tmp_path / method, "identity", evaluate)
        restored = load_packed_model(tmp_path / method / "checkpoint", dtype=torch.float16)
        for name, module in restored.named_modules():
            if isinstance(module, QuantLinear):
                assert module.qweight.codebook.levels == (3 if ".mlp." in name else 16)
    assert seen == ["ternary_ptq", "ternary_qat"]
    assert prepared[0].keys() == prepared[1].keys()
    for name in prepared[0]:
        assert source_state_digest(prepared[0][name]) == source_state_digest(prepared[1][name])
