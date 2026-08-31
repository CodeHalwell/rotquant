"""Joint transformer-block rotation training and exact held-out selection."""
from types import SimpleNamespace

import torch
from torch import nn

from rotquant.block_train import (
    BlockRotationTrainConfig,
    FakeQuantButterflyLinear,
    _packed_cpu_block,
    collect_block_calls,
    collect_teacher_calls,
    find_transformer_blocks,
    train_and_patch_blocks,
    train_fake_quant_block,
)
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig
from rotquant.quantize import (
    QuantConfig,
    Quantizer,
    _group_scales_rms,
    _storage_scales,
)
from rotquant.rotate import ButterflyRotation, Identity


class ToyBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, hidden_states, position_ids=None, **kwargs):
        del position_ids, kwargs
        return self.norm(hidden_states + self.fc2(torch.relu(self.fc1(hidden_states))))


class ToyTransformer(nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(d) for _ in range(n)])

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, position_ids=positions)
        return x


class ToyCausalLM(nn.Module):
    def __init__(self, vocab=32, d=16, n=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([ToyBlock(d) for _ in range(n)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        positions = torch.arange(
            hidden.shape[1], device=hidden.device).unsqueeze(0)
        for layer in self.layers:
            hidden = layer(hidden, position_ids=positions)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _qcfg():
    return QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=16)


def _pcfg(steps=3):
    return PatchConfig(
        quant=_qcfg(), rotation="butterfly", block=16, fallback=True,
        train_rotation={
            "objective": "block", "steps": steps, "lr": 1e-2,
            "train_batches": 1, "validation_batches": 1,
            "selection_batches": 1,
            "assignment_scale": "rms", "selection_min_improvement": 0.0,
        },
    )


def test_find_and_capture_replayable_transformer_blocks():
    torch.manual_seed(0)
    model = ToyTransformer()
    blocks = find_transformer_blocks(model)
    assert [name for name, _ in blocks] == ["layers.0", "layers.1"]
    calls = collect_block_calls(
        model, [torch.randn(1, 8, 16) for _ in range(3)], "cpu",
        blocks=blocks, max_batches=3, storage_dtype=torch.float32)
    assert all(len(records) == 3 for records in calls.values())
    assert calls["layers.0"][0].args[0].shape == (1, 8, 16)
    assert calls["layers.1"][0].output.shape == (1, 8, 16)
    assert all(not block._forward_hooks for _, block in blocks)


def test_joint_block_training_selects_checkpoint_on_validation_calls():
    torch.manual_seed(1)
    model = ToyTransformer(n=1)
    blocks = find_transformer_blocks(model)
    calls = collect_block_calls(
        model, [torch.randn(1, 8, 16) for _ in range(3)], "cpu",
        blocks=blocks, max_batches=3, storage_dtype=torch.float32)
    seeds = {"layers.0.fc1": 0, "layers.0.fc2": 1}
    _, selection, stats = train_fake_quant_block(
        model.layers[0], "layers.0", calls["layers.0"], _qcfg(),
        _pcfg(steps=5), seeds, "cpu",
        BlockRotationTrainConfig(**_pcfg(steps=5).train_rotation))
    assert stats["layers"] == 2 and len(selection) == 1
    assert selection[0] is calls["layers.0"][2]
    assert stats["final_validation_mse"] <= stats["initial_validation_mse"] + 1e-8
    assert 0 <= stats["best_step"] <= 5


def test_joint_block_training_can_stop_early_on_validation_plateau():
    torch.manual_seed(4)
    model = ToyTransformer(n=1)
    blocks = find_transformer_blocks(model)
    calls = collect_block_calls(
        model, [torch.randn(1, 8, 16) for _ in range(3)], "cpu",
        blocks=blocks, max_batches=3, storage_dtype=torch.float32)
    cfg = _pcfg(steps=10)
    cfg.train_rotation.update({
        "early_stopping_patience": 2,
        "validation_min_improvement": 0.99,
    })
    seeds = {"layers.0.fc1": 0, "layers.0.fc2": 1}
    _, _, stats = train_fake_quant_block(
        model.layers[0], "layers.0", calls["layers.0"], _qcfg(), cfg,
        seeds, "cpu", BlockRotationTrainConfig(**cfg.train_rotation))
    assert stats["stopped_early"]
    assert stats["steps_run"] == 2


def test_learned_scale_proxy_starts_at_exact_packed_scale_search():
    torch.manual_seed(5)
    linear = nn.Linear(16, 16)
    cfg = QuantConfig(
        bits=3, codebook="gaussian", scale="mse_search", group_size=16)
    fake = FakeQuantButterflyLinear(
        linear, cfg, block=16, seed=3, learn_scales=True)
    packed = QuantLinear.from_linear(
        linear, cfg, weight_rotation=fake.rotation,
        act_rotation=fake.rotation, fallback=True, fallback_dtype=torch.float32)
    assert torch.equal(fake._assigned_weight(), packed.qweight.dequantize())


def test_learned_scales_survive_exact_packing_without_extra_storage():
    torch.manual_seed(6)
    model = ToyTransformer(n=1)
    blocks = find_transformer_blocks(model)
    calls = collect_block_calls(
        model, [torch.randn(1, 8, 16) for _ in range(3)], "cpu",
        blocks=blocks, max_batches=3, storage_dtype=torch.float32)
    cfg = _pcfg(steps=2)
    cfg.train_rotation.update({"learn_scales": True, "scale_lr": 2e-2})
    seeds = {"layers.0.fc1": 0, "layers.0.fc2": 1}
    states, _, stats = train_fake_quant_block(
        model.layers[0], "layers.0", calls["layers.0"], _qcfg(), cfg,
        seeds, "cpu", BlockRotationTrainConfig(**cfg.train_rotation))
    assert stats["learned_scales"]
    assert all("scale_multiplier" in state for state in states.values())

    candidate = _packed_cpu_block(
        model.layers[0], "layers.0", _qcfg(), cfg, seeds, states)
    reference = _packed_cpu_block(
        model.layers[0], "layers.0", _qcfg(), cfg, seeds, None)
    for relative in ("fc1", "fc2"):
        source = getattr(model.layers[0], relative)
        quantized = getattr(candidate, relative)
        state = states[relative]
        rotated = quantized.act_rotation.rotate_weight(source.weight).float()
        expected = _storage_scales(
            _group_scales_rms(rotated, _qcfg().group_size)
            * state["scale_multiplier"],
            _qcfg().scale_bits,
        )
        assert torch.equal(quantized.qweight.scales, expected)
        assert (quantized.packed_state_bytes()
                == getattr(reference, relative).packed_state_bytes())


def test_block_training_packs_model_and_reports_selection():
    torch.manual_seed(2)
    model = ToyTransformer(n=2).eval()
    batches = [torch.randn(1, 8, 16) for _ in range(3)]
    calls = collect_block_calls(model, batches, "cpu", max_batches=3,
                                storage_dtype=torch.float32)
    stats = {}
    train_and_patch_blocks(model, _pcfg(steps=2), calls, stats_out=stats)
    assert all(isinstance(block.fc1, QuantLinear)
               and isinstance(block.fc2, QuantLinear) for block in model.layers)
    output = model(batches[0])
    assert output.shape == batches[0].shape and torch.isfinite(output).all()
    summary = stats["rotation_train"]
    assert summary["objective"] == "block"
    assert summary["blocks"] == 2 and summary["layers"] == 4
    assert summary["validation_batches"] == 1
    assert summary["selection_device"] == "cpu"
    assert 0.0 <= summary["selection_acceptance_rate"] <= 1.0


def test_propagation_uses_deployed_prefix_outputs_for_later_blocks():
    torch.manual_seed(7)
    model = ToyTransformer(n=2).eval()
    batches = [torch.randn(1, 8, 16) for _ in range(3)]
    calls = collect_block_calls(model, batches, "cpu", max_batches=3,
                                storage_dtype=torch.float32)
    cfg = _pcfg(steps=1)
    cfg.train_rotation["propagate_quantized_inputs"] = True
    stats = {}
    train_and_patch_blocks(model, cfg, calls, stats_out=stats)
    summary = stats["rotation_train"]
    assert summary["propagate_quantized_inputs"]
    assert summary["final_input_drift_mse"] > 0
    assert summary["mean_input_drift_mse"] > 0


def test_end_to_end_distillation_commits_packed_parameters():
    torch.manual_seed(8)
    model = ToyCausalLM().eval()
    batches = [{
        "input_ids": torch.randint(0, 32, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    } for _ in range(6)]
    blocks = find_transformer_blocks(model)
    calls = collect_block_calls(
        model, batches[:3], "cpu", blocks=blocks, max_batches=3,
        storage_dtype=torch.float32)
    teacher_calls = collect_teacher_calls(
        model, batches[3:], "cpu", max_batches=3,
        storage_dtype=torch.float32)
    cfg = _pcfg(steps=1)
    cfg.train_rotation.update({
        "learn_scales": True,
        "propagate_quantized_inputs": True,
        "distill_steps": 2,
        "distill_train_batches": 1,
        "distill_validation_batches": 1,
        "distill_selection_batches": 1,
        "distill_early_stopping_patience": 0,
        "distill_lora_rank": 2,
        "distill_lora_alpha": 4.0,
        "distill_lora_lr": 1e-3,
        "distill_lora_init": "residual_svd",
        "distill_lora_svd_oversample": 1,
        "distill_code_refresh_interval": 1,
        "distill_existing_patterns": ["norm"],
        "distill_existing_lr": 1e-4,
        "distill_train_rotations": False,
        "distill_train_scales": False,
    })
    stats = {}
    train_and_patch_blocks(
        model, cfg, calls, distill_calls=teacher_calls, stats_out=stats)
    distillation = stats["distillation"]
    assert distillation["steps_run"] == 2
    assert 0 <= distillation["best_step"] <= 2
    assert (distillation["candidate_validation_loss"]
            <= distillation["initial_validation_loss"] + 1e-8)
    assert isinstance(distillation["accepted"], bool)
    assert distillation["lora_rank"] == 2
    assert distillation["lora_init"] == "residual_svd"
    assert distillation["code_refresh_interval"] == 1
    assert distillation["existing_parameter_names"]
    assert not distillation["train_rotations"]
    assert not distillation["train_scales"]
    assert (distillation["adapter_parameter_bytes"] > 0
            if distillation["lora_retained"]
            else distillation["adapter_parameter_bytes"] == 0)
    quant_linears = [module for module in model.modules()
                     if isinstance(module, QuantLinear)]
    assert quant_linears
    assert all(module._log_scale_multiplier is None
               for module in quant_linears)
    assert all((module.lora_A is not None) == distillation["lora_retained"]
               for module in quant_linears)
    assert torch.isfinite(model(**batches[0]).logits).all()


def test_residual_svd_lora_starts_closer_to_source_weight():
    torch.manual_seed(9)
    source = nn.Linear(16, 24, bias=False)
    qlinear = QuantLinear.from_linear(
        source, _qcfg(), weight_rotation=Identity(16),
        act_rotation=Identity(16), fallback=True)
    residual = source.weight.detach().float() - qlinear.qweight.dequantize().float()
    baseline_error = residual.pow(2).mean()
    matrix_a, matrix_b = qlinear.enable_lora(
        4, 8.0, init="residual_svd", residual=residual,
        oversample=2, niter=2)
    recovered = matrix_b @ matrix_a * (qlinear.lora_alpha / qlinear.lora_rank)
    assert (residual - recovered).pow(2).mean() < baseline_error


def test_code_refresh_reassigns_for_current_rotation():
    torch.manual_seed(10)
    source = nn.Linear(16, 16, bias=False)
    rotation = ButterflyRotation(16, block=16, seed=0)
    qlinear = QuantLinear.from_linear(
        source, _qcfg(), weight_rotation=rotation,
        act_rotation=rotation, fallback=True)
    qlinear.retain_recovery_source(source.weight)
    with torch.no_grad():
        rotation.theta.add_(0.05)
    qlinear.refresh_quantization()
    expected = Quantizer(_qcfg()).quantize_weight(
        rotation.rotate_weight(source.weight.half().float())).dequantize()
    assert torch.allclose(qlinear.qweight.dequantize(), expected)
    qlinear.drop_recovery_source()
    assert qlinear._recovery_source_weight is None


def test_rejected_block_uses_parameter_free_fwht():
    from rotquant.block_train import _patch_source_block

    block = ToyBlock()
    cfg = _pcfg(steps=1)
    seeds = {"layers.0.fc1": 0, "layers.0.fc2": 1}
    assert _patch_source_block(block, "layers.0", _qcfg(), cfg, seeds, None) == 2
    assert sum(p.numel() for p in block.fc1.act_rotation.parameters()) == 0
    assert sum(p.numel() for p in block.fc2.act_rotation.parameters()) == 0
