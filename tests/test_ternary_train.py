from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import rotquant.ternary_train as training
from rotquant.block_train import collect_block_calls
from rotquant.linear import QuantLinear
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import Identity
from rotquant.ternary import TernaryConfig, TernaryLinear
from rotquant.ternary_state import source_state_digest
from rotquant.ternary_train import TernaryTrainConfig, train_and_patch_ternary


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.attn = nn.Linear(8, 8)
        self.in_proj_a = nn.Linear(8, 8)

    def forward(self, hidden):
        return hidden + self.mlp(hidden) + self.attn(hidden) + 0.01 * self.in_proj_a(hidden)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _fixture(tmp_path=None, **kwargs):
    torch.manual_seed(19)
    model = ToyModel().eval()
    config = TernaryTrainConfig(
        quant=TernaryConfig(4, 4), train_batches=2, validation_batches=1,
        validation_interval=1, checkpoint_interval=1,
        checkpoint_dir=None if tmp_path is None else str(tmp_path), **kwargs)
    train = [{"input_ids": torch.tensor([[1, 2, 3, 4]])},
             {"input_ids": torch.tensor([[4, 5, 6, 7]])}]
    validation = [{"input_ids": torch.tensor([[8, 9, 10, 11]])}]
    return model, config, train, validation


def test_scope_and_protected_paths_and_no_training_parameter_leaks():
    model, config, train, validation = _fixture(steps=2, scope="mlp")
    protected = model.layers[0].in_proj_a.weight.detach().clone()
    vocabulary = model.lm_head.weight.detach().clone()
    stats = {}
    train_and_patch_ternary(model, train, validation, config, stats_out=stats)
    assert isinstance(model.layers[0].mlp[0], QuantLinear)
    assert isinstance(model.layers[0].attn, nn.Linear)
    assert isinstance(model.layers[0].in_proj_a, nn.Linear)
    assert torch.equal(protected, model.layers[0].in_proj_a.weight)
    assert torch.equal(vocabulary, model.lm_head.weight)
    assert not any(isinstance(module, TernaryLinear) for module in model.modules())
    for result in stats["ternary_train"]["blocks"].values():
        assert result["validation_loss"] <= result["initial_validation_loss"]
        assert result["packed_validation_loss"] == result["validation_loss"]


def test_steps_zero_packs_baseline_without_optimizer_updates(monkeypatch):
    model, config, train, validation = _fixture(steps=0)
    def forbidden(_):
        raise AssertionError("PTQ baseline called update")
    monkeypatch.setattr(training._BlockTrainer, "update", forbidden)
    stats = {}
    train_and_patch_ternary(model, train, validation, config, stats_out=stats)
    assert isinstance(model.layers[0].attn, QuantLinear)
    assert all(row["steps_completed"] == 0 for row in stats["ternary_train"]["blocks"].values())


def test_next_block_capture_uses_deployed_prefix(monkeypatch):
    model, config, train, validation = _fixture(steps=0)
    source = copy.deepcopy(model)
    observed = {}
    original = training.train_ternary_block
    def recording(block, name, train_calls, validation_calls, config, **kwargs):
        observed[name] = train_calls[0].args[0].clone()
        return original(block, name, train_calls, validation_calls, config, **kwargs)
    monkeypatch.setattr(training, "train_ternary_block", recording)
    train_and_patch_ternary(model, train, validation, config)
    hidden = model.embed(train[0]["input_ids"])
    assert torch.equal(observed["layers.1"], model.layers[0](hidden))
    assert not torch.equal(observed["layers.1"], source.layers[0](hidden))


def test_completed_resume_reproduces_exact_packed_logits(tmp_path):
    model, config, train, validation = _fixture(tmp_path, steps=2)
    fresh = copy.deepcopy(model)
    train_and_patch_ternary(model, train, validation, config)
    train_and_patch_ternary(fresh, train, validation, replace(config, resume=True))
    assert torch.equal(model(**validation[0]).logits, fresh(**validation[0]).logits)
    state = torch.load(tmp_path / "layers.0.pt", weights_only=True)
    assert state["complete"]
    assert "optimizer" not in state and "parameters" not in state


def test_partial_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    model, config, train, validation = _fixture(tmp_path, steps=4, early_stopping_patience=0)
    fresh, uninterrupted = copy.deepcopy(model), copy.deepcopy(model)
    original = training._BlockTrainer.update
    def interrupt(self):
        if self.step == 2:
            raise RuntimeError("simulated interruption")
        original(self)
    monkeypatch.setattr(training._BlockTrainer, "update", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        train_and_patch_ternary(model, train, validation, config)
    monkeypatch.setattr(training._BlockTrainer, "update", original)
    train_and_patch_ternary(fresh, train, validation, replace(config, resume=True))
    train_and_patch_ternary(uninterrupted, train, validation, replace(config, checkpoint_dir=None))
    assert torch.equal(fresh(**validation[0]).logits, uninterrupted(**validation[0]).logits)
    completed = torch.load(tmp_path / "layers.0.pt", weights_only=True)
    assert completed["stats"]["optimiser_seen_tokens"] == 16


def test_early_stopped_partial_resume_does_not_update_and_restores_initial_best(tmp_path, monkeypatch):
    model, config, train, validation = _fixture(tmp_path, steps=5, early_stopping_patience=1)
    fresh, baseline = copy.deepcopy(model), copy.deepcopy(model)
    original_save = training.save_training_state
    def worsen(self):
        with torch.no_grad():
            for name, parameter in self.student.named_parameters():
                if name.endswith("log_scales"):
                    parameter.add_(4)
        self.step += 1
    def interrupt_completion(path, state):
        if state["complete"]:
            raise RuntimeError("completion interrupted")
        original_save(path, state)
    monkeypatch.setattr(training._BlockTrainer, "update", worsen)
    monkeypatch.setattr(training, "save_training_state", interrupt_completion)
    with pytest.raises(RuntimeError, match="completion"):
        train_and_patch_ternary(model, train, validation, replace(config, max_blocks=1))
    partial = torch.load(tmp_path / "layers.0.pt", weights_only=True)
    assert partial["stale"] == 1 and partial["best_step"] == 0
    def forbidden(_):
        raise AssertionError("early-stopped resume took another optimiser step")
    monkeypatch.setattr(training._BlockTrainer, "update", forbidden)
    monkeypatch.setattr(training, "save_training_state", original_save)
    train_and_patch_ternary(fresh, train, validation, replace(config, resume=True, max_blocks=1))
    train_and_patch_ternary(baseline, train, validation,
                           replace(config, steps=0, checkpoint_dir=None, max_blocks=1))
    assert torch.equal(fresh(**validation[0]).logits, baseline(**validation[0]).logits)


@pytest.mark.parametrize("changed", ["source", "data", "config", "source_id"])
def test_resume_fails_closed_on_identity_changes(tmp_path, changed):
    model, config, train, validation = _fixture(tmp_path, steps=0, max_blocks=1)
    fresh = copy.deepcopy(model)
    train_and_patch_ternary(model, train, validation, config)
    config = replace(config, resume=True)
    if changed == "source":
        with torch.no_grad():
            fresh.layers[0].mlp[0].weight[0, 0] += 1
    elif changed == "data":
        train[0]["input_ids"][0, 0] = 19
    elif changed == "config":
        config = replace(config, lr=0.2)
    else:
        config = replace(config, source_id="other-revision")
    with pytest.raises(ValueError, match="identity mismatch"):
        train_and_patch_ternary(fresh, train, validation, config)


def test_validation_overlap_and_short_loaders_fail():
    model, config, train, validation = _fixture(steps=0)
    with pytest.raises(ValueError, match="overlap"):
        train_and_patch_ternary(model, train, train[:1], config)
    with pytest.raises(ValueError, match="required 2"):
        train_and_patch_ternary(model, train[:1], validation, config)
    validation[0]["attention_mask"] = torch.tensor([[1, 1, 0, 0]])
    with pytest.raises(ValueError, match="unpadded"):
        train_and_patch_ternary(model, train, validation, config)


def test_block_checkpoint_requires_explicit_resume(tmp_path):
    model, config, train, validation = _fixture(tmp_path, steps=0, max_blocks=1)
    fresh = copy.deepcopy(model)
    train_and_patch_ternary(model, train, validation, config)
    with pytest.raises(FileExistsError, match="enable resume"):
        train_and_patch_ternary(fresh, train, validation, config)


def test_low_level_empty_calls_fail():
    source = ToyBlock()
    with pytest.raises(ValueError, match="nonempty"):
        training.train_ternary_block(source, "block", [], [], TernaryTrainConfig())


def test_block_capture_and_training_preserve_source_weights():
    model, config, train, validation = _fixture(steps=2, max_blocks=1)
    source = model.layers[0]
    weights = copy.deepcopy(source.state_dict())
    calls = collect_block_calls(model, train + validation, "cpu",
                                blocks=[("layers.0", source)], max_batches=3,
                                storage_dtype=torch.float32)["layers.0"]
    training.train_ternary_block(source, "layers.0", calls[:2], calls[2:], config)
    for name, value in source.state_dict().items():
        assert torch.equal(value, weights[name])


def test_mlp_qat_preserves_existing_packed_attention_and_hashes_its_state(tmp_path):
    model, config, train, validation = _fixture(tmp_path, steps=2, scope="mlp")
    for block in model.layers:
        source = block.attn
        quantizer = Quantizer(QuantConfig(bits=4, group_size=4, scale_bits=8))
        block.attn = QuantLinear(quantizer.quantize_weight(source.weight), Identity(8), source.bias)
    original = copy.deepcopy(model)
    original_hash = source_state_digest(model.layers[0])
    model.layers[0].attn.qweight.packed.data[0] ^= 1
    assert source_state_digest(model.layers[0]) != original_hash
    model.layers[0].attn.qweight.packed.data[0] ^= 1
    model.layers[0].attn.qweight.scale_steps[0] += 0.01
    assert source_state_digest(model.layers[0]) != original_hash
    model = copy.deepcopy(original)
    train_and_patch_ternary(model, train, validation, config)
    for actual, expected in zip(model.layers, original.layers):
        assert torch.equal(actual.attn.qweight.packed.data, expected.attn.qweight.packed.data)
        assert torch.equal(actual.attn.qweight.scales, expected.attn.qweight.scales)
        assert torch.equal(actual.attn.qweight.scale_steps, expected.attn.qweight.scale_steps)
        assert torch.equal(actual.attn.qweight.scale_offsets, expected.attn.qweight.scale_offsets)
        assert actual.mlp[0].qweight.codebook.levels == 3
