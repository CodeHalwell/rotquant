from __future__ import annotations

import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rotquant.block_train import BlockCall, TeacherCall
from rotquant.linear import QuantLinear
from rotquant.teacher_targets import capture_union_targets, distillation_loss
from rotquant.ternary import TernaryConfig, TernaryLinear, packed_ternary_state
from rotquant.ternary_global import GlobalTernaryTrainer, TernaryGlobalConfig, train_global_ternary
from rotquant.ternary_global_data import chunked_loss
from rotquant.ternary_initialise import activation_aware_initialise
from rotquant.ternary_latents import retained_site_latents
from rotquant.ternary_state import source_state_digest
from rotquant.ternary_train import TernaryTrainConfig, train_ternary_block


class GlobalToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 8)
        self.embed.weight.data.copy_(torch.ones_like(self.embed.weight))
        self.proj = TernaryLinear(nn.Linear(8, 8, bias=False), TernaryConfig(4, 4)).to_quant_linear()
        self.protected = nn.Linear(8, 8)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def example(ids):
    inputs = {"input_ids": torch.tensor([ids]),
              "response_mask": torch.tensor([[False] + [True] * (len(ids) - 1)])}
    teacher = torch.zeros(1, len(ids), 8)
    teacher[:, :, 4] = 10
    return TeacherCall(inputs=inputs, logits=teacher)


def behaviour(_model):
    return {"data_id": "behaviour", "examples": 2, "completion_rate": 1.,
            "repetition_rate": 0., "format_rate": 1.}


def fixture(tmp_path=None, steps=12):
    torch.manual_seed(81)
    model = GlobalToy().eval()
    config = TernaryGlobalConfig(steps=steps, lr=.2, scale_lr=.02, required_ids=(7,),
                                 validation_interval=1, checkpoint_interval=1,
                                 early_stopping_patience=0, source_id="source", data_id="train",
                                 behaviour_data_id="behaviour", gradient_checkpointing=False,
                                 ce_weight=0, topk=8, loss_chunk_tokens=2,
                                 checkpoint_dir=None if tmp_path is None else str(tmp_path))
    return model, config, [example([0, 1, 2, 3])], [example([0, 2, 3, 4, 5])]


def test_global_changes_assignments_and_keeps_nonternary_weights_exact():
    model, config, train, validation = fixture()
    embed, protected = source_state_digest(model.embed), source_state_digest(model.protected)
    stats = {}
    train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour, stats_out=stats)
    result = stats["ternary_global"]
    assert result["accepted"] and result["assignments"]["changed"] > 0
    assert result["selected_validation_loss"] < result["initial_validation_loss"]
    assert result["initialisation"]["proj"] == "packed_reconstruction"
    assert source_state_digest(model.embed) == embed
    assert source_state_digest(model.protected) == protected
    assert isinstance(model.proj, QuantLinear)
    assert not any(isinstance(module, TernaryLinear) for module in model.modules())


def test_behaviour_rejection_restores_original_assignments_and_scales():
    model, config, train, validation = fixture(steps=3)
    original = packed_ternary_state(model.proj)
    evaluations = 0
    def regress(model):
        nonlocal evaluations
        metrics = behaviour(model)
        metrics["completion_rate"] = 1. if evaluations == 0 else 0.
        evaluations += 1
        return metrics
    stats = {}
    train_global_ternary(model, train, validation, config, behavioural_evaluator=regress, stats_out=stats)
    report = stats["ternary_global"]
    assert not report["accepted"] and report["selection"] == "restored_baseline"
    assert report["best_step"] == 0 and report["assignments"]["changed"] == 0
    assert all(row["rejected_metrics"] == ["completion_rate"] for row in report["history"])
    for name in ("packed", "scales", "signs"):
        assert torch.equal(packed_ternary_state(model.proj)[name], original[name])


def test_global_complete_resume_exact_and_training_artifact_separate(tmp_path):
    model, config, train, validation = fixture(tmp_path, steps=3)
    fresh = copy.deepcopy(model)
    train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour)
    train_global_ternary(fresh, train, validation, replace(config, resume=True),
                         behavioural_evaluator=behaviour)
    assert torch.equal(model(input_ids=validation[0].inputs["input_ids"]).logits,
                       fresh(input_ids=validation[0].inputs["input_ids"]).logits)
    state = torch.load(tmp_path / "global.pt", weights_only=True)
    assert "optimizer" not in state and "parameters" not in state
    assert state["stats"]["selected_latents"]["training_only"]
    latent = torch.load(tmp_path / "selected_latents/global.latents.pt", weights_only=True)
    assert latent["modules"]["proj"]["latent_weight"].dtype == torch.float32


def test_global_partial_resume_matches_uninterrupted_and_restores_on_exception(tmp_path, monkeypatch):
    model, config, train, validation = fixture(tmp_path, steps=4)
    fresh, uninterrupted = copy.deepcopy(model), copy.deepcopy(model)
    original = source_state_digest(model)
    update = GlobalTernaryTrainer.update
    def interrupt(self):
        if self.step == 2:
            raise RuntimeError("interruption")
        update(self)
    monkeypatch.setattr(GlobalTernaryTrainer, "update", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour)
    assert source_state_digest(model) == original
    monkeypatch.setattr(GlobalTernaryTrainer, "update", update)
    train_global_ternary(fresh, train, validation, replace(config, resume=True),
                         behavioural_evaluator=behaviour)
    train_global_ternary(uninterrupted, train, validation, replace(config, checkpoint_dir=None),
                         behavioural_evaluator=behaviour)
    assert source_state_digest(fresh) == source_state_digest(uninterrupted)


@pytest.mark.parametrize("change", ["source", "data", "objective", "behaviour"])
def test_global_resume_rejects_changed_contract(tmp_path, change):
    model, config, train, validation = fixture(tmp_path, steps=0)
    fresh = copy.deepcopy(model)
    train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour)
    config = replace(config, resume=True)
    if change == "source":
        fresh.embed.weight.data[0, 0] += 1
    elif change == "data":
        train = [example([1, 1, 2, 3])]
    elif change == "objective":
        config = replace(config, ce_weight=.2)
    else:
        config = replace(config, behaviour_data_id="changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        train_global_ternary(fresh, train, validation, config, behavioural_evaluator=behaviour)


def test_retained_latents_require_exact_original_packed_forward():
    model, _, _, _ = fixture(steps=0)
    trainable = TernaryLinear.from_quant_linear(model.proj)
    state = {"latent_weight": trainable.latent_weight.detach().clone(),
             "log_scales": trainable.log_scales.detach().clone()}
    restored = TernaryLinear.from_quant_linear(model.proj, state)
    values = torch.randn(2, 8)
    assert torch.equal(model.proj(values), restored(values))
    state["latent_weight"].zero_()
    with pytest.raises(ValueError, match="reproduce"):
        TernaryLinear.from_quant_linear(model.proj, state)


def test_chunked_loss_and_gradient_match_dense_union_objective():
    _, config, train, _ = fixture(steps=0)
    call = train[0]
    logits = torch.randn(1, 4, 8, requires_grad=True)
    target = capture_union_targets(call.logits, 4, 1., required_ids=(7,),
                                    response_ids=call.inputs["input_ids"])
    dense = distillation_loss(logits, target, call.inputs["input_ids"],
                               response_mask=call.inputs["response_mask"], ce_weight=0)
    dense_grad, = torch.autograd.grad(dense, logits)
    chunked = chunked_loss(logits, call, target, config)
    chunked_grad, = torch.autograd.grad(chunked, logits)
    torch.testing.assert_close(dense, chunked)
    torch.testing.assert_close(dense_grad, chunked_grad)


def test_activation_covariance_initialisation_never_increases_weighted_error():
    torch.manual_seed(41)
    module = TernaryLinear(nn.Linear(8, 8), TernaryConfig(4, 4))
    activations = torch.randn(16, 8) @ torch.diag(torch.arange(1., 9.))
    covariance = activations.T @ activations / len(activations)
    report = activation_aware_initialise(module, covariance, iterations=3, row_batch=4)
    assert report["weighted_error_after"] <= report["weighted_error_before"]
    assert set(module.hard_codes().unique().tolist()) <= {-1., 0., 1.}
    assert module.to_quant_linear().qweight.scales.dtype == torch.float16


def test_scale_only_recovery_keeps_every_ternary_assignment_fixed():
    model, config, train, validation = fixture(steps=3)
    stats = {}
    train_global_ternary(model, train, validation, replace(config, learn_assignments=False),
                         behavioural_evaluator=behaviour, stats_out=stats)
    result = stats["ternary_global"]
    assert result["learn_assignments"] is False
    assert result["assignments"]["changed"] == 0
    assert result["accepted"]


def test_selected_block_latents_are_separate_hash_bound_and_reload_exact(tmp_path):
    torch.manual_seed(13)
    source = nn.Sequential(nn.Linear(8, 8)).eval()
    calls = []
    for _ in range(3):
        value = torch.randn(1, 4, 8)
        calls.append(BlockCall((value,), {}, source(value).detach()))
    config = TernaryTrainConfig(quant=TernaryConfig(4, 4), steps=1,
                                checkpoint_dir=str(tmp_path), retain_latents=True,
                                validation_interval=1)
    result, stats = train_ternary_block(source, "block", calls[:2], calls[2:], config)
    model = nn.ModuleDict({"block": result})
    latents, identities = retained_site_latents(model, [stats["selected_latents"]])
    restored = TernaryLinear.from_quant_linear(result[0], latents["block.0"])
    value = calls[-1].args[0]
    assert torch.equal(restored(value), result[0](value)) and identities
    deployed = torch.load(tmp_path / "block.pt", weights_only=True)
    assert "latent_weight" not in repr(deployed["packed"])
    artifact = tmp_path / "selected_latents/block.latents.pt"
    with artifact.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        retained_site_latents(model, [stats["selected_latents"]])


def test_dense_kl_audit_is_independent_periodic_and_bound_to_resume(tmp_path):
    model, config, train, validation = fixture(tmp_path, steps=2)
    fresh = copy.deepcopy(model)
    config = replace(config, topk=2, dense_audit_interval=1)
    audit = copy.deepcopy(validation[0])
    stats = {}
    train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour,
                         dense_audit_calls=[audit], stats_out=stats)
    report = stats["ternary_global"]
    assert report["initial_dense_audit"]["training_use"] is False
    for row in report["history"]:
        assert row["dense_audit"]["positions"] == 4
        assert row["dense_audit"]["dense_kl"] >= row["dense_audit"]["coarsened_kl"] - 1e-5
    audit.logits[0, 0, 1] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        train_global_ternary(fresh, train, validation, replace(config, resume=True),
                             behavioural_evaluator=behaviour, dense_audit_calls=[audit])


def test_half_logits_objective_avoids_fp16_reduction_overflow():
    _, config, train, _ = fixture(steps=0)
    call = train[0]
    logits = torch.full((1, 4, 8), 30000., dtype=torch.float16, requires_grad=True)
    target = capture_union_targets(call.logits, 8, 1., required_ids=(7,))
    loss = chunked_loss(logits, call, target, config)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_global_leaves_existing_packed_w4_representation_unchanged():
    from rotquant.quantize import QuantConfig, Quantizer
    from rotquant.rotate import Identity
    model, config, train, validation = fixture(steps=1)
    original = model.protected
    model.protected = QuantLinear(Quantizer(QuantConfig(bits=4, group_size=4)).quantize_weight(
        original.weight), Identity(8), original.bias)
    protected = source_state_digest(model.protected)
    train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour)
    assert source_state_digest(model.protected) == protected


@pytest.mark.parametrize("checkpointing", [False, True])
def test_tiny_real_qwen_hybrid_global_update_has_finite_gradients(checkpointing):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from rotquant._internal import get_parent
    from rotquant.ternary_global_state import vocabulary_state
    from rotquant.ternary_state import content_digest
    from rotquant.vocabulary import VocabularyConfig, install_packed_vocabulary, quantize_vocabulary
    from scripts.preflight_packed_validation import tiny_model
    torch.manual_seed(73)
    model = tiny_model(dtype=torch.float16).eval()
    vocabulary = quantize_vocabulary(model.get_input_embeddings().weight,
                                     VocabularyConfig(bits=6, chunk_rows=64))
    install_packed_vocabulary(model, vocabulary)
    protected_vocabulary = content_digest(vocabulary_state(model))
    calls = []
    for ids in ([1, 2, 3, 4], [5, 6, 7, 8]):
        inputs = {"input_ids": torch.tensor([ids])}
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits.detach()
        inputs["response_mask"] = torch.tensor([[False, True, True, True]])
        calls.append(TeacherCall(inputs=inputs, logits=logits))
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "language_model.layers." in name and ".mlp." in name:
            owner, attribute = get_parent(model, name)
            setattr(owner, attribute, TernaryLinear(module, TernaryConfig(128, 128)).to_quant_linear())
    config = TernaryGlobalConfig(steps=1, required_ids=(7,), source_id="qwen", data_id="tiny",
                                 behaviour_data_id="behaviour", validation_interval=1,
                                 gradient_checkpointing=checkpointing)
    stats = {}
    train_global_ternary(model, calls[:1], calls[1:], config, behavioural_evaluator=behaviour,
                         stats_out=stats)
    report = stats["ternary_global"]
    assert report["steps_completed"] == 1 and report["optimiser_seen_tokens"] == 3
    assert math.isfinite(report["packed_validation_loss"])
    assert content_digest(vocabulary_state(model)) == protected_vocabulary
    assert not any(isinstance(module, TernaryLinear) for module in model.modules())


def test_dense_audit_interval_is_independent_of_validation_interval():
    model, config, train, validation = fixture(steps=5)
    config = replace(config, validation_interval=3, dense_audit_interval=2)
    stats = {}
    train_global_ternary(model, train, validation, config, behavioural_evaluator=behaviour,
                         dense_audit_calls=[example([0, 3, 4, 5, 6])], stats_out=stats)
    history = stats['ternary_global']['history']
    assert [row['step'] for row in history] == [2, 3, 4, 5]
    assert [row['step'] for row in history if 'dense_audit' in row] == [2, 4, 5]
