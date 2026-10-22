"""Recovery preserves teacher mass and the exact fixed-code packed representation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from rotquant.block_train import (
    BlockRotationTrainConfig,
    TeacherCall,
    _load_distill_checkpoint,
    _teacher_call_loss,
    capture_topk_teacher,
    collect_teacher_calls,
    distill_packed_model,
    distillation_loss_contract,
)
from rotquant.linear import QuantLinear
from rotquant.native_v3 import encode_native_v3
from rotquant.quantize import QuantConfig
from rotquant.rotate import RandomizedHadamard


class LogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(logits, dtype=torch.float32).clone())

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.values)


def loss_config(temperature=1.0, **kwargs):
    return BlockRotationTrainConfig(
        distill_temperature=temperature, distill_ce_weight=0, **kwargs)


def teacher_fixture(logits, topk=2, temperature=1.0):
    values = torch.tensor(logits, dtype=torch.float32).view(1, 1, -1).repeat(1, 4, 1)
    inputs = {"input_ids": torch.arange(4).remainder(values.shape[-1]).view(1, -1)}
    return LogitModel(values), capture_topk_teacher(inputs, values, topk, temperature)


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("topk", [1, 2, 4, 9])
def test_identical_teacher_student_has_zero_loss_and_gradient(temperature, topk):
    model, call = teacher_fixture([2, 1, 0, -1], topk, temperature)
    loss = _teacher_call_loss(model, call, "cpu", torch.float32, loss_config(temperature))
    loss.backward()
    assert abs(loss.item()) < 1e-7
    torch.testing.assert_close(model.values.grad, torch.zeros_like(model.values), atol=2e-7, rtol=0)


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("topk", [1, 3, 4])
def test_extreme_logits_keep_loss_and_tail_gradients_finite(temperature, topk):
    model, call = teacher_fixture([20000, 1000, -1000, -20000], topk, temperature)
    with torch.no_grad():
        model.values[..., 0] = -20000
        model.values[..., -1] = 20000
    loss = _teacher_call_loss(model, call, "cpu", torch.float32, loss_config(temperature))
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(model.values.grad).all()


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_full_vocabulary_matches_dense_kl_and_gradient(temperature):
    model, call = teacher_fixture([2, 1, 0, -1], 4, temperature)
    dense = TeacherCall(inputs=call.inputs, logits=model.values.detach().clone())
    with torch.no_grad():
        model.values.add_(torch.tensor([0.3, -0.1, 0.8, -0.4]))
    config = loss_config(temperature)
    sparse_loss = _teacher_call_loss(model, call, "cpu", torch.float32, config)
    sparse_grad, = torch.autograd.grad(sparse_loss, model.values)
    dense_loss = _teacher_call_loss(model, dense, "cpu", torch.float32, config)
    dense_grad, = torch.autograd.grad(dense_loss, model.values)
    torch.testing.assert_close(sparse_loss, dense_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(sparse_grad, dense_grad, atol=1e-7, rtol=1e-5)


def test_topk_tail_kl_matches_manually_aggregated_probabilities():
    model, call = teacher_fixture([2, 1, 0, -1])
    with torch.no_grad():
        model.values.add_(torch.tensor([0.3, -0.1, 0.8, -0.4]))
    teacher = torch.tensor([2., 1., 0., -1.]).softmax(-1)
    student = model.values[0, 0].softmax(-1)
    p = torch.cat((teacher[:2], teacher[2:].sum().view(1)))
    q = torch.cat((student[:2], student[2:].sum().view(1)))
    expected = (p * (p.log() - q.log())).sum()
    actual = _teacher_call_loss(model, call, "cpu", torch.float32, loss_config())
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


def test_tail_redistribution_is_honestly_a_coarsened_loss():
    model, call = teacher_fixture([2, 1, 0, -1])
    dense = TeacherCall(inputs=call.inputs, logits=model.values.detach().clone())
    with torch.no_grad():
        model.values[..., 2:] = torch.tensor([-1., 0.])
    coarse = _teacher_call_loss(model, call, "cpu", torch.float32, loss_config())
    full = _teacher_call_loss(model, dense, "cpu", torch.float32, loss_config())
    assert abs(coarse.item()) < 1e-7 and full.item() > 0.01


def test_mask_and_next_token_shift_apply_to_kl_and_ce():
    model, call = teacher_fixture([2, 1, 0, -1])
    call.inputs["attention_mask"] = torch.tensor([[1, 1, 0, 0]])
    with torch.no_grad():
        model.values.add_(torch.tensor([0.3, -0.1, 0.8, -0.4]))
        model.values[:, 1:].mul_(10)
    config = replace(loss_config(), distill_ce_weight=0.3)
    loss = _teacher_call_loss(model, call, "cpu", torch.float32, config)
    one = replace(call, inputs={"input_ids": call.inputs["input_ids"][:, :2]},
                  topk_indices=call.topk_indices[:, :2],
                  topk_log_probs=call.topk_log_probs[:, :2],
                  tail_log_prob=call.tail_log_prob[:, :2])
    first_model = LogitModel(model.values[:, :2].detach())
    first_kl = _teacher_call_loss(first_model, one, "cpu", torch.float32, loss_config())
    expected = first_kl + 0.3 * F.cross_entropy(model.values[:, 0], call.inputs["input_ids"][:, 1])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.values.grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(model.values.grad[:, 1:]) == 0


def test_all_masked_positions_produce_zero_finite_loss_and_gradient():
    model, call = teacher_fixture([2, 1, 0, -1])
    call.inputs["attention_mask"] = torch.zeros(1, 4, dtype=torch.long)
    loss = _teacher_call_loss(model, call, "cpu", torch.float32,
                              replace(loss_config(), distill_ce_weight=0.1))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(model.values.grad, torch.zeros_like(model.values))


def test_topk_capture_keeps_fp32_probability_mass_at_requested_temperature():
    model, reference = teacher_fixture([2, 1, 0, -1], temperature=2)
    call, = collect_teacher_calls(model, [reference.inputs], "cpu", 1,
                                  storage_dtype=torch.float16, topk=2, temperature=2)
    assert call.temperature == 2 and call.vocabulary_size == 4
    assert call.topk_log_probs.dtype == call.tail_log_prob.dtype == torch.float32
    assert call.topk_logits is None and call.logits is None
    mass = call.topk_log_probs.exp().sum(-1) + call.tail_log_prob.exp()
    torch.testing.assert_close(mass, torch.ones_like(mass))


def test_legacy_targets_and_temperature_changes_fail_closed():
    model, call = teacher_fixture([2, 1, 0, -1])
    legacy = TeacherCall(inputs=call.inputs, topk_indices=call.topk_indices,
                         topk_logits=model.values.detach()[..., :2])
    with pytest.raises(ValueError, match="recapture"):
        _teacher_call_loss(model, legacy, "cpu", torch.float32, loss_config())
    with pytest.raises(ValueError, match="temperature changed"):
        _teacher_call_loss(model, call, "cpu", torch.float32, loss_config(2))


def test_invalid_probability_mass_fails_closed():
    model, call = teacher_fixture([2, 1, 0, -1])
    broken = replace(call, topk_log_probs=F.log_softmax(call.topk_log_probs, dim=-1))
    with pytest.raises(ValueError, match="preserve total mass"):
        _teacher_call_loss(model, broken, "cpu", torch.float32, loss_config())


def test_recovery_checkpoints_reject_legacy_objectives_and_changed_temperature(tmp_path):
    path = tmp_path / "recovery.pt"
    contract = distillation_loss_contract(loss_config())
    torch.save({"format": "rotquant-distill-v1"}, path)
    with pytest.raises(ValueError, match="legacy"):
        _load_distill_checkpoint(str(path), contract)
    torch.save({"format": "rotquant-distill-v2", "loss_contract": contract}, path)
    assert _load_distill_checkpoint(str(path), contract)["loss_contract"] == contract
    with pytest.raises(ValueError, match="loss changed"):
        _load_distill_checkpoint(str(path), {**contract, "temperature": 2})


class PackedToyLM(nn.Module):
    def __init__(self, scale_bits):
        super().__init__()
        self.embedding = nn.Embedding(16, 128)
        self.head = nn.Linear(128, 16, bias=False)
        self.scale_bits = scale_bits

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    def quantize(self):
        rotation = RandomizedHadamard(128, seed=0)
        self.head = QuantLinear.from_linear(
            self.head, QuantConfig(bits=4, group_size=128, error_comp="gptq",
                                   scale_bits=self.scale_bits, scale_quant_group_size=4),
            weight_rotation=rotation, H=torch.eye(128))


def packed_recovery_fixture(scale_bits):
    torch.manual_seed(11)
    model = PackedToyLM(scale_bits)
    batches = [{"input_ids": torch.randint(0, 16, (1, 6))} for _ in range(3)]
    calls = collect_teacher_calls(model, batches, "cpu", 3, topk=4, temperature=2)
    model.quantize()
    config = loss_config(2, distill_steps=2, distill_train_rotations=False,
                         distill_train_scales=True, distill_early_stopping_patience=0,
                         distill_teacher_topk=4)
    return model, calls, config


@pytest.mark.parametrize("scale_bits", [8, 16])
@pytest.mark.parametrize("reject", [False, True])
def test_scale_only_gptq_recovery_keeps_codes_and_exact_deployed_selection(scale_bits, reject):
    model, calls, config = packed_recovery_fixture(scale_bits)
    config = replace(config, distill_selection_min_improvement=0.9999 if reject else 0)
    codes = model.head.qweight.packed.data.clone()
    original = encode_native_v3(model.head.qweight).to_bytes()
    byte_count = model.head.packed_state_bytes()
    stats = distill_packed_model(model, calls, "cpu", torch.float32, config)
    final_selection = _teacher_call_loss(model, calls[-1], "cpu", torch.float32, config).item()
    expected = stats["selection_candidate_loss"] if stats["accepted"] else stats["selection_reference_loss"]
    assert final_selection == pytest.approx(expected, abs=1e-7)
    assert torch.equal(model.head.qweight.packed.data, codes)
    assert model.head.packed_state_bytes() == byte_count
    assert model.head._log_scale_multiplier is None
    assert stats["accepted"] is not reject
    assert stats["selection_kl_objective"] == "coarsened_topk_tail_kl"
    encode_native_v3(model.head.qweight).validate()
    if reject:
        assert encode_native_v3(model.head.qweight).to_bytes() == original


def test_gptq_refresh_is_rejected_before_recovery_mutation():
    model, calls, config = packed_recovery_fixture(8)
    packed = encode_native_v3(model.head.qweight).to_bytes()
    with pytest.raises(ValueError, match="calibrated Hessians and packed-state rollback"):
        distill_packed_model(model, calls, "cpu", torch.float32,
                             replace(config, distill_code_refresh_interval=1))
    assert model.head._log_scale_multiplier is None
    assert encode_native_v3(model.head.qweight).to_bytes() == packed


def test_legacy_resume_is_rejected_before_recovery_mutation(tmp_path):
    model, calls, config = packed_recovery_fixture(8)
    path = tmp_path / "legacy.pt"
    torch.save({"format": "rotquant-distill-v1"}, path)
    with pytest.raises(ValueError, match="legacy"):
        distill_packed_model(model, calls, "cpu", torch.float32, replace(
            config, distill_resume=True, distill_checkpoint_path=str(path)))
    assert model.head._log_scale_multiplier is None


@pytest.mark.parametrize("failure", ["loss", "gradient", "evaluation"])
def test_recovery_rejects_nonfinite_values_before_optimizer_step(monkeypatch, failure):
    from rotquant import block_train

    model, calls, config = packed_recovery_fixture(16)
    original = encode_native_v3(model.head.qweight).to_bytes()
    real_loss = block_train._teacher_call_loss

    def injected_loss(*args, **kwargs):
        loss = real_loss(*args, **kwargs)
        if failure == "evaluation":
            return loss * float("nan")
        if torch.is_grad_enabled():
            if failure == "loss":
                return loss * float("nan")
            loss.register_hook(lambda gradient: gradient * float("nan"))
        return loss

    def forbidden_step(*args, **kwargs):
        pytest.fail("optimizer must not consume non-finite values")

    monkeypatch.setattr(block_train, "_teacher_call_loss", injected_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_step)
    with pytest.raises(FloatingPointError, match="non-finite"):
        distill_packed_model(model, calls, "cpu", torch.float32, config)
    assert encode_native_v3(model.head.qweight).to_bytes() == original


def test_scale8_initial_training_and_commit_preserve_exact_payload():
    model, _, _ = packed_recovery_fixture(8)
    original = encode_native_v3(model.head.qweight).to_bytes()
    weight = model.head.qweight.dequantize().clone()
    model.head.enable_scale_finetuning()
    assert torch.equal(model.head._weight(), weight)
    model.head.commit_scale_finetuning()
    assert encode_native_v3(model.head.qweight).to_bytes() == original


@pytest.mark.parametrize("scale_bits", [8, 16])
@pytest.mark.parametrize("reject", [False, True])
def test_early_stopped_resume_preserves_selected_payload(scale_bits, reject, tmp_path, monkeypatch):
    path = tmp_path / "recovery.pt"
    outcomes = []
    for attempt in range(2):
        model, calls, config = packed_recovery_fixture(scale_bits)
        original = encode_native_v3(model.head.qweight).to_bytes()
        config = replace(config, distill_steps=5, distill_early_stopping_patience=1,
                         distill_validation_interval=1, distill_checkpoint_interval=1,
                         distill_checkpoint_path=str(path), distill_resume=True,
                         distill_selection_min_improvement=.9999 if reject else 0)
        validation = iter([1., .5, .5])

        def fixed_validation(model, selected, *args, validation=validation, calls=calls):
            return next(validation, .5) if selected[0] is calls[1] else 1.

        monkeypatch.setattr("rotquant.block_train._distill_eval_loss", fixed_validation)
        stats = distill_packed_model(model, calls, "cpu", torch.float32, config)
        deployed = encode_native_v3(model.head.qweight).to_bytes()
        outcomes.append(deployed)
        assert stats["steps_run"] == 2 and stats["best_step"] == 1
        assert stats["stopped_early"] and stats["resumed"] is bool(attempt)
        assert stats["accepted"] is not reject
        if reject:
            assert deployed == original
    assert outcomes[0] == outcomes[1]
    checkpoint = torch.load(path, weights_only=True)
    assert any(torch.count_nonzero(parameter) for parameter in checkpoint["best_snapshot"])
