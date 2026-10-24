from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rotquant.block_train import (
    BlockRotationTrainConfig,
    _teacher_call_loss,
    collect_teacher_calls,
    distillation_loss_contract,
)
from rotquant.teacher_targets import capture_union_targets, dense_kl_audit, distillation_loss


def fixture(topk=2):
    torch.manual_seed(20)
    teacher = torch.randn(2, 5, 9)
    ids = torch.tensor([[1, 8, 3, 7, 8], [1, 2, 8, 8, 8]])
    return teacher, ids, capture_union_targets(teacher, topk, 2, (7, 8), ids)


def test_union_tracks_each_next_response_id_and_required_stops_without_double_counting():
    _, ids, target = fixture()
    for row in range(2):
        for pos in range(4):
            active = target.indices[row, pos][target.valid_mask[row, pos]].tolist()
            assert len(active) == len(set(active))
            assert {7, 8, ids[row, pos + 1].item()} <= set(active)
    mass = target.log_probs.exp().sum(-1) + target.tail_log_prob.exp()
    torch.testing.assert_close(mass, torch.ones_like(mass))


def test_union_dense_limit_has_exact_loss_and_gradient_with_masked_prompt_padding():
    teacher, ids, target = fixture(9)
    student = (teacher + 0.3 * torch.randn_like(teacher)).requires_grad_()
    response = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    loss = distillation_loss(student, target, ids, response_mask=response,
                             attention_mask=attention, ce_weight=0)
    logp, logq = (value.log_softmax(-1) for value in (teacher / 2, student / 2))
    kl = (logp.exp() * (logp - logq)).sum(-1)[:, :-1]
    mask = (response * attention)[:, 1:]
    expected = 4 * (kl * mask).sum() / mask.sum()
    actual_grad, = torch.autograd.grad(loss, student, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, student)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
    assert torch.count_nonzero(actual_grad[mask.new_tensor([[1, 0, 0, 0, 1],
                                                          [1, 0, 1, 1, 1]]).bool()]) == 0


def test_union_keeps_stop_identity_hidden_by_old_tail_bucket():
    teacher = torch.tensor([[[4., 3., 1., -2.], [4., 3., 1., -2.]]])
    student = teacher.clone()
    student[..., 2:] = student[..., 2:].flip(-1)
    ids = torch.tensor([[0, 0]])
    coarse = capture_union_targets(teacher, 2, 1)
    protected = capture_union_targets(teacher, 2, 1, (3,))
    old = distillation_loss(student, coarse, ids, ce_weight=0)
    new = distillation_loss(student, protected, ids, ce_weight=0)
    assert old.abs() < 1e-6 and new > 0.05
    audit = dense_kl_audit(student, teacher, protected)
    assert audit["dense_kl"] == pytest.approx(new.item(), abs=1e-6)
    assert abs(audit["unresolved_tail_kl"]) < 1e-6


def test_identical_union_loss_and_all_masked_gradients_are_finite():
    teacher, ids, target = fixture()
    student = teacher.clone().requires_grad_()
    loss = distillation_loss(student, target, ids, ce_weight=0)
    assert loss.abs() < 1e-6
    zero = distillation_loss(student, target, ids, response_mask=torch.zeros_like(ids))
    grad, = torch.autograd.grad(zero, student)
    assert zero == 0 and torch.isfinite(grad).all() and grad.count_nonzero() == 0


def test_union_rejects_tampered_mask_or_required_ids():
    teacher, ids, target = fixture()
    broken = replace(target, valid_mask=torch.ones_like(target.valid_mask))
    with pytest.raises(ValueError, match="duplicate"):
        broken.validate()
    with pytest.raises(ValueError, match="exceeds vocabulary"):
        capture_union_targets(teacher, 2, 1, (9,), ids)


class StrictModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.values = nn.Parameter(logits)

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache
        return SimpleNamespace(logits=self.values)


def test_capture_and_legacy_recovery_integrate_response_union_without_model_metadata_kwargs():
    teacher, ids, _ = fixture()
    model = StrictModel(teacher.clone())
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "response_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])}
    call, = collect_teacher_calls(model, [batch], "cpu", 1, topk=2, temperature=2,
                                  required_ids=(7, 8), include_response_ids=True)
    assert call.inputs["response_mask"].equal(batch["response_mask"])
    assert call.topk_valid_mask is not None
    config = BlockRotationTrainConfig(distill_teacher_required_ids=(7, 8),
                                      distill_teacher_include_response_ids=True)
    loss = _teacher_call_loss(model, call, "cpu", torch.float32, config)
    assert torch.isfinite(loss)
    contract = distillation_loss_contract(config)
    assert contract["target"] == "full-mass-topk-union-tail-v1"
    assert distillation_loss_contract(BlockRotationTrainConfig())["target"] == "full-mass-topk-tail-v1"
