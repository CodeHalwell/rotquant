"""Full-mass sparse teacher distributions with explicit response and stop identities."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SparseTeacherTarget:
    indices: torch.Tensor
    log_probs: torch.Tensor
    valid_mask: torch.Tensor
    tail_log_prob: torch.Tensor
    temperature: float
    vocabulary_size: int
    contract: str = "full-mass-topk-union-tail-v1"

    def validate(self) -> None:
        if self.indices.shape != self.log_probs.shape or self.indices.shape != self.valid_mask.shape:
            raise ValueError("teacher union shapes disagree")
        if self.indices.shape[:-1] != self.tail_log_prob.shape:
            raise ValueError("teacher tail shape disagrees")
        if self.indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("teacher indices must be integers")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("teacher union mask must be boolean")
        if (self.indices < 0).any() or (self.indices >= self.vocabulary_size).any():
            raise ValueError("teacher indices exceed vocabulary")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("teacher temperature must be finite and positive")
        validate_union_mass(self)


def validate_union_mass(target: SparseTeacherTarget) -> None:
    selected = target.indices.masked_fill(~target.valid_mask, target.vocabulary_size)
    ordered = selected.sort(-1).values
    repeated = ordered[..., 1:] == ordered[..., :-1]
    if (repeated & (ordered[..., 1:] < target.vocabulary_size)).any():
        raise ValueError("teacher union contains duplicate active IDs")
    if not torch.isneginf(target.log_probs[~target.valid_mask]).all():
        raise ValueError("inactive teacher slots must have zero mass")
    buckets = torch.cat((target.log_probs, target.tail_log_prob.unsqueeze(-1)), -1)
    mass = torch.logsumexp(buckets.float(), -1)
    if not torch.allclose(mass, torch.zeros_like(mass), atol=2e-5):
        raise ValueError("teacher union must preserve full probability mass")
    full = target.valid_mask.sum(-1) == target.vocabulary_size
    if not torch.isneginf(target.tail_log_prob[full]).all():
        raise ValueError("full-vocabulary teacher unions cannot have tail mass")


def union_indices(log_probs, topk, required_ids, response_ids):
    if topk < 1:
        raise ValueError("teacher topk must be positive")
    ids = tuple(sorted({int(x) for x in required_ids}))
    if any(x < 0 or x >= log_probs.shape[-1] for x in ids):
        raise ValueError("required teacher ID exceeds vocabulary")
    parts = [log_probs.topk(min(topk, log_probs.shape[-1]), -1).indices]
    required = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
    parts.append(required.expand(*log_probs.shape[:-1], len(ids)))
    if response_ids is not None:
        if response_ids.shape != log_probs.shape[:-1]:
            raise ValueError("response IDs must match teacher sequence positions")
        response = response_ids.to(log_probs.device, torch.long)
        if (response < 0).any() or (response >= log_probs.shape[-1]).any():
            raise ValueError("response ID exceeds vocabulary")
        next_ids = torch.cat((response[..., 1:], response[..., -1:]), -1)
        parts.append(next_ids.unsqueeze(-1))
    indices = torch.cat(parts, -1).sort(-1).values
    valid = torch.ones_like(indices, dtype=torch.bool)
    valid[..., 1:] = indices[..., 1:] != indices[..., :-1]
    return indices, valid


def remainder_log_probability(log_probs, indices, valid_mask):
    remainder = log_probs.scatter(-1, indices, -torch.inf)
    full = valid_mask.sum(-1) == log_probs.shape[-1]
    safe = remainder.masked_fill(full.unsqueeze(-1), 0)
    return safe.logsumexp(-1).masked_fill(full, -torch.inf)


def capture_union_targets(logits, topk, temperature, required_ids=(), response_ids=None):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("teacher temperature must be finite and positive")
    if logits.ndim != 3 or not torch.isfinite(logits).all():
        raise ValueError("teacher logits must be finite batch/sequence/vocabulary tensors")
    with torch.no_grad():
        log_probs = F.log_softmax(logits.float() / temperature, -1)
        indices, valid = union_indices(log_probs, topk, required_ids, response_ids)
        selected = log_probs.gather(-1, indices).masked_fill(~valid, -torch.inf)
        tail = remainder_log_probability(log_probs, indices, valid)
    target = SparseTeacherTarget(indices.cpu().to(torch.int32), selected.cpu(), valid.cpu(),
                                 tail.cpu(), float(temperature), logits.shape[-1])
    target.validate()
    return target


def sparse_teacher_kl(student_log_probs, target):
    target.validate()
    if student_log_probs.shape[:-1] != target.tail_log_prob.shape:
        raise ValueError("teacher and student sequence shapes disagree")
    if student_log_probs.shape[-1] != target.vocabulary_size:
        raise ValueError("teacher and student vocabulary sizes disagree")
    device = student_log_probs.device
    indices = target.indices.to(device, torch.long)
    valid = target.valid_mask.to(device)
    student = student_log_probs.gather(-1, indices).masked_fill(~valid, -torch.inf)
    student_tail = remainder_log_probability(student_log_probs, indices, valid)
    student = torch.cat((student, student_tail.unsqueeze(-1)), -1)
    teacher = torch.cat((target.log_probs.to(device), target.tail_log_prob.to(device).unsqueeze(-1)), -1)
    positive = teacher.exp() > 0
    ratio = teacher.masked_fill(~positive, 0) - student.masked_fill(~positive, 0)
    return (teacher.exp() * ratio).sum(-1)


def next_token_mask(input_ids, response_mask=None, attention_mask=None):
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for value in (response_mask, attention_mask):
        if value is not None:
            if value.shape != input_ids.shape:
                raise ValueError("response/attention mask must match input IDs")
            mask = mask & value.to(input_ids.device, torch.bool)
    return mask[:, 1:].float()


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def distillation_loss(student_logits, target, input_ids, *, response_mask=None,
                      attention_mask=None, kl_weight=1.0, ce_weight=0.1):
    if kl_weight < 0 or ce_weight < 0 or kl_weight + ce_weight <= 0:
        raise ValueError("distillation weights must be nonnegative and nonzero")
    labels = input_ids.to(student_logits.device)[:, 1:]
    mask = next_token_mask(input_ids, response_mask, attention_mask).to(student_logits.device)
    student = student_logits.float()
    loss = student.sum() * 0
    if kl_weight:
        log_probs = F.log_softmax(student / target.temperature, -1)
        loss = loss + kl_weight * target.temperature ** 2 * masked_mean(
            sparse_teacher_kl(log_probs, target)[:, :-1], mask)
    if ce_weight:
        ce = F.cross_entropy(student[:, :-1].reshape(-1, student.shape[-1]),
                             labels.reshape(-1), reduction="none").reshape_as(labels)
        loss = loss + ce_weight * masked_mean(ce, mask)
    return loss


def dense_kl_audit(student_logits, teacher_logits, target, *, response_mask=None,
                   attention_mask=None):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("dense audit requires identical teacher/student shapes")
    target.validate()
    temperature = target.temperature
    student = F.log_softmax(student_logits.float() / temperature, -1)
    teacher = F.log_softmax(teacher_logits.float() / temperature, -1)
    dense = (teacher.exp() * (teacher - student)).sum(-1)[:, :-1]
    coarse = sparse_teacher_kl(student, target)[:, :-1]
    ids = torch.zeros(student.shape[:-1], device=student.device, dtype=torch.long)
    mask = next_token_mask(ids, response_mask, attention_mask)
    return {"dense_kl": float(masked_mean(dense, mask).detach()),
            "coarsened_kl": float(masked_mean(coarse, mask).detach()),
            "unresolved_tail_kl": float(masked_mean(dense - coarse, mask).detach()),
            "positions": int(mask.sum().item()), "temperature": temperature}
