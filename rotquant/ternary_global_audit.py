"""Bounded, held-out dense KL audits, independent of the sparse training objective."""
from __future__ import annotations

import torch

from .teacher_targets import capture_union_targets, dense_kl_audit
from .ternary_global_data import call_mask, model_logits, sequence_keys
from .ternary_state import content_digest


def prepare_dense_audit(train, calls):
    calls = list(calls)
    if len(calls) > 1:
        raise ValueError("dense KL audit is bounded to one held-out example")
    for call in calls:
        ids = call.inputs["input_ids"]
        if call.logits is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > 2048:
            raise ValueError("dense KL audit requires one dense held-out sequence of at most 2048 tokens")
        if call.logits.shape[:2] != ids.shape or not torch.isfinite(call.logits).all():
            raise ValueError("dense KL audit logits are invalid")
        call_mask(call)
    if sequence_keys(train) & sequence_keys(calls):
        raise ValueError("dense KL audit overlaps training tokens")
    identity = content_digest([{"inputs": call.inputs, "logits": call.logits} for call in calls])
    return calls, identity


@torch.no_grad()
def audit_dense_kl(model, calls, config, device):
    if not calls:
        return None
    was_training = model.training
    model.eval()
    try:
        call = calls[0]
        logits = model_logits(model, call, device)
        total, positions = {"dense_kl": 0., "coarsened_kl": 0., "unresolved_tail_kl": 0.}, 0
        for start in range(0, logits.shape[1] - 1, config.loss_chunk_tokens):
            end = min(start + config.loss_chunk_tokens + 1, logits.shape[1])
            report = audit_chunk(logits[:, start:end], call, start, end, config)
            for name in total:
                total[name] += report[name] * report["positions"]
            positions += report["positions"]
        return {**{name: value / positions for name, value in total.items()},
                "positions": positions, "temperature": config.temperature, "training_use": False}
    finally:
        model.train(was_training)


def audit_chunk(student, call, start, end, config):
    teacher = call.logits[:, start:end].to(student.device)
    ids = call.inputs["input_ids"][:, start:end]
    target = capture_union_targets(teacher, config.topk, config.temperature, config.required_ids, ids)
    return dense_kl_audit(student, teacher, target,
                          response_mask=call_mask(call)[:, start:end])
