"""Masked, sequence-chunked whole-model teacher loss and content identities."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, replace

import torch
from torch.utils.checkpoint import checkpoint

from .block_train import _tree_to
from .teacher_targets import SparseTeacherTarget, capture_union_targets, distillation_loss
from .ternary_state import content_digest


def call_mask(call):
    inputs = call.inputs
    ids = inputs["input_ids"]
    mask = inputs.get("response_mask", torch.ones_like(ids, dtype=torch.bool)).bool()
    mask = mask & inputs.get("attention_mask", torch.ones_like(ids)).bool()
    if mask.shape != ids.shape or not mask[:, 1:].any():
        raise ValueError("global QAT requires nonempty unpadded response targets")
    return mask


def teacher_target(call, config):
    if call.logits is not None:
        return capture_union_targets(call.logits, config.topk, config.temperature,
                                     config.required_ids, call.inputs["input_ids"])
    valid = getattr(call, "topk_valid_mask", None)
    if valid is None:
        raise ValueError("global QAT requires explicit union teacher targets; recapture legacy targets")
    target = SparseTeacherTarget(call.topk_indices, call.topk_log_probs, valid,
                                 call.tail_log_prob, call.temperature, call.vocabulary_size)
    target.validate()
    if target.temperature != config.temperature:
        raise ValueError("global QAT teacher objective temperature differs")
    for token in config.required_ids:
        if not ((target.indices == token) & target.valid_mask).any(-1).all():
            raise ValueError("global QAT teacher union is missing a required stop/format ID")
    next_ids = call.inputs["input_ids"][:, 1:].to(target.indices.device)
    present = ((target.indices[:, :-1] == next_ids.unsqueeze(-1)) & target.valid_mask[:, :-1]).any(-1)
    if not present[call_mask(call)[:, 1:].to(present.device)].all():
        raise ValueError("global QAT teacher union is missing supervised response IDs")
    return target


def prepare_calls(train, validation, config):
    if not train or not validation:
        raise ValueError("global QAT requires independent nonempty train and validation calls")
    if sequence_keys(train) & sequence_keys(validation):
        raise ValueError("global QAT train and validation token sequences overlap")
    for call in (*train, *validation):
        call_mask(call)
    return [(call, teacher_target(call, config)) for call in train], [
        (call, teacher_target(call, config)) for call in validation]


def sequence_keys(calls):
    result = set()
    for call in calls:
        ids = call.inputs["input_ids"]
        masks = call.inputs.get("attention_mask", torch.ones_like(ids)).bool()
        result.update(content_digest(row[mask]) for row, mask in zip(ids, masks))
    return result


def calls_identity(records):
    return content_digest([{"inputs": call.inputs, "target": asdict(target)}
                           for call, target in records])


def sliced_target(target, start, end):
    return replace(target, indices=target.indices[:, start:end],
                   log_probs=target.log_probs[:, start:end],
                   valid_mask=target.valid_mask[:, start:end],
                   tail_log_prob=target.tail_log_prob[:, start:end])


def chunked_loss(logits, call, target, config):
    inputs = call.inputs
    total, count = logits[..., :1].float().sum() * 0, 0
    mask = call_mask(call)
    for start in range(0, logits.shape[1] - 1, config.loss_chunk_tokens):
        end = min(start + config.loss_chunk_tokens + 1, logits.shape[1])
        positions = int(mask[:, start + 1:end].sum())
        if positions:
            selected = sliced_target(target, start, end)
            ids, selected_mask = inputs["input_ids"][:, start:end], mask[:, start:end]
            def loss_chunk(value, target=selected, ids=ids, selected_mask=selected_mask):
                return distillation_loss(value, target, ids, response_mask=selected_mask,
                                         kl_weight=config.kl_weight, ce_weight=config.ce_weight)
            loss = checkpoint(loss_chunk, logits[:, start:end], use_reentrant=False)
            total, count = total + positions * loss, count + positions
    return total / count


def model_loss(model, record, config, device):
    call, target = record
    logits = model_logits(model, call, device)
    return chunked_loss(logits, call, target, config)


def model_logits(model, call, device):
    inputs = {key: value for key, value in call.inputs.items()
              if key not in {"response_mask", "labels"}}
    inputs = _tree_to(inputs, device, None)
    inputs["use_cache"] = False
    output = model(**inputs)
    return output.logits if hasattr(output, "logits") else output[0]


@torch.no_grad()
def evaluate_loss(model, records, config, device):
    return evaluate_forward(model, records, config, device)[0]


@torch.no_grad()
def evaluate_forward(model, records, config, device, *, exact=False):
    was_training = model.training
    model.eval()
    try:
        return validation_mean(model, records, config, device, exact=exact)
    finally:
        model.train(was_training)


def validation_mean(model, records, config, device, *, exact=False):
    total, count = 0., 0
    digests = []
    for record in records:
        positions = int(call_mask(record[0])[:, 1:].sum())
        logits = model_logits(model, record[0], device)
        loss = chunked_loss(logits, record[0], record[1], config)
        if not torch.isfinite(loss):
            raise FloatingPointError("global ternary validation loss is nonfinite")
        total, count = total + float(loss) * positions, count + positions
        if exact:
            digests.append(logits_digest(logits))
    return total / count, digests


def logits_digest(logits):
    digest = hashlib.sha256(str((logits.shape, logits.dtype)).encode())
    for rows in logits.detach().reshape(-1, logits.shape[-1]).split(32):
        chunk = rows.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        digest.update(chunk)
    return digest.hexdigest()
