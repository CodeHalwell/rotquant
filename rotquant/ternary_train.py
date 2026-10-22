"""Bounded sequential block QAT with independent validation and safe resume.

Capture one unmodified source block after its predecessors have been deployed.
Its source targets and the student therefore see identical, quantised-prefix
inputs. Resume starts from a fresh source model and reapplies completed blocks.
"""
from __future__ import annotations

import copy
import fnmatch
import itertools
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from ._internal import get_parent
from .block_train import (
    _primary_tensor,
    _record_hidden,
    _tree_to,
    collect_block_calls,
    find_transformer_blocks,
)
from .ternary import (
    TernaryConfig,
    TernaryLinear,
    packed_ternary_state,
    restore_packed_ternary,
)
from .ternary_numerics import qwen_fp32_training
from .ternary_state import (
    checkpoint_contract,
    content_digest,
    load_training_state,
    parameter_snapshot,
    restore_parameters,
    save_training_state,
)


@dataclass
class TernaryTrainConfig:
    quant: TernaryConfig = field(default_factory=TernaryConfig)
    scope: str = "all"
    steps: int = 100
    lr: float = 1e-4
    scale_lr: float = 1e-3
    train_batches: int = 8
    validation_batches: int = 2
    validation_interval: int = 10
    early_stopping_patience: int = 4
    min_improvement: float = 0.0
    max_grad_norm: float = 1.0
    checkpoint_dir: str | None = None
    checkpoint_interval: int = 10
    resume: bool = False
    max_blocks: int = 0
    source_id: str = ""
    data_id: str = ""
    exclude: tuple[str, ...] = ("in_proj_a", "in_proj_b", "lm_head", "embed_out")

    def __post_init__(self):
        if isinstance(self.quant, dict):
            self.quant = TernaryConfig(**self.quant)
        if not isinstance(self.quant, TernaryConfig):
            raise TypeError("quant must be TernaryConfig or its dictionary")
        self.exclude = tuple(self.exclude)
        if self.scope not in {"mlp", "all"}:
            raise ValueError("scope must be mlp or all")
        self._validate_numbers()
        if self.resume and not self.checkpoint_dir:
            raise ValueError("resume requires checkpoint_dir")

    def _validate_numbers(self):
        minimums = {"steps": 0, "train_batches": 1, "validation_batches": 1,
                    "validation_interval": 1, "early_stopping_patience": 0,
                    "checkpoint_interval": 1, "max_blocks": 0}
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("lr", "scale_lr", "max_grad_norm", "min_improvement"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lr == 0 or self.scale_lr == 0:
            raise ValueError("learning rates must be positive")


def _matches(name, patterns):
    return any(fnmatch.fnmatchcase(name, pattern) or pattern in name for pattern in patterns)


def eligible_ternary_linears(block, config):
    targets = []
    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear) or _matches(name, config.exclude):
            continue
        if config.scope == "mlp" and not _matches(name, ("mlp", "fc1", "fc2", "feed_forward")):
            continue
        targets.append((name, module))
    return targets


def _make_student(source, global_name, config, device):
    # Hybrid recurrent blocks can overflow in FP16 backward even with finite
    # forward loss. Latent optimisation uses FP32; validation below restores
    # every frozen tensor's source dtype and uses deployed forward values.
    student = copy.deepcopy(source).to(device).float().eval()
    student.requires_grad_(False)
    for name, linear in eligible_ternary_linears(student, config):
        # Stable per-site seeds do not depend on process hash randomisation.
        seed = (config.quant.seed + int(content_digest(global_name + "." + name)[:8], 16)) % 2**31
        module = TernaryLinear(linear, replace(config.quant, seed=seed))
        parent, attribute = get_parent(student, name)
        setattr(parent, attribute, module)
    return student


def _block_loss(student, records, device, dtype):
    total = None
    for record in records:
        predicted = _primary_tensor(student(
            *_tree_to(record.args, device, dtype),
            **_tree_to(record.kwargs, device, dtype))).float()
        target = record.output.to(device=device, dtype=torch.float32)
        loss = (predicted - target).square().mean() / target.square().mean().clamp_min(1e-12)
        total = loss if total is None else total + loss
    return total / len(records)


def _optimizer(student, config):
    latent, scales = [], []
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            (scales if name.endswith("log_scales") else latent).append(parameter)
    return torch.optim.Adam([
        {"params": latent, "lr": config.lr}, {"params": scales, "lr": config.scale_lr}])


def _pack_student(student, source, device):
    result = copy.deepcopy(source).to(device).eval()
    result.requires_grad_(False)
    for name, module in list(student.named_modules()):
        if isinstance(module, TernaryLinear):
            dtype = source.get_submodule(name).weight.dtype
            packed = module.to_quant_linear().to(device=device, dtype=dtype)
            parent, attribute = get_parent(result, name)
            setattr(parent, attribute, packed)
    return result


@contextmanager
def _source_precision(student, dtypes):
    """Temporarily restore frozen source precision without rounding latents."""
    restored = []
    try:
        tensors = list(student.named_parameters()) + list(student.named_buffers())
        for name, value in tensors:
            dtype = dtypes.get(name, value.dtype)
            if not value.requires_grad and value.is_floating_point() and value.dtype != dtype:
                restored.append((value, value.data))
                value.data = value.data.to(dtype)
        yield
    finally:
        for value, original in restored:
            value.data = original


def _restore_completed(source, packed, config, device):
    result = copy.deepcopy(source).to(device).eval()
    targets = dict(eligible_ternary_linears(result, config))
    if targets.keys() != packed.keys():
        raise ValueError("ternary checkpoint target modules differ")
    for name, linear in targets.items():
        parent, attribute = get_parent(result, name)
        setattr(parent, attribute, restore_packed_ternary(packed[name], linear))
    return result


class _BlockTrainer:
    def __init__(self, student, train, validation, config, contract, path, device, source_dtypes):
        self.student, self.train, self.validation = student, train, validation
        self.config, self.contract, self.path = config, contract, path
        self.device, self.source_dtypes = device, source_dtypes
        self.optimizer = _optimizer(student, config)
        self.step, self.stale, self.best_step = 0, 0, 0
        self.optimiser_seen_tokens = 0
        self.initial_loss = self.evaluate()
        self.best_loss = self.initial_loss
        self.best = parameter_snapshot(student)
        self.history = [{"step": 0, "validation_loss": self.initial_loss}]

    def evaluate(self):
        with torch.no_grad(), _source_precision(self.student, self.source_dtypes):
            loss = float(_block_loss(self.student, self.validation, self.device, None))
        if not math.isfinite(loss):
            raise FloatingPointError("ternary validation loss is nonfinite")
        return loss

    def restore(self, state):
        restore_parameters(self.student, state["parameters"])
        self.optimizer.load_state_dict(state["optimizer"])
        for name in ("step", "stale", "best_step", "initial_loss", "best_loss", "best", "history",
                     "optimiser_seen_tokens"):
            setattr(self, name, state[name])
        # Validate the rollback snapshot as well as the current candidate.
        current = parameter_snapshot(self.student)
        restore_parameters(self.student, self.best)
        restore_parameters(self.student, current)

    def save(self):
        state = {name: getattr(self, name) for name in
                 ("step", "stale", "best_step", "initial_loss", "best_loss", "best", "history",
                  "optimiser_seen_tokens")}
        state.update(contract=self.contract, complete=False,
                     parameters=parameter_snapshot(self.student), optimizer=self.optimizer.state_dict())
        save_training_state(self.path, state)

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        record = self.train[self.step % len(self.train)]
        with qwen_fp32_training(self.student):
            loss = _block_loss(self.student, [record], self.device, torch.float32)
            if not torch.isfinite(loss):
                raise FloatingPointError("ternary training loss is nonfinite")
            loss.backward()
        parameters = [p for p in self.student.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm or float("inf"))
        if not torch.isfinite(norm):
            raise FloatingPointError("ternary training gradient is nonfinite")
        self.optimizer.step()
        self.step += 1
        self.optimiser_seen_tokens += math.prod(_record_hidden(record).shape[:-1])

    def validate(self):
        loss = self.evaluate()
        self.history.append({"step": self.step, "validation_loss": loss})
        if loss < self.best_loss - self.config.min_improvement:
            self.best_loss, self.best_step = loss, self.step
            self.best, self.stale = parameter_snapshot(self.student), 0
        else:
            self.stale += 1

    def finished(self):
        patience = self.config.early_stopping_patience
        return self.step >= self.config.steps or bool(patience and self.stale >= patience)

    def run(self):
        while not self.finished():
            self.update()
            if self.step % self.config.validation_interval == 0 or self.step == self.config.steps:
                self.validate()
            if self.step % self.config.checkpoint_interval == 0 or self.finished():
                self.save()
        restore_parameters(self.student, self.best)
        return {"steps_completed": self.step, "best_step": self.best_step,
                "initial_validation_loss": self.initial_loss, "validation_loss": self.evaluate(),
                "history": self.history, "early_stopped": self.step < self.config.steps,
                "training_compute_dtype": "float32", "validation_precision": "source",
                "training_kernels": "qwen_torch_reference",
                "optimiser_seen_tokens": self.optimiser_seen_tokens}


def _attempt_stats(started, device):
    peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
    return {"wall_seconds_this_attempt": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": peak}


def train_ternary_block(source, name, train_calls, validation_calls, config, *, device=None):
    """Return a deployed block and validation statistics, never the latent model."""
    if not train_calls or not validation_calls:
        raise ValueError("ternary training requires independent nonempty call sets")
    parameter = next(source.parameters())
    device = device or parameter.device
    started = time.perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    contract = checkpoint_contract(source, name, config, train_calls, validation_calls)
    path = None if config.checkpoint_dir is None else Path(config.checkpoint_dir) / f"{name}.pt"
    state = load_training_state(path, contract, config.resume)
    if state is not None and state["complete"]:
        result = _restore_completed(source, state["packed"], config, device)
        stats = {**state["stats"], **_attempt_stats(started, device)}
        return result, stats
    student = _make_student(source, name, config, device)
    trainer = _BlockTrainer(student, train_calls, validation_calls, config, contract, path,
                            device, {key: tensor.dtype for key, tensor in source.state_dict().items()})
    if state is not None:
        trainer.restore(state)
    stats = trainer.run()
    packed = {name: packed_ternary_state(module.to_quant_linear())
              for name, module in student.named_modules() if isinstance(module, TernaryLinear)}
    result = _pack_student(student, source, device)
    stats["packed_validation_loss"] = float(_evaluate_packed(result, validation_calls, device, None))
    stats.update(_attempt_stats(started, device))
    stats["source_compute_dtype"] = str(parameter.dtype).removeprefix("torch.")
    save_training_state(path, {"contract": contract, "complete": True, "packed": packed, "stats": stats})
    return result, stats


@torch.no_grad()
def _evaluate_packed(module, records, device, dtype):
    return _block_loss(module, records, device, dtype)


def _bounded_batches(loader, count):
    batches = list(itertools.islice(iter(loader), count))
    if len(batches) != count:
        raise ValueError(f"ternary loader supplied {len(batches)} batches; required {count}")
    return batches


def _batch_keys(batch):
    if isinstance(batch, dict) and torch.is_tensor(batch.get("input_ids")):
        ids = batch["input_ids"]
        mask = batch.get("attention_mask", torch.ones_like(ids))
        if mask.shape != ids.shape or not (mask == 1).all():
            raise ValueError("ternary block reconstruction requires unpadded token windows")
        return {content_digest(row[keep.bool()]) for row, keep in zip(ids, mask)}
    return {content_digest(batch)}


def _separate_batches(train_loader, validation_loader, config):
    train = _bounded_batches(train_loader, config.train_batches)
    validation = _bounded_batches(validation_loader, config.validation_batches)
    train_keys = set().union(*(_batch_keys(batch) for batch in train))
    validation_keys = set().union(*(_batch_keys(batch) for batch in validation))
    if train_keys & validation_keys:
        raise ValueError("ternary training and validation data overlap")
    return train, validation


def train_and_patch_ternary(model, train_dataloader, validation_dataloader,
                            config: TernaryTrainConfig, *, device=None, stats_out=None):
    """Sequential deployed-prefix training; only one block's calls are retained."""
    train, validation = _separate_batches(train_dataloader, validation_dataloader, config)
    device = device or next(model.parameters()).device
    # Keep names only: retaining original block objects would keep a second
    # full model alive as the sequential replacements accumulate.
    blocks = [name for name, _ in find_transformer_blocks(model)]
    if config.max_blocks:
        blocks = blocks[:config.max_blocks]
    results = {}
    for name in blocks:
        source = model.get_submodule(name)
        if not eligible_ternary_linears(source, config):
            continue
        dtype = next(source.parameters()).dtype
        records = collect_block_calls(model, train + validation, device,
                                      blocks=[(name, source)], max_batches=len(train + validation),
                                      storage_dtype=dtype)[name]
        result, stats = train_ternary_block(source, name, records[:len(train)],
                                          records[len(train):], config, device=device)
        parent, attribute = get_parent(model, name)
        setattr(parent, attribute, result)
        results[name] = stats
        del records, source, result
    if not results:
        raise ValueError("no eligible ternary matrices were found")
    if stats_out is not None:
        stats_out["ternary_train"] = {"blocks": results, "deployed_prefix_capture": True,
                                     "resident_block_call_sets": 1, "scope": config.scope}
    return model
