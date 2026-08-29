"""Sequential transformer-block reconstruction for structured rotations.

This is the stronger follow-up to independent linear reconstruction. Each
transformer block is replayed with its original call arguments, all butterfly
rotations inside that block are trained jointly under fake 3-bit quantisation,
and the candidate is accepted only when an exact final-quantizer copy beats the
plain-FWHT block on disjoint held-out calls.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import QuantLinear
from .patch import PatchConfig, _cpu_staging_linear, _get_parent
from .quantize import (
    QuantConfig, Quantizer, _expand_scales, _group_scales_rms,
    _quantize_groups, _storage_scales,
)
from .rotate import ButterflyRotation, RandomizedHadamard
from .utils import get_logger

logger = get_logger()


@dataclass
class BlockCall:
    args: tuple
    kwargs: dict
    output: torch.Tensor


@dataclass
class TeacherCall:
    inputs: dict
    logits: torch.Tensor


@dataclass
class BlockRotationTrainConfig:
    steps: int = 8
    lr: float = 3e-3
    objective: str = "block"
    train_batches: int = 1
    validation_batches: int = 1
    selection_batches: int = 1
    assignment_scale: Optional[str] = "rms"
    max_grad_norm: float = 1.0
    restore_best: bool = True
    early_stopping_patience: int = 0
    validation_min_improvement: float = 0.0
    selection_min_improvement: float = 0.0
    learn_scales: bool = False
    scale_lr: Optional[float] = None
    scale_multiplier_min: float = 0.5
    scale_multiplier_max: float = 1.5
    propagate_quantized_inputs: bool = False
    distill_steps: int = 0
    distill_lr: float = 2e-4
    distill_scale_lr: float = 5e-3
    distill_train_batches: int = 1
    distill_validation_batches: int = 1
    distill_selection_batches: int = 1
    distill_temperature: float = 2.0
    distill_kl_weight: float = 1.0
    distill_ce_weight: float = 0.1
    distill_angle_l2: float = 1e-4
    distill_max_grad_norm: float = 1.0
    distill_early_stopping_patience: int = 3
    distill_validation_min_improvement: float = 0.001
    distill_selection_min_improvement: float = 0.0
    distill_scale_multiplier_min: float = 0.8
    distill_scale_multiplier_max: float = 1.25
    distill_lora_rank: int = 0
    distill_lora_alpha: float = 8.0
    distill_lora_lr: float = 1e-3
    distill_train_rotations: bool = True
    distill_train_scales: bool = True

    def __post_init__(self) -> None:
        if self.objective != "block":
            raise ValueError("BlockRotationTrainConfig requires objective='block'")
        if self.steps < 1 or self.lr <= 0:
            raise ValueError("block training requires steps >= 1 and lr > 0")
        if (self.train_batches < 1 or self.validation_batches < 1
                or self.selection_batches < 1):
            raise ValueError(
                "block training requires train_batches, validation_batches, "
                "and selection_batches >= 1")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if not 0 <= self.validation_min_improvement < 1:
            raise ValueError("validation_min_improvement must be in [0, 1)")
        if not 0 <= self.selection_min_improvement < 1:
            raise ValueError("selection_min_improvement must be in [0, 1)")
        if (self.scale_multiplier_min <= 0
                or self.scale_multiplier_max < self.scale_multiplier_min):
            raise ValueError(
                "scale multiplier bounds require 0 < min <= max")
        if self.scale_lr is not None and self.scale_lr <= 0:
            raise ValueError("scale_lr must be > 0 when provided")
        if self.distill_steps < 0:
            raise ValueError("distill_steps must be >= 0")
        if self.distill_steps:
            if self.distill_lr <= 0 or self.distill_scale_lr <= 0:
                raise ValueError("distillation learning rates must be > 0")
            if (self.distill_train_batches < 1
                    or self.distill_validation_batches < 1
                    or self.distill_selection_batches < 1):
                raise ValueError(
                    "distillation train, validation, and selection batches "
                    "must be >= 1")
            if self.distill_temperature <= 0:
                raise ValueError("distill_temperature must be > 0")
            if (self.distill_kl_weight < 0 or self.distill_ce_weight < 0
                    or self.distill_kl_weight + self.distill_ce_weight <= 0):
                raise ValueError("distillation loss weights must be nonnegative")
            if self.distill_angle_l2 < 0 or self.distill_max_grad_norm < 0:
                raise ValueError("distillation regularization must be nonnegative")
            if self.distill_early_stopping_patience < 0:
                raise ValueError(
                    "distill_early_stopping_patience must be >= 0")
            if not 0 <= self.distill_validation_min_improvement < 1:
                raise ValueError(
                    "distill_validation_min_improvement must be in [0, 1)")
            if not 0 <= self.distill_selection_min_improvement < 1:
                raise ValueError(
                    "distill_selection_min_improvement must be in [0, 1)")
            if (self.distill_scale_multiplier_min <= 0
                    or self.distill_scale_multiplier_max
                    < self.distill_scale_multiplier_min):
                raise ValueError(
                    "distillation scale bounds require 0 < min <= max")
            if self.distill_lora_rank < 0:
                raise ValueError("distill_lora_rank must be >= 0")
            if self.distill_lora_rank and (
                    self.distill_lora_alpha <= 0 or self.distill_lora_lr <= 0):
                raise ValueError(
                    "enabled distillation LoRA requires alpha and lr > 0")
            if (not self.distill_train_rotations
                    and not self.distill_train_scales
                    and not self.distill_lora_rank):
                raise ValueError(
                    "distillation must train rotations, scales, or LoRA")


def _tree_copy_cpu(value, storage_dtype=torch.float16):
    if torch.is_tensor(value):
        dtype = storage_dtype if value.is_floating_point() else value.dtype
        return value.detach().to(device="cpu", dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_tree_copy_cpu(v, storage_dtype) for v in value)
    if isinstance(value, list):
        return [_tree_copy_cpu(v, storage_dtype) for v in value]
    if isinstance(value, dict):
        return {k: _tree_copy_cpu(v, storage_dtype) for k, v in value.items()}
    return value


def _tree_to(value, device, dtype=torch.float32):
    if torch.is_tensor(value):
        target_dtype = dtype if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, tuple):
        return tuple(_tree_to(v, device, dtype) for v in value)
    if isinstance(value, list):
        return [_tree_to(v, device, dtype) for v in value]
    if isinstance(value, dict):
        return {k: _tree_to(v, device, dtype) for k, v in value.items()}
    return value


def _primary_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _primary_tensor(output[0])
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"cannot extract tensor output from {type(output).__name__}")


def _record_hidden(record: BlockCall) -> torch.Tensor:
    if record.args and torch.is_tensor(record.args[0]):
        return record.args[0]
    hidden = record.kwargs.get("hidden_states")
    if torch.is_tensor(hidden):
        return hidden
    raise TypeError("block call has no tensor hidden_states input")


def _replace_record_hidden(record: BlockCall,
                           hidden: torch.Tensor) -> BlockCall:
    if tuple(hidden.shape) != tuple(_record_hidden(record).shape):
        raise ValueError(
            "propagated hidden-state shape changed from "
            f"{tuple(_record_hidden(record).shape)} to {tuple(hidden.shape)}")
    if record.args and torch.is_tensor(record.args[0]):
        args = (hidden,) + record.args[1:]
        kwargs = record.kwargs
    else:
        args = record.args
        kwargs = dict(record.kwargs)
        kwargs["hidden_states"] = hidden
    return BlockCall(args=args, kwargs=kwargs, output=record.output)


def _records_with_hidden(records: Sequence[BlockCall],
                         hidden_states: Sequence[torch.Tensor]
                         ) -> List[BlockCall]:
    if len(records) != len(hidden_states):
        raise ValueError("propagated hidden-state count does not match block calls")
    return [_replace_record_hidden(record, hidden)
            for record, hidden in zip(records, hidden_states)]


def _input_drift(records: Sequence[BlockCall],
                 source_records: Sequence[BlockCall]) -> float:
    errors = []
    for record, source in zip(records, source_records):
        current = _record_hidden(record).float()
        original = _record_hidden(source).float()
        errors.append(float(
            ((current - original).pow(2).mean()
             / original.pow(2).mean().clamp_min(1e-12)).item()))
    return sum(errors) / max(len(errors), 1)


@torch.no_grad()
def _replay_block_hidden(block: nn.Module, records: Sequence[BlockCall],
                         device, compute_dtype: torch.dtype,
                         storage_dtype: torch.dtype = torch.float16,
                         ) -> List[torch.Tensor]:
    """Run a deployed block and retain only its bounded CPU hidden outputs."""
    block.eval()
    outputs = []
    for record in records:
        args = _tree_to(record.args, device, compute_dtype)
        kwargs = _tree_to(record.kwargs, device, compute_dtype)
        hidden = _primary_tensor(block(*args, **kwargs))
        outputs.append(_tree_copy_cpu(hidden, storage_dtype))
    return outputs


def find_transformer_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Find the principal ``ModuleList`` of transformer blocks."""
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        counts = [sum(isinstance(m, nn.Linear) for m in block.modules())
                  for block in module]
        if counts and min(counts) > 0:
            candidates.append((sum(counts), name, module))
    if not candidates:
        raise ValueError("could not find a transformer block ModuleList")
    _, name, blocks = max(candidates, key=lambda item: item[0])
    return [(f"{name}.{i}", block) for i, block in enumerate(blocks)]


@torch.no_grad()
def collect_block_calls(model: nn.Module, dataloader: Iterable, device,
                        blocks: Optional[List[Tuple[str, nn.Module]]] = None,
                        max_batches: int = 2,
                        storage_dtype: torch.dtype = torch.float16,
                        ) -> Dict[str, List[BlockCall]]:
    """Capture bounded, replayable source-model calls and outputs per block."""
    blocks = blocks or find_transformer_blocks(model)
    calls: Dict[str, List[BlockCall]] = {name: [] for name, _ in blocks}
    handles = []

    def make_hook(name):
        def hook(_module, args, kwargs, output):
            if len(calls[name]) >= max_batches:
                return
            calls[name].append(BlockCall(
                args=_tree_copy_cpu(args, storage_dtype),
                kwargs=_tree_copy_cpu(kwargs, storage_dtype),
                output=_tree_copy_cpu(_primary_tensor(output), storage_dtype),
            ))
        return hook

    for name, block in blocks:
        handles.append(block.register_forward_hook(make_hook(name), with_kwargs=True))
    model.eval()
    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            if isinstance(batch, dict):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                # Replayed training calls must not carry a mutable DynamicCache
                # from one optimiser step into the next.
                batch.setdefault("use_cache", False)
                model(**batch)
            elif isinstance(batch, (tuple, list)):
                model(*[v.to(device) if torch.is_tensor(v) else v for v in batch])
            else:
                model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, records in calls.items()
               if len(records) < max_batches]
    if missing:
        raise RuntimeError(
            f"captured fewer than {max_batches} calls for blocks: {missing[:3]}")
    logger.info("Captured %d replay calls for %d transformer blocks",
                max_batches, len(blocks))
    return calls


@torch.no_grad()
def collect_teacher_calls(model: nn.Module, dataloader: Iterable, device,
                          max_batches: int,
                          storage_dtype: torch.dtype = torch.float16,
                          ) -> List[TeacherCall]:
    """Capture bounded full-model inputs and source logits for distillation."""
    calls = []
    model.eval()
    for index, batch in enumerate(dataloader):
        if index >= max_batches:
            break
        if not isinstance(batch, dict):
            raise TypeError("model-level distillation requires dict batches")
        inputs = {key: _tree_copy_cpu(value, storage_dtype)
                  for key, value in batch.items()}
        work = {key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in batch.items()}
        work.setdefault("use_cache", False)
        output = model(**work)
        logits = output.logits if hasattr(output, "logits") else output[0]
        calls.append(TeacherCall(
            inputs=inputs,
            logits=_tree_copy_cpu(logits, storage_dtype),
        ))
    if len(calls) < max_batches:
        raise RuntimeError(
            f"captured {len(calls)} teacher calls, expected {max_batches}")
    logger.info("Captured %d teacher-logit calls", len(calls))
    return calls


class FakeQuantButterflyLinear(nn.Module):
    """Frozen dense linear with trainable butterfly and per-forward fake quant."""

    def __init__(self, linear: nn.Linear, quant_cfg: QuantConfig, *, block: int,
                 seed: int, learn_scales: bool = False,
                 scale_multiplier_min: float = 0.5,
                 scale_multiplier_max: float = 1.5):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("weight", linear.weight.detach().float().clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float().clone())
        else:
            self.bias = None
        self.rotation = ButterflyRotation(
            self.in_features, block=block, seed=seed,
            device=self.weight.device, dtype=torch.float32)
        self.quantizer = Quantizer(quant_cfg)
        self.scale_multiplier_min = scale_multiplier_min
        self.scale_multiplier_max = scale_multiplier_max
        self.register_parameter("log_scale_multiplier", None)
        if learn_scales:
            if quant_cfg.scale == "turboquant":
                raise ValueError(
                    "learned block scales currently require rms or mse_search")
            with torch.no_grad():
                rotated = self.rotation.rotate_weight(self.weight)
                rms = _group_scales_rms(rotated, quant_cfg.group_size)
                initial = self.quantizer._select_scales(rotated)
                multiplier = (initial / rms.clamp_min(1e-12)).clamp(
                    scale_multiplier_min, scale_multiplier_max)
            self.log_scale_multiplier = nn.Parameter(multiplier.log())

    def scale_multiplier(self) -> Optional[torch.Tensor]:
        if self.log_scale_multiplier is None:
            return None
        return self.log_scale_multiplier.exp().clamp(
            self.scale_multiplier_min, self.scale_multiplier_max)

    def _learned_scales(self, rotated: torch.Tensor) -> torch.Tensor:
        # RMS follows the changing rotation, but is deliberately detached: the
        # rotation surrogate remains the established activation-path gradient,
        # while the stored group scales receive their own direct gradient.
        rms = _group_scales_rms(
            rotated, self.quantizer.cfg.group_size).detach()
        scales = rms * self.scale_multiplier()
        stored = _storage_scales(scales, self.quantizer.cfg.scale_bits)
        # Match packed fp16 values in the forward pass while keeping a stable
        # fp32 identity gradient through the storage rounding operation.
        return scales + (stored.to(scales.dtype) - scales).detach()

    def _assigned_weight(self) -> torch.Tensor:
        with torch.no_grad():
            rotated = self.rotation.rotate_weight(self.weight)
        if self.log_scale_multiplier is not None:
            scales = self._learned_scales(rotated)
            with torch.no_grad():
                _, indices = _quantize_groups(
                    rotated, scales.detach(), self.quantizer.codebook,
                    self.quantizer.cfg.group_size)
            centroids = self.quantizer.codebook.centroids.to(rotated.device)
            normalized = centroids[indices]
            return normalized * _expand_scales(
                scales, self.quantizer.cfg.group_size, self.in_features)
        with torch.no_grad():
            scales = self.quantizer._select_scales(rotated)
            q, _ = _quantize_groups(
                rotated, scales, self.quantizer.codebook,
                self.quantizer.cfg.group_size)
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.rotation.rotate_activation(x)
        return F.linear(xr, self._assigned_weight().to(xr.dtype), self.bias)


def _target_linears(module: nn.Module) -> List[Tuple[str, nn.Linear]]:
    return [(name, child) for name, child in module.named_modules()
            if isinstance(child, nn.Linear)]


def _block_loss(block: nn.Module, records: Sequence[BlockCall], device,
                *, grad: bool) -> torch.Tensor:
    losses = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for record in records:
            args = _tree_to(record.args, device, torch.float32)
            kwargs = _tree_to(record.kwargs, device, torch.float32)
            target = record.output.to(device=device, dtype=torch.float32)
            pred = _primary_tensor(block(*args, **kwargs)).float()
            losses.append((pred - target).pow(2).mean()
                          / target.pow(2).mean().clamp_min(1e-12))
    return torch.stack(losses).mean()


def _fake_quant_block(source: nn.Module, global_name: str,
                      quant_cfg: QuantConfig, patch_cfg: PatchConfig,
                      seed_by_name: Dict[str, int], device,
                      config: BlockRotationTrainConfig):
    block = copy.deepcopy(source).to(device=device, dtype=torch.float32).eval()
    proxy_cfg = replace(
        quant_cfg,
        error_comp="none",
        scale=(quant_cfg.scale if config.learn_scales else
               ((patch_cfg.train_rotation or {}).get("assignment_scale")
                or quant_cfg.scale)),
    )
    fake: Dict[str, FakeQuantButterflyLinear] = {}
    for relative, linear in _target_linears(block):
        full_name = f"{global_name}.{relative}"
        if full_name not in seed_by_name:
            continue
        replacement = FakeQuantButterflyLinear(
            linear, proxy_cfg, block=patch_cfg.block,
            seed=seed_by_name[full_name], learn_scales=config.learn_scales,
            scale_multiplier_min=config.scale_multiplier_min,
            scale_multiplier_max=config.scale_multiplier_max).to(device)
        parent, attr = _get_parent(block, relative)
        setattr(parent, attr, replacement)
        fake[relative] = replacement
    if not fake:
        raise ValueError(f"block {global_name} contains no targeted linear layers")
    block.requires_grad_(False)
    # Keep dropout/norm in eval mode while allowing butterfly caches to track
    # changing theta during optimisation.
    for module in fake.values():
        module.rotation.requires_grad_(True)
        module.rotation.train(True)
        if module.log_scale_multiplier is not None:
            module.log_scale_multiplier.requires_grad_(True)
    return block, fake


def _rotation_states(fake: Dict[str, FakeQuantButterflyLinear]
                     ) -> Dict[str, Dict[str, torch.Tensor]]:
    states = {}
    for name, module in fake.items():
        state = {
            "theta": module.rotation.theta.detach().cpu().float().clone(),
        }
        multiplier = module.scale_multiplier()
        if multiplier is not None:
            state["scale_multiplier"] = (
                multiplier.detach().cpu().float().clone())
        states[name] = state
    return states


def _restore_rotation_states(fake: Dict[str, FakeQuantButterflyLinear], states) -> None:
    with torch.no_grad():
        for name, state in states.items():
            module = fake[name]
            module.rotation.theta.copy_(
                state["theta"].to(module.rotation.theta.device))
            if module.log_scale_multiplier is not None:
                module.log_scale_multiplier.copy_(
                    state["scale_multiplier"].to(
                        module.log_scale_multiplier.device).log())


def train_fake_quant_block(source: nn.Module, global_name: str,
                           records: Sequence[BlockCall], quant_cfg: QuantConfig,
                           patch_cfg: PatchConfig, seed_by_name: Dict[str, int],
                           device, config: BlockRotationTrainConfig):
    """Optimise a block and select its checkpoint on disjoint validation calls.

    The returned records have not influenced either gradients or checkpoint
    selection.  They are reserved for the exact packed-candidate versus FWHT
    gate in :func:`train_and_patch_blocks`.
    """
    required = (config.train_batches + config.validation_batches
                + config.selection_batches)
    if len(records) < required:
        raise ValueError(f"block {global_name} needs {required} captured calls")
    train_records = records[:config.train_batches]
    validation_start = config.train_batches
    validation_end = validation_start + config.validation_batches
    validation_records = records[validation_start:validation_end]
    selection_records = records[validation_end:required]
    block, fake = _fake_quant_block(
        source, global_name, quant_cfg, patch_cfg, seed_by_name, device, config)
    rotation_params = [module.rotation.theta for module in fake.values()]
    scale_params = [module.log_scale_multiplier for module in fake.values()
                    if module.log_scale_multiplier is not None]
    params = rotation_params + scale_params
    param_groups = [{"params": rotation_params, "lr": config.lr}]
    if scale_params:
        param_groups.append({
            "params": scale_params,
            "lr": config.scale_lr if config.scale_lr is not None else config.lr,
        })
    optimizer = torch.optim.Adam(param_groups)

    initial = float(_block_loss(block, train_records, device, grad=False).item())
    initial_validation = float(
        _block_loss(block, validation_records, device, grad=False).item())
    best_validation, best_step = initial_validation, 0
    patience_validation = initial_validation
    best_states = _rotation_states(fake)
    stale_steps = 0
    steps_run = 0
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        loss = _block_loss(block, train_records, device, grad=True)
        loss.backward()
        if config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
        optimizer.step()

        steps_run = step
        validation = float(
            _block_loss(block, validation_records, device, grad=False).item())
        if validation < best_validation:
            best_validation, best_step = validation, step
            best_states = _rotation_states(fake)
        threshold = patience_validation * (
            1.0 - config.validation_min_improvement)
        if validation < threshold:
            patience_validation = validation
            stale_steps = 0
        else:
            stale_steps += 1
        if (config.early_stopping_patience
                and stale_steps >= config.early_stopping_patience):
            break

    if config.restore_best:
        _restore_rotation_states(fake, best_states)
    selected_validation = float(
        _block_loss(block, validation_records, device, grad=False).item())
    selected_train = float(
        _block_loss(block, train_records, device, grad=False).item())
    trained_states = _rotation_states(fake)
    scale_multipliers = [state["scale_multiplier"].reshape(-1)
                         for state in trained_states.values()
                         if "scale_multiplier" in state]
    del optimizer, block
    stats = {
        "initial_mse": initial,
        "final_mse": selected_train,
        "initial_validation_mse": initial_validation,
        "final_validation_mse": selected_validation,
        "best_validation_mse": best_validation,
        "best_step": best_step,
        "steps_run": steps_run,
        "stopped_early": steps_run < config.steps,
        "layers": len(fake),
        "learned_scales": bool(scale_multipliers),
    }
    if scale_multipliers:
        multipliers = torch.cat(scale_multipliers)
        stats.update({
            "scale_multiplier_mean": float(multipliers.mean().item()),
            "scale_multiplier_min": float(multipliers.min().item()),
            "scale_multiplier_max": float(multipliers.max().item()),
        })
    return trained_states, selection_records, stats


def _scale_override(linear: nn.Linear, rotation: nn.Module,
                    quant_cfg: QuantConfig,
                    state: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    multiplier = state.get("scale_multiplier")
    if multiplier is None:
        return None
    with torch.no_grad():
        rotated = rotation.rotate_weight(linear.weight.detach()).float()
        rms = _group_scales_rms(rotated, quant_cfg.group_size)
        return rms * multiplier.to(device=rms.device, dtype=rms.dtype)


def _packed_cpu_block(source: nn.Module, global_name: str,
                      quant_cfg: QuantConfig, patch_cfg: PatchConfig,
                      seed_by_name: Dict[str, int],
                      states: Optional[Dict[str, Dict[str, torch.Tensor]]],
                      device="cpu") -> nn.Module:
    """Build an exact packed block for held-out selection on ``device``.

    The historical name is retained for compatibility. CPU remains the default
    and the MPS path still selects on CPU, while high-memory CUDA experiments
    can keep this otherwise dominant scale-search/replay work on the GPU.
    """
    device = torch.device(device)
    block = copy.deepcopy(source).to(device=device, dtype=torch.float32).eval()
    for relative, linear in _target_linears(block):
        full_name = f"{global_name}.{relative}"
        if full_name not in seed_by_name:
            continue
        rotation = ButterflyRotation(
            linear.in_features, block=patch_cfg.block,
            seed=seed_by_name[full_name], device=device)
        state = states[relative] if states is not None else None
        if state is not None:
            rotation.theta.data.copy_(state["theta"])
        rotation.requires_grad_(False).eval()
        scales_override = (
            _scale_override(linear, rotation, quant_cfg, state)
            if state is not None else None)
        qlinear = QuantLinear.from_linear(
            linear, quant_cfg, weight_rotation=rotation, act_rotation=rotation,
            scales_override=scales_override,
            fallback=True, fallback_dtype=torch.float32)
        parent, attr = _get_parent(block, relative)
        setattr(parent, attr, qlinear)
    return block


def _patch_source_block(source: nn.Module, global_name: str,
                        quant_cfg: QuantConfig, patch_cfg: PatchConfig,
                        seed_by_name: Dict[str, int],
                        states: Optional[Dict[str, Dict[str, torch.Tensor]]]
                        ) -> int:
    targets = _target_linears(source)
    patched = 0
    for relative, linear in targets:
        full_name = f"{global_name}.{relative}"
        if full_name not in seed_by_name:
            continue
        source_device, source_dtype = linear.weight.device, linear.weight.dtype
        stage = source_device.type == "mps" and patch_cfg.fallback
        work = _cpu_staging_linear(linear) if stage else linear
        state = states[relative] if states is not None else None
        if state is None:
            rotation = RandomizedHadamard(
                linear.in_features, block=patch_cfg.block,
                seed=seed_by_name[full_name], device=work.weight.device)
        else:
            rotation = ButterflyRotation(
                linear.in_features, block=patch_cfg.block,
                seed=seed_by_name[full_name], device=work.weight.device)
            rotation.theta.data.copy_(
                state["theta"].to(rotation.theta.device))
        rotation.requires_grad_(False).eval()
        scales_override = (
            _scale_override(work, rotation, quant_cfg, state)
            if state is not None else None)
        qlinear = QuantLinear.from_linear(
            work, quant_cfg, weight_rotation=rotation, act_rotation=rotation,
            scales_override=scales_override,
            fallback=patch_cfg.fallback, fallback_dtype=source_dtype)
        if stage:
            qlinear.to(device=source_device, dtype=source_dtype)
        parent, attr = _get_parent(source, relative)
        setattr(parent, attr, qlinear)
        patched += 1
    return patched


def _teacher_call_loss(model: nn.Module, call: TeacherCall, device,
                       compute_dtype: torch.dtype,
                       config: BlockRotationTrainConfig) -> torch.Tensor:
    inputs = _tree_to(call.inputs, device, compute_dtype)
    if "input_ids" not in inputs:
        raise ValueError("distillation batches require input_ids")
    inputs.setdefault("use_cache", False)
    output = model(**inputs)
    student = output.logits if hasattr(output, "logits") else output[0]
    student = student[:, :-1].float()
    teacher = call.logits.to(device=device, dtype=torch.float32)[:, :-1]
    labels = inputs["input_ids"][:, 1:]
    mask = inputs.get("attention_mask")
    if mask is None:
        token_mask = torch.ones_like(labels, dtype=torch.float32)
    else:
        token_mask = mask[:, 1:].float()
    denominator = token_mask.sum().clamp_min(1.0)

    total = student.new_zeros(())
    if config.distill_kl_weight:
        temperature = config.distill_temperature
        teacher_prob = F.softmax(teacher / temperature, dim=-1)
        student_log_prob = F.log_softmax(student / temperature, dim=-1)
        token_kl = F.kl_div(
            student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        kl = (token_kl * token_mask).sum() / denominator
        total = total + config.distill_kl_weight * kl * temperature ** 2
    if config.distill_ce_weight:
        token_ce = F.cross_entropy(
            student.reshape(-1, student.shape[-1]), labels.reshape(-1),
            reduction="none").reshape_as(labels)
        ce = (token_ce * token_mask).sum() / denominator
        total = total + config.distill_ce_weight * ce
    return total


@torch.no_grad()
def _distill_eval_loss(model: nn.Module, calls: Sequence[TeacherCall], device,
                       compute_dtype: torch.dtype,
                       config: BlockRotationTrainConfig) -> float:
    values = [float(_teacher_call_loss(
        model, call, device, compute_dtype, config).item()) for call in calls]
    return sum(values) / max(len(values), 1)


def _parameter_snapshot(parameters: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [parameter.detach().cpu().float().clone()
            for parameter in parameters]


def _restore_parameter_snapshot(parameters: Sequence[torch.Tensor],
                                snapshot: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, snapshot):
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def distill_packed_model(model: nn.Module, calls: Sequence[TeacherCall], device,
                         compute_dtype: torch.dtype,
                         config: BlockRotationTrainConfig) -> Dict[str, Any]:
    """Fine-tune deployed rotations/scales with fixed 3-bit code indices."""
    required = (config.distill_train_batches
                + config.distill_validation_batches
                + config.distill_selection_batches)
    if len(calls) < required:
        raise ValueError(f"distillation requires {required} teacher calls")
    train_end = config.distill_train_batches
    validation_end = train_end + config.distill_validation_batches
    train_calls = calls[:train_end]
    validation_calls = calls[train_end:validation_end]
    selection_calls = calls[validation_end:required]

    model.eval().requires_grad_(False)
    rotation_params = []
    rotation_modules = []
    seen_rotations = set()
    scale_params = []
    lora_params = []
    quant_linears = []
    for module in model.modules():
        if not isinstance(module, QuantLinear):
            continue
        quant_linears.append(module)
        if config.distill_train_scales:
            scale_params.append(module.enable_scale_finetuning(
                config.distill_scale_multiplier_min,
                config.distill_scale_multiplier_max))
        if config.distill_lora_rank:
            lora_params.extend(module.enable_lora(
                config.distill_lora_rank, config.distill_lora_alpha))
        rotation = module.act_rotation
        if (config.distill_train_rotations
                and isinstance(rotation, ButterflyRotation)
                and id(rotation) not in seen_rotations):
            seen_rotations.add(id(rotation))
            rotation.theta.requires_grad_(True)
            rotation.train(True)
            rotation_modules.append(rotation)
            rotation_params.append(rotation.theta)
    if not scale_params and not rotation_params and not lora_params:
        raise ValueError("distillation found no trainable packed parameters")

    parameters = rotation_params + scale_params + lora_params
    initial_snapshot = _parameter_snapshot(parameters)
    angle_reference = [parameter.detach().clone()
                       for parameter in rotation_params]
    groups = []
    if rotation_params:
        groups.append({"params": rotation_params, "lr": config.distill_lr})
    if scale_params:
        groups.append({"params": scale_params, "lr": config.distill_scale_lr})
    if lora_params:
        groups.append({"params": lora_params, "lr": config.distill_lora_lr})
    optimizer = torch.optim.Adam(groups)

    initial_train = _distill_eval_loss(
        model, train_calls, device, compute_dtype, config)
    initial_validation = _distill_eval_loss(
        model, validation_calls, device, compute_dtype, config)
    best_validation = initial_validation
    patience_validation = initial_validation
    best_step = 0
    best_snapshot = initial_snapshot
    stale_steps = 0
    steps_run = 0

    for step in range(1, config.distill_steps + 1):
        optimizer.zero_grad()
        train_value = 0.0
        for call in train_calls:
            loss = _teacher_call_loss(
                model, call, device, compute_dtype, config)
            train_value += float(loss.detach().item()) / len(train_calls)
            (loss / len(train_calls)).backward()
        if rotation_params and config.distill_angle_l2:
            angle_loss = torch.stack([
                (parameter - reference).pow(2).mean()
                for parameter, reference in zip(
                    rotation_params, angle_reference)
            ]).mean()
            (config.distill_angle_l2 * angle_loss).backward()
        if config.distill_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                parameters, config.distill_max_grad_norm)
        optimizer.step()
        steps_run = step

        validation = _distill_eval_loss(
            model, validation_calls, device, compute_dtype, config)
        if validation < best_validation:
            best_validation = validation
            best_step = step
            best_snapshot = _parameter_snapshot(parameters)
        threshold = patience_validation * (
            1.0 - config.distill_validation_min_improvement)
        if validation < threshold:
            patience_validation = validation
            stale_steps = 0
        else:
            stale_steps += 1
        if (config.distill_early_stopping_patience
                and stale_steps >= config.distill_early_stopping_patience):
            break

    _restore_parameter_snapshot(parameters, best_snapshot)
    candidate_snapshot = _parameter_snapshot(parameters)
    candidate_train = _distill_eval_loss(
        model, train_calls, device, compute_dtype, config)
    candidate_validation = _distill_eval_loss(
        model, validation_calls, device, compute_dtype, config)
    candidate_selection = _distill_eval_loss(
        model, selection_calls, device, compute_dtype, config)
    _restore_parameter_snapshot(parameters, initial_snapshot)
    reference_selection = _distill_eval_loss(
        model, selection_calls, device, compute_dtype, config)
    accepted = candidate_selection <= reference_selection * (
        1.0 - config.distill_selection_min_improvement)
    if accepted:
        _restore_parameter_snapshot(parameters, candidate_snapshot)

    final_train = _distill_eval_loss(
        model, train_calls, device, compute_dtype, config)
    final_validation = _distill_eval_loss(
        model, validation_calls, device, compute_dtype, config)
    scale_multiplier_values = [
        module.scale_finetuning_multiplier().detach().cpu().reshape(-1)
        for module in quant_linears
        if module.scale_finetuning_multiplier() is not None]
    scale_multipliers = (
        torch.cat(scale_multiplier_values) if scale_multiplier_values else None)
    for module in quant_linears:
        module.commit_scale_finetuning()
        if accepted:
            module.commit_lora()
        else:
            module.disable_lora()
    for rotation in rotation_modules:
        rotation.requires_grad_(False).eval()
    del optimizer

    logger.info(
        "end-to-end distillation: train %.6f -> candidate %.6f -> deployed "
        "%.6f; validation %.6f -> candidate %.6f -> deployed %.6f at step "
        "%d/%d; held-out %.6f vs %.6f (%s)",
        initial_train, candidate_train, final_train,
        initial_validation, candidate_validation, final_validation,
        best_step, steps_run, candidate_selection, reference_selection,
        "accepted" if accepted else "restored block model")
    return {
        "steps": config.distill_steps,
        "steps_run": steps_run,
        "best_step": best_step,
        "stopped_early": steps_run < config.distill_steps,
        "accepted": accepted,
        "lora_rank": config.distill_lora_rank,
        "lora_retained": bool(accepted and config.distill_lora_rank),
        "train_rotations": config.distill_train_rotations,
        "train_scales": config.distill_train_scales,
        "adapter_parameter_bytes": sum(
            module.adapter_state_bytes() for module in quant_linears),
        "train_batches": config.distill_train_batches,
        "validation_batches": config.distill_validation_batches,
        "selection_batches": config.distill_selection_batches,
        "initial_train_loss": initial_train,
        "candidate_train_loss": candidate_train,
        "final_train_loss": final_train,
        "initial_validation_loss": initial_validation,
        "candidate_validation_loss": candidate_validation,
        "best_validation_loss": best_validation,
        "final_validation_loss": final_validation,
        "selection_candidate_loss": candidate_selection,
        "selection_reference_loss": reference_selection,
        "scale_multiplier_mean": (
            float(scale_multipliers.mean().item())
            if scale_multipliers is not None else None),
        "scale_multiplier_min": (
            float(scale_multipliers.min().item())
            if scale_multipliers is not None else None),
        "scale_multiplier_max": (
            float(scale_multipliers.max().item())
            if scale_multipliers is not None else None),
    }


def train_and_patch_blocks(model: nn.Module, patch_cfg: PatchConfig,
                           calls: Dict[str, List[BlockCall]],
                           distill_calls: Optional[Sequence[TeacherCall]] = None,
                           stats_out: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Train, validate, exact-held-out-select, and pack transformer blocks."""
    if patch_cfg.rotation not in ("butterfly", "learned_butterfly", "structured"):
        raise ValueError("block rotation training requires rotation='butterfly'")
    if patch_cfg.mode not in ("consistent", "fused_inverse"):
        raise ValueError("block rotation training requires a consistent patch mode")
    if patch_cfg.quant.error_comp == "gptq":
        raise ValueError("block rotation training with GPTQ is not implemented")
    config = BlockRotationTrainConfig(**(patch_cfg.train_rotation or {}))
    if config.learn_scales and patch_cfg.quant.scale == "turboquant":
        raise ValueError(
            "learned block scales currently require rms or mse_search")
    blocks = find_transformer_blocks(model)
    include = tuple(patch_cfg.include) if patch_cfg.include is not None else None
    exclude = tuple(patch_cfg.exclude or ())
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)
               and (include is None or any(term in name for term in include))
               and not any(term in name for term in exclude)]
    seed_by_name = {name: patch_cfg.seed + i for i, (name, _) in enumerate(targets)}
    covered = {f"{block_name}.{relative}"
               for block_name, block in blocks
               for relative, _ in _target_linears(block)}
    device = next(model.parameters()).device
    compute_dtype = next(model.parameters()).dtype
    # Exact candidate/reference packing is expensive for large blocks. CUDA has
    # the bandwidth and, for the intended Qwen run, memory to select one block
    # at a time on-device. MPS keeps its established CPU scale-search path.
    selection_device = device if device.type == "cuda" else torch.device("cpu")
    block_stats = []
    total_patched = 0
    propagated_hidden = None

    for index, (global_name, block) in enumerate(blocks):
        source_records = calls[global_name]
        records = (
            _records_with_hidden(source_records, propagated_hidden)
            if config.propagate_quantized_inputs and propagated_hidden is not None
            else source_records)
        states, selection_records, stats = train_fake_quant_block(
            block, global_name, records, patch_cfg.quant,
            patch_cfg, seed_by_name, device, config)
        stats["input_drift_mse"] = _input_drift(records, source_records)
        candidate = _packed_cpu_block(
            block, global_name, patch_cfg.quant, patch_cfg, seed_by_name, states,
            device=selection_device)
        candidate_error = float(_block_loss(
            candidate, selection_records, selection_device, grad=False).item())
        del candidate
        reference = _packed_cpu_block(
            block, global_name, patch_cfg.quant, patch_cfg, seed_by_name, None,
            device=selection_device)
        reference_error = float(_block_loss(
            reference, selection_records, selection_device, grad=False).item())
        del reference
        accepted = candidate_error <= reference_error * (
            1.0 - config.selection_min_improvement)
        selected_states = states if accepted else None
        stats.update({
            "selection_candidate_mse": candidate_error,
            "selection_reference_mse": reference_error,
            "selection_accepted": accepted,
        })
        total_patched += _patch_source_block(
            block, global_name, patch_cfg.quant, patch_cfg,
            seed_by_name, selected_states)
        if config.propagate_quantized_inputs and index + 1 < len(blocks):
            propagated_hidden = _replay_block_hidden(
                block, records, device, compute_dtype,
                storage_dtype=source_records[0].output.dtype)
        block_stats.append(stats)
        logger.info(
            "block-trained %d/%d %s: input drift %.6f; train %.6f -> %.6f; "
            "validation %.6f -> %.6f at step %d/%d; held-out %.6f vs FWHT "
            "%.6f (%s)",
            index + 1, len(blocks), global_name,
            stats["input_drift_mse"],
            stats["initial_mse"], stats["final_mse"],
            stats["initial_validation_mse"], stats["final_validation_mse"],
            stats["best_step"], stats["steps_run"],
            candidate_error, reference_error,
            "accepted" if accepted else "restored FWHT")

    uncovered = [name for name, _ in targets if name not in covered]
    if uncovered:
        logger.warning("block training left %d targeted linears unpatched: %s",
                       len(uncovered), uncovered[:3])

    distill_stats = None
    if config.distill_steps:
        if distill_calls is None:
            raise ValueError(
                "distill_steps > 0 requires captured teacher-logit calls")
        distill_stats = distill_packed_model(
            model, distill_calls, device, compute_dtype, config)

    if stats_out is not None and block_stats:
        n = len(block_stats)
        stats_out["rotation_train"] = {
            "objective": "block",
            "blocks": n,
            "layers": total_patched,
            "steps": config.steps,
            "learn_scales": config.learn_scales,
            "propagate_quantized_inputs": config.propagate_quantized_inputs,
            "scale_lr": (config.scale_lr if config.scale_lr is not None
                         else config.lr) if config.learn_scales else None,
            "train_batches": config.train_batches,
            "validation_batches": config.validation_batches,
            "selection_batches": config.selection_batches,
            "early_stopping_patience": config.early_stopping_patience,
            "mean_initial_mse": sum(s["initial_mse"] for s in block_stats) / n,
            "mean_final_mse": sum(s["final_mse"] for s in block_stats) / n,
            "mean_initial_validation_mse": sum(
                s["initial_validation_mse"] for s in block_stats) / n,
            "mean_final_validation_mse": sum(
                s["final_validation_mse"] for s in block_stats) / n,
            "mean_best_step": sum(s["best_step"] for s in block_stats) / n,
            "mean_steps_run": sum(s["steps_run"] for s in block_stats) / n,
            "early_stopping_rate": (
                sum(float(s["stopped_early"]) for s in block_stats) / n),
            "mean_input_drift_mse": sum(
                s["input_drift_mse"] for s in block_stats) / n,
            "final_input_drift_mse": block_stats[-1]["input_drift_mse"],
            "selection_acceptance_rate": (
                sum(float(s["selection_accepted"]) for s in block_stats) / n),
            "selection_device": selection_device.type,
            "mean_selection_reference_mse": (
                sum(s["selection_reference_mse"] for s in block_stats) / n),
            "mean_selection_deployed_mse": sum(
                s["selection_candidate_mse"] if s["selection_accepted"]
                else s["selection_reference_mse"] for s in block_stats
            ) / n,
        }
        if config.learn_scales:
            stats_out["rotation_train"].update({
                "mean_scale_multiplier": sum(
                    s["scale_multiplier_mean"] for s in block_stats) / n,
                "min_scale_multiplier": min(
                    s["scale_multiplier_min"] for s in block_stats),
                "max_scale_multiplier": max(
                    s["scale_multiplier_max"] for s in block_stats),
            })
        if distill_stats is not None:
            stats_out["distillation"] = distill_stats
    logger.info("Block-trained and patched %d linear layers", total_patched)
    return model
