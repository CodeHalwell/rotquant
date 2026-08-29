"""Model-specific mixed-precision allocation for RotQuant.

The allocator deliberately separates three concepts which are often conflated
under the name *dynamic quantisation*:

* a cheap, layer-local reconstruction score used to screen every candidate;
* an optional source-model logit-KL score measured by perturbing one projection
  at a time; and
* a byte-budgeted assignment of quantiser configurations to projections.

Nothing changes precision at inference time.  The resulting policy is a static,
model-specific mixed-precision recipe, analogous to modern dynamic GGUF recipes.
It can subsequently be passed through RotQuant's block training and held-out
selection stages.
"""
from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .linear import QuantLinear
from .patch import _cpu_staging_linear, _get_parent, _make_rotations
from .quantize import QuantConfig
from .utils import get_logger

logger = get_logger()


@dataclass(frozen=True)
class DynamicQuantConfig:
    """Configuration for model-specific precision allocation.

    ``target_bpw`` applies to packed projection weights, including their stored
    scale metadata but excluding rotations (constant across candidate widths).
    Patterns use shell-style matching when they contain a glob character and
    substring matching otherwise.
    """

    candidate_bits: tuple[int, ...] = (3, 4)
    target_bpw: float = 3.625
    max_tokens: int = 32
    local_weight: float = 1.0
    global_kl_weight: float = 1.0
    global_kl_temperature: float = 1.0
    global_kl_batches: int = 0
    # Each rule is ``{"match": "pattern", "min_bits": 4}``,
    # ``max_bits`` or an exact ``bits`` value. Later rules take precedence.
    rules: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        bits = tuple(sorted({int(value) for value in self.candidate_bits}))
        if not bits or any(value < 1 or value > 16 for value in bits):
            raise ValueError("candidate_bits must contain integers in [1, 16]")
        object.__setattr__(self, "candidate_bits", bits)
        if self.target_bpw <= 0:
            raise ValueError("target_bpw must be > 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.local_weight < 0 or self.global_kl_weight < 0:
            raise ValueError("dynamic score weights must be nonnegative")
        if self.local_weight + self.global_kl_weight <= 0:
            raise ValueError("at least one dynamic score weight must be positive")
        if self.global_kl_temperature <= 0:
            raise ValueError("global_kl_temperature must be > 0")
        if self.global_kl_batches < 0:
            raise ValueError("global_kl_batches must be >= 0")
        object.__setattr__(self, "rules", tuple(self.rules or ()))
        for rule in self.rules:
            if "match" not in rule:
                raise ValueError("every dynamic rule requires a match pattern")
            controls = {key for key in ("bits", "min_bits", "max_bits")
                        if key in rule}
            if not controls:
                raise ValueError(
                    "dynamic rules require bits, min_bits, or max_bits")


@dataclass
class CandidateScore:
    bits: int
    config: QuantConfig
    packed_bytes: int
    local_error: float
    global_kl: float
    score: float


def _matches(name: str, pattern: str) -> bool:
    if any(char in pattern for char in "*?["):
        return fnmatch.fnmatchcase(name, pattern)
    return pattern in name


def _allowed_bits(name: str, config: DynamicQuantConfig) -> tuple[int, ...]:
    minimum = min(config.candidate_bits)
    maximum = max(config.candidate_bits)
    exact: int | None = None
    for rule in config.rules:
        if not _matches(name, str(rule["match"])):
            continue
        if "bits" in rule:
            exact = int(rule["bits"])
        if "min_bits" in rule:
            minimum = max(minimum, int(rule["min_bits"]))
        if "max_bits" in rule:
            maximum = min(maximum, int(rule["max_bits"]))
    if exact is not None:
        if exact not in config.candidate_bits:
            raise ValueError(
                f"dynamic rule selects unavailable {exact}-bit candidate for {name}")
        return (exact,)
    allowed = tuple(value for value in config.candidate_bits
                    if minimum <= value <= maximum)
    if not allowed:
        raise ValueError(f"dynamic rules leave no candidate precision for {name}")
    return allowed


def _tree_to(value, device, dtype):
    if torch.is_tensor(value):
        target_dtype = dtype if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_tree_to(item, device, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to(item, device, dtype)
                for key, item in value.items()}
    return value


@torch.no_grad()
def teacher_logit_kl(model: nn.Module, calls: Sequence[Any], device,
                     dtype: torch.dtype, temperature: float = 1.0,
                     max_batches: int | None = None) -> float:
    """Mean token KL from stored source logits to ``model`` logits."""

    losses = []
    selected = calls if max_batches is None else calls[:max_batches]
    for call in selected:
        inputs = _tree_to(call.inputs, device, dtype)
        inputs.setdefault("use_cache", False)
        output = model(**inputs)
        student = (output.logits if hasattr(output, "logits")
                   else output[0]).float()
        teacher = call.logits.to(device=device, dtype=torch.float32)
        # A TeacherCall can retain either all positions or just a final one.
        if teacher.shape[-2] != student.shape[-2]:
            student = student[..., -teacher.shape[-2]:, :]
        mask = inputs.get("attention_mask")
        if mask is None:
            token_mask = torch.ones(
                student.shape[:-1], device=student.device, dtype=torch.float32)
        else:
            token_mask = mask[..., -student.shape[-2]:].float()
        scale = float(temperature)
        teacher_prob = F.softmax(teacher / scale, dim=-1)
        student_log_prob = F.log_softmax(student / scale, dim=-1)
        token_kl = F.kl_div(
            student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        losses.append(float(
            ((token_kl * token_mask).sum()
             / token_mask.sum().clamp_min(1.0) * scale**2).item()))
    if not losses:
        raise ValueError("teacher_logit_kl requires at least one teacher call")
    return sum(losses) / len(losses)


def _local_error(source: nn.Linear, candidate: QuantLinear,
                 activations: torch.Tensor | None, max_tokens: int) -> float:
    device = candidate.qweight.packed.data.device
    with torch.no_grad():
        rotated_weight = candidate.act_rotation.rotate_weight(
            source.weight.detach().to(device=device, dtype=torch.float32))
        quantized_weight = candidate.qweight.dequantize().float()
        if activations is None:
            denominator = rotated_weight.pow(2).mean().clamp_min(1e-12)
            return float(((quantized_weight - rotated_weight).pow(2).mean()
                          / denominator).item())
        inputs = activations[:max_tokens].to(device=device, dtype=torch.float32)
        rotated_inputs = candidate.act_rotation.rotate_activation(inputs)
        reference = F.linear(rotated_inputs, rotated_weight)
        quantized = F.linear(rotated_inputs, quantized_weight)
        denominator = reference.pow(2).mean().clamp_min(1e-12)
        return float(((quantized - reference).pow(2).mean()
                      / denominator).item())


def _candidate_linear(source: nn.Linear, quant: QuantConfig, patch_cfg,
                      seed: int) -> QuantLinear:
    source_device = source.weight.device
    source_dtype = source.weight.dtype
    stage = source_device.type == "mps"
    work = _cpu_staging_linear(source) if stage else source
    weight_rotation, act_rotation = _make_rotations(
        work.in_features, patch_cfg, seed, device=work.weight.device)
    candidate = QuantLinear.from_linear(
        work, quant, weight_rotation=weight_rotation,
        act_rotation=act_rotation, fallback=True, fallback_dtype=source_dtype)
    if stage:
        candidate.to(device=source_device, dtype=source_dtype)
    return candidate


def _score_candidates(model: nn.Module, targets, patch_cfg,
                      config: DynamicQuantConfig,
                      activations: dict[str, torch.Tensor],
                      teacher_calls: Sequence[Any] | None,
                      ) -> dict[str, list[CandidateScore]]:
    device = next(model.parameters()).device
    dtype = next(parameter.dtype for parameter in model.parameters()
                 if parameter.is_floating_point())
    scores: dict[str, list[CandidateScore]] = {}
    for index, (name, source) in enumerate(targets):
        layer_scores = []
        for bits in _allowed_bits(name, config):
            quant = replace(patch_cfg.quant, bits=bits)
            candidate = _candidate_linear(
                source, quant, patch_cfg, patch_cfg.seed + index)
            local = _local_error(
                source, candidate, activations.get(name), config.max_tokens)
            global_kl = 0.0
            if teacher_calls is not None and config.global_kl_batches:
                parent, attr = _get_parent(model, name)
                original = getattr(parent, attr)
                setattr(parent, attr, candidate)
                try:
                    global_kl = teacher_logit_kl(
                        model, teacher_calls, device, dtype,
                        temperature=config.global_kl_temperature,
                        max_batches=config.global_kl_batches)
                finally:
                    setattr(parent, attr, original)
            score = (config.local_weight * local
                     + config.global_kl_weight * global_kl)
            layer_scores.append(CandidateScore(
                bits=bits,
                config=quant,
                packed_bytes=candidate.packed_state_bytes(),
                local_error=local,
                global_kl=global_kl,
                score=score,
            ))
            del candidate
        scores[name] = sorted(layer_scores, key=lambda item: item.bits)
        logger.info(
            "dynamic scored %d/%d %s: %s", index + 1, len(targets), name,
            ", ".join(
                f"{item.bits}b local={item.local_error:.3g} "
                f"kl={item.global_kl:.3g}" for item in scores[name]))
    return scores


def select_dynamic_quantization(
    model: nn.Module,
    patch_cfg,
    *,
    activations: dict[str, torch.Tensor] | None = None,
    teacher_calls: Sequence[Any] | None = None,
) -> tuple[dict[str, QuantConfig], dict[str, Any]]:
    """Return a per-projection quantizer recipe and serializable diagnostics."""

    config = DynamicQuantConfig(**(patch_cfg.dynamic or {}))
    include = tuple(patch_cfg.include) if patch_cfg.include is not None else None
    exclude = tuple(patch_cfg.exclude or ())
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)
               and (include is None or any(term in name for term in include))
               and not any(term in name for term in exclude)]
    if not targets:
        raise ValueError("dynamic quantization found no target linear layers")
    target_by_name = dict(targets)
    scores = _score_candidates(
        model, targets, patch_cfg, config, activations or {}, teacher_calls)

    selected = {name: len(items) - 1 for name, items in scores.items()}
    total_weights = sum(
        target_by_name[name].weight.numel() for name in selected)
    target_bits = config.target_bpw * total_weights

    def stored_bits() -> int:
        return sum(scores[name][index].packed_bytes * 8
                   for name, index in selected.items())

    # Multiple-choice rate allocation: begin at each tensor's highest allowed
    # precision, then repeatedly take the least harmful next downgrade per byte.
    while stored_bits() > target_bits:
        best = None
        for order, name in enumerate(selected):
            index = selected[name]
            if index == 0:
                continue
            current = scores[name][index]
            lower = scores[name][index - 1]
            savings = current.packed_bytes - lower.packed_bytes
            if savings <= 0:
                continue
            penalty = max(0.0, lower.score - current.score)
            # A deterministic infinitesimal tie-break prefers the larger saving,
            # then source module order.
            ratio = penalty / savings
            key = (ratio, -savings, order)
            if best is None or key < best[0]:
                best = (key, name)
        if best is None:
            break
        selected[best[1]] -= 1

    achieved_bits = stored_bits()
    recipe = {
        name: scores[name][index].config
        for name, index in selected.items()
    }
    count_by_bits: dict[str, int] = {}
    details = []
    for name, index in selected.items():
        item = scores[name][index]
        count_by_bits[str(item.bits)] = count_by_bits.get(str(item.bits), 0) + 1
        details.append({
            "name": name,
            "bits": item.bits,
            "packed_bytes": item.packed_bytes,
            "local_error": item.local_error,
            "global_kl": item.global_kl,
            "score": item.score,
        })
    stats = {
        "config": asdict(config),
        "layers": len(targets),
        "target_bpw": config.target_bpw,
        "achieved_bpw": achieved_bits / total_weights,
        "target_reached": achieved_bits <= target_bits,
        "packed_bytes": achieved_bits // 8,
        "weights": total_weights,
        "counts_by_bits": count_by_bits,
        "details": details,
    }
    logger.info(
        "dynamic allocation: %.4f bpw target -> %.4f bpw (%s)",
        config.target_bpw, stats["achieved_bpw"], count_by_bits)
    return recipe, stats
