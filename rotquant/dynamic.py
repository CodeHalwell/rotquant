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
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._internal import cpu_staging_linear, get_parent
from .linear import QuantLinear
from .patch import commit_rotation_storage, make_rotations
from .quantize import QuantConfig
from .utils import get_logger

logger = get_logger(__name__)


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
    # A complete-model byte target takes precedence over ``target_bpw``.  It
    # includes every persistent source tensor that remains outside the selected
    # projections plus each candidate's codes, scales, codebook, bias, and
    # rotation state.  This is the mode used for same-size provider comparisons.
    target_complete_bytes: int | None = None
    target_tolerance_fraction: float = 0.01
    require_target_match: bool = False
    max_tokens: int = 32
    local_weight: float = 1.0
    global_kl_weight: float = 1.0
    global_kl_temperature: float = 1.0
    global_kl_batches: int = 0
    # Candidate importance scoring should be substantially cheaper than the
    # final GPTQ pack.  These settings change the scoring proxy only; the recipe
    # retains the base quantizer's exact error compensation and scale strategy.
    scoring_error_comp: str = "none"
    scoring_scale: str = "rms"
    # ``random`` is a matched-budget negative control. It follows a seeded
    # random downgrade order while retaining the exact same candidate formats
    # and byte accounting as the sensitivity-guided allocator.
    allocation: str = "greedy"
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
        if self.target_complete_bytes is not None and (
            isinstance(self.target_complete_bytes, bool)
            or int(self.target_complete_bytes) < 1
        ):
            raise ValueError("target_complete_bytes must be a positive integer")
        if not 0 <= self.target_tolerance_fraction <= 0.25:
            raise ValueError("target_tolerance_fraction must be in [0, 0.25]")
        if not isinstance(self.require_target_match, bool):
            raise TypeError("require_target_match must be boolean")
        if self.require_target_match and self.target_complete_bytes is None:
            raise ValueError(
                "require_target_match requires target_complete_bytes"
            )
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
        if self.scoring_error_comp not in {"none", "gptq"}:
            raise ValueError("scoring_error_comp must be 'none' or 'gptq'")
        if self.scoring_scale not in {"rms", "mse_search", "turboquant"}:
            raise ValueError(
                "scoring_scale must be 'rms', 'mse_search', or 'turboquant'"
            )
        if self.allocation not in {"greedy", "random"}:
            raise ValueError("dynamic allocation must be 'greedy' or 'random'")
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
    registered_bytes: int
    codebook_bytes: int
    complete_bytes: int
    local_error: float
    global_kl: float
    score: float


# A stage runner evaluates matched arms in one Python process. Candidate
# reconstruction scores are source-model facts, so random and greedy allocation
# may share them when the runner supplies the same content-addressed context.
_CANDIDATE_SCORE_CACHE: dict[str, dict[str, list[CandidateScore]]] = {}


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
    """Expected summed output distortion for one invocation of ``source``.

    A per-layer relative mean is not comparable across projections with
    different output widths.  Summing output dimensions gives the sampled
    counterpart of ``tr(ΔW H ΔWᵀ)``; with no activations, an identity input
    covariance reduces that expression to the squared Frobenius norm.
    """
    device = candidate.qweight.packed.data.device
    with torch.no_grad():
        rotated_weight = candidate.act_rotation.rotate_weight(
            source.weight.detach().to(device=device, dtype=torch.float32))
        quantized_weight = candidate.qweight.dequantize().float()
        if activations is None:
            return float(
                (quantized_weight - rotated_weight).pow(2).sum().item()
            )
        inputs = activations[:max_tokens].to(device=device, dtype=torch.float32)
        rotated_inputs = candidate.act_rotation.rotate_activation(inputs)
        reference = F.linear(rotated_inputs, rotated_weight)
        quantized = F.linear(rotated_inputs, quantized_weight)
        token_errors = (quantized - reference).pow(2).reshape(
            -1, reference.shape[-1]
        ).sum(dim=-1)
        return float(token_errors.mean().item())


def _persistent_registered_tensors(module: nn.Module) -> list[torch.Tensor]:
    """Return unique parameters and persistent buffers owned by ``module``."""

    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for submodule in module.modules():
        for parameter in submodule._parameters.values():
            if parameter is not None and id(parameter) not in seen:
                seen.add(id(parameter))
                tensors.append(parameter)
        nonpersistent = getattr(submodule, "_non_persistent_buffers_set", set())
        for name, buffer in submodule._buffers.items():
            if (
                buffer is not None
                and name not in nonpersistent
                and id(buffer) not in seen
            ):
                seen.add(id(buffer))
                tensors.append(buffer)
    return tensors


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _candidate_codebook_bytes(candidate: QuantLinear) -> int:
    tensors = []
    for codebook in (
        candidate.qweight.codebook,
        candidate.qweight.residual_codebook,
    ):
        centroids = getattr(codebook, "centroids", None)
        if centroids is not None:
            tensors.append(centroids)
    # A primary and residual codebook are distinct today, but retain unique
    # accounting so future shared codebooks do not get charged twice.
    unique = {id(tensor): tensor for tensor in tensors}
    return _tensor_bytes(tuple(unique.values()))


def _candidate_linear(source: nn.Linear, quant: QuantConfig, patch_cfg,
                      seed: int) -> QuantLinear:
    source_device = source.weight.device
    source_dtype = source.weight.dtype
    stage = source_device.type == "mps"
    work = cpu_staging_linear(source) if stage else source
    weight_rotation, act_rotation = make_rotations(
        work.in_features, patch_cfg, seed, device=work.weight.device)
    commit_rotation_storage(weight_rotation, patch_cfg.rotation_storage_dtype)
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
            scoring_quant = replace(
                quant,
                error_comp=config.scoring_error_comp,
                scale=config.scoring_scale,
            )
            candidate = _candidate_linear(
                source, scoring_quant, patch_cfg, patch_cfg.seed + index)
            local = _local_error(
                source, candidate, activations.get(name), config.max_tokens)
            global_kl = 0.0
            if teacher_calls is not None and config.global_kl_batches:
                parent, attr = get_parent(model, name)
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
            registered_bytes = _tensor_bytes(
                _persistent_registered_tensors(candidate)
            )
            codebook_bytes = _candidate_codebook_bytes(candidate)
            packed_bytes = candidate.packed_state_bytes()
            layer_scores.append(CandidateScore(
                bits=bits,
                config=quant,
                packed_bytes=packed_bytes,
                registered_bytes=registered_bytes,
                codebook_bytes=codebook_bytes,
                complete_bytes=packed_bytes + registered_bytes + codebook_bytes,
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
    score_cache_key: str | None = None,
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
    scores = (
        _CANDIDATE_SCORE_CACHE.get(score_cache_key)
        if score_cache_key is not None else None
    )
    score_cache_hit = scores is not None
    if scores is None:
        scores = _score_candidates(
            model, targets, patch_cfg, config, activations or {}, teacher_calls
        )
        if score_cache_key is not None:
            _CANDIDATE_SCORE_CACHE[score_cache_key] = scores
    else:
        if set(scores) != set(target_by_name):
            raise ValueError("dynamic candidate score cache target mismatch")
        logger.info(
            "Reusing dynamic candidate scores for %d layers (key=%s)",
            len(scores), score_cache_key,
        )

    selected = {name: len(items) - 1 for name, items in scores.items()}
    randomizer = random.Random(patch_cfg.seed)
    total_weights = sum(
        target_by_name[name].weight.numel() for name in selected)
    target_tensor_ids = {
        id(tensor)
        for _name, module in targets
        for tensor in _persistent_registered_tensors(module)
    }
    fixed_tensors = [
        tensor for tensor in _persistent_registered_tensors(model)
        if id(tensor) not in target_tensor_ids
    ]
    fixed_complete_bytes = _tensor_bytes(fixed_tensors)
    target_bits = config.target_bpw * total_weights

    def stored_bits() -> int:
        return sum(scores[name][index].packed_bytes * 8
                   for name, index in selected.items())

    def selected_complete_bytes() -> int:
        return fixed_complete_bytes + sum(
            scores[name][index].complete_bytes
            for name, index in selected.items()
        )

    if config.target_complete_bytes is not None:
        target_bytes = int(config.target_complete_bytes)
        tolerance_bytes = round(
            target_bytes * config.target_tolerance_fraction
        )

        def over_budget() -> bool:
            return selected_complete_bytes() > target_bytes + tolerance_bytes

        def move_savings(current: CandidateScore, lower: CandidateScore) -> int:
            return current.complete_bytes - lower.complete_bytes
    else:
        target_bytes = None
        tolerance_bytes = None

        def over_budget() -> bool:
            return stored_bits() > target_bits

        def move_savings(current: CandidateScore, lower: CandidateScore) -> int:
            return current.packed_bytes - lower.packed_bytes

    # Multiple-choice rate allocation: begin at each tensor's highest allowed
    # precision, then repeatedly take the least harmful next downgrade per byte.
    while over_budget():
        moves = []
        for order, name in enumerate(selected):
            index = selected[name]
            if index == 0:
                continue
            current = scores[name][index]
            lower = scores[name][index - 1]
            savings = move_savings(current, lower)
            if savings <= 0:
                continue
            penalty = max(0.0, lower.score - current.score)
            # A deterministic infinitesimal tie-break prefers the larger saving,
            # then source module order.
            ratio = penalty / savings
            key = (ratio, -savings, order)
            moves.append((key, name))
        if not moves:
            break
        if config.allocation == "random":
            _, selected_name = randomizer.choice(moves)
        else:
            _, selected_name = min(moves)
        selected[selected_name] -= 1

    achieved_bits = stored_bits()
    achieved_complete_bytes = selected_complete_bytes()
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
            "registered_bytes": item.registered_bytes,
            "codebook_bytes": item.codebook_bytes,
            "complete_bytes": item.complete_bytes,
            "local_error": item.local_error,
            "global_kl": item.global_kl,
            "score": item.score,
        })
    stats = {
        "config": asdict(config),
        "candidate_score_cache_key": score_cache_key,
        "candidate_score_cache_hit": score_cache_hit,
        "layers": len(targets),
        "target_bpw": config.target_bpw,
        "achieved_bpw": achieved_bits / total_weights,
        "target_reached": (
            achieved_complete_bytes <= int(config.target_complete_bytes)
            + int(tolerance_bytes or 0)
            if config.target_complete_bytes is not None
            else achieved_bits <= target_bits
        ),
        "packed_bytes": achieved_bits // 8,
        "fixed_complete_bytes": fixed_complete_bytes,
        "estimated_complete_bytes": achieved_complete_bytes,
        "target_complete_bytes": config.target_complete_bytes,
        "target_tolerance_bytes": tolerance_bytes,
        "target_error_bytes": (
            achieved_complete_bytes - int(config.target_complete_bytes)
            if config.target_complete_bytes is not None else None
        ),
        "within_target_tolerance": (
            abs(achieved_complete_bytes - int(config.target_complete_bytes))
            <= int(tolerance_bytes or 0)
            if config.target_complete_bytes is not None else None
        ),
        "weights": total_weights,
        "counts_by_bits": count_by_bits,
        "details": details,
        "candidate_table": [
            {
                "name": name,
                "bits": item.bits,
                "packed_bytes": item.packed_bytes,
                "registered_bytes": item.registered_bytes,
                "codebook_bytes": item.codebook_bytes,
                "complete_bytes": item.complete_bytes,
                "local_error": item.local_error,
                "global_kl": item.global_kl,
                "score": item.score,
                "selected": selected[name] == index,
            }
            for name, items in scores.items()
            for index, item in enumerate(items)
        ],
    }
    logger.info(
        "dynamic allocation: %.4f bpw target -> %.4f bpw; "
        "complete=%d target=%s (%s)",
        config.target_bpw, stats["achieved_bpw"], achieved_complete_bytes,
        config.target_complete_bytes, count_by_bits)
    if config.require_target_match and not stats["within_target_tolerance"]:
        raise ValueError(
            "dynamic allocation missed complete-model byte target: "
            f"estimated={achieved_complete_bytes}, "
            f"target={config.target_complete_bytes}, "
            f"tolerance={tolerance_bytes}"
        )
    return recipe, stats
