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
import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._internal import cpu_staging_linear, get_parent
from .adapters import resolve_model_adapter
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
    # Serialized checkpoints include deterministic framing and required
    # tokenizer/config files that are not registered model tensors.  For a
    # provider-size comparison, target the complete exported artifact and
    # subtract a measured conservative overhead during allocation.  The final
    # exported artifact remains the authority for the release gate.
    target_artifact_bytes: int | None = None
    artifact_overhead_bytes: int = 0
    target_tolerance_fraction: float = 0.01
    require_target_match: bool = False
    max_tokens: int = 32
    local_weight: float = 1.0
    global_kl_weight: float = 0.0
    global_kl_temperature: float = 1.0
    global_kl_batches: int = 0
    # ``inherit`` scores exactly the quantizer that will be deployed. Cheaper
    # proxies remain available as explicit experimental controls, but are never
    # silently substituted for the final pack.
    scoring_error_comp: str = "inherit"
    scoring_scale: str = "inherit"
    # Absolute summed output error is the empirical tr(dW H dW^T). Relative
    # error is safer when combining heterogeneous projection families. Robust
    # median normalization puts local error and marginal teacher KL on
    # commensurable scales before their user weights are applied.
    local_normalization: str = "relative"
    score_normalization: str = "median"
    # ``random`` is a matched-budget negative control. It follows a seeded
    # random downgrade order while retaining the exact same candidate formats
    # and byte accounting as the sensitivity-guided allocator. ``pareto`` is a
    # bucketed multiple-choice knapsack solver; unlike greedy downgrades it can
    # choose non-adjacent formats and reject dominated candidates globally.
    allocation: str = "pareto"
    allocation_granularity_bytes: int = 262_144
    # Optional exact-byte single/pair exchange passes after the bucketed
    # dynamic program. This repairs bucket-frontier approximation error while
    # preserving the same additive measured objective and byte interval.
    refinement_passes: int = 0
    # Persist an incomplete candidate table periodically when the runner gives
    # the allocator a content-addressed cache key.  This turns a multi-hour
    # model screen into resumable work without writing to Drive after every
    # projection.
    score_checkpoint_interval: int = 8
    # Global bounds are allocation-only controls, so a broad 2--8-bit candidate
    # screen can be reused by conservative adjacent-bit policies.
    allocation_min_bits: int | None = None
    allocation_max_bits: int | None = None
    # Automatically keep the most sensitive fraction of projections at or
    # above ``protect_min_bits``. Sensitivity is measured from the already
    # collected final-quantizer candidate table, not from model-name folklore.
    protect_top_fraction: float = 0.0
    protect_min_bits: int | None = None
    protect_metric: str = "global_kl"
    # Optional fail-closed audit. When both local and marginal-KL measurements
    # exist, abort allocation if their downgrade rankings correlate less than
    # this threshold instead of trusting a demonstrably invalid proxy.
    min_proxy_rank_correlation: float | None = None
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
        for field_name in ("target_complete_bytes", "target_artifact_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or int(value) < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            self.target_complete_bytes is not None
            and self.target_artifact_bytes is not None
        ):
            raise ValueError(
                "target_complete_bytes and target_artifact_bytes are mutually "
                "exclusive"
            )
        if (
            isinstance(self.artifact_overhead_bytes, bool)
            or int(self.artifact_overhead_bytes) < 0
        ):
            raise ValueError("artifact_overhead_bytes must be non-negative")
        if self.target_artifact_bytes is None and self.artifact_overhead_bytes:
            raise ValueError(
                "artifact_overhead_bytes requires target_artifact_bytes"
            )
        if (
            self.target_artifact_bytes is not None
            and self.artifact_overhead_bytes >= self.target_artifact_bytes
        ):
            raise ValueError(
                "artifact_overhead_bytes must be smaller than "
                "target_artifact_bytes"
            )
        if not 0 <= self.target_tolerance_fraction <= 0.25:
            raise ValueError("target_tolerance_fraction must be in [0, 0.25]")
        if not isinstance(self.require_target_match, bool):
            raise TypeError("require_target_match must be boolean")
        if self.require_target_match and (
            self.target_complete_bytes is None
            and self.target_artifact_bytes is None
        ):
            raise ValueError(
                "require_target_match requires a complete or artifact byte target"
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
        if self.global_kl_weight > 0 and self.global_kl_batches == 0:
            raise ValueError(
                "global_kl_weight > 0 requires global_kl_batches > 0"
            )
        if self.scoring_error_comp not in {"inherit", "none", "gptq"}:
            raise ValueError(
                "scoring_error_comp must be 'inherit', 'none', or 'gptq'"
            )
        if self.scoring_scale not in {
            "inherit", "rms", "mse_search", "turboquant"
        }:
            raise ValueError(
                "scoring_scale must be 'inherit', 'rms', 'mse_search', or "
                "'turboquant'"
            )
        if self.local_normalization not in {"absolute", "relative"}:
            raise ValueError(
                "local_normalization must be 'absolute' or 'relative'"
            )
        if self.score_normalization not in {"none", "median"}:
            raise ValueError("score_normalization must be 'none' or 'median'")
        if self.allocation not in {
            "greedy", "pareto", "random", "random_pareto"
        }:
            raise ValueError(
                "dynamic allocation must be 'greedy', 'pareto', 'random', "
                "or 'random_pareto'"
            )
        if (
            isinstance(self.allocation_granularity_bytes, bool)
            or int(self.allocation_granularity_bytes) < 1
        ):
            raise ValueError("allocation_granularity_bytes must be positive")
        if (
            isinstance(self.refinement_passes, bool)
            or int(self.refinement_passes) < 0
        ):
            raise ValueError("refinement_passes must be non-negative")
        if (
            isinstance(self.score_checkpoint_interval, bool)
            or int(self.score_checkpoint_interval) < 1
        ):
            raise ValueError("score_checkpoint_interval must be positive")
        for field_name in ("allocation_min_bits", "allocation_max_bits"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or int(value) not in bits
            ):
                raise ValueError(
                    f"{field_name} must be one of candidate_bits or None"
                )
        if (
            self.allocation_min_bits is not None
            and self.allocation_max_bits is not None
            and self.allocation_min_bits > self.allocation_max_bits
        ):
            raise ValueError(
                "allocation_min_bits must not exceed allocation_max_bits"
            )
        if not 0.0 <= self.protect_top_fraction <= 1.0:
            raise ValueError("protect_top_fraction must be in [0, 1]")
        if self.protect_min_bits is not None and (
            isinstance(self.protect_min_bits, bool)
            or int(self.protect_min_bits) not in bits
        ):
            raise ValueError("protect_min_bits must be one of candidate_bits")
        if self.protect_top_fraction and self.protect_min_bits is None:
            raise ValueError(
                "protect_top_fraction requires protect_min_bits"
            )
        if self.protect_metric not in {
            "score", "local_error", "local_relative_error", "global_kl"
        }:
            raise ValueError("unknown protect_metric")
        if self.protect_metric == "global_kl" \
                and self.protect_top_fraction \
                and self.global_kl_batches == 0:
            raise ValueError(
                "global_kl protection requires global_kl_batches > 0"
            )
        if self.min_proxy_rank_correlation is not None and not (
            -1.0 <= float(self.min_proxy_rank_correlation) <= 1.0
        ):
            raise ValueError("min_proxy_rank_correlation must be in [-1, 1]")
        if self.min_proxy_rank_correlation is not None \
                and self.global_kl_batches == 0:
            raise ValueError(
                "min_proxy_rank_correlation requires global_kl_batches > 0"
            )
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
    reference_energy: float
    local_relative_error: float
    global_kl: float
    score: float
    normalized_local: float = 0.0
    normalized_global_kl: float = 0.0


# A stage runner evaluates matched arms in one Python process. Candidate
# reconstruction scores are source-model facts, so random and greedy allocation
# may share them when the runner supplies the same content-addressed context.
_CANDIDATE_SCORE_CACHE: dict[str, dict[str, list[CandidateScore]]] = {}
_CANDIDATE_CACHE_SCHEMA = 1


def _candidate_cache_path(key: str) -> Path | None:
    root = os.environ.get("ROTQUANT_DYNAMIC_SCORE_CACHE_DIR")
    if not root:
        return None
    digest = hashlib.sha256(key.encode()).hexdigest()
    return Path(root) / f"candidate-scores-v{_CANDIDATE_CACHE_SCHEMA}-{digest}.json"


def _load_candidate_score_cache(
    key: str,
) -> dict[str, list[CandidateScore]] | None:
    path = _candidate_cache_path(key)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != _CANDIDATE_CACHE_SCHEMA
            or payload.get("key") != key
        ):
            return None
        return {
            name: [
                CandidateScore(
                    bits=int(row["bits"]),
                    config=QuantConfig(**row["config"]),
                    packed_bytes=int(row["packed_bytes"]),
                    registered_bytes=int(row["registered_bytes"]),
                    codebook_bytes=int(row["codebook_bytes"]),
                    complete_bytes=int(row["complete_bytes"]),
                    local_error=float(row["local_error"]),
                    reference_energy=float(row["reference_energy"]),
                    local_relative_error=float(row["local_relative_error"]),
                    global_kl=float(row["global_kl"]),
                    score=0.0,
                )
                for row in rows
            ]
            for name, rows in payload["scores"].items()
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid dynamic score cache %s: %s", path, exc)
        return None


def _write_candidate_score_cache(
    key: str, scores: Mapping[str, Sequence[CandidateScore]]
) -> None:
    path = _candidate_cache_path(key)
    if path is None:
        return
    payload = {
        "schema": _CANDIDATE_CACHE_SCHEMA,
        "key": key,
        "scores": {
            name: [
                {
                    "bits": item.bits,
                    "config": asdict(item.config),
                    "packed_bytes": item.packed_bytes,
                    "registered_bytes": item.registered_bytes,
                    "codebook_bytes": item.codebook_bytes,
                    "complete_bytes": item.complete_bytes,
                    "local_error": item.local_error,
                    "reference_energy": item.reference_energy,
                    "local_relative_error": item.local_relative_error,
                    "global_kl": item.global_kl,
                }
                for item in items
            ]
            for name, items in scores.items()
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        logger.warning("Could not persist dynamic score cache %s: %s", path, exc)


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
                 activations: torch.Tensor | None,
                 max_tokens: int) -> tuple[float, float, float]:
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
            error = float(
                (quantized_weight - rotated_weight).pow(2).sum().item()
            )
            energy = float(rotated_weight.pow(2).sum().item())
            return error, energy, error / max(energy, 1e-30)
        inputs = activations[:max_tokens].to(device=device, dtype=torch.float32)
        rotated_inputs = candidate.act_rotation.rotate_activation(inputs)
        source_bias = (
            source.bias.detach().to(device=device, dtype=torch.float32)
            if source.bias is not None else None
        )
        candidate_bias = (
            candidate.bias.detach().to(device=device, dtype=torch.float32)
            if candidate.bias is not None else None
        )
        reference = F.linear(rotated_inputs, rotated_weight, source_bias)
        quantized = F.linear(rotated_inputs, quantized_weight, candidate_bias)
        token_errors = (quantized - reference).pow(2).reshape(
            -1, reference.shape[-1]
        ).sum(dim=-1)
        token_energy = reference.pow(2).reshape(
            -1, reference.shape[-1]
        ).sum(dim=-1)
        error = float(token_errors.mean().item())
        energy = float(token_energy.mean().item())
        return error, energy, error / max(energy, 1e-30)


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


def _candidate_context(source: nn.Linear, patch_cfg, seed: int,
                       hessian: torch.Tensor | None):
    """Build one deterministic rotation/Hessian context for every bit width."""

    source_device = source.weight.device
    source_dtype = source.weight.dtype
    stage = source_device.type == "mps"
    work = cpu_staging_linear(source) if stage else source
    weight_rotation, act_rotation = make_rotations(
        work.in_features, patch_cfg, seed, device=work.weight.device)
    commit_rotation_storage(weight_rotation, patch_cfg.rotation_storage_dtype)
    rotated_hessian = hessian
    if rotated_hessian is not None:
        rotated_hessian = rotated_hessian.to(work.weight.device)
        if patch_cfg.rotation not in ("none", "identity"):
            from ._internal import rotate_hessian
            rotated_hessian = rotate_hessian(weight_rotation, rotated_hessian)
    return (
        work, weight_rotation, act_rotation, rotated_hessian,
        source_device, source_dtype, stage,
    )


def _candidate_linear(source: nn.Linear, quant: QuantConfig, context,
                      activation_mean: torch.Tensor | None) -> QuantLinear:
    (work, weight_rotation, act_rotation, rotated_hessian,
     source_device, source_dtype, stage) = context
    candidate = QuantLinear.from_linear(
        work, quant, weight_rotation=weight_rotation,
        act_rotation=act_rotation, H=rotated_hessian,
        fallback=True, fallback_dtype=source_dtype)
    if quant.bias_correction in {"mean", "length_mean"}:
        if activation_mean is None:
            raise ValueError(
                "faithful dynamic scoring requires an activation mean for "
                "bias-corrected candidates"
            )
        candidate.apply_mean_bias_correction(source.weight, activation_mean)
    if stage:
        candidate.to(device=source_device, dtype=source_dtype)
    return candidate


def _score_candidates(model: nn.Module, targets, adapter, patch_cfg,
                      config: DynamicQuantConfig,
                      activations: dict[str, torch.Tensor],
                      hessians: Mapping[str, torch.Tensor],
                      activation_means: Mapping[str, torch.Tensor],
                      teacher_calls: Sequence[Any] | None,
                      existing_scores: Mapping[
                          str, Sequence[CandidateScore]
                      ] | None = None,
                      checkpoint=None,
                      ) -> dict[str, list[CandidateScore]]:
    device = next(model.parameters()).device
    dtype = next(parameter.dtype for parameter in model.parameters()
                 if parameter.is_floating_point())
    scores: dict[str, list[CandidateScore]] = {}
    scored_since_checkpoint = 0
    for index, (name, source_module, source) in enumerate(targets):
        allowed_bits = _allowed_bits(name, config)
        cached = list((existing_scores or {}).get(name, ()))
        if (
            len(cached) == len(allowed_bits)
            and tuple(sorted(item.bits for item in cached)) == allowed_bits
        ):
            scores[name] = sorted(
                (replace(item) for item in cached), key=lambda item: item.bits
            )
            logger.info(
                "dynamic resume %d/%d %s from candidate-score checkpoint",
                index + 1, len(targets), name,
            )
            continue
        layer_scores = []
        source_hessian = hessians.get(name)
        effective_error_comp = (
            patch_cfg.quant.error_comp
            if config.scoring_error_comp == "inherit"
            else config.scoring_error_comp
        )
        if effective_error_comp == "gptq" and source_hessian is None:
            raise ValueError(
                f"faithful GPTQ candidate scoring requires a Hessian for {name}"
            )
        context = _candidate_context(
            source, patch_cfg, patch_cfg.seed + index, source_hessian
        )
        for bits in allowed_bits:
            quant = replace(patch_cfg.quant, bits=bits)
            scoring_quant = replace(
                quant,
                error_comp=(
                    quant.error_comp
                    if config.scoring_error_comp == "inherit"
                    else config.scoring_error_comp
                ),
                scale=(
                    quant.scale
                    if config.scoring_scale == "inherit"
                    else config.scoring_scale
                ),
            )
            candidate = _candidate_linear(
                source, scoring_quant, context, activation_means.get(name))
            local, reference_energy, local_relative = _local_error(
                source, candidate, activations.get(name), config.max_tokens)
            global_kl = 0.0
            if teacher_calls is not None and config.global_kl_batches:
                parent, attr = get_parent(model, name)
                original = getattr(parent, attr)
                adapter.replace_quantized_module(
                    parent, attr, source_module, candidate
                )
                try:
                    global_kl = teacher_logit_kl(
                        model, teacher_calls, device, dtype,
                        temperature=config.global_kl_temperature,
                        max_batches=config.global_kl_batches)
                finally:
                    setattr(parent, attr, original)
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
                reference_energy=reference_energy,
                local_relative_error=local_relative,
                global_kl=global_kl,
                score=0.0,
            ))
            del candidate
        scores[name] = sorted(layer_scores, key=lambda item: item.bits)
        logger.info(
            "dynamic scored %d/%d %s: %s", index + 1, len(targets), name,
            ", ".join(
                f"{item.bits}b local={item.local_error:.3g} "
                f"rel={item.local_relative_error:.3g} "
                f"kl={item.global_kl:.3g}" for item in scores[name]))
        scored_since_checkpoint += 1
        if (
            checkpoint is not None
            and scored_since_checkpoint >= config.score_checkpoint_interval
        ):
            checkpoint(scores)
            scored_since_checkpoint = 0
        del context, source_hessian
    if checkpoint is not None and scored_since_checkpoint:
        checkpoint(scores)
    return scores


def _clone_candidate_scores(
    scores: Mapping[str, Sequence[CandidateScore]],
) -> dict[str, list[CandidateScore]]:
    """Detach objective-specific fields from the content-addressed raw cache."""

    return {
        name: [replace(item) for item in items]
        for name, items in scores.items()
    }


def _median_positive(values: Sequence[float]) -> float:
    positive = [float(value) for value in values if value > 0 and math.isfinite(value)]
    return statistics.median(positive) if positive else 1.0


def _compose_candidate_scores(
    scores: Mapping[str, Sequence[CandidateScore]],
    config: DynamicQuantConfig,
) -> dict[str, Any]:
    """Combine raw local and marginal-KL measurements on robust scales.

    Every layer's best measured candidate is its zero point. Subtracting that
    layer-specific constant does not change the allocation optimum, and avoids
    charging a format for irreducible source/teacher numerical differences.
    """

    local_attr = (
        "local_error"
        if config.local_normalization == "absolute"
        else "local_relative_error"
    )
    local_penalties: list[float] = []
    global_penalties: list[float] = []
    floors: dict[str, tuple[float, float]] = {}
    for name, items in scores.items():
        local_floor = min(float(getattr(item, local_attr)) for item in items)
        global_floor = min(float(item.global_kl) for item in items)
        floors[name] = (local_floor, global_floor)
        local_penalties.extend(
            max(0.0, float(getattr(item, local_attr)) - local_floor)
            for item in items
        )
        global_penalties.extend(
            max(0.0, float(item.global_kl) - global_floor)
            for item in items
        )
    if config.score_normalization == "median":
        local_scale = _median_positive(local_penalties)
        global_scale = _median_positive(global_penalties)
    else:
        local_scale = global_scale = 1.0
    for name, items in scores.items():
        local_floor, global_floor = floors[name]
        for item in items:
            item.normalized_local = max(
                0.0, float(getattr(item, local_attr)) - local_floor
            ) / local_scale
            item.normalized_global_kl = max(
                0.0, float(item.global_kl) - global_floor
            ) / global_scale
            item.score = (
                config.local_weight * item.normalized_local
                + config.global_kl_weight * item.normalized_global_kl
            )
    return {
        "local_metric": local_attr,
        "normalization": config.score_normalization,
        "local_scale": local_scale,
        "global_kl_scale": global_scale,
    }


def _rankdata(values: Sequence[float]) -> list[float]:
    """Return average ranks without requiring SciPy in the core package."""

    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _score_diagnostics(
    scores: Mapping[str, Sequence[CandidateScore]],
    config: DynamicQuantConfig,
) -> dict[str, Any]:
    """Audit monotonicity and local/global downgrade-rank agreement."""

    local_attr = (
        "local_error"
        if config.local_normalization == "absolute"
        else "local_relative_error"
    )
    local_deltas: list[float] = []
    global_deltas: list[float] = []
    local_violations = global_violations = comparisons = 0
    for items in scores.values():
        ordered = sorted(items, key=lambda item: item.bits)
        for lower, higher in pairwise(ordered):
            lower_local = float(getattr(lower, local_attr))
            higher_local = float(getattr(higher, local_attr))
            local_deltas.append(lower_local - higher_local)
            global_deltas.append(float(lower.global_kl - higher.global_kl))
            local_violations += int(higher_local > lower_local + 1e-12)
            global_violations += int(higher.global_kl > lower.global_kl + 1e-12)
            comparisons += 1
    correlation = None
    if config.global_kl_batches and local_deltas:
        correlation = _pearson(
            _rankdata(local_deltas), _rankdata(global_deltas)
        )
    return {
        "adjacent_comparisons": comparisons,
        "local_monotonicity_violation_fraction": (
            local_violations / comparisons if comparisons else None
        ),
        "global_kl_monotonicity_violation_fraction": (
            global_violations / comparisons
            if comparisons and config.global_kl_batches else None
        ),
        "local_global_spearman": correlation,
    }


def _metric_value(item: CandidateScore, metric: str) -> float:
    if metric == "score":
        return float(item.score)
    return float(getattr(item, metric))


def _allocation_candidates(
    scores: Mapping[str, Sequence[CandidateScore]],
    config: DynamicQuantConfig,
) -> tuple[dict[str, list[CandidateScore]], list[dict[str, Any]]]:
    """Apply allocation-only bounds and measured automatic protection."""

    candidates: dict[str, list[CandidateScore]] = {}
    for name, items in scores.items():
        allowed = [
            item for item in items
            if (
                config.allocation_min_bits is None
                or item.bits >= config.allocation_min_bits
            ) and (
                config.allocation_max_bits is None
                or item.bits <= config.allocation_max_bits
            )
        ]
        if not allowed:
            raise ValueError(
                f"allocation bit bounds leave no candidate for {name}"
            )
        candidates[name] = allowed

    protected: list[dict[str, Any]] = []
    count = math.ceil(config.protect_top_fraction * len(candidates))
    if count:
        sensitivities = []
        for order, (name, items) in enumerate(candidates.items()):
            low = min(items, key=lambda item: item.bits)
            high = max(items, key=lambda item: item.bits)
            sensitivity = max(
                0.0,
                _metric_value(low, config.protect_metric)
                - _metric_value(high, config.protect_metric),
            )
            sensitivities.append((-sensitivity, order, name, sensitivity))
        for _negative, _order, name, sensitivity in sorted(sensitivities)[:count]:
            original = candidates[name]
            kept = [
                item for item in original
                if item.bits >= int(config.protect_min_bits or 0)
            ]
            if not kept:
                raise ValueError(
                    f"protect_min_bits leaves no candidate for {name}"
                )
            candidates[name] = kept
            protected.append({
                "name": name,
                "metric": config.protect_metric,
                "sensitivity": sensitivity,
                "min_bits": config.protect_min_bits,
            })
    return candidates, protected


def _pareto_selection(
    candidates: Mapping[str, Sequence[CandidateScore]],
    *,
    size_attr: str,
    fixed_bytes: int,
    target_bytes: int,
    tolerance_bytes: int,
    granularity_bytes: int,
) -> tuple[dict[str, CandidateScore], dict[str, Any]]:
    """Solve a bucketed multiple-choice rate-distortion problem.

    States are indexed by quantized savings relative to the all-highest-precision
    recipe. Exact bytes are retained in every state and are used for the final
    tolerance check; bucketing only bounds search memory.
    """

    names = list(candidates)
    highest = {
        name: max(candidates[name], key=lambda item: item.bits)
        for name in names
    }
    high_total = fixed_bytes + sum(
        int(getattr(item, size_attr)) for item in highest.values()
    )
    positive_steps = [
        int(getattr(highest[name], size_attr)) - int(getattr(item, size_attr))
        for name in names
        for item in candidates[name]
        if int(getattr(highest[name], size_attr)) > int(getattr(item, size_attr))
    ]
    # Unit tests and genuinely small models need exact arithmetic. Large LLMs
    # retain the configured bounded-memory buckets even when their tensor byte
    # sizes share a much smaller storage-alignment gcd.
    if high_total <= 64 * granularity_bytes and positive_steps:
        exact_granularity = positive_steps[0]
        for step in positive_steps[1:]:
            exact_granularity = math.gcd(exact_granularity, step)
        granularity_bytes = max(1, exact_granularity)
    upper = target_bytes + tolerance_bytes
    lower = max(0, target_bytes - tolerance_bytes)
    required_max = max(0, high_total - lower)
    largest_step = max(
        (
            int(getattr(highest[name], size_attr))
            - min(int(getattr(item, size_attr)) for item in candidates[name])
        )
        for name in names
    )
    max_units = max(
        1,
        math.ceil((required_max + largest_step) / granularity_bytes),
    )
    # A coarse bucket can contain several useful paths: the lowest-distortion
    # state need not have enough exact savings, while the largest-saving state
    # may be unnecessarily destructive. Retain a small nondominated frontier
    # instead of collapsing every bucket to one point.
    states_per_bucket = 4
    # unit -> [(additive score, exact savings, choice indices), ...]
    states: dict[int, list[tuple[float, int, tuple[int, ...]]]] = {
        0: [(0.0, 0, ())]
    }
    peak_states = 1
    for name in names:
        high_size = int(getattr(highest[name], size_attr))
        layer = list(candidates[name])
        proposed: dict[int, list[tuple[float, int, tuple[int, ...]]]] = {}
        for bucket_states in states.values():
            for base_score, base_savings, choices in bucket_states:
                for index, item in enumerate(layer):
                    savings = (
                        base_savings + high_size
                        - int(getattr(item, size_attr))
                    )
                    unit = min(max_units, savings // granularity_bytes)
                    proposed.setdefault(unit, []).append((
                        base_score + float(item.score),
                        savings,
                        choices + (index,),
                    ))
        next_states = {}
        for unit, bucket_states in proposed.items():
            # Descending savings makes the best score seen so far the exact
            # two-objective Pareto dominance test within this byte bucket.
            frontier = []
            best_score = math.inf
            for state in sorted(bucket_states, key=lambda value: (-value[1], value[0])):
                if state[0] < best_score:
                    frontier.append(state)
                    best_score = state[0]
            if len(frontier) > states_per_bucket:
                by_score = sorted(frontier, key=lambda value: value[0])
                by_savings = sorted(frontier, key=lambda value: -value[1])
                retained = []
                for state in (*by_score[:2], *by_savings[:2]):
                    if state not in retained:
                        retained.append(state)
                frontier = retained[:states_per_bucket]
            next_states[unit] = frontier
        states = next_states
        peak_states = max(
            peak_states, sum(len(bucket) for bucket in states.values())
        )

    feasible = [
        state for bucket in states.values() for state in bucket
        if lower <= high_total - state[1] <= upper
    ]
    if feasible:
        chosen = min(
            feasible,
            key=lambda state: (
                state[0], abs((high_total - state[1]) - target_bytes)
            ),
        )
    else:
        under_budget = [
            state for bucket in states.values() for state in bucket
            if high_total - state[1] <= upper
        ]
        pool = under_budget or [
            state for bucket in states.values() for state in bucket
        ]
        chosen = min(
            pool,
            key=lambda state: (
                abs((high_total - state[1]) - target_bytes), state[0]
            ),
        )
    selected = {
        name: list(candidates[name])[index]
        for name, index in zip(names, chosen[2])
    }
    achieved = high_total - chosen[1]
    return selected, {
        "solver": "bucketed_multiple_choice_pareto",
        "granularity_bytes": granularity_bytes,
        "states_per_bucket": states_per_bucket,
        "peak_states": peak_states,
        "additive_score": chosen[0],
        "search_high_bytes": high_total,
        "search_achieved_bytes": achieved,
        "search_within_tolerance": lower <= achieved <= upper,
    }


def _refine_selection(
    candidates: Mapping[str, Sequence[CandidateScore]],
    selected: Mapping[str, CandidateScore],
    *,
    size_attr: str,
    fixed_bytes: int,
    target_bytes: int,
    tolerance_bytes: int,
    max_passes: int,
) -> tuple[dict[str, CandidateScore], dict[str, Any]]:
    """Improve an allocation with exact-byte single and pair exchanges.

    The bucketed dynamic program deliberately keeps a bounded frontier.  This
    deterministic local search repairs a missed exact-byte combination without
    pretending to model cross-layer quantization interactions: every accepted
    move strictly lowers the same measured additive objective and remains in
    the registered byte interval.
    """

    current = dict(selected)
    lower = max(0, target_bytes - tolerance_bytes)
    upper = target_bytes + tolerance_bytes

    def allocation_bytes() -> int:
        return fixed_bytes + sum(
            int(getattr(item, size_attr)) for item in current.values()
        )

    def allocation_score() -> float:
        return sum(float(item.score) for item in current.values())

    start_bytes = allocation_bytes()
    start_score = allocation_score()
    history: list[dict[str, Any]] = []
    for pass_index in range(int(max_passes)):
        base_bytes = allocation_bytes()
        moves: list[tuple[str, CandidateScore, int, float]] = []
        for name, items in candidates.items():
            chosen = current[name]
            for item in items:
                if item.bits == chosen.bits:
                    continue
                moves.append((
                    name,
                    item,
                    int(getattr(item, size_attr))
                    - int(getattr(chosen, size_attr)),
                    float(item.score) - float(chosen.score),
                ))

        best: tuple[
            tuple[float, int, int, str, str],
            tuple[str, CandidateScore],
            tuple[str, CandidateScore] | None,
        ] | None = None

        def consider(
            delta_bytes: int,
            delta_score: float,
            first: tuple[str, CandidateScore],
            second: tuple[str, CandidateScore] | None = None,
            current_base_bytes: int = base_bytes,
        ) -> None:
            nonlocal best
            achieved = current_base_bytes + delta_bytes
            if not lower <= achieved <= upper or delta_score >= -1e-12:
                return
            names = (
                first[0]
                if second is None
                else "/".join(sorted((first[0], second[0])))
            )
            bits = (
                str(first[1].bits)
                if second is None
                else "/".join(map(str, sorted((first[1].bits, second[1].bits))))
            )
            key = (
                delta_score,
                abs(achieved - target_bytes),
                1 if second is None else 2,
                names,
                bits,
            )
            if best is None or key < best[0]:
                best = (key, first, second)

        for name, item, delta_bytes, delta_score in moves:
            consider(delta_bytes, delta_score, (name, item))
        for left_index, left in enumerate(moves):
            for right in moves[left_index + 1:]:
                if left[0] == right[0]:
                    continue
                consider(
                    left[2] + right[2],
                    left[3] + right[3],
                    (left[0], left[1]),
                    (right[0], right[1]),
                )
        if best is None:
            break
        _key, first, second = best
        changes = []
        chosen_moves = (first,) if second is None else (first, second)
        for name, item in chosen_moves:
            previous = current[name]
            current[name] = item
            changes.append({
                "name": name,
                "from_bits": previous.bits,
                "to_bits": item.bits,
            })
        history.append({
            "pass": pass_index + 1,
            "changes": changes,
            "bytes": allocation_bytes(),
            "additive_score": allocation_score(),
        })

    end_bytes = allocation_bytes()
    end_score = allocation_score()
    return current, {
        "requested_passes": int(max_passes),
        "applied_passes": len(history),
        "start_bytes": start_bytes,
        "end_bytes": end_bytes,
        "start_additive_score": start_score,
        "end_additive_score": end_score,
        "score_improvement": start_score - end_score,
        "within_tolerance": lower <= end_bytes <= upper,
        "history": history,
    }


def select_dynamic_quantization(
    model: nn.Module,
    patch_cfg,
    *,
    activations: dict[str, torch.Tensor] | None = None,
    hessians: Mapping[str, torch.Tensor] | None = None,
    activation_means: Mapping[str, torch.Tensor] | None = None,
    teacher_calls: Sequence[Any] | None = None,
    score_cache_key: str | None = None,
) -> tuple[dict[str, QuantConfig], dict[str, Any]]:
    """Return a per-projection quantizer recipe and serializable diagnostics."""

    config = DynamicQuantConfig(**(patch_cfg.dynamic or {}))
    if config.global_kl_batches and not teacher_calls:
        raise ValueError(
            "global_kl_batches requires source-model teacher calls"
        )
    include = tuple(patch_cfg.include) if patch_cfg.include is not None else None
    exclude = tuple(patch_cfg.exclude or ())
    adapter = resolve_model_adapter(model, patch_cfg.adapter)
    targets = [
        (name, module, adapter.to_linear(module))
        for name, module in adapter.iter_quantizable_modules(model)
        if (include is None or any(term in name for term in include))
        and not any(term in name for term in exclude)
    ]
    if not targets:
        raise ValueError(
            f"dynamic quantization found no targets through adapter={adapter.name}"
        )
    target_by_name = {name: linear for name, _module, linear in targets}
    source_by_name = {name: module for name, module, _linear in targets}
    raw_scores = (
        _CANDIDATE_SCORE_CACHE.get(score_cache_key)
        if score_cache_key is not None else None
    )
    score_cache_source = "memory" if raw_scores is not None else None
    if raw_scores is None and score_cache_key is not None:
        raw_scores = _load_candidate_score_cache(score_cache_key)
        if raw_scores is not None:
            _CANDIDATE_SCORE_CACHE[score_cache_key] = raw_scores
            score_cache_source = "disk"
    score_cache_hit = raw_scores is not None
    expected_names = set(target_by_name)
    if raw_scores is not None and not set(raw_scores) <= expected_names:
        raise ValueError("dynamic candidate score cache target mismatch")
    complete_cache = raw_scores is not None and all(
        name in raw_scores
        and tuple(sorted(item.bits for item in raw_scores[name]))
        == _allowed_bits(name, config)
        for name in expected_names
    )
    if not complete_cache:
        partial_count = len(raw_scores or {})
        if partial_count:
            logger.info(
                "Resuming partial dynamic candidate scores for %d/%d layers "
                "(key=%s)",
                partial_count, len(targets), score_cache_key,
            )

        def checkpoint(partial_scores):
            if score_cache_key is None:
                return
            _CANDIDATE_SCORE_CACHE[score_cache_key] = partial_scores
            _write_candidate_score_cache(score_cache_key, partial_scores)

        raw_scores = _score_candidates(
            model, targets, adapter, patch_cfg, config,
            activations or {}, hessians or {}, activation_means or {},
            teacher_calls,
            existing_scores=raw_scores,
            checkpoint=checkpoint if score_cache_key is not None else None,
        )
        if score_cache_key is not None:
            _CANDIDATE_SCORE_CACHE[score_cache_key] = raw_scores
            _write_candidate_score_cache(score_cache_key, raw_scores)
        score_cache_source = (
            "computed-resume" if partial_count else "computed"
        )
    else:
        logger.info(
            "Reusing dynamic candidate scores for %d layers (key=%s)",
            len(raw_scores), score_cache_key,
        )

    scores = _clone_candidate_scores(raw_scores)
    score_composition = _compose_candidate_scores(scores, config)
    score_diagnostics = _score_diagnostics(scores, config)
    correlation = score_diagnostics["local_global_spearman"]
    if config.min_proxy_rank_correlation is not None and (
        correlation is None
        or correlation < config.min_proxy_rank_correlation
    ):
        raise ValueError(
            "dynamic local/global proxy correlation gate failed: "
            f"observed={correlation}, "
            f"required={config.min_proxy_rank_correlation}"
        )
    candidates, protected = _allocation_candidates(scores, config)
    randomizer = random.Random(patch_cfg.seed)
    total_weights = sum(
        target_by_name[name].weight.numel() for name in candidates
    )
    target_tensor_ids = {
        id(tensor)
        for module in source_by_name.values()
        for tensor in _persistent_registered_tensors(module)
    }
    fixed_tensors = [
        tensor for tensor in _persistent_registered_tensors(model)
        if id(tensor) not in target_tensor_ids
    ]
    fixed_complete_bytes = _tensor_bytes(fixed_tensors)
    target_bits = config.target_bpw * total_weights
    if config.target_artifact_bytes is not None:
        artifact_target_bytes = int(config.target_artifact_bytes)
        target_bytes = (
            artifact_target_bytes - int(config.artifact_overhead_bytes)
        )
        tolerance_bytes = round(
            artifact_target_bytes * config.target_tolerance_fraction
        )
        size_attr = "complete_bytes"
        fixed_search_bytes = fixed_complete_bytes
    elif config.target_complete_bytes is not None:
        target_bytes = int(config.target_complete_bytes)
        tolerance_bytes = round(
            target_bytes * config.target_tolerance_fraction
        )
        size_attr = "complete_bytes"
        fixed_search_bytes = fixed_complete_bytes
    else:
        target_bytes = math.floor(target_bits / 8)
        tolerance_bytes = 0
        size_attr = "packed_bytes"
        fixed_search_bytes = 0

    solver_stats: dict[str, Any]
    if config.allocation in {"pareto", "random_pareto"}:
        allocation_candidates = candidates
        if config.allocation == "random_pareto":
            # Preserve the exact same candidate palette and byte solver while
            # deliberately destroying any relationship between measured
            # sensitivity and the chosen allocation.
            allocation_candidates = {
                name: [
                    replace(item, score=randomizer.random())
                    for item in items
                ]
                for name, items in candidates.items()
            }
        selected, solver_stats = _pareto_selection(
            allocation_candidates,
            size_attr=size_attr,
            fixed_bytes=fixed_search_bytes,
            target_bytes=target_bytes,
            tolerance_bytes=tolerance_bytes,
            granularity_bytes=config.allocation_granularity_bytes,
        )
        if config.allocation == "random_pareto":
            solver_stats["solver"] = "seeded_random_multiple_choice_pareto"
            solver_stats["random_seed"] = patch_cfg.seed
        if config.refinement_passes:
            selected, refinement_stats = _refine_selection(
                allocation_candidates,
                selected,
                size_attr=size_attr,
                fixed_bytes=fixed_search_bytes,
                target_bytes=target_bytes,
                tolerance_bytes=tolerance_bytes,
                max_passes=config.refinement_passes,
            )
            solver_stats["refinement"] = refinement_stats
        if config.allocation == "random_pareto":
            selected = {
                name: next(
                    item for item in candidates[name]
                    if item.bits == random_item.bits
                )
                for name, random_item in selected.items()
            }
    else:
        selected_indices = {
            name: len(items) - 1 for name, items in candidates.items()
        }

        def search_bytes() -> int:
            return fixed_search_bytes + sum(
                int(getattr(candidates[name][index], size_attr))
                for name, index in selected_indices.items()
            )

        while search_bytes() > target_bytes + tolerance_bytes:
            moves = []
            for order, name in enumerate(selected_indices):
                index = selected_indices[name]
                if index == 0:
                    continue
                current = candidates[name][index]
                lower = candidates[name][index - 1]
                savings = int(getattr(current, size_attr)) - int(
                    getattr(lower, size_attr)
                )
                if savings <= 0:
                    continue
                penalty = max(0.0, lower.score - current.score)
                moves.append(((penalty / savings, -savings, order), name))
            if not moves:
                break
            if config.allocation == "random":
                _, selected_name = randomizer.choice(moves)
            else:
                _, selected_name = min(moves)
            selected_indices[selected_name] -= 1
        selected = {
            name: candidates[name][index]
            for name, index in selected_indices.items()
        }
        solver_stats = {
            "solver": (
                "seeded_random_adjacent_downgrade"
                if config.allocation == "random"
                else "greedy_adjacent_rate_distortion"
            ),
            "search_achieved_bytes": search_bytes(),
            "search_within_tolerance": (
                abs(search_bytes() - target_bytes) <= tolerance_bytes
                if (
                    config.target_complete_bytes is not None
                    or config.target_artifact_bytes is not None
                )
                else search_bytes() <= target_bytes
            ),
        }

    achieved_bits = sum(item.packed_bytes * 8 for item in selected.values())
    achieved_complete_bytes = fixed_complete_bytes + sum(
        item.complete_bytes for item in selected.values()
    )
    recipe = {
        name: item.config for name, item in selected.items()
    }
    count_by_bits: dict[str, int] = {}
    details = []
    for name, item in selected.items():
        count_by_bits[str(item.bits)] = count_by_bits.get(str(item.bits), 0) + 1
        details.append({
            "name": name,
            "bits": item.bits,
            "packed_bytes": item.packed_bytes,
            "registered_bytes": item.registered_bytes,
            "codebook_bytes": item.codebook_bytes,
            "complete_bytes": item.complete_bytes,
            "local_error": item.local_error,
            "reference_energy": item.reference_energy,
            "local_relative_error": item.local_relative_error,
            "global_kl": item.global_kl,
            "normalized_local": item.normalized_local,
            "normalized_global_kl": item.normalized_global_kl,
            "score": item.score,
        })
    eligible_bits = {
        name: {item.bits for item in items}
        for name, items in candidates.items()
    }
    allocation_payload = [
        {"name": name, "quant": asdict(item.config)}
        for name, item in sorted(selected.items())
    ]
    allocation_fingerprint = hashlib.sha256(json.dumps(
        allocation_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()
    target_domain_bytes = (
        int(config.target_artifact_bytes)
        if config.target_artifact_bytes is not None
        else config.target_complete_bytes
    )
    achieved_domain_bytes = (
        achieved_complete_bytes + int(config.artifact_overhead_bytes)
        if config.target_artifact_bytes is not None
        else achieved_complete_bytes
    )
    stats = {
        "config": asdict(config),
        "adapter": adapter.name,
        "candidate_score_cache_key": score_cache_key,
        "candidate_score_cache_hit": score_cache_hit,
        "candidate_score_cache_source": score_cache_source,
        "layers": len(targets),
        "candidate_scoring_matches_deployed": (
            config.scoring_error_comp == "inherit"
            and config.scoring_scale == "inherit"
        ),
        "score_composition": score_composition,
        "score_diagnostics": score_diagnostics,
        "protected_layers": protected,
        "solver": solver_stats,
        "target_bpw": config.target_bpw,
        "achieved_bpw": achieved_bits / total_weights,
        "target_reached": (
            achieved_domain_bytes <= int(target_domain_bytes)
            + int(tolerance_bytes or 0)
            if target_domain_bytes is not None
            else achieved_bits <= target_bits
        ),
        "packed_bytes": achieved_bits // 8,
        "fixed_complete_bytes": fixed_complete_bytes,
        "estimated_complete_bytes": achieved_complete_bytes,
        "target_complete_bytes": config.target_complete_bytes,
        "target_artifact_bytes": config.target_artifact_bytes,
        "artifact_overhead_bytes": config.artifact_overhead_bytes,
        "estimated_artifact_bytes": (
            achieved_complete_bytes + int(config.artifact_overhead_bytes)
            if config.target_artifact_bytes is not None else None
        ),
        "target_tolerance_bytes": (
            tolerance_bytes if target_domain_bytes is not None else None
        ),
        "target_error_bytes": (
            achieved_domain_bytes - int(target_domain_bytes)
            if target_domain_bytes is not None else None
        ),
        "within_target_tolerance": (
            abs(achieved_domain_bytes - int(target_domain_bytes))
            <= int(tolerance_bytes)
            if target_domain_bytes is not None else None
        ),
        "allocation_fingerprint": allocation_fingerprint,
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
                "reference_energy": item.reference_energy,
                "local_relative_error": item.local_relative_error,
                "global_kl": item.global_kl,
                "normalized_local": item.normalized_local,
                "normalized_global_kl": item.normalized_global_kl,
                "score": item.score,
                "eligible": item.bits in eligible_bits[name],
                "selected": selected[name].bits == item.bits,
            }
            for name, items in scores.items()
            for item in items
        ],
    }
    logger.info(
        "dynamic allocation: %.4f bpw target -> %.4f bpw; "
        "complete=%d complete_target=%s artifact_target=%s (%s)",
        config.target_bpw, stats["achieved_bpw"], achieved_complete_bytes,
        config.target_complete_bytes, config.target_artifact_bytes,
        count_by_bits)
    if config.require_target_match and not stats["within_target_tolerance"]:
        raise ValueError(
            "dynamic allocation missed byte target: "
            f"estimated={achieved_domain_bytes}, "
            f"target={target_domain_bytes}, "
            f"tolerance={tolerance_bytes}"
        )
    return recipe, stats
