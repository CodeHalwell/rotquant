"""Auditable, size-matched quantization competition contracts.

The benchmark runner may live in Python, llama.cpp, or another inference
engine.  These small data contracts make its outputs comparable: both
artifacts must use the same model, prompt manifest, tokenizer/chat template,
and generation policy, while deployed bytes are checked explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CompetitiveEvalProtocol:
    """Identity of a held-out distribution and trajectory evaluation."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    prompt_manifest_sha256: str
    calibration_manifest_sha256: str
    chat_template_sha256: str
    domains: tuple[str, ...]
    prompt_count: int = 300
    generation_tokens: int = 32
    temperature: float = 0.0
    include_auxiliary_heads: bool = False

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "prompt_manifest_sha256",
            "calibration_manifest_sha256",
            "chat_template_sha256",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.prompt_manifest_sha256 == self.calibration_manifest_sha256:
            raise ValueError("calibration and held-out prompt manifests must be disjoint")
        for name in (
            "prompt_manifest_sha256",
            "calibration_manifest_sha256",
            "chat_template_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.domains or any(not domain.strip() for domain in self.domains):
            raise ValueError("at least one non-empty evaluation domain is required")
        if self.prompt_count < 1 or self.generation_tokens < 1:
            raise ValueError("prompt_count and generation_tokens must be positive")
        if self.temperature != 0.0:
            raise ValueError("competitive trajectory evaluation must use greedy decoding")

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domains"] = list(self.domains)
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArtifactEvaluation:
    """Metrics for one deployed artifact under one protocol fingerprint."""

    name: str
    format: str
    protocol_fingerprint: str
    artifact_bytes: int
    scored_tokens: int
    trajectory_prompts: int
    trajectory_tokens: int
    mean_teacher_kl: float
    median_teacher_kl: float
    p95_teacher_kl: float
    top1_agreement: float
    trajectory_token_agreement: float
    exact_trajectory_rate: float
    mean_matching_prefix: float

    def __post_init__(self) -> None:
        if not self.name or not self.format or not self.protocol_fingerprint:
            raise ValueError("name, format, and protocol_fingerprint are required")
        _require_sha256("protocol_fingerprint", self.protocol_fingerprint)
        if self.artifact_bytes < 1 or self.scored_tokens < 1:
            raise ValueError("artifact_bytes and scored_tokens must be positive")
        if self.trajectory_prompts < 1 or self.trajectory_tokens < 1:
            raise ValueError("trajectory dimensions must be positive")
        for name in ("mean_teacher_kl", "median_teacher_kl", "p95_teacher_kl"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "top1_agreement",
            "trajectory_token_agreement",
            "exact_trajectory_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not 0 <= self.mean_matching_prefix <= self.trajectory_tokens:
            raise ValueError("mean_matching_prefix is outside the generated-token range")


def compare_matched_artifacts(
    candidate: ArtifactEvaluation,
    baseline: ArtifactEvaluation,
    protocol: CompetitiveEvalProtocol,
    *,
    max_size_delta_fraction: float = 0.01,
) -> dict[str, Any]:
    """Compare two artifacts only when protocol and deployed size are matched."""

    if not 0 <= max_size_delta_fraction < 1:
        raise ValueError("max_size_delta_fraction must be in [0, 1)")
    if candidate.protocol_fingerprint != baseline.protocol_fingerprint:
        raise ValueError("artifact evaluations use different protocol fingerprints")
    if candidate.protocol_fingerprint != protocol.fingerprint:
        raise ValueError("artifact evaluation does not match the supplied protocol")
    for evaluation in (candidate, baseline):
        if evaluation.trajectory_prompts != protocol.prompt_count:
            raise ValueError(
                f"{evaluation.name} did not evaluate every registered prompt"
            )
        if evaluation.trajectory_tokens != protocol.generation_tokens:
            raise ValueError(
                f"{evaluation.name} used the wrong trajectory length"
            )
    if candidate.scored_tokens != baseline.scored_tokens:
        raise ValueError("artifact evaluations contain different scored-token counts")
    size_delta = candidate.artifact_bytes / baseline.artifact_bytes - 1.0
    if abs(size_delta) > max_size_delta_fraction:
        raise ValueError(
            "artifacts are not size matched: "
            f"delta={size_delta:.4%}, tolerance={max_size_delta_fraction:.4%}"
        )
    return {
        "candidate": candidate.name,
        "baseline": baseline.name,
        "protocol_fingerprint": candidate.protocol_fingerprint,
        "candidate_artifact_bytes": candidate.artifact_bytes,
        "baseline_artifact_bytes": baseline.artifact_bytes,
        "size_delta_fraction": size_delta,
        "mean_teacher_kl_delta": (
            candidate.mean_teacher_kl - baseline.mean_teacher_kl
        ),
        "median_teacher_kl_delta": (
            candidate.median_teacher_kl - baseline.median_teacher_kl
        ),
        "p95_teacher_kl_delta": candidate.p95_teacher_kl - baseline.p95_teacher_kl,
        "top1_agreement_delta": candidate.top1_agreement - baseline.top1_agreement,
        "trajectory_token_agreement_delta": (
            candidate.trajectory_token_agreement
            - baseline.trajectory_token_agreement
        ),
        "exact_trajectory_rate_delta": (
            candidate.exact_trajectory_rate - baseline.exact_trajectory_rate
        ),
        "mean_matching_prefix_delta": (
            candidate.mean_matching_prefix - baseline.mean_matching_prefix
        ),
    }


__all__ = [
    "ArtifactEvaluation",
    "CompetitiveEvalProtocol",
    "compare_matched_artifacts",
]
