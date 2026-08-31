"""Engine-neutral ingestion and paired reporting for competitive evaluations."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rotquant.eval.competition import (
    ArtifactEvaluation,
    CompetitiveEvalProtocol,
    compare_matched_artifacts,
)
from rotquant.eval.data_manifest import DatasetManifest, token_sequence_sha256

RUN_REPORT_SCHEMA_VERSION = 1
FAILURE_STAGES = frozenset({"load", "prefill", "logits", "generation", "scoring"})
ARTIFACT_IDENTITY_SCHEME = "rotquant-artifact-bundle-v1"


def _require_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _content_fingerprint(payload: dict[str, Any], field: str) -> str:
    content = dict(payload)
    content.pop(field, None)
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finalize_fingerprint(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _content_fingerprint(result, field)
    return result


def _verify_fingerprint(payload: dict[str, Any], field: str) -> None:
    expected = payload.get(field)
    _require_sha256(field, expected)
    if expected != _content_fingerprint(payload, field):
        raise ValueError(f"{field} does not match the report contents")


@dataclass(frozen=True)
class ArtifactFile:
    """One file counted in a deployed artifact's size and identity."""

    name: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("artifact file name must not be empty")
        _require_sha256("artifact file sha256", self.sha256)
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, Integral)
            or self.bytes <= 0
        ):
            raise ValueError("artifact file bytes must be a positive integer")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> ArtifactFile:
        return cls(**payload)


def artifact_identity(files: tuple[ArtifactFile, ...]) -> str:
    """Return a raw file hash for one file or a canonical bundle hash for many."""

    if not isinstance(files, tuple) or not files:
        raise ValueError("artifact files must be a non-empty tuple")
    if len(files) == 1:
        return files[0].sha256
    payload = {
        "scheme": ARTIFACT_IDENTITY_SCHEME,
        "files": [
            file.manifest() for file in sorted(files, key=lambda entry: entry.name)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifact_files(
    artifacts: tuple[tuple[str, str | Path], ...],
) -> tuple[ArtifactFile, ...]:
    """Measure explicitly named files; intended for metadata creation tools."""

    if not isinstance(artifacts, tuple) or not artifacts:
        raise ValueError("artifacts must be a non-empty tuple of (name, path) pairs")
    files = []
    for name, raw_path in artifacts:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"artifact path is not a file: {path}")
        files.append(
            ArtifactFile(
                name=name,
                sha256=_hash_file(path),
                bytes=path.stat().st_size,
            )
        )
    return tuple(files)


@dataclass(frozen=True)
class RunMetadata:
    """Artifact and engine identity, independent of the inference backend."""

    name: str
    format: str
    artifact_sha256: str
    artifact_bytes: int
    artifact_files: tuple[ArtifactFile, ...]
    engine: str
    engine_revision: str
    protocol_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("name", "format", "engine", "engine_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        _require_sha256("artifact_sha256", self.artifact_sha256)
        _require_sha256("protocol_fingerprint", self.protocol_fingerprint)
        if (
            isinstance(self.artifact_bytes, bool)
            or not isinstance(self.artifact_bytes, Integral)
            or self.artifact_bytes <= 0
        ):
            raise ValueError("artifact_bytes must be a positive integer")
        if not isinstance(self.artifact_files, tuple) or not self.artifact_files:
            raise ValueError("artifact_files must be a non-empty tuple")
        if any(not isinstance(file, ArtifactFile) for file in self.artifact_files):
            raise ValueError("artifact_files must contain ArtifactFile values")
        names = [file.name for file in self.artifact_files]
        if len(set(names)) != len(names):
            raise ValueError("artifact file names must be unique")
        if sum(file.bytes for file in self.artifact_files) != self.artifact_bytes:
            raise ValueError("artifact_bytes does not equal the sum of artifact files")
        if artifact_identity(self.artifact_files) != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match the artifact file manifest")
        if self.engine_revision.strip().casefold() in {
            "main",
            "master",
            "latest",
            "head",
        }:
            raise ValueError("engine_revision must pin an immutable revision")

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_files"] = [file.manifest() for file in self.artifact_files]
        return payload

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> RunMetadata:
        values = dict(payload)
        values["artifact_files"] = tuple(
            ArtifactFile.from_manifest(file) for file in values["artifact_files"]
        )
        return cls(**values)


@dataclass(frozen=True)
class PromptObservation:
    """Raw per-token outputs for one held-out prompt."""

    item_sha256: str
    domain: str
    teacher_kl: tuple[float, ...]
    top1_matches: tuple[bool, ...]
    source_continuation: tuple[int, ...]
    candidate_continuation: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_sha256("item_sha256", self.item_sha256)
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must not be empty")
        if not isinstance(self.teacher_kl, tuple) or not self.teacher_kl:
            raise ValueError("teacher_kl must be a non-empty tuple")
        for index, value in enumerate(self.teacher_kl):
            _finite_nonnegative(f"teacher_kl[{index}]", value)
        if (
            not isinstance(self.top1_matches, tuple)
            or len(self.top1_matches) != len(self.teacher_kl)
            or any(not isinstance(value, bool) for value in self.top1_matches)
        ):
            raise ValueError("top1_matches must contain one bool per teacher KL value")
        for name in ("source_continuation", "candidate_continuation"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a non-empty tuple")
            if any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
                for value in values
            ):
                raise ValueError(f"{name} values must be non-negative integers")

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "teacher_kl",
            "top1_matches",
            "source_continuation",
            "candidate_continuation",
        ):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> PromptObservation:
        values = dict(payload)
        for name in (
            "teacher_kl",
            "top1_matches",
            "source_continuation",
            "candidate_continuation",
        ):
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class RunFailure:
    """A structured infrastructure/scoring failure, never a wrong answer."""

    stage: Literal["load", "prefill", "logits", "generation", "scoring"]
    error_type: str
    message: str
    item_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in FAILURE_STAGES:
            raise ValueError("failure stage is not registered")
        for name in ("error_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.item_sha256 is not None:
            _require_sha256("item_sha256", self.item_sha256)

    def manifest(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> RunFailure:
        return cls(**payload)


def _matching_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = 0
    for left_token, right_token in zip(left, right, strict=True):
        if left_token != right_token:
            break
        length += 1
    return length


def _prompt_metrics(observation: PromptObservation) -> dict[str, Any]:
    trajectory_matches = np.equal(
        observation.source_continuation,
        observation.candidate_continuation,
    )
    return {
        "item_sha256": observation.item_sha256,
        "domain": observation.domain,
        "source_continuation_sha256": token_sequence_sha256(
            observation.source_continuation
        ),
        "candidate_continuation_sha256": token_sequence_sha256(
            observation.candidate_continuation
        ),
        "scored_tokens": len(observation.teacher_kl),
        "mean_teacher_kl": float(np.mean(observation.teacher_kl)),
        "top1_agreement": float(np.mean(observation.top1_matches)),
        "trajectory_token_agreement": float(np.mean(trajectory_matches)),
        "exact_trajectory": bool(np.all(trajectory_matches)),
        "matching_prefix": _matching_prefix(
            observation.source_continuation,
            observation.candidate_continuation,
        ),
    }


def _summary(per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_count": len(per_prompt),
        "scored_tokens": sum(row["scored_tokens"] for row in per_prompt),
        "mean_teacher_kl": float(np.mean([row["mean_teacher_kl"] for row in per_prompt])),
        "top1_agreement": float(np.mean([row["top1_agreement"] for row in per_prompt])),
        "trajectory_token_agreement": float(
            np.mean([row["trajectory_token_agreement"] for row in per_prompt])
        ),
        "exact_trajectory_rate": float(
            np.mean([row["exact_trajectory"] for row in per_prompt])
        ),
        "mean_matching_prefix": float(
            np.mean([row["matching_prefix"] for row in per_prompt])
        ),
    }


def aggregate_competitive_run(
    *,
    protocol: CompetitiveEvalProtocol,
    prompt_manifest: DatasetManifest,
    metadata: RunMetadata,
    observations: tuple[PromptObservation, ...],
    failures: tuple[RunFailure, ...] = (),
) -> dict[str, Any]:
    """Validate raw outputs and produce a completed or incomplete report.

    A report containing a load/generation/scoring failure remains useful for
    diagnostics, but it never receives an ``artifact_evaluation`` and therefore
    cannot enter a quality comparison.
    """

    if prompt_manifest.role != "evaluation":
        raise ValueError("prompt_manifest must be an evaluation manifest")
    if prompt_manifest.fingerprint != protocol.prompt_manifest_sha256:
        raise ValueError("prompt manifest does not match the supplied protocol")
    if metadata.protocol_fingerprint != protocol.fingerprint:
        raise ValueError("run metadata does not match the supplied protocol")

    expected = {item.token_sha256: item for item in prompt_manifest.items}
    if len(observations) != len({observation.item_sha256 for observation in observations}):
        raise ValueError("observations contain duplicate prompt identities")
    observed = {observation.item_sha256: observation for observation in observations}
    unexpected = set(observed).difference(expected)
    if unexpected:
        raise ValueError(f"observations contain {len(unexpected)} unexpected prompt(s)")
    failed_items = [failure.item_sha256 for failure in failures if failure.item_sha256]
    if len(failed_items) != len(set(failed_items)):
        raise ValueError("failures contain duplicate prompt identities")
    unexpected_failures = set(failed_items).difference(expected)
    if unexpected_failures:
        raise ValueError(f"failures contain {len(unexpected_failures)} unexpected prompt(s)")
    collision = set(observed).intersection(failed_items)
    if collision:
        raise ValueError("a prompt cannot have both an observation and a failure")

    for item_sha256, observation in observed.items():
        if observation.domain != expected[item_sha256].domain:
            raise ValueError(
                f"observation domain does not match manifest for {item_sha256}"
            )
        if len(observation.teacher_kl) != protocol.generation_tokens:
            raise ValueError(
                "each observation must contain one KL/top-1 score per generated token"
            )
        if (
            len(observation.source_continuation) != protocol.generation_tokens
            or len(observation.candidate_continuation) != protocol.generation_tokens
        ):
            raise ValueError(
                "each observation must contain the registered trajectory length"
            )

    missing = set(expected).difference(observed).difference(failed_items)
    ordered_observations = [
        observed[item.token_sha256]
        for item in prompt_manifest.items
        if item.token_sha256 in observed
    ]
    per_prompt = [_prompt_metrics(observation) for observation in ordered_observations]
    failure_counts = dict(sorted(Counter(failure.stage for failure in failures).items()))
    report: dict[str, Any] = {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "status": "incomplete" if failures or missing else "completed",
        "metadata": metadata.manifest(),
        "prompt_manifest_sha256": prompt_manifest.fingerprint,
        "expected_prompt_count": protocol.prompt_count,
        "observed_prompt_count": len(observations),
        "failure_counts": failure_counts,
        "failures": [failure.manifest() for failure in failures],
        "missing_item_sha256": sorted(missing),
        "per_prompt": per_prompt,
    }
    if failures or missing:
        return _finalize_fingerprint(report, "run_report_sha256")

    all_kl = np.asarray(
        [value for observation in ordered_observations for value in observation.teacher_kl],
        dtype=np.float64,
    )
    overall = _summary(per_prompt)
    domain_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_prompt:
        domain_rows[row["domain"]].append(row)
    domain_summaries = {
        domain: _summary(rows) for domain, rows in sorted(domain_rows.items())
    }
    evaluation = ArtifactEvaluation(
        name=metadata.name,
        format=metadata.format,
        protocol_fingerprint=metadata.protocol_fingerprint,
        artifact_bytes=int(metadata.artifact_bytes),
        scored_tokens=int(overall["scored_tokens"]),
        trajectory_prompts=protocol.prompt_count,
        trajectory_tokens=protocol.generation_tokens,
        mean_teacher_kl=float(np.mean(all_kl)),
        median_teacher_kl=float(np.median(all_kl)),
        p95_teacher_kl=float(np.percentile(all_kl, 95)),
        max_teacher_kl=float(np.max(all_kl)),
        top1_agreement=float(overall["top1_agreement"]),
        trajectory_token_agreement=float(overall["trajectory_token_agreement"]),
        exact_trajectory_rate=float(overall["exact_trajectory_rate"]),
        mean_matching_prefix=float(overall["mean_matching_prefix"]),
    )
    report["artifact_evaluation"] = evaluation.manifest()
    report["domain_summaries"] = domain_summaries
    return _finalize_fingerprint(report, "run_report_sha256")


def _paired_delta(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [float(left[metric]) - float(right[metric]) for left, right in zip(candidate, baseline, strict=True)],
        dtype=np.float64,
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    bootstrapped = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(bootstrapped, [2.5, 97.5])
    return float(lower), float(upper)


def compare_run_reports(
    *,
    candidate_report: dict[str, Any],
    baseline_report: dict[str, Any],
    protocol: CompetitiveEvalProtocol,
    max_size_delta_fraction: float = 0.01,
    bootstrap_draws: int = 2_000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Create paired prompt/domain deltas without declaring a winner."""

    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, Integral) or bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    for name, report in (("candidate", candidate_report), ("baseline", baseline_report)):
        _verify_fingerprint(report, "run_report_sha256")
        if report.get("schema_version") != RUN_REPORT_SCHEMA_VERSION:
            raise ValueError(f"{name} report uses an unsupported schema version")
        if report.get("status") != "completed" or "artifact_evaluation" not in report:
            raise ValueError(f"{name} report is incomplete and cannot be compared")
        if report.get("prompt_manifest_sha256") != protocol.prompt_manifest_sha256:
            raise ValueError(f"{name} report uses a different prompt manifest")

    candidate_evaluation = ArtifactEvaluation.from_manifest(
        candidate_report["artifact_evaluation"]
    )
    baseline_evaluation = ArtifactEvaluation.from_manifest(
        baseline_report["artifact_evaluation"]
    )
    candidate_metadata = RunMetadata.from_manifest(candidate_report["metadata"])
    baseline_metadata = RunMetadata.from_manifest(baseline_report["metadata"])
    for name, metadata, evaluation in (
        ("candidate", candidate_metadata, candidate_evaluation),
        ("baseline", baseline_metadata, baseline_evaluation),
    ):
        if metadata.protocol_fingerprint != protocol.fingerprint:
            raise ValueError(f"{name} metadata uses a different protocol")
        if (
            metadata.name != evaluation.name
            or metadata.format != evaluation.format
            or metadata.artifact_bytes != evaluation.artifact_bytes
        ):
            raise ValueError(f"{name} metadata does not match its artifact evaluation")
    artifact_comparison = compare_matched_artifacts(
        candidate_evaluation,
        baseline_evaluation,
        protocol,
        max_size_delta_fraction=max_size_delta_fraction,
    )

    candidate_by_id = {
        row["item_sha256"]: row for row in candidate_report["per_prompt"]
    }
    baseline_by_id = {
        row["item_sha256"]: row for row in baseline_report["per_prompt"]
    }
    if len(candidate_by_id) != len(candidate_report["per_prompt"]):
        raise ValueError("candidate report contains duplicate prompt identities")
    if len(baseline_by_id) != len(baseline_report["per_prompt"]):
        raise ValueError("baseline report contains duplicate prompt identities")
    if set(candidate_by_id) != set(baseline_by_id):
        raise ValueError("reports do not contain the same prompt identities")
    identities = list(protocol.prompt_item_sha256)
    if set(identities) != set(candidate_by_id):
        raise ValueError("reports do not contain every protocol prompt identity")
    candidate = [candidate_by_id[identity] for identity in identities]
    baseline = [baseline_by_id[identity] for identity in identities]
    for left, right in zip(candidate, baseline, strict=True):
        if left["domain"] != right["domain"]:
            raise ValueError("paired prompt domains differ")
        if left["source_continuation_sha256"] != right["source_continuation_sha256"]:
            raise ValueError("paired reports use different source continuations")

    metrics = (
        "mean_teacher_kl",
        "top1_agreement",
        "trajectory_token_agreement",
        "exact_trajectory",
        "matching_prefix",
    )
    deltas = {
        metric: _paired_delta(candidate, baseline, metric) for metric in metrics
    }
    paired_summary = {}
    for index, (metric, values) in enumerate(deltas.items()):
        lower, upper = _bootstrap_mean_interval(
            values,
            draws=int(bootstrap_draws),
            seed=int(bootstrap_seed) + index,
        )
        paired_summary[metric] = {
            "mean_delta": float(np.mean(values)),
            "bootstrap_95_ci": [lower, upper],
        }

    domain_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(candidate):
        domain_indices[row["domain"]].append(index)
    domain_deltas = {}
    for domain, indices in sorted(domain_indices.items()):
        domain_deltas[domain] = {
            metric: float(np.mean(values[indices])) for metric, values in deltas.items()
        }
    lower_is_better = {"mean_teacher_kl"}
    worst_domain_deltas = {}
    for metric in metrics:
        ordered = sorted(
            (
                (domain, values[metric])
                for domain, values in domain_deltas.items()
            ),
            key=lambda entry: entry[1],
            reverse=metric in lower_is_better,
        )
        domain, value = ordered[0]
        worst_domain_deltas[metric] = {
            "domain": domain,
            "mean_delta": value,
        }

    per_prompt_deltas = []
    for index, identity in enumerate(identities):
        per_prompt_deltas.append(
            {
                "item_sha256": identity,
                "domain": candidate[index]["domain"],
                **{metric: float(values[index]) for metric, values in deltas.items()},
            }
        )
    comparison = {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "interpretation": "candidate minus baseline; no quality winner is inferred",
        "protocol_fingerprint": protocol.fingerprint,
        "prompt_manifest_sha256": protocol.prompt_manifest_sha256,
        "candidate_metadata": candidate_report["metadata"],
        "baseline_metadata": baseline_report["metadata"],
        "artifact_comparison": artifact_comparison,
        "paired_prompt_deltas": paired_summary,
        "domain_deltas": domain_deltas,
        "worst_domain_deltas": worst_domain_deltas,
        "per_prompt_deltas": per_prompt_deltas,
        "bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(bootstrap_seed),
            "confidence": 0.95,
        },
    }
    return _finalize_fingerprint(comparison, "comparison_sha256")


__all__ = [
    "ARTIFACT_IDENTITY_SCHEME",
    "FAILURE_STAGES",
    "RUN_REPORT_SCHEMA_VERSION",
    "ArtifactFile",
    "PromptObservation",
    "RunFailure",
    "RunMetadata",
    "aggregate_competitive_run",
    "artifact_identity",
    "compare_run_reports",
    "inspect_artifact_files",
]
