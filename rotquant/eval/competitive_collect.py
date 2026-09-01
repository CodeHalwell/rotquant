"""Reusable primitives for exact competitive source/candidate collection."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .competitive_run import ArtifactFile, PromptObservation


@dataclass(frozen=True)
class SourceReference:
    """Full teacher logits and greedy continuation for one registered prompt."""

    item_sha256: str
    protocol_fingerprint: str
    teacher_logits: torch.Tensor
    continuation: tuple[int, ...]


def _digest_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("ascii"), dtype=np.uint8).copy()


def _decode_digest(value: np.ndarray, name: str) -> str:
    try:
        result = value.astype(np.uint8, copy=False).tobytes().decode("ascii")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid {name} in source reference") from error
    if (
        len(result) != 64
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"invalid {name} in source reference")
    return result


def source_reference_path(root: str | Path, item_sha256: str) -> Path:
    return Path(root) / f"{item_sha256}.npz"


def verify_artifact_files(
    candidate: str | Path,
    artifact_files: tuple[ArtifactFile, ...],
) -> None:
    """Verify that the model being loaded is the artifact named by metadata."""

    root = Path(candidate)
    if root.is_file():
        if len(artifact_files) != 1:
            raise ValueError("a single-file candidate needs one artifact metadata file")
        resolved = ((artifact_files[0], root),)
    elif root.is_dir():
        entries = []
        for artifact in artifact_files:
            relative = Path(artifact.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("artifact metadata names must be safe relative paths")
            entries.append((artifact, root / relative))
        resolved = tuple(entries)
    else:
        raise ValueError(
            "competitive candidate must be a local file or directory so its "
            "artifact identity can be verified"
        )
    for artifact, path in resolved:
        if not path.is_file():
            raise ValueError(f"candidate artifact file is missing: {path}")
        if path.stat().st_size != artifact.bytes:
            raise ValueError(f"candidate artifact size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError(f"candidate artifact SHA-256 mismatch: {path}")


def save_source_reference(path: str | Path, reference: SourceReference) -> None:
    """Atomically store an object-free NumPy reference record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logits = reference.teacher_logits.detach().cpu()
    if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]:
        raise ValueError("teacher_logits must have shape [tokens, vocabulary]")
    continuation = np.asarray(reference.continuation, dtype=np.int64)
    if continuation.ndim != 1 or len(continuation) != logits.shape[0]:
        raise ValueError("continuation length must match teacher-logit tokens")
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            item_sha256=_digest_array(reference.item_sha256),
            protocol_fingerprint=_digest_array(reference.protocol_fingerprint),
            teacher_logits=logits.to(torch.float16).numpy(),
            continuation=continuation,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_source_reference(
    path: str | Path,
    *,
    expected_item_sha256: str | None = None,
    expected_protocol_fingerprint: str | None = None,
    expected_tokens: int | None = None,
) -> SourceReference:
    """Load and validate a source record without enabling NumPy pickles."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "item_sha256",
            "protocol_fingerprint",
            "teacher_logits",
            "continuation",
        }
        if set(payload.files) != required:
            raise ValueError("source reference fields do not match the schema")
        item_sha256 = _decode_digest(payload["item_sha256"], "item_sha256")
        protocol_fingerprint = _decode_digest(
            payload["protocol_fingerprint"], "protocol_fingerprint"
        )
        logits = np.asarray(payload["teacher_logits"])
        continuation = np.asarray(payload["continuation"])
    if expected_item_sha256 is not None and item_sha256 != expected_item_sha256:
        raise ValueError("source reference item identity mismatch")
    if (
        expected_protocol_fingerprint is not None
        and protocol_fingerprint != expected_protocol_fingerprint
    ):
        raise ValueError("source reference protocol mismatch")
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("source teacher_logits must have shape [tokens, vocabulary]")
    if logits.dtype != np.float16:
        raise ValueError("source teacher_logits must use float16 storage")
    if continuation.ndim != 1 or continuation.shape[0] != logits.shape[0]:
        raise ValueError("source continuation and logits lengths differ")
    if expected_tokens is not None and logits.shape[0] != expected_tokens:
        raise ValueError("source reference uses the wrong continuation length")
    if not np.issubdtype(continuation.dtype, np.integer):
        raise ValueError("source continuation must contain integer token IDs")
    return SourceReference(
        item_sha256=item_sha256,
        protocol_fingerprint=protocol_fingerprint,
        teacher_logits=torch.from_numpy(logits.copy()),
        continuation=tuple(int(value) for value in continuation),
    )


def model_logits(output: Any) -> torch.Tensor:
    logits = output.logits if hasattr(output, "logits") else output[0]
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("model output must contain [batch, tokens, vocabulary] logits")
    return logits


def continuation_prediction_logits(
    output: Any,
    *,
    prompt_tokens: int,
    continuation_tokens: int,
) -> torch.Tensor:
    """Select logits that predict each teacher-forced continuation token."""

    if prompt_tokens < 1 or continuation_tokens < 1:
        raise ValueError("prompt and continuation lengths must be positive")
    logits = model_logits(output)
    start = prompt_tokens - 1
    stop = start + continuation_tokens
    if logits.shape[1] < stop:
        raise ValueError("model output is too short for continuation scoring")
    return logits[0, start:stop, :]


def score_candidate(
    *,
    item_sha256: str,
    domain: str,
    source: SourceReference,
    candidate_logits: torch.Tensor,
    candidate_continuation: tuple[int, ...],
) -> PromptObservation:
    """Compute full-distribution teacher KL and top-1 agreement per token."""

    teacher = source.teacher_logits.to(
        device=candidate_logits.device, dtype=torch.float32
    )
    candidate = candidate_logits.to(dtype=torch.float32)
    if candidate.shape != teacher.shape:
        raise ValueError(
            "source and candidate logits must use identical token/vocabulary shapes"
        )
    teacher_log_prob = F.log_softmax(teacher, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    candidate_log_prob = F.log_softmax(candidate, dim=-1)
    token_kl = torch.sum(
        teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1
    ).clamp_min(0.0)
    top1 = teacher.argmax(dim=-1).eq(candidate.argmax(dim=-1))
    return PromptObservation(
        item_sha256=item_sha256,
        domain=domain,
        teacher_kl=tuple(float(value) for value in token_kl.detach().cpu()),
        top1_matches=tuple(bool(value) for value in top1.detach().cpu()),
        source_continuation=source.continuation,
        candidate_continuation=candidate_continuation,
    )


__all__ = [
    "SourceReference",
    "continuation_prediction_logits",
    "load_source_reference",
    "model_logits",
    "save_source_reference",
    "score_candidate",
    "source_reference_path",
    "verify_artifact_files",
]
