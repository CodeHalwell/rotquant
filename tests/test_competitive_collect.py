from __future__ import annotations

import hashlib

import torch

from rotquant.eval.competitive_collect import (
    SourceReference,
    continuation_prediction_logits,
    load_source_reference,
    save_source_reference,
    score_candidate,
    verify_artifact_files,
)
from rotquant.eval.competitive_run import ArtifactFile


def _digest(character: str) -> str:
    return character * 64


class _Output:
    def __init__(self, logits):
        self.logits = logits


def test_continuation_logits_use_last_prompt_position_first():
    logits = torch.arange(1 * 7 * 3).reshape(1, 7, 3)
    selected = continuation_prediction_logits(
        _Output(logits), prompt_tokens=3, continuation_tokens=2
    )
    assert torch.equal(selected, logits[0, 2:4])


def test_source_reference_round_trip_is_object_free_and_validated(tmp_path):
    reference = SourceReference(
        item_sha256=_digest("a"),
        protocol_fingerprint=_digest("b"),
        teacher_logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        continuation=(3, 4),
    )
    path = tmp_path / "reference.npz"
    save_source_reference(path, reference)
    restored = load_source_reference(
        path,
        expected_item_sha256=_digest("a"),
        expected_protocol_fingerprint=_digest("b"),
        expected_tokens=2,
    )
    assert restored.continuation == (3, 4)
    assert restored.teacher_logits.dtype == torch.float16
    assert torch.equal(restored.teacher_logits, reference.teacher_logits.half())


def test_candidate_scoring_returns_per_token_kl_top1_and_trajectories():
    source = SourceReference(
        item_sha256=_digest("a"),
        protocol_fingerprint=_digest("b"),
        teacher_logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        continuation=(1, 2),
    )
    observation = score_candidate(
        item_sha256=_digest("a"),
        domain="code",
        source=source,
        candidate_logits=torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        candidate_continuation=(1, 9),
    )
    assert len(observation.teacher_kl) == 2
    assert observation.teacher_kl[0] >= 0
    assert observation.top1_matches == (True, False)
    assert observation.source_continuation == (1, 2)
    assert observation.candidate_continuation == (1, 9)


def test_candidate_artifact_is_rehashed_before_collection(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"packed-weights")
    artifact = ArtifactFile(
        name="weights.bin",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
    )
    verify_artifact_files(tmp_path, (artifact,))
    path.write_bytes(b"changed-weights")
    try:
        verify_artifact_files(tmp_path, (artifact,))
    except ValueError as error:
        assert "size mismatch" in str(error) or "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("mutated artifact passed verification")
