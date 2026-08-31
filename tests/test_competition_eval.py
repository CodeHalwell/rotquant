"""External-artifact competition contracts."""

import pytest

from rotquant.eval.competition import (
    ArtifactEvaluation,
    CompetitiveEvalProtocol,
    compare_matched_artifacts,
)


def _protocol(**overrides):
    values = {
        "model_id": "org/model",
        "model_revision": "model-sha",
        "tokenizer_revision": "tokenizer-sha",
        "prompt_manifest_sha256": "a" * 64,
        "calibration_manifest_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "domains": ("agentic", "code", "math", "multilingual", "long_document"),
    }
    values.update(overrides)
    return CompetitiveEvalProtocol(**values)


def _evaluation(name, fingerprint, *, size=1_000_000, kl=0.2, top1=0.8):
    return ArtifactEvaluation(
        name=name,
        format="gguf",
        protocol_fingerprint=fingerprint,
        artifact_bytes=size,
        scored_tokens=10_000,
        trajectory_prompts=300,
        trajectory_tokens=32,
        mean_teacher_kl=kl,
        median_teacher_kl=kl / 2,
        p95_teacher_kl=kl * 2,
        top1_agreement=top1,
        trajectory_token_agreement=0.5,
        exact_trajectory_rate=0.2,
        mean_matching_prefix=8.0,
    )


def test_protocol_is_content_addressed_and_rejects_leakage():
    protocol = _protocol()
    assert len(protocol.fingerprint) == 64
    assert protocol.fingerprint == _protocol().fingerprint
    assert protocol.manifest()["prompt_count"] == 300
    assert protocol.manifest()["generation_tokens"] == 32

    with pytest.raises(ValueError, match="disjoint"):
        _protocol(calibration_manifest_sha256="a" * 64)
    with pytest.raises(ValueError, match="greedy"):
        _protocol(temperature=0.7)


def test_comparison_requires_same_protocol_and_size():
    fingerprint = _protocol().fingerprint
    candidate = _evaluation("rotquant", fingerprint, size=995_000, kl=0.1, top1=0.9)
    baseline = _evaluation("provider", fingerprint)
    comparison = compare_matched_artifacts(candidate, baseline, _protocol())

    assert comparison["size_delta_fraction"] == pytest.approx(-0.005)
    assert comparison["mean_teacher_kl_delta"] == pytest.approx(-0.1)
    assert comparison["top1_agreement_delta"] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="not size matched"):
        compare_matched_artifacts(
            _evaluation("large", fingerprint, size=1_100_000), baseline, _protocol()
        )
    with pytest.raises(ValueError, match="different protocol"):
        compare_matched_artifacts(
            _evaluation("other", _protocol(model_revision="other").fingerprint),
            baseline,
            _protocol(),
        )
