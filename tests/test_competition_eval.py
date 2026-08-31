"""External-artifact competition contracts."""

import pytest

from rotquant.eval.competition import (
    REGISTERED_DOMAIN_QUOTAS,
    ArtifactEvaluation,
    CompetitiveEvalProtocol,
    compare_matched_artifacts,
)


def _protocol(**overrides):
    prompt_items = tuple(f"{index:064x}" for index in range(1, 301))
    calibration_items = tuple(f"{index:064x}" for index in range(1001, 1005))
    values = {
        "model_id": "org/model",
        "model_revision": "model-sha",
        "tokenizer_id": "org/tokenizer",
        "tokenizer_revision": "tokenizer-sha",
        "prompt_manifest_sha256": "a" * 64,
        "calibration_manifest_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "domains": ("agentic", "code", "math", "multilingual", "long_document"),
        "domain_counts": tuple(REGISTERED_DOMAIN_QUOTAS.items()),
        "prompt_item_sha256": prompt_items,
        "calibration_item_sha256": calibration_items,
    }
    values.update(overrides)
    return CompetitiveEvalProtocol(**values)


def _evaluation(name, fingerprint, *, size=1_000_000, kl=0.2, top1=0.8):
    return ArtifactEvaluation(
        name=name,
        format="gguf",
        protocol_fingerprint=fingerprint,
        artifact_bytes=size,
        scored_tokens=9_600,
        trajectory_prompts=300,
        trajectory_tokens=32,
        mean_teacher_kl=kl,
        median_teacher_kl=kl / 2,
        p95_teacher_kl=kl * 2,
        max_teacher_kl=kl * 3,
        top1_agreement=top1,
        trajectory_token_agreement=0.5,
        exact_trajectory_rate=0.2,
        mean_matching_prefix=8.0,
    )


def test_protocol_is_content_addressed_and_rejects_leakage():
    protocol = _protocol()
    assert sum(REGISTERED_DOMAIN_QUOTAS.values()) == protocol.prompt_count
    assert len(protocol.fingerprint) == 64
    assert protocol.fingerprint == _protocol().fingerprint
    assert protocol.manifest()["prompt_count"] == 300
    assert protocol.manifest()["generation_tokens"] == 32
    assert CompetitiveEvalProtocol.from_manifest(protocol.manifest()) == protocol
    assert protocol.fingerprint == _protocol(
        domains=tuple(reversed(protocol.domains)),
        domain_counts=tuple(reversed(protocol.domain_counts)),
    ).fingerprint

    with pytest.raises(ValueError, match="disjoint"):
        _protocol(calibration_item_sha256=("0" * 63 + "1",))
    with pytest.raises(ValueError, match="greedy"):
        _protocol(temperature=0.7)
    with pytest.raises(ValueError, match="requires 300 prompts"):
        _protocol(prompt_count=1)
    with pytest.raises(ValueError, match="requires 32 generated tokens"):
        _protocol(generation_tokens=1)
    with pytest.raises(ValueError, match="chat_template_sha256"):
        _protocol(chat_template_sha256=None)
    with pytest.raises(ValueError, match="domain quotas"):
        _protocol(
            domain_counts=(("agentic", 59),) + tuple(REGISTERED_DOMAIN_QUOTAS.items())[1:]
        )
    with pytest.raises(ValueError, match="immutable"):
        _protocol(model_revision="main")


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


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 1.5, True, 0, -1])
def test_artifact_evaluation_rejects_invalid_byte_counts(value):
    with pytest.raises(ValueError, match="artifact_bytes must be a positive integer"):
        _evaluation("invalid", _protocol().fingerprint, size=value)


def test_artifact_evaluation_round_trip():
    evaluation = _evaluation("rotquant", _protocol().fingerprint)
    assert ArtifactEvaluation.from_manifest(evaluation.manifest()) == evaluation
