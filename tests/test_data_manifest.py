"""Content-addressed calibration and held-out dataset manifests."""

import json

import pytest

from rotquant.eval.competition import REGISTERED_DOMAIN_QUOTAS
from rotquant.eval.data_manifest import (
    DatasetManifest,
    DatasetSource,
    ManifestItem,
    find_near_duplicates,
    protocol_from_manifests,
    read_dataset_manifest,
    token_sequence_sha256,
    validate_disjoint,
    write_dataset_manifest,
)

CHAT_HASH = "c" * 64
TOKENIZER_REVISION = "0123456789abcdef"


def _source(source_id="source", *, revision="v1", redistribution="allowed"):
    return DatasetSource(
        source_id=source_id,
        revision=revision,
        split="test",
        licenses=("MIT",),
        url=f"https://example.com/{source_id}",
        redistribution=redistribution,
    )


def _item(item_id, domain, tokens, *, source_id="source"):
    return ManifestItem(
        item_id=item_id,
        domain=domain,
        source_id=source_id,
        source_record_id=f"row-{item_id}",
        token_ids=tuple(tokens),
        licenses=("MIT",),
        metadata=(("language", "en"),),
    )


def _calibration(items=None, **overrides):
    values = {
        "role": "calibration",
        "tokenizer_id": "org/tokenizer",
        "tokenizer_revision": TOKENIZER_REVISION,
        "chat_template_sha256": CHAT_HASH,
        "sources": (_source(),),
        "items": tuple(items or [_item("cal-0", "general", (1, 2, 3))]),
        "transformations": ("apply_chat_template:add_generation_prompt=true",),
        "seed": 17,
    }
    values.update(overrides)
    return DatasetManifest(**values)


def _evaluation(*, token_offset=10_000, replacements=None, **overrides):
    replacements = replacements or {}
    items = []
    index = 0
    for domain, quota in REGISTERED_DOMAIN_QUOTAS.items():
        for domain_index in range(quota):
            item_id = f"{domain}-{domain_index}"
            items.append(
                replacements.get(
                    item_id,
                    _item(item_id, domain, (token_offset + index, token_offset + index + 1)),
                )
            )
            index += 2
    values = {
        "role": "evaluation",
        "tokenizer_id": "org/tokenizer",
        "tokenizer_revision": TOKENIZER_REVISION,
        "chat_template_sha256": CHAT_HASH,
        "sources": (_source(),),
        "items": tuple(items),
        "transformations": ("apply_chat_template:add_generation_prompt=true",),
        "seed": 17,
    }
    values.update(overrides)
    return DatasetManifest(**values)


def test_token_identity_is_cross_platform_and_validated():
    expected = token_sequence_sha256((1, 256, 65_535))
    assert expected == "c6c837acbd66d164827297e72c25c7574e54029e3be2775176ade351bf99c6a2"
    assert expected == token_sequence_sha256([1, 256, 65_535])
    assert len(expected) == 64
    assert token_sequence_sha256((1, 2)) != token_sequence_sha256((1, 2, 0))
    with pytest.raises(ValueError, match=r"token_ids\[0\]"):
        token_sequence_sha256((True,))
    with pytest.raises(ValueError, match=r"token_ids\[0\]"):
        token_sequence_sha256((-1,))


def test_manifest_round_trip_and_tamper_detection(tmp_path):
    manifest = _calibration()
    path = tmp_path / "manifest.json"
    write_dataset_manifest(path, manifest)

    assert read_dataset_manifest(path) == manifest
    assert read_dataset_manifest(path).fingerprint == manifest.fingerprint
    payload = json.loads(path.read_text())
    payload["items"][0]["token_ids"][0] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="token hash mismatch"):
        read_dataset_manifest(path)


def test_public_summary_omits_source_derived_tokens():
    manifest = _calibration(sources=(_source(redistribution="gated"),))
    summary = manifest.public_summary()
    assert not manifest.redistributable
    assert summary["replayable"] is False
    assert "token_ids" not in summary["items"][0]
    assert summary["items"][0]["token_sha256"] == manifest.items[0].token_sha256


def test_license_order_does_not_change_manifest_fingerprint():
    first_source = _source()
    first_source = DatasetSource(
        source_id=first_source.source_id,
        revision=first_source.revision,
        split=first_source.split,
        licenses=("MIT", "Apache-2.0"),
        url=first_source.url,
    )
    second_source = DatasetSource(
        source_id=first_source.source_id,
        revision=first_source.revision,
        split=first_source.split,
        licenses=tuple(reversed(first_source.licenses)),
        url=first_source.url,
    )
    first_item = ManifestItem(
        item_id="cal-0",
        domain="general",
        source_id="source",
        source_record_id="row",
        token_ids=(1, 2, 3),
        licenses=("MIT", "Apache-2.0"),
    )
    second_item = ManifestItem(
        item_id="cal-0",
        domain="general",
        source_id="source",
        source_record_id="row",
        token_ids=(1, 2, 3),
        licenses=tuple(reversed(first_item.licenses)),
    )
    assert _calibration(
        sources=(first_source,), items=(first_item,)
    ).fingerprint == _calibration(
        sources=(second_source,), items=(second_item,)
    ).fingerprint


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD"])
def test_source_rejects_mutable_revisions(revision):
    with pytest.raises(ValueError, match="immutable"):
        _source(revision=revision)


def test_evaluation_requires_exact_balanced_registered_domains():
    manifest = _evaluation()
    assert len(manifest.items) == 300
    assert manifest.domain_counts == REGISTERED_DOMAIN_QUOTAS

    values = manifest.payload()
    values.pop("schema_version")
    values.pop("token_identity_scheme")
    items = list(manifest.items[:-1])
    with pytest.raises(ValueError, match="domain quotas"):
        DatasetManifest(
            role="evaluation",
            tokenizer_id=manifest.tokenizer_id,
            tokenizer_revision=manifest.tokenizer_revision,
            chat_template_sha256=manifest.chat_template_sha256,
            sources=manifest.sources,
            items=tuple(items),
            transformations=manifest.transformations,
            seed=manifest.seed,
        )


def test_manifest_rejects_duplicate_tokens_and_undeclared_licenses():
    duplicate = _item("cal-1", "general", (1, 2, 3))
    with pytest.raises(ValueError, match="duplicate token"):
        _calibration(items=[_item("cal-0", "general", (1, 2, 3)), duplicate])
    bad_license = ManifestItem(
        item_id="bad-license",
        domain="general",
        source_id="source",
        source_record_id="row",
        token_ids=(5,),
        licenses=("Apache-2.0",),
    )
    with pytest.raises(ValueError, match="undeclared licenses"):
        _calibration(items=[bad_license])


def test_disjointness_checks_exact_and_near_duplicates():
    evaluation = _evaluation()
    exact = _calibration(
        items=[_item("cal-exact", "general", evaluation.items[0].token_ids)]
    )
    with pytest.raises(ValueError, match="not disjoint"):
        validate_disjoint(exact, evaluation)

    near_calibration = _calibration(
        items=[_item("cal-near", "general", (80, 81, 82, 83, 84, 85, 86))]
    )
    replacement = _item(
        "agentic-0", "agentic", (80, 81, 82, 83, 84, 85, 99)
    )
    near_evaluation = _evaluation(replacements={"agentic-0": replacement})
    findings = find_near_duplicates(
        near_calibration, near_evaluation, ngram_size=2, threshold=0.7
    )
    assert findings[0].left_item_id == "cal-near"
    assert findings[0].right_item_id == "agentic-0"
    assert findings[0].containment > findings[0].jaccard
    with pytest.raises(ValueError, match="near duplicates"):
        validate_disjoint(
            near_calibration,
            near_evaluation,
            ngram_size=2,
            near_duplicate_threshold=0.7,
        )


def test_short_sequence_containment_is_detected_by_default():
    calibration = _calibration(
        items=[_item("cal-short", "general", (80, 81, 82, 83))]
    )
    replacement = _item(
        "agentic-0", "agentic", (0, 80, 81, 82, 83, 99)
    )
    evaluation = _evaluation(replacements={"agentic-0": replacement})

    findings = find_near_duplicates(calibration, evaluation)
    assert findings[0].containment == 1.0
    with pytest.raises(ValueError, match="near duplicates"):
        validate_disjoint(calibration, evaluation)
    with pytest.raises(ValueError, match="near duplicates"):
        protocol_from_manifests(
            model_id="Qwen/Qwen3.5-4B",
            model_revision="model-commit",
            calibration=calibration,
            evaluation=evaluation,
        )


def test_protocol_is_built_from_verified_manifests():
    calibration = _calibration()
    evaluation = _evaluation()
    protocol = protocol_from_manifests(
        model_id="Qwen/Qwen3.5-4B",
        model_revision="model-commit",
        calibration=calibration,
        evaluation=evaluation,
    )
    assert protocol.prompt_manifest_sha256 == evaluation.fingerprint
    assert protocol.calibration_manifest_sha256 == calibration.fingerprint
    assert dict(protocol.domain_counts) == REGISTERED_DOMAIN_QUOTAS
    assert protocol.prompt_item_sha256 == evaluation.token_hashes


def test_manifest_fingerprint_binds_item_order():
    manifest = _calibration(
        items=[
            _item("cal-0", "general", (1, 2, 3)),
            _item("cal-1", "general", (4, 5, 6)),
        ]
    )
    assert manifest.fingerprint != _calibration(items=tuple(reversed(manifest.items))).fingerprint
