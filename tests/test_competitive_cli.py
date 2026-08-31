"""Prepared-data manifest builder used by the command-line pipeline."""

import pytest

from rotquant.eval.competition import REGISTERED_DOMAIN_QUOTAS
from rotquant.eval.data_manifest import DatasetSource, chat_template_sha256
from scripts import build_competitive_manifest as builder


def _source(*, licenses=("MIT",)):
    return DatasetSource(
        source_id="fixture",
        revision="v1",
        split="test",
        licenses=licenses,
        url="https://example.com/fixture",
    )


def _evaluation_records():
    records = []
    token = 10_000
    for domain, count in REGISTERED_DOMAIN_QUOTAS.items():
        for index in range(count):
            records.append(
                {
                    "item_id": f"{domain}-{index}",
                    "domain": domain,
                    "source_id": "fixture",
                    "source_record_id": f"row-{domain}-{index}",
                    "token_ids": [token, token + 1],
                    "metadata": {"ordinal": index, "language": "en"},
                }
            )
            token += 2
    return records


def test_prepared_token_ids_build_registered_manifest_without_transformers(monkeypatch):
    monkeypatch.setattr(
        builder,
        "_load_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("tokenizer should not be loaded"),
    )
    manifest = builder.build_manifest(
        role="evaluation",
        records=_evaluation_records(),
        sources=(_source(),),
        transformations=("fixture-adapter:v1",),
        tokenizer_id="org/tokenizer",
        tokenizer_revision="tokenizer-v1",
        seed=17,
        supplied_chat_template_sha256="c" * 64,
    )

    assert len(manifest.items) == 300
    assert manifest.domain_counts == REGISTERED_DOMAIN_QUOTAS
    assert manifest.items[0].metadata == (("language", "en"), ("ordinal", "0"))


def test_mixed_license_source_requires_per_record_license():
    record = {
        "item_id": "calibration-0",
        "domain": "code",
        "source_id": "fixture",
        "source_record_id": "row-0",
        "token_ids": [1, 2, 3],
    }
    with pytest.raises(ValueError, match="mixed-license"):
        builder.build_manifest(
            role="calibration",
            records=[record],
            sources=(_source(licenses=("MIT", "Apache-2.0")),),
            transformations=("fixture-adapter:v1",),
            tokenizer_id="org/tokenizer",
            tokenizer_revision="tokenizer-v1",
            seed=17,
            supplied_chat_template_sha256="c" * 64,
        )


class _FakeTokenizer:
    template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    def get_chat_template(self, **_kwargs):
        return self.template

    def apply_chat_template(self, _messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        return [41, 42, 43]


def test_messages_bind_the_resolved_template(monkeypatch):
    monkeypatch.setattr(builder, "_load_tokenizer", lambda *_args: _FakeTokenizer())
    record = {
        "item_id": "calibration-0",
        "domain": "agentic",
        "source_id": "fixture",
        "source_record_id": "row-0",
        "messages": [{"role": "user", "content": "hello"}],
    }
    manifest = builder.build_manifest(
        role="calibration",
        records=[record],
        sources=(_source(),),
        transformations=("apply-chat-template:v1",),
        tokenizer_id="org/tokenizer",
        tokenizer_revision="tokenizer-v1",
        seed=17,
        supplied_chat_template_sha256=None,
    )
    assert manifest.chat_template_sha256 == chat_template_sha256(
        _FakeTokenizer.template
    )
    assert manifest.items[0].token_ids == (41, 42, 43)

    with pytest.raises(ValueError, match="does not match"):
        builder.build_manifest(
            role="calibration",
            records=[record],
            sources=(_source(),),
            transformations=("apply-chat-template:v1",),
            tokenizer_id="org/tokenizer",
            tokenizer_revision="tokenizer-v1",
            seed=17,
            supplied_chat_template_sha256="f" * 64,
        )
