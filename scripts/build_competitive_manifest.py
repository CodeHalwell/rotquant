#!/usr/bin/env python3
"""Build a replayable, content-addressed calibration/evaluation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from rotquant.eval.data_manifest import (
    DatasetManifest,
    DatasetSource,
    ManifestItem,
    chat_template_sha256,
    read_dataset_manifest,
    validate_disjoint,
    write_dataset_manifest,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise TypeError(f"record on {path}:{line_number} must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _load_source_config(path: Path) -> tuple[tuple[DatasetSource, ...], tuple[str, ...]]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("source config must be an object")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source config must contain a non-empty sources list")
    sources = tuple(DatasetSource.from_manifest(source) for source in raw_sources)
    transformations = payload.get("transformations", [])
    if not isinstance(transformations, list):
        raise TypeError("transformations must be a list")
    return sources, tuple(transformations)


def _metadata_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise TypeError("record metadata must be an object")
    pairs = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("record metadata keys must be strings")
        rendered = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
        pairs.append((key, rendered))
    return tuple(sorted(pairs))


def _load_tokenizer(tokenizer_id: str, revision: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised in eval environments
        raise RuntimeError(
            "transformers is required for text/messages input; install rotquant[eval] "
            "or provide pre-tokenized token_ids"
        ) from error
    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=revision,
        trust_remote_code=False,
    )


def _record_token_ids(record: dict[str, Any], tokenizer: Any | None) -> tuple[int, ...]:
    supplied = [name for name in ("token_ids", "messages", "text") if name in record]
    if len(supplied) != 1:
        raise ValueError("each record must contain exactly one of token_ids, messages, or text")
    if "token_ids" in record:
        if not isinstance(record["token_ids"], list):
            raise ValueError("record token_ids must be a list")
        return tuple(record["token_ids"])
    if tokenizer is None:  # pragma: no cover - defensive
        raise RuntimeError("a tokenizer is required for text/messages input")
    if "messages" in record:
        if not isinstance(record["messages"], list):
            raise ValueError("record messages must be a list")
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if "tools" in record:
            kwargs["tools"] = record["tools"]
        return tuple(tokenizer.apply_chat_template(record["messages"], **kwargs))
    encoded = tokenizer(record["text"], add_special_tokens=True)
    return tuple(encoded["input_ids"])


def _selected_chat_templates(records: list[dict[str, Any]], tokenizer: Any) -> set[str]:
    templates = set()
    for record in records:
        if "messages" not in record:
            continue
        kwargs = {"tools": record["tools"]} if "tools" in record else {}
        template = tokenizer.get_chat_template(**kwargs)
        if not isinstance(template, str) or not template:
            raise ValueError("tokenizer did not resolve a non-empty chat template")
        templates.add(template)
    return templates


def build_manifest(
    *,
    role: str,
    records: list[dict[str, Any]],
    sources: tuple[DatasetSource, ...],
    transformations: tuple[str, ...],
    tokenizer_id: str,
    tokenizer_revision: str,
    seed: int,
    supplied_chat_template_sha256: str | None,
) -> DatasetManifest:
    """Build a manifest; exposed separately so the CLI path is unit-testable."""

    source_by_id = {source.source_id: source for source in sources}
    needs_tokenizer = any("token_ids" not in record for record in records)
    tokenizer = (
        _load_tokenizer(tokenizer_id, tokenizer_revision) if needs_tokenizer else None
    )
    if tokenizer is not None:
        templates = _selected_chat_templates(records, tokenizer)
        if len(templates) > 1:
            raise ValueError(
                "records resolve to multiple chat templates; split them into separate manifests"
            )
        resolved_hash = (
            chat_template_sha256(next(iter(templates))) if templates else None
        )
        if (
            supplied_chat_template_sha256 is not None
            and resolved_hash is not None
            and supplied_chat_template_sha256 != resolved_hash
        ):
            raise ValueError("supplied chat-template hash does not match the tokenizer")
        effective_chat_hash = resolved_hash or supplied_chat_template_sha256
    else:
        effective_chat_hash = supplied_chat_template_sha256
    if effective_chat_hash is None:
        raise ValueError(
            "--chat-template-sha256 is required when no messages resolve a template"
        )

    items = []
    for index, record in enumerate(records):
        missing = [
            name
            for name in ("item_id", "domain", "source_id", "source_record_id")
            if name not in record
        ]
        if missing:
            raise ValueError(
                f"record {index} is missing required fields: {', '.join(missing)}"
            )
        source = source_by_id.get(record["source_id"])
        if source is None:
            raise ValueError(
                f"record {index} references unknown source {record['source_id']!r}"
            )
        licenses = record.get("licenses")
        if licenses is None:
            if len(source.licenses) != 1:
                raise ValueError(
                    f"record {index} must declare its license for a mixed-license source"
                )
            licenses = list(source.licenses)
        if not isinstance(licenses, list):
            raise TypeError(f"record {index} licenses must be a list")
        items.append(
            ManifestItem(
                item_id=record["item_id"],
                domain=record["domain"],
                source_id=record["source_id"],
                source_record_id=record["source_record_id"],
                token_ids=_record_token_ids(record, tokenizer),
                licenses=tuple(licenses),
                metadata=_metadata_pairs(record.get("metadata")),
            )
        )
    return DatasetManifest(
        role=role,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        chat_template_sha256=effective_chat_hash,
        sources=sources,
        items=tuple(items),
        transformations=transformations,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Prepared JSONL records")
    parser.add_argument("--sources", type=Path, required=True, help="Pinned source YAML/JSON")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chat-template-sha256")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-summary-output",
        type=Path,
        help="Optional token-free summary path for publication",
    )
    parser.add_argument("--near-against", type=Path)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources, transformations = _load_source_config(args.sources)
    manifest = build_manifest(
        role=args.role,
        records=_load_jsonl(args.input),
        sources=sources,
        transformations=transformations,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        seed=args.seed,
        supplied_chat_template_sha256=args.chat_template_sha256,
    )
    if args.near_against:
        other = read_dataset_manifest(args.near_against)
        calibration, evaluation = (
            (manifest, other) if manifest.role == "calibration" else (other, manifest)
        )
        validate_disjoint(
            calibration,
            evaluation,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
    write_dataset_manifest(args.output, manifest)
    if args.public_summary_output:
        write_dataset_manifest(
            args.public_summary_output,
            manifest,
            public_summary=True,
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": manifest.fingerprint,
                "role": manifest.role,
                "items": len(manifest.items),
                "domain_counts": manifest.domain_counts,
                "redistributable": manifest.redistributable,
                "public_summary_output": (
                    str(args.public_summary_output)
                    if args.public_summary_output
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
