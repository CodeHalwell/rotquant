"""Content-addressed datasets for competitive quantization evaluation.

The manifest binds a prompt to the *post-template token sequence* consumed by
the model.  This catches differences in source text normalization, tokenizer
revision, and chat-template application that a source-row identifier cannot.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

from rotquant.eval.competition import (
    REGISTERED_DOMAIN_QUOTAS,
    CompetitiveEvalProtocol,
)
from rotquant.utils import write_result

MANIFEST_SCHEMA_VERSION = 1
TOKEN_IDENTITY_SCHEME = "rotquant-token-sequence-v1-le-u64"
REDISTRIBUTION_POLICIES = frozenset({"allowed", "gated", "metadata_only"})


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def chat_template_sha256(template: str) -> str:
    """Hash the exact rendered-template program selected by the tokenizer."""

    if not isinstance(template, str) or not template:
        raise ValueError("chat template must be a non-empty string")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def token_sequence_sha256(token_ids: Iterable[int]) -> str:
    """Hash token IDs with an architecture-independent binary encoding.

    Torch/NumPy native byte representations depend on dtype and endianness.
    This format always uses an unsigned 64-bit little-endian length followed by
    unsigned 64-bit little-endian IDs, prefixed by a versioned domain tag.
    """

    values = tuple(token_ids)
    digest = hashlib.sha256()
    digest.update(TOKEN_IDENTITY_SCHEME.encode("ascii") + b"\0")
    digest.update(struct.pack("<Q", len(values)))
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
            or value >= 2**64
        ):
            raise ValueError(
                f"token_ids[{index}] must be an integer in the unsigned 64-bit range"
            )
        digest.update(struct.pack("<Q", int(value)))
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSource:
    """Pinned source and licensing metadata for one dataset snapshot."""

    source_id: str
    revision: str
    split: str
    licenses: tuple[str, ...]
    url: str
    subset: str | None = None
    redistribution: Literal["allowed", "gated", "metadata_only"] = "allowed"

    def __post_init__(self) -> None:
        for name in ("source_id", "revision", "split", "url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.revision.strip().casefold() in {"main", "master", "latest", "head"}:
            raise ValueError("revision must pin an immutable commit, tag, or release")
        if (
            not isinstance(self.licenses, tuple)
            or not self.licenses
            or any(not isinstance(value, str) or not value.strip() for value in self.licenses)
        ):
            raise ValueError("licenses must be a non-empty tuple of license identifiers")
        if len(set(self.licenses)) != len(self.licenses):
            raise ValueError("licenses must not contain duplicates")
        if self.redistribution not in REDISTRIBUTION_POLICIES:
            raise ValueError(
                "redistribution must be allowed, gated, or metadata_only"
            )
        if self.subset is not None and (
            not isinstance(self.subset, str) or not self.subset.strip()
        ):
            raise ValueError("subset must be a non-empty string when supplied")

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["licenses"] = list(self.licenses)
        return payload

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> DatasetSource:
        values = dict(payload)
        values["licenses"] = tuple(values["licenses"])
        return cls(**values)


@dataclass(frozen=True)
class ManifestItem:
    """One replayable item after normalization and tokenization."""

    item_id: str
    domain: str
    source_id: str
    source_record_id: str
    token_ids: tuple[int, ...]
    licenses: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("item_id", "domain", "source_id", "source_record_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("token_ids must be a non-empty tuple")
        token_sequence_sha256(self.token_ids)
        if (
            not isinstance(self.licenses, tuple)
            or not self.licenses
            or any(not isinstance(value, str) or not value.strip() for value in self.licenses)
        ):
            raise ValueError("item licenses must be a non-empty tuple")
        if len(set(self.licenses)) != len(self.licenses):
            raise ValueError("item licenses must not contain duplicates")
        if not isinstance(self.metadata, tuple) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not all(isinstance(value, str) for value in entry)
            for entry in self.metadata
        ):
            raise ValueError("metadata must be a tuple of string pairs")
        if tuple(sorted(self.metadata)) != self.metadata:
            raise ValueError("metadata pairs must be sorted")
        if len(dict(self.metadata)) != len(self.metadata):
            raise ValueError("metadata keys must be unique")

    @property
    def token_sha256(self) -> str:
        return token_sequence_sha256(self.token_ids)

    def manifest(self, *, include_token_ids: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item_id": self.item_id,
            "domain": self.domain,
            "source_id": self.source_id,
            "source_record_id": self.source_record_id,
            "token_sha256": self.token_sha256,
            "licenses": list(self.licenses),
            "metadata": {key: value for key, value in self.metadata},
        }
        if include_token_ids:
            payload["token_ids"] = list(self.token_ids)
        return payload

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> ManifestItem:
        if "token_ids" not in payload:
            raise ValueError("a replayable manifest item must contain token_ids")
        expected_hash = payload.get("token_sha256")
        item = cls(
            item_id=payload["item_id"],
            domain=payload["domain"],
            source_id=payload["source_id"],
            source_record_id=payload["source_record_id"],
            token_ids=tuple(payload["token_ids"]),
            licenses=tuple(payload["licenses"]),
            metadata=tuple(sorted(dict(payload.get("metadata", {})).items())),
        )
        if expected_hash is not None and expected_hash != item.token_sha256:
            raise ValueError(f"token hash mismatch for manifest item {item.item_id!r}")
        return item


@dataclass(frozen=True)
class DatasetManifest:
    """A deterministic calibration or held-out evaluation dataset."""

    role: Literal["calibration", "evaluation"]
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_sha256: str
    sources: tuple[DatasetSource, ...]
    items: tuple[ManifestItem, ...]
    transformations: tuple[str, ...]
    seed: int
    schema_version: int = MANIFEST_SCHEMA_VERSION
    token_identity_scheme: str = TOKEN_IDENTITY_SCHEME

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema version {self.schema_version}"
            )
        if self.token_identity_scheme != TOKEN_IDENTITY_SCHEME:
            raise ValueError("unsupported token identity scheme")
        if self.role not in {"calibration", "evaluation"}:
            raise ValueError("role must be calibration or evaluation")
        for name in ("tokenizer_id", "tokenizer_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.tokenizer_revision.strip().casefold() in {
            "main",
            "master",
            "latest",
            "head",
        }:
            raise ValueError("tokenizer_revision must pin an immutable revision")
        _require_sha256("chat_template_sha256", self.chat_template_sha256)
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("sources must be a non-empty tuple")
        if not isinstance(self.items, tuple) or not self.items:
            raise ValueError("items must be a non-empty tuple")
        if any(not isinstance(source, DatasetSource) for source in self.sources):
            raise ValueError("sources must contain DatasetSource values")
        if any(not isinstance(item, ManifestItem) for item in self.items):
            raise ValueError("items must contain ManifestItem values")
        if not isinstance(self.transformations, tuple) or not self.transformations or any(
            not isinstance(value, str) or not value.strip()
            for value in self.transformations
        ):
            raise ValueError(
                "transformations must be a non-empty tuple of non-empty strings"
            )

        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_id values must be unique")
        sources = {source.source_id: source for source in self.sources}
        item_ids = [item.item_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item_id values must be unique")
        token_hashes = [item.token_sha256 for item in self.items]
        if len(set(token_hashes)) != len(token_hashes):
            raise ValueError("manifest must not contain duplicate token sequences")
        for item in self.items:
            if item.source_id not in sources:
                raise ValueError(
                    f"item {item.item_id!r} references unknown source {item.source_id!r}"
                )
            undeclared = set(item.licenses).difference(sources[item.source_id].licenses)
            if undeclared:
                raise ValueError(
                    f"item {item.item_id!r} uses undeclared licenses: "
                    + ", ".join(sorted(undeclared))
                )

        if self.role == "evaluation" and self.domain_counts != REGISTERED_DOMAIN_QUOTAS:
            expected = ", ".join(
                f"{domain}={count}"
                for domain, count in REGISTERED_DOMAIN_QUOTAS.items()
            )
            raise ValueError(f"evaluation manifest requires domain quotas: {expected}")

    @property
    def token_hashes(self) -> tuple[str, ...]:
        return tuple(item.token_sha256 for item in self.items)

    @property
    def domain_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.domain for item in self.items).items()))

    @property
    def redistributable(self) -> bool:
        return all(source.redistribution == "allowed" for source in self.sources)

    def payload(self, *, include_token_ids: bool = True) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_identity_scheme": self.token_identity_scheme,
            "role": self.role,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "sources": [
                source.manifest()
                for source in sorted(self.sources, key=lambda source: source.source_id)
            ],
            "items": [
                item.manifest(include_token_ids=include_token_ids) for item in self.items
            ],
            "transformations": list(self.transformations),
            "seed": int(self.seed),
        }

    @property
    def fingerprint(self) -> str:
        return _sha256(self.payload())

    def manifest(self) -> dict[str, Any]:
        payload = self.payload()
        payload["manifest_sha256"] = self.fingerprint
        return payload

    def public_summary(self) -> dict[str, Any]:
        """Return auditable identities without redistributing source-derived tokens."""

        payload = self.payload(include_token_ids=False)
        payload["replayable"] = False
        payload["full_manifest_sha256"] = self.fingerprint
        payload["summary_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> DatasetManifest:
        values = dict(payload)
        expected_hash = values.pop("manifest_sha256", None)
        manifest = cls(
            schema_version=values["schema_version"],
            token_identity_scheme=values["token_identity_scheme"],
            role=values["role"],
            tokenizer_id=values["tokenizer_id"],
            tokenizer_revision=values["tokenizer_revision"],
            chat_template_sha256=values["chat_template_sha256"],
            sources=tuple(
                DatasetSource.from_manifest(source) for source in values["sources"]
            ),
            items=tuple(ManifestItem.from_manifest(item) for item in values["items"]),
            transformations=tuple(values["transformations"]),
            seed=values["seed"],
        )
        if expected_hash is not None and expected_hash != manifest.fingerprint:
            raise ValueError("dataset manifest fingerprint does not match its contents")
        return manifest


@dataclass(frozen=True)
class NearDuplicate:
    left_item_id: str
    right_item_id: str
    jaccard: float
    containment: float


def _token_shingles(token_ids: tuple[int, ...], ngram_size: int) -> frozenset[tuple[int, ...]]:
    if len(token_ids) < ngram_size:
        return frozenset({token_ids})
    return frozenset(
        token_ids[index : index + ngram_size]
        for index in range(len(token_ids) - ngram_size + 1)
    )


def find_near_duplicates(
    left: DatasetManifest,
    right: DatasetManifest,
    *,
    ngram_size: int = 5,
    threshold: float = 0.8,
    max_findings: int = 100,
) -> tuple[NearDuplicate, ...]:
    """Find cross-manifest token n-gram duplicates using an inverted index."""

    if isinstance(ngram_size, bool) or not isinstance(ngram_size, Integral) or ngram_size <= 0:
        raise ValueError("ngram_size must be a positive integer")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if isinstance(max_findings, bool) or not isinstance(max_findings, Integral) or max_findings <= 0:
        raise ValueError("max_findings must be a positive integer")

    left_shingles = [
        _token_shingles(item.token_ids, int(ngram_size)) for item in left.items
    ]
    inverted: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for index, shingles in enumerate(left_shingles):
        for shingle in shingles:
            inverted[shingle].add(index)

    findings: list[NearDuplicate] = []
    for right_item in right.items:
        right_shingles = _token_shingles(right_item.token_ids, int(ngram_size))
        candidates: set[int] = set()
        for shingle in right_shingles:
            candidates.update(inverted.get(shingle, ()))
        for index in sorted(candidates):
            intersection = len(left_shingles[index].intersection(right_shingles))
            union = len(left_shingles[index].union(right_shingles))
            score = intersection / union
            containment = intersection / min(
                len(left_shingles[index]), len(right_shingles)
            )
            if max(score, containment) >= threshold:
                findings.append(
                    NearDuplicate(
                        left_item_id=left.items[index].item_id,
                        right_item_id=right_item.item_id,
                        jaccard=score,
                        containment=containment,
                    )
                )
                if len(findings) >= max_findings:
                    return tuple(findings)
    return tuple(findings)


def validate_disjoint(
    calibration: DatasetManifest,
    evaluation: DatasetManifest,
    *,
    near_duplicate_threshold: float | None = None,
    ngram_size: int = 5,
) -> None:
    """Require identical tokenization and no calibration/evaluation leakage."""

    if calibration.role != "calibration" or evaluation.role != "evaluation":
        raise ValueError("expected calibration and evaluation manifests, in that order")
    for name in ("tokenizer_id", "tokenizer_revision", "chat_template_sha256"):
        if getattr(calibration, name) != getattr(evaluation, name):
            raise ValueError(f"calibration and evaluation {name} values differ")
    overlap = set(calibration.token_hashes).intersection(evaluation.token_hashes)
    if overlap:
        raise ValueError(
            "calibration and evaluation manifests are not disjoint: "
            f"{len(overlap)} exact token sequence(s) overlap"
        )
    if near_duplicate_threshold is not None:
        findings = find_near_duplicates(
            calibration,
            evaluation,
            ngram_size=ngram_size,
            threshold=near_duplicate_threshold,
        )
        if findings:
            first = findings[0]
            raise ValueError(
                "calibration and evaluation manifests contain near duplicates: "
                f"{first.left_item_id!r} vs {first.right_item_id!r} "
                f"(Jaccard={first.jaccard:.4f}, containment={first.containment:.4f})"
            )


def protocol_from_manifests(
    *,
    model_id: str,
    model_revision: str,
    calibration: DatasetManifest,
    evaluation: DatasetManifest,
    include_auxiliary_heads: bool = False,
    near_duplicate_threshold: float | None = None,
) -> CompetitiveEvalProtocol:
    """Build the registered competition protocol after leakage validation."""

    validate_disjoint(
        calibration,
        evaluation,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    return CompetitiveEvalProtocol(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=evaluation.tokenizer_id,
        tokenizer_revision=evaluation.tokenizer_revision,
        prompt_manifest_sha256=evaluation.fingerprint,
        calibration_manifest_sha256=calibration.fingerprint,
        chat_template_sha256=evaluation.chat_template_sha256,
        domains=tuple(sorted(evaluation.domain_counts)),
        domain_counts=tuple(evaluation.domain_counts.items()),
        prompt_item_sha256=evaluation.token_hashes,
        calibration_item_sha256=calibration.token_hashes,
        include_auxiliary_heads=include_auxiliary_heads,
    )


def read_dataset_manifest(path: str | Path) -> DatasetManifest:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return DatasetManifest.from_manifest(payload)


def write_dataset_manifest(
    path: str | Path,
    manifest: DatasetManifest,
    *,
    public_summary: bool = False,
) -> None:
    write_result(
        str(path),
        manifest.public_summary() if public_summary else manifest.manifest(),
    )


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "REDISTRIBUTION_POLICIES",
    "TOKEN_IDENTITY_SCHEME",
    "DatasetManifest",
    "DatasetSource",
    "ManifestItem",
    "NearDuplicate",
    "chat_template_sha256",
    "find_near_duplicates",
    "protocol_from_manifests",
    "read_dataset_manifest",
    "token_sequence_sha256",
    "validate_disjoint",
    "write_dataset_manifest",
]
