"""Versioned storage contracts shared by checkpoints and native runtimes.

The format module deliberately contains no model or serialization code.  It is
the small, stable boundary that producers and runtimes can depend on without
importing Transformers.  Version 1 checkpoints written before the explicit
``packing`` block remain valid: the missing block means the v1 defaults below.
Version 4 is written only when a checkpoint actually stores entropy-coded code
streams (``code_encoding``); a reader that predates it rejects the version
instead of misreading the streams.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, NoReturn

FORMAT_NAME = "rotquant-packed"
FORMAT_VERSION = 2
VOCABULARY_FORMAT_VERSION = 3
ENTROPY_CODED_FORMAT_VERSION = 4
# v3 adds a shared packed vocabulary; v4 adds entropy-coded primary code
# streams to the v2/v3 layout and is written only when one is stored.
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3, 4)

PACKED_WORD_BITS = 32
PACKED_WORD_DTYPE = "int32"
PACKED_BIT_ORDER = "lsb_first"
PACKED_LOGICAL_ORDER = "row_major"
MIN_STORAGE_BITS = 1
MAX_STORAGE_BITS = 16
OPTIMIZED_PROFILE_BITS = tuple(range(1, 9))


class FormatValidationError(ValueError):
    """Raised when a RotQuant artifact violates its declared contract."""


@dataclass(frozen=True, slots=True)
class PackingContract:
    """Machine-readable definition of the generic packed-code bitstream."""

    word_bits: int = PACKED_WORD_BITS
    word_dtype: str = PACKED_WORD_DTYPE
    bit_order: str = PACKED_BIT_ORDER
    logical_order: str = PACKED_LOGICAL_ORDER
    min_code_bits: int = MIN_STORAGE_BITS
    max_code_bits: int = MAX_STORAGE_BITS
    optimized_profile_bits: tuple[int, ...] = OPTIMIZED_PROFILE_BITS

    def to_manifest(self) -> dict[str, Any]:
        return {
            "word_bits": self.word_bits,
            "word_dtype": self.word_dtype,
            "bit_order": self.bit_order,
            "logical_order": self.logical_order,
            "min_code_bits": self.min_code_bits,
            "max_code_bits": self.max_code_bits,
            "optimized_profile_bits": list(self.optimized_profile_bits),
        }


CURRENT_PACKING = PackingContract()


@dataclass(frozen=True, slots=True)
class RansRowsEncoding:
    """Machine-readable definition of the ``rans-rows-v1`` code-stream encoding.

    One static frequency table per tensor (``2**probability_bits`` total), one
    independent rANS stream per output row with a 32-bit state kept in
    ``[state_lower_bound, 2**state_bits)`` and 16-bit renormalisation words.
    Decoding yields the code indices that the fixed-rate packing contract then
    stores; the encoding changes no code.
    """

    name: str = "rans-rows-v1"
    version: int = 1
    probability_bits: int = 14
    state_bits: int = 32
    state_lower_bound: int = 1 << 16
    renormalization_bits: int = 16
    row_split: str = "output_row"
    max_code_bits: int = 14
    frequencies_dtype: str = "int32"
    row_words_dtype: str = "int32"
    words_dtype: str = "int16"
    decoded_digest: str = "sha256_int32_le_words"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "probability_bits": self.probability_bits,
            "state_bits": self.state_bits,
            "state_lower_bound": self.state_lower_bound,
            "renormalization_bits": self.renormalization_bits,
            "row_split": self.row_split,
            "max_code_bits": self.max_code_bits,
            "tensor_dtypes": {
                "frequencies": self.frequencies_dtype,
                "row_words": self.row_words_dtype,
                "words": self.words_dtype,
            },
            "decoded_digest": self.decoded_digest,
        }


RANS_ROWS_V1 = RansRowsEncoding()
CODE_ENCODINGS: dict[str, RansRowsEncoding] = {RANS_ROWS_V1.name: RANS_ROWS_V1}


def packed_word_count(numel: int, bits: int) -> int:
    """Return the exact number of 32-bit words for ``numel`` packed codes."""

    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("numel must be a non-negative integer")
    validate_storage_bits(bits)
    return (numel * bits + PACKED_WORD_BITS - 1) // PACKED_WORD_BITS


def validate_storage_bits(bits: int) -> None:
    """Validate the generic on-disk packer's supported code width."""

    if (
        isinstance(bits, bool)
        or not isinstance(bits, int)
        or not MIN_STORAGE_BITS <= bits <= MAX_STORAGE_BITS
    ):
        raise ValueError(
            f"bits must be an integer in [{MIN_STORAGE_BITS}, {MAX_STORAGE_BITS}]"
        )


def validate_profile_bits(bits: int) -> None:
    """Validate a public, kernel-targeted RotQuant profile width."""

    if (
        isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits not in OPTIMIZED_PROFILE_BITS
    ):
        choices = ", ".join(str(value) for value in OPTIMIZED_PROFILE_BITS)
        raise ValueError(f"profile bits must be one of: {choices}")


def _fail(path: str, message: str) -> NoReturn:
    raise FormatValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _artifact_name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty relative filename")
    candidate = PurePath(value)
    if candidate.is_absolute() or len(candidate.parts) != 1 or "\\" in value:
        _fail(path, "must be a filename, not a path")
    return value


def _is_count(value: Any, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _packed_spec(
    value: Any,
    path: str,
    *,
    code_encoding: str | None = None,
    primary: bool = False,
) -> bool:
    """Validate one code stream; return whether it is entropy coded.

    ``primary`` marks the main code stream of a backbone module, the only
    stream that may be entropy coded. ``code_encoding`` is the encoding the
    manifest declares (format v4), or ``None``.
    """
    spec = _mapping(value, path)
    encoding = spec.get("encoding")
    if encoding is None:
        tensor = spec.get("tensor")
        if not isinstance(tensor, str) or not tensor:
            _fail(f"{path}.tensor", "must be a non-empty tensor key")
    shape = spec.get("shape")
    if not isinstance(shape, list):
        _fail(f"{path}.shape", "must be an array")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
        for dim in shape
    ):
        _fail(f"{path}.shape", "dimensions must be non-negative integers")
    numel = spec.get("numel")
    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        _fail(f"{path}.numel", "must be a non-negative integer")
    logical_numel = math.prod(shape)
    if logical_numel != numel:
        _fail(f"{path}.numel", f"declares {numel}, but shape contains {logical_numel}")
    bits = spec.get("bits")
    if isinstance(bits, bool) or not isinstance(bits, int):
        _fail(f"{path}.bits", "must be an integer")
    try:
        validate_storage_bits(bits)
    except ValueError as exc:
        _fail(f"{path}.bits", str(exc))
    if encoding is None:
        return False
    if not primary:
        _fail(f"{path}.encoding",
              "only primary backbone code streams may be entropy coded")
    if code_encoding is None:
        _fail(f"{path}.encoding",
              f"entropy-coded streams require format v{ENTROPY_CODED_FORMAT_VERSION} "
              "with a code_encoding declaration")
    if encoding != code_encoding:
        _fail(f"{path}.encoding",
              f"unsupported code encoding {encoding!r}; the manifest declares "
              f"{code_encoding!r}")
    contract = CODE_ENCODINGS[code_encoding]
    if "tensor" in spec:
        _fail(f"{path}.tensor", "must be absent from an entropy-coded stream")
    keys = []
    for field in ("frequencies", "row_words", "words"):
        key = spec.get(field)
        if not isinstance(key, str) or not key:
            _fail(f"{path}.{field}", "must be a non-empty tensor key")
        keys.append(key)
    if len(set(keys)) != len(keys):
        _fail(path, "rANS tensors must use distinct keys")
    rows, columns = spec.get("rows"), spec.get("columns")
    if not _is_count(rows, 1) or not _is_count(columns, 1):
        _fail(f"{path}.rows", "rows and columns must be positive integers")
    if rows * columns != numel:
        _fail(f"{path}.rows", f"rows * columns must equal numel ({numel})")
    if bits > contract.max_code_bits:
        _fail(f"{path}.bits",
              f"{contract.name} codes at most {contract.max_code_bits} bits")
    if not _is_count(spec.get("word_count"), 2 * rows):
        _fail(f"{path}.word_count", "must count at least two state words per row")
    if not _is_sha256(spec.get("decoded_sha256")):
        _fail(f"{path}.decoded_sha256", "must be a SHA-256 hex digest")
    return True


def _optional_packed_spec(value: Any, path: str, code_encoding: str | None) -> None:
    if value is not None:
        _packed_spec(value, path, code_encoding=code_encoding)


def _code_encoding(value: Any, version: int) -> str | None:
    """Return the declared code-stream encoding, failing closed on unknown ones."""
    if version < ENTROPY_CODED_FORMAT_VERSION:
        if value is not None:
            _fail("code_encoding", f"requires format v{ENTROPY_CODED_FORMAT_VERSION}")
        return None
    declaration = _mapping(value, "code_encoding")
    name = declaration.get("name")
    contract = CODE_ENCODINGS.get(name) if isinstance(name, str) else None
    if contract is None:
        _fail("code_encoding.name",
              f"unsupported code encoding {name!r}; supported={tuple(CODE_ENCODINGS)}")
    if declaration != contract.to_manifest():
        _fail("code_encoding", f"does not match the {contract.name} contract")
    return contract.name


def _validate_module(
    value: Any,
    index: int,
    version: int,
    code_encoding: str | None = None,
    *,
    allow_encoding: bool = True,
) -> str:
    path = f"quantized_modules[{index}]"
    module = _mapping(value, path)
    name = module.get("name")
    if not isinstance(name, str) or not name:
        _fail(f"{path}.name", "must be a non-empty module name")
    aliases = module.get("aliases", [])
    if (not isinstance(aliases, list) or any(not isinstance(alias, str) or not alias
                                           or alias == name for alias in aliases)
            or len(aliases) != len(set(aliases))):
        _fail(f"{path}.aliases", "must contain unique non-empty alias names")
    for field in ("in_features", "out_features"):
        feature_count = module.get(field)
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count < 1
        ):
            _fail(f"{path}.{field}", "must be a positive integer")
    if not isinstance(module.get("has_bias"), bool):
        _fail(f"{path}.has_bias", "must be a boolean")
    lora_rank = module.get("lora_rank", 0)
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 0:
        _fail(f"{path}.lora_rank", "must be a non-negative integer")
    rotation = _mapping(module.get("rotation"), f"{path}.rotation")
    rotation_dim = rotation.get("dim")
    if (
        isinstance(rotation_dim, bool)
        or not isinstance(rotation_dim, int)
        or rotation_dim != module["in_features"]
    ):
        _fail(f"{path}.rotation.dim", "must match in_features")
    rotation_kind = rotation.get("kind")
    if rotation_kind not in {"identity", "fwht", "butterfly", "dense", "learned"}:
        _fail(f"{path}.rotation.kind", "is not supported by format version 1")
    if rotation_kind in {"fwht", "butterfly"}:
        block = rotation.get("block")
        if isinstance(block, bool) or not isinstance(block, int) or block < 1:
            _fail(f"{path}.rotation.block", "must be a positive integer")
    if rotation_kind in {"butterfly", "dense", "learned"}:
        storage_dtype = rotation.get("storage_dtype", "float32")
        choices = {"float16", "bfloat16", "float32"}
        if rotation_kind != "butterfly":
            choices.add("float64")
        if storage_dtype not in choices:
            _fail(
                f"{path}.rotation.storage_dtype",
                "unsupported rotation storage dtype",
            )

    qweight = _mapping(module.get("qweight"), f"{path}.qweight")
    for field in ("in_features", "out_features"):
        feature_count = qweight.get(field)
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count != module[field]
        ):
            _fail(f"{path}.qweight.{field}", f"must match module {field}")
    group_size = qweight.get("group_size")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 1
    ):
        _fail(f"{path}.qweight.group_size", "must be a positive integer")
    scale_bits = qweight.get("scale_bits_main", 16.0)
    if scale_bits not in {8.0, 16.0, 32.0}:
        _fail(f"{path}.qweight.scale_bits_main", "must be 8, 16, or 32")
    if version == 1 and float(scale_bits) == 8.0:
        _fail(f"{path}.qweight.scale_bits_main", "8-bit scales require format v2")
    if qweight.get("scales") is not None and not isinstance(
        qweight.get("scales"), str
    ):
        _fail(f"{path}.qweight.scales", "must be a tensor key or null")
    if float(scale_bits) == 8.0 and qweight.get("scales") is not None:
        for field in ("scale_offsets", "scale_steps"):
            if not isinstance(qweight.get(field), str):
                _fail(
                    f"{path}.qweight.{field}",
                    "must be a tensor key for 8-bit scales",
                )
        scale_group = qweight.get("scale_quant_group_size", 256)
        if (isinstance(scale_group, bool) or not isinstance(scale_group, int)
                or scale_group < 2):
            _fail(
                f"{path}.qweight.scale_quant_group_size",
                "must be an integer >= 2",
            )
    encoded = _packed_spec(
        qweight.get("packed"), f"{path}.qweight.packed",
        code_encoding=code_encoding, primary=allow_encoding,
    )
    packed_numel = qweight["packed"]["numel"]
    codebook = _mapping(qweight.get("codebook"), f"{path}.qweight.codebook")
    codebook_kind = codebook.get("kind", "scalar")
    if codebook_kind not in {"scalar", "vector"}:
        _fail(f"{path}.qweight.codebook.kind", "must be 'scalar' or 'vector'")
    if version == 1 and codebook_kind == "vector":
        _fail(f"{path}.qweight.codebook.kind", "vector codebooks require format v2")
    dimension = codebook.get("dimension", 1)
    if (isinstance(dimension, bool) or not isinstance(dimension, int)
            or dimension < 1):
        _fail(f"{path}.qweight.codebook.dimension", "must be a positive integer")
    if codebook_kind == "scalar" and dimension != 1:
        _fail(f"{path}.qweight.codebook.dimension", "must be 1 for scalar codebooks")
    if module["in_features"] % dimension:
        _fail(f"{path}.qweight.codebook.dimension", "must divide in_features")
    expected_numel = (
        module["in_features"] * module["out_features"] // dimension)
    if packed_numel != expected_numel:
        _fail(
            f"{path}.qweight.packed.numel",
            f"must equal out_features * in_features / dimension ({expected_numel})",
        )
    if encoded:
        if codebook_kind != "scalar":
            _fail(f"{path}.qweight.packed.encoding",
                  "only scalar-codebook streams may be entropy coded")
        if (qweight["packed"]["rows"], qweight["packed"]["columns"]) != (
                module["out_features"], module["in_features"]):
            _fail(f"{path}.qweight.packed.rows",
                  "entropy-coded streams hold one row per output feature")
    _optional_packed_spec(
        qweight.get("residual_packed"), f"{path}.qweight.residual_packed", code_encoding
    )
    _optional_packed_spec(qweight.get("sketch"), f"{path}.qweight.sketch", code_encoding)
    if not isinstance(codebook.get("centroids"), str):
        _fail(f"{path}.qweight.codebook.centroids", "must be a tensor key")
    return name


def validate_checkpoint_manifest(manifest: Any) -> None:
    """Fail closed if a checkpoint manifest violates a supported schema.

    Unknown fields are allowed so producers can add optional metadata without a
    format bump.  Fields that affect binary interpretation are validated
    exactly.  Legacy v1 manifests may omit ``packing`` and use the v1 defaults.
    """

    root = _mapping(manifest, "manifest")
    if root.get("format") != FORMAT_NAME:
        _fail("format", f"expected {FORMAT_NAME!r}")
    version = root.get("format_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in SUPPORTED_FORMAT_VERSIONS
    ):
        _fail(
            "format_version",
            f"unsupported version {version!r}; supported={SUPPORTED_FORMAT_VERSIONS}",
        )
    _artifact_name(root.get("model_state"), "model_state")
    _artifact_name(root.get("packed_state"), "packed_state")

    packing = root.get("packing")
    if packing is not None and packing != CURRENT_PACKING.to_manifest():
        _fail("packing", "does not match the RotQuant v1 packing contract")
    code_encoding = _code_encoding(root.get("code_encoding"), version)

    architecture = root.get("architecture")
    if architecture is not None:
        architecture = _mapping(architecture, "architecture")
        if not isinstance(architecture.get("adapter"), str):
            _fail("architecture.adapter", "must be a string")
        model_type = architecture.get("model_type")
        if model_type is not None and not isinstance(model_type, str):
            _fail("architecture.model_type", "must be a string or null")

    modules = root.get("quantized_modules")
    if not isinstance(modules, list) or not modules:
        _fail("quantized_modules", "must be a non-empty array")
    names = [
        _validate_module(module, index, version, code_encoding)
        for index, module in enumerate(modules)
    ]
    names.extend(alias for module in modules for alias in module.get("aliases", []))
    if len(names) != len(set(names)):
        _fail("quantized_modules", "module names must be unique")
    vocabulary = root.get("tied_vocabulary")
    if vocabulary is not None:
        if version not in (VOCABULARY_FORMAT_VERSION, ENTROPY_CODED_FORMAT_VERSION):
            _fail("tied_vocabulary", "requires format v3 (or v4 with entropy-coded backbone)")
        vocabulary = _mapping(vocabulary, "tied_vocabulary")
        aliases = [vocabulary.get(key) for key in ("embedding", "head")]
        if (any(not isinstance(name, str) or not name for name in aliases)
                or aliases[0] == aliases[1] or any(name in names for name in aliases)):
            _fail("tied_vocabulary", "requires two distinct non-backbone aliases")
        config = _mapping(vocabulary.get("config"), "tied_vocabulary.config")
        if (isinstance(config.get("bits"), bool) or not isinstance(config.get("bits"), int)
                or config.get("bits") not in (6, 8)):
            _fail("tied_vocabulary.config.bits", "must be 6 or 8")
        for field in ("group_size", "block", "chunk_rows", "seed"):
            value = config.get(field)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < (0 if field == "seed" else 1)):
                _fail(f"tied_vocabulary.config.{field}", "invalid integer")
        shape = vocabulary.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape)):
            _fail("tied_vocabulary.shape", "must be a positive [vocab,hidden] pair")
        if (config["block"] & (config["block"] - 1)
                or shape[1] % config["block"] or shape[1] % config["group_size"]
                or shape[1] * config["bits"] % 32):
            _fail("tied_vocabulary.config", "unsupported rotation/group/row alignment")
        if vocabulary.get("execution_dtype") not in {"float16", "bfloat16", "float32"}:
            _fail("tied_vocabulary.execution_dtype", "unsupported execution dtype")
        if vocabulary.get("projection_mode", "rotated") not in {"rotated", "dense_equivalent"}:
            _fail("tied_vocabulary.projection_mode", "unsupported projection mode")
        source_digest = vocabulary.get("source_digest")
        if (not isinstance(source_digest, str) or len(source_digest) != 64
                or any(char not in "0123456789abcdef" for char in source_digest)):
            _fail("tied_vocabulary.source_digest", "must be a SHA-256 hex digest")
        chunks = vocabulary.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            _fail("tied_vocabulary.chunks", "must be nonempty")
        rows = 0
        centroid_key = None
        for index, chunk in enumerate(chunks):
            chunk = _mapping(chunk, f"tied_vocabulary.chunks[{index}]")
            _validate_module({"name": f"vocabulary_chunk_{index}",
                              "in_features": shape[1], "out_features": chunk.get("out_features"),
                              "has_bias": False, "rotation": {"kind": "fwht", "dim": shape[1],
                               "block": config.get("block")}, "qweight": chunk}, index, version,
                             allow_encoding=False)
            if (chunk["packed"]["bits"] != config["bits"]
                    or chunk.get("scale_bits_main", 16) != 16
                    or chunk.get("group_size") != config.get("group_size")
                    or not isinstance(chunk.get("scales"), str)
                    or chunk["codebook"].get("kind", "scalar") != "scalar"
                    or chunk.get("residual_packed") is not None or chunk.get("sketch") is not None):
                _fail("tied_vocabulary.chunks", "unsupported vocabulary format")
            key = chunk["codebook"]["centroids"]
            if centroid_key is not None and key != centroid_key:
                _fail("tied_vocabulary.chunks", "must share one codebook")
            centroid_key = key
            rows += chunk["out_features"]
        if rows != shape[0]:
            _fail("tied_vocabulary.chunks", "row counts do not cover the vocabulary")


__all__ = [
    "CODE_ENCODINGS",
    "CURRENT_PACKING",
    "ENTROPY_CODED_FORMAT_VERSION",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MAX_STORAGE_BITS",
    "MIN_STORAGE_BITS",
    "OPTIMIZED_PROFILE_BITS",
    "PACKED_BIT_ORDER",
    "PACKED_LOGICAL_ORDER",
    "PACKED_WORD_BITS",
    "PACKED_WORD_DTYPE",
    "RANS_ROWS_V1",
    "VOCABULARY_FORMAT_VERSION",
    "FormatValidationError",
    "PackingContract",
    "RansRowsEncoding",
    "packed_word_count",
    "validate_checkpoint_manifest",
    "validate_profile_bits",
    "validate_storage_bits",
]
