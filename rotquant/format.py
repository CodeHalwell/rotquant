"""Versioned storage contracts shared by checkpoints and native runtimes.

The format module deliberately contains no model or serialization code.  It is
the small, stable boundary that producers and runtimes can depend on without
importing Transformers.  Version 1 checkpoints written before the explicit
``packing`` block remain valid: the missing block means the v1 defaults below.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, NoReturn

FORMAT_NAME = "rotquant-packed"
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, 2)

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


def _packed_spec(value: Any, path: str) -> None:
    spec = _mapping(value, path)
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


def _optional_packed_spec(value: Any, path: str) -> None:
    if value is not None:
        _packed_spec(value, path)


def _validate_module(value: Any, index: int, version: int) -> str:
    path = f"quantized_modules[{index}]"
    module = _mapping(value, path)
    name = module.get("name")
    if not isinstance(name, str) or not name:
        _fail(f"{path}.name", "must be a non-empty module name")
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
    _packed_spec(qweight.get("packed"), f"{path}.qweight.packed")
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
    _optional_packed_spec(
        qweight.get("residual_packed"), f"{path}.qweight.residual_packed"
    )
    _optional_packed_spec(qweight.get("sketch"), f"{path}.qweight.sketch")
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
        _validate_module(module, index, version)
        for index, module in enumerate(modules)
    ]
    if len(names) != len(set(names)):
        _fail("quantized_modules", "module names must be unique")


__all__ = [
    "CURRENT_PACKING",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MAX_STORAGE_BITS",
    "MIN_STORAGE_BITS",
    "OPTIMIZED_PROFILE_BITS",
    "PACKED_BIT_ORDER",
    "PACKED_LOGICAL_ORDER",
    "PACKED_WORD_BITS",
    "PACKED_WORD_DTYPE",
    "FormatValidationError",
    "PackingContract",
    "packed_word_count",
    "validate_checkpoint_manifest",
    "validate_profile_bits",
    "validate_storage_bits",
]
