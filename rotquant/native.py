"""Portable native block layout for 1--8-bit RotQuant scalar weights.

Version 2 is intentionally independent of GGUF and any one kernel backend.  It
stores one fp16 scale followed by an LSB-first code bitstream for each logical
group.  Arbitrary scalar codebooks travel with the encoded matrix, allowing a
runtime to specialize packing and GEMM without embedding one hard-coded table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .format import PACKED_BIT_ORDER, validate_profile_bits
from .pack import unpack_indices
from .quantize import QuantizedWeight

NATIVE_FORMAT_NAME = "rotquant-native-blocks"
NATIVE_FORMAT_VERSION = 2
NATIVE_SCALE_DTYPE = np.dtype("<f2")


@dataclass(frozen=True, slots=True)
class NativeLayout:
    """Binary interpretation of one native matrix's group blocks."""

    bits: int
    group_size: int = 128

    def __post_init__(self) -> None:
        validate_profile_bits(self.bits)
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 1
        ):
            raise ValueError("group_size must be a positive integer")

    @property
    def code_bytes_per_group(self) -> int:
        return (self.group_size * self.bits + 7) // 8

    @property
    def bytes_per_group(self) -> int:
        return NATIVE_SCALE_DTYPE.itemsize + self.code_bytes_per_group

    def groups_for(self, in_features: int) -> int:
        if (
            isinstance(in_features, bool)
            or not isinstance(in_features, int)
            or in_features < 1
        ):
            raise ValueError("in_features must be a positive integer")
        return math.ceil(in_features / self.group_size)

    def row_bytes(self, in_features: int) -> int:
        return self.groups_for(in_features) * self.bytes_per_group

    def to_manifest(self) -> dict[str, object]:
        return {
            "format": NATIVE_FORMAT_NAME,
            "format_version": NATIVE_FORMAT_VERSION,
            "bits": self.bits,
            "group_size": self.group_size,
            "scale_dtype": "float16-le",
            "bit_order": PACKED_BIT_ORDER,
            "code_bytes_per_group": self.code_bytes_per_group,
            "bytes_per_group": self.bytes_per_group,
        }


@dataclass(frozen=True, slots=True)
class NativeEncodedMatrix:
    """Self-describing reference representation of one quantized matrix."""

    qdata: np.ndarray
    codebook: np.ndarray
    layout: NativeLayout
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        if not isinstance(self.qdata, np.ndarray) or self.qdata.dtype != np.int8:
            raise TypeError("qdata must be an int8 NumPy array")
        if self.qdata.ndim != 2:
            raise ValueError("qdata must have shape [out_features, row_bytes]")
        if self.out_features < 1 or self.in_features < 1:
            raise ValueError("matrix dimensions must be positive")
        expected_shape = (
            self.out_features,
            self.layout.row_bytes(self.in_features),
        )
        if self.qdata.shape != expected_shape:
            raise ValueError(
                f"qdata shape {self.qdata.shape} does not match {expected_shape}"
            )
        if (
            not isinstance(self.codebook, np.ndarray)
            or self.codebook.dtype != np.float32
            or self.codebook.shape != (1 << self.layout.bits,)
        ):
            raise ValueError(
                "codebook must be a float32 vector with exactly 2**bits entries"
            )
        if not np.isfinite(self.codebook).all():
            raise ValueError("codebook must contain only finite values")

    @property
    def persistent_bytes(self) -> int:
        return self.qdata.nbytes + self.codebook.nbytes

    def to_manifest(self) -> dict[str, object]:
        return {
            **self.layout.to_manifest(),
            "in_features": self.in_features,
            "out_features": self.out_features,
            "qdata_bytes": self.qdata.nbytes,
            "codebook_bytes": self.codebook.nbytes,
        }


def _pack_group_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``[..., group_size]`` indices into an LSB-first byte stream."""

    group_size = codes.shape[-1]
    output = np.zeros(
        (*codes.shape[:-1], (group_size * bits + 7) // 8), dtype=np.uint8
    )
    values = codes.astype(np.uint16, copy=False)
    for element in range(group_size):
        bit_position = element * bits
        byte_index = bit_position // 8
        offset = bit_position % 8
        value = values[..., element]
        output[..., byte_index] |= ((value << offset) & 0xFF).astype(np.uint8)
        if offset + bits > 8:
            output[..., byte_index + 1] |= (value >> (8 - offset)).astype(
                np.uint8
            )
    return output


def _unpack_group_codes(
    data: np.ndarray, *, bits: int, group_size: int
) -> np.ndarray:
    """Inverse of :func:`_pack_group_codes`."""

    output = np.empty((*data.shape[:-1], group_size), dtype=np.uint8)
    source = data.astype(np.uint16, copy=False)
    mask = (1 << bits) - 1
    for element in range(group_size):
        bit_position = element * bits
        byte_index = bit_position // 8
        offset = bit_position % 8
        value = source[..., byte_index] >> offset
        if offset + bits > 8:
            value |= source[..., byte_index + 1] << (8 - offset)
        output[..., element] = (value & mask).astype(np.uint8)
    return output


def pack_native_blocks(
    indices: np.ndarray,
    scales: np.ndarray,
    layout: NativeLayout,
) -> np.ndarray:
    """Encode logical indices and per-group scales as native v2 row blocks."""

    indices = np.asarray(indices)
    if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must be a two-dimensional integer array")
    out_features, in_features = indices.shape
    if out_features < 1 or in_features < 1:
        raise ValueError("indices must have non-empty matrix dimensions")
    if indices.size and (
        int(indices.min()) < 0 or int(indices.max()) >= (1 << layout.bits)
    ):
        raise ValueError("code index is out of range for the native layout")

    groups = layout.groups_for(in_features)
    scales = np.asarray(scales)
    if scales.shape != (out_features, groups):
        raise ValueError(
            f"scale shape {scales.shape} does not match {(out_features, groups)}"
        )
    if (
        not np.isfinite(scales).all()
        or np.any(scales < 0)
        or np.any(scales > np.finfo(NATIVE_SCALE_DTYPE).max)
    ):
        raise ValueError(
            "scales must be finite, non-negative, and representable as float16"
        )

    padded = np.zeros(
        (out_features, groups * layout.group_size), dtype=np.uint8
    )
    padded[:, :in_features] = indices.astype(np.uint8, copy=False)
    grouped = padded.reshape(out_features, groups, layout.group_size)
    codes = _pack_group_codes(grouped, layout.bits)

    scale_values = np.ascontiguousarray(scales, dtype=NATIVE_SCALE_DTYPE)
    scale_bytes = scale_values.view(np.uint8).reshape(
        out_features, groups, NATIVE_SCALE_DTYPE.itemsize
    )
    blocks = np.concatenate((scale_bytes, codes), axis=-1)
    return np.ascontiguousarray(blocks.reshape(out_features, -1)).view(np.int8)


def unpack_native_blocks(
    qdata: np.ndarray,
    *,
    in_features: int,
    layout: NativeLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode native v2 rows into logical indices and fp16 scales."""

    raw = np.ascontiguousarray(np.asarray(qdata)).view(np.uint8)
    if raw.ndim != 2 or raw.shape[1] != layout.row_bytes(in_features):
        raise ValueError("qdata row size does not match the declared native layout")
    out_features = raw.shape[0]
    groups = layout.groups_for(in_features)
    blocks = raw.reshape(out_features, groups, layout.bytes_per_group)
    scale_bytes = np.ascontiguousarray(
        blocks[..., : NATIVE_SCALE_DTYPE.itemsize]
    )
    scales = scale_bytes.reshape(
        out_features, groups * NATIVE_SCALE_DTYPE.itemsize
    ).view(NATIVE_SCALE_DTYPE)
    code_bytes = blocks[..., NATIVE_SCALE_DTYPE.itemsize :]
    padded = _unpack_group_codes(
        code_bytes, bits=layout.bits, group_size=layout.group_size
    ).reshape(out_features, groups * layout.group_size)
    return padded[:, :in_features], scales.reshape(out_features, groups)


def encode_quantized_weight(
    qweight: QuantizedWeight,
    *,
    layout: NativeLayout | None = None,
) -> NativeEncodedMatrix:
    """Convert a scalar ``QuantizedWeight`` without re-quantizing it."""

    resolved = layout or NativeLayout(
        bits=qweight.packed.bits, group_size=qweight.group_size
    )
    if resolved.bits != qweight.packed.bits:
        raise ValueError("native layout bits do not match the packed weight")
    if resolved.group_size != qweight.group_size:
        raise ValueError("native layout group size does not match the packed weight")
    if qweight.scales is None:
        raise ValueError("native v2 requires per-group scales")
    if qweight.scale_group_size not in (None, qweight.group_size):
        raise ValueError("native v2 does not yet support per-row scales")
    if qweight.residual_packed is not None or qweight.sketch is not None:
        raise ValueError("native v2 does not yet support residual or sketch streams")

    codebook = (
        qweight.codebook.centroids.detach().cpu().numpy().astype(np.float32, copy=True)
    )
    if codebook.shape != (1 << resolved.bits,):
        raise ValueError("codebook size does not match packed weight bits")
    indices = (
        unpack_indices(qweight.packed)
        .reshape(qweight.out_features, qweight.in_features)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
    )
    scales = qweight.scales.detach().cpu().numpy()
    qdata = pack_native_blocks(indices, scales, resolved)
    return NativeEncodedMatrix(
        qdata=qdata,
        codebook=codebook,
        layout=resolved,
        in_features=qweight.in_features,
        out_features=qweight.out_features,
    )


def reference_dequantize(encoded: NativeEncodedMatrix) -> np.ndarray:
    """Materialize the exact float32 matrix represented by native v2 blocks."""

    indices, scales = unpack_native_blocks(
        encoded.qdata,
        in_features=encoded.in_features,
        layout=encoded.layout,
    )
    weight = encoded.codebook[indices]
    expanded = np.repeat(scales.astype(np.float32), encoded.layout.group_size, axis=1)
    return weight * expanded[:, : encoded.in_features]


def reference_matmul(x: np.ndarray, encoded: NativeEncodedMatrix) -> np.ndarray:
    """Dense correctness oracle for the native matrix representation."""

    values = np.asarray(x, dtype=np.float32)
    if values.shape[-1] != encoded.in_features:
        raise ValueError("activation dimension does not match encoded matrix")
    return values @ reference_dequantize(encoded).T


def reference_streaming_matmul(
    x: np.ndarray, encoded: NativeEncodedMatrix
) -> np.ndarray:
    """Group-streaming matmul that never reconstructs the full dense weight.

    This remains a NumPy reference rather than an optimized kernel, but its
    allocation pattern mirrors the intended fused implementations: only one
    decoded group of weights is live at a time.
    """

    values = np.asarray(x, dtype=np.float32)
    if values.shape[-1] != encoded.in_features:
        raise ValueError("activation dimension does not match encoded matrix")
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, encoded.in_features)
    output = np.zeros((flat.shape[0], encoded.out_features), dtype=np.float32)

    layout = encoded.layout
    groups = layout.groups_for(encoded.in_features)
    raw = np.ascontiguousarray(encoded.qdata).view(np.uint8)
    blocks = raw.reshape(encoded.out_features, groups, layout.bytes_per_group)
    for group in range(groups):
        start = group * layout.group_size
        stop = min(start + layout.group_size, encoded.in_features)
        width = stop - start
        scale_bytes = np.ascontiguousarray(
            blocks[:, group, : NATIVE_SCALE_DTYPE.itemsize]
        )
        scales = scale_bytes.reshape(
            encoded.out_features, NATIVE_SCALE_DTYPE.itemsize
        ).view(NATIVE_SCALE_DTYPE).reshape(encoded.out_features).astype(np.float32)
        codes = _unpack_group_codes(
            blocks[:, group, NATIVE_SCALE_DTYPE.itemsize :],
            bits=layout.bits,
            group_size=layout.group_size,
        )[:, :width]
        group_weight = encoded.codebook[codes] * scales[:, None]
        output += flat[:, start:stop] @ group_weight.T
    return output.reshape(*original_shape, encoded.out_features)


__all__ = [
    "NATIVE_FORMAT_NAME",
    "NATIVE_FORMAT_VERSION",
    "NATIVE_SCALE_DTYPE",
    "NativeEncodedMatrix",
    "NativeLayout",
    "encode_quantized_weight",
    "pack_native_blocks",
    "reference_dequantize",
    "reference_matmul",
    "reference_streaming_matmul",
    "unpack_native_blocks",
]
