"""Exact scalar-weight wire format for the next native runtime.

This is a matrix primitive, NOT a GGUF/model loader. It preserves canonical
int32 code words and fp16 or affine-uint8 scale storage without re-encoding.
Rotations, the vocabulary's inverse-rotation/rounding, and model graph execution
remain separate operator responsibilities. Native-v2/GGUF-v1 bytes are unchanged.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

from .quantize import QuantizedWeight

NATIVE_V3_MAGIC = b"RQNATV3\0"
NATIVE_V3_VERSION = 3
NATIVE_V3_HEADER = struct.Struct("<8sIIIIQQIIIIQ")


def positive_integer(value, name):
    if type(value) is not int or not 0 < value <= (1 << 63) - 1:
        raise ValueError(f"{name} must be a positive signed-64-bit integer")


@dataclass(frozen=True)
class NativeV3Layout:
    bits: int
    group_size: int
    out_features: int
    in_features: int
    scale_bits: int
    scale_quant_group_size: int = 0

    def __post_init__(self):
        for name in ("bits", "group_size", "out_features", "in_features", "scale_bits"):
            positive_integer(getattr(self, name), name)
        if self.bits not in range(1, 9) or self.scale_bits not in (8, 16):
            raise ValueError("native v3 supports 1–8-bit scalar codes and 8/16-bit scales")
        if self.group_size > (1 << 32) - 1:
            raise ValueError("group_size exceeds the wire field")
        block = self.scale_quant_group_size
        if type(block) is not int or not 0 <= block <= (1 << 32) - 1:
            raise ValueError("invalid scale_quant_group_size")
        if (self.scale_bits == 8 and block < 2) or (self.scale_bits == 16 and block != 0):
            raise ValueError("scale8 needs a metadata block >=2; scale16 requires block=0")

    @property
    def groups_per_row(self):
        return (self.in_features + self.group_size - 1) // self.group_size

    @property
    def scale_count(self):
        return self.out_features * self.groups_per_row

    @property
    def word_count(self):
        return (self.out_features * self.in_features * self.bits + 31) // 32

    @property
    def metadata_count(self):
        return ((self.scale_count + self.scale_quant_group_size - 1)
                // self.scale_quant_group_size) if self.scale_bits == 8 else 0

    @property
    def payload_bytes(self):
        return (self.word_count * 4 + self.scale_count * (self.scale_bits // 8)
                + self.metadata_count * 4 + (1 << self.bits) * 4)

    def header(self):
        return NATIVE_V3_HEADER.pack(
            NATIVE_V3_MAGIC, NATIVE_V3_VERSION, NATIVE_V3_HEADER.size,
            self.bits, self.group_size, self.out_features, self.in_features,
            self.scale_bits, self.scale_quant_group_size, 0, 0, self.payload_bytes)


@dataclass(frozen=True)
class NativeV3Matrix:
    layout: NativeV3Layout
    words: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    steps: np.ndarray
    codebook: np.ndarray

    def __post_init__(self):
        self.validate()

    def validate(self):
        spec = self.layout
        fields = (
            (self.words, np.dtype("<i4"), (spec.word_count,), "words"),
            (self.scales, np.dtype("u1" if spec.scale_bits == 8 else "<f2"),
             (spec.out_features, spec.groups_per_row), "scales"),
            (self.offsets, np.dtype("<f2"), (spec.metadata_count,), "offsets"),
            (self.steps, np.dtype("<f2"), (spec.metadata_count,), "steps"),
            (self.codebook, np.dtype("<f4"), (1 << spec.bits,), "codebook"),
        )
        for value, dtype, shape, name in fields:
            if not isinstance(value, np.ndarray) or value.dtype != dtype:
                raise TypeError(f"{name} must have exact storage dtype {dtype}")
            if value.shape != shape or not value.flags.c_contiguous:
                raise ValueError(f"{name} must have contiguous shape {shape}")
            if name != "words" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            if name in {"scales", "offsets", "steps"} and np.any(value < 0):
                raise ValueError(f"{name} must be non-negative")
        used = spec.out_features * spec.in_features * spec.bits % 32
        if used and (int(self.words[-1]) & 0xFFFFFFFF) >> used:
            raise ValueError("non-zero unused bits in final code word")
        # Bound validation memory independently of the number of matrix values.
        for start in range(0, spec.scale_count, 65536):
            scales = self.decoded_scales(start, min(start + 65536, spec.scale_count))
            peak = float(np.max(np.abs(self.codebook))) * float(np.max(scales))
            if not math.isfinite(peak) or peak > float(np.finfo(np.float32).max):
                raise ValueError("decoded weights overflow float32")

    def decoded_scales(self, start, stop):
        """Decode a bounded flattened scale slice with separate fp32 mul/add."""
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= self.layout.scale_count):
            raise ValueError("invalid scale slice")
        value = self.scales.reshape(-1)[start:stop].astype(np.float32)
        if self.layout.scale_bits == 8:
            block = np.arange(start, stop, dtype=np.int64) // self.layout.scale_quant_group_size
            value = value * self.steps[block].astype(np.float32)
            value = self.offsets[block].astype(np.float32) + value
        return value

    @property
    def persistent_bytes(self):
        return NATIVE_V3_HEADER.size + self.layout.payload_bytes

    def to_bytes(self):
        self.validate()
        return self.layout.header() + b"".join(
            value.tobytes(order="C") for value in
            (self.words, self.scales, self.offsets, self.steps, self.codebook))

    def to_manifest(self):
        data = self.to_bytes()
        return {
            "format": "rotquant-native-matrix", "format_version": NATIVE_V3_VERSION,
            **self.layout.__dict__, "header_bytes": NATIVE_V3_HEADER.size,
            "payload_bytes": self.layout.payload_bytes, "total_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "code_order": "row-major-lsb-first-int32-no-row-padding",
            "scale8_reconstruction": "fp32(offset) + fp32(code) * fp32(step); no FMA",
            "scope": "rotated-basis scalar matrix; not a model or vocabulary operator",
        }

    @classmethod
    def from_bytes(cls, data: bytes):
        if not isinstance(data, bytes) or len(data) < NATIVE_V3_HEADER.size:
            raise ValueError("expected immutable bytes with a complete native-v3 header")
        (magic, version, header, bits, group, out, inf, sbits, block,
         flags, reserved, payload) = NATIVE_V3_HEADER.unpack_from(data)
        if (magic != NATIVE_V3_MAGIC or version != NATIVE_V3_VERSION
                or header != NATIVE_V3_HEADER.size or flags or reserved):
            raise ValueError("unsupported native-v3 header, flags or version")
        layout = NativeV3Layout(bits, group, out, inf, sbits, block)
        if payload != layout.payload_bytes or len(data) != header + payload:
            raise ValueError("native-v3 payload length mismatch")
        position = header

        def take(dtype, shape):
            nonlocal position
            result = np.frombuffer(data, dtype=dtype, count=math.prod(shape),
                                   offset=position).reshape(shape)
            position += result.nbytes
            return result

        return cls(layout, take("<i4", (layout.word_count,)),
                   take("u1" if sbits == 8 else "<f2", (out, layout.groups_per_row)),
                   take("<f2", (layout.metadata_count,)),
                   take("<f2", (layout.metadata_count,)), take("<f4", (1 << bits,)))


def encode_native_v3(qweight: QuantizedWeight) -> NativeV3Matrix:
    """Copy compact stored tensors only: no code unpacking or scale re-encoding."""
    qweight.packed.validate()
    if (qweight.residual_packed is not None or qweight.sketch is not None
            or qweight.scale_group_size not in (None, qweight.group_size)):
        raise ValueError("native v3 does not support residual/sketch/per-row scale layouts")
    if qweight.scales is None or qweight.scale_bits_main not in (8, 16):
        raise ValueError("native v3 requires stored 8/16-bit scales")
    # Scalar producers pack a flat stream; the logical matrix shape belongs to
    # QuantizedWeight, whose dequantizer reshapes by these dimensions too.
    if (qweight.packed.numel != qweight.out_features * qweight.in_features
            or qweight.codebook.centroids.ndim != 1):
        raise ValueError("native v3 requires a scalar matrix with matching dimensions")
    layout = NativeV3Layout(
        qweight.packed.bits, qweight.group_size, qweight.out_features,
        qweight.in_features, int(qweight.scale_bits_main),
        qweight.scale_quant_group_size if qweight.scale_bits_main == 8 else 0)

    def stored(tensor):
        return tensor.detach().cpu().contiguous().numpy().copy()

    if layout.scale_bits == 8:
        if qweight.scale_offsets is None or qweight.scale_steps is None:
            raise ValueError("scale8 metadata is missing")
        offsets, steps = stored(qweight.scale_offsets), stored(qweight.scale_steps)
    else:
        if qweight.scale_offsets is not None or qweight.scale_steps is not None:
            raise ValueError("scale16 must not carry affine metadata")
        offsets = np.empty(0, dtype="<f2")
        steps = np.empty(0, dtype="<f2")
    result = NativeV3Matrix(layout, stored(qweight.packed.data), stored(qweight.scales),
                            offsets, steps, stored(qweight.codebook.centroids))
    # An immutable payload owns independent compact bytes; no source aliasing.
    return NativeV3Matrix.from_bytes(result.to_bytes())


def decode_native_v3_rows(matrix: NativeV3Matrix, start=0, count=None):
    """Bounded-row float32 oracle; callers explicitly choose the dense output."""
    spec = matrix.layout
    count = spec.out_features - start if count is None else count
    if type(start) is not int or type(count) is not int or start < 0 or count < 1:
        raise ValueError("invalid row range")
    if start + count > spec.out_features:
        raise ValueError("row range exceeds matrix")
    result = np.empty((count, spec.in_features), dtype=np.float32)
    words = matrix.words.view(np.uint32)
    for row in range(start, start + count):
        positions = (row * spec.in_features + np.arange(spec.in_features, dtype=np.int64)) * spec.bits
        indices, shift = positions // 32, positions % 32
        codes = words[indices].astype(np.uint64) >> shift.astype(np.uint64)
        spill = shift + spec.bits > 32
        codes[spill] |= words[indices[spill] + 1].astype(np.uint64) << (32 - shift[spill]).astype(np.uint64)
        codes &= (1 << spec.bits) - 1
        lo = row * spec.groups_per_row
        scales = matrix.decoded_scales(lo, lo + spec.groups_per_row)
        # Index only the requested columns; repeat(group_size) can otherwise
        # allocate billions of padded values for a tiny partial group.
        groups = np.arange(spec.in_features, dtype=np.int64) // spec.group_size
        result[row - start] = matrix.codebook[codes] * scales[groups]
    return result
