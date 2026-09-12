"""Lossless matrix assembly for the W5/W6/W8 llama.cpp model integration.

GGUF v2 wraps native-matrix v3. Qwen layout permutations are separate index
tensors: never reorder affine scale codes without their original metadata.
This module does not train, re-quantize, or expand a vocabulary to dense weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .native_v3 import NativeV3Layout, NativeV3Matrix

GGUF_VERSION = 2
ROTATION_BLOCK = 128


def checkpoint_matrix(handle, spec: dict) -> NativeV3Matrix:
    """Read one validated checkpoint matrix from an open safetensors handle."""
    if (spec.get("codebook", {}).get("kind", "scalar") != "scalar"
            or spec.get("codebook", {}).get("dimension", 1) != 1):
        raise ValueError("native GGUF v2 requires scalar codebooks")
    if spec.get("residual_packed") is not None or spec.get("sketch") is not None:
        raise ValueError("native GGUF v2 does not support residual/sketch streams")
    if spec.get("scale_group_size") not in (None, spec["group_size"]):
        raise ValueError("native GGUF v2 requires group scales")
    packed = spec["packed"]
    if (packed["numel"] != spec["out_features"] * spec["in_features"]
            or math.prod(packed["shape"]) != packed["numel"]):
        raise ValueError("checkpoint packed dimensions disagree")
    sbits = spec.get("scale_bits_main", 16)
    if sbits not in (8, 16):
        raise ValueError("native GGUF v2 requires 8/16-bit scales")
    layout = NativeV3Layout(packed["bits"], spec["group_size"], spec["out_features"],
                            spec["in_features"], int(sbits),
                            spec.get("scale_quant_group_size", 256) if sbits == 8 else 0)

    def tensor(key):
        if not key:
            raise ValueError("missing required checkpoint tensor")
        return handle.get_tensor(key).detach().cpu().contiguous().numpy()

    if sbits == 8:
        offsets, steps = tensor(spec.get("scale_offsets")), tensor(spec.get("scale_steps"))
    else:
        if spec.get("scale_offsets") or spec.get("scale_steps"):
            raise ValueError("FP16 scales must not carry affine metadata")
        offsets, steps = np.empty(0, dtype="<f2"), np.empty(0, dtype="<f2")
    return NativeV3Matrix(layout, tensor(packed["tensor"]), tensor(spec.get("scales")),
                          offsets, steps, tensor(spec["codebook"]["centroids"]))


def join_vocabulary_chunks(chunks: list[NativeV3Matrix]) -> NativeV3Matrix:
    """Concatenate word-aligned FP16-scale rows without changing codes or scales."""
    if not chunks:
        raise ValueError("vocabulary chunks are empty")
    first = chunks[0]
    layout = first.layout
    if layout.bits not in (6, 8) or layout.scale_bits != 16:
        raise ValueError("shared vocabulary requires W6/W8 with FP16 scales")
    if layout.in_features * layout.bits % 32:
        raise ValueError("vocabulary rows must end on word boundaries")
    rows = 0
    for chunk in chunks:
        chunk.validate()
        current = chunk.layout
        if ((current.bits, current.group_size, current.in_features, current.scale_bits)
                != (layout.bits, layout.group_size, layout.in_features, 16)
                or not np.array_equal(chunk.codebook.view("<u4"), first.codebook.view("<u4"))):
            raise ValueError("vocabulary chunks do not share one format/codebook")
        rows += current.out_features
    combined = NativeV3Layout(layout.bits, layout.group_size, rows, layout.in_features, 16)
    return NativeV3Matrix(combined, np.concatenate([c.words for c in chunks]),
                          np.concatenate([c.scales for c in chunks]),
                          np.empty(0, dtype="<f2"), np.empty(0, dtype="<f2"), first.codebook)


def qwen35_permutations(name: str, hparams: dict):
    """Return source-row and source-column-to-engine-column maps for GDN."""
    nk = int(hparams.get("linear_num_key_heads", 0))
    nv = int(hparams.get("linear_num_value_heads", 0))
    if ".linear_attn." not in name or not nk or not nv or nk == nv:
        return None, None
    if nv % nk:
        raise ValueError("GDN value heads must be divisible by key heads")
    dk, dv = int(hparams["linear_key_head_dim"]), int(hparams["linear_value_head_dim"])

    def order(width):
        return np.arange(nv * width, dtype=np.int32).reshape(nk, nv // nk, width).transpose(
            1, 0, 2).reshape(-1).copy()

    rows = order(dv)
    if name.endswith(".in_proj_qkv"):
        return np.concatenate([np.arange(2 * nk * dk, dtype=np.int32), rows + 2 * nk * dk]), None
    if name.endswith(".in_proj_z"):
        return rows, None
    if name.endswith((".in_proj_a", ".in_proj_b")):
        return order(1), None
    if name.endswith(".out_proj"):
        return None, np.argsort(rows).astype(np.int32)
    return None, None


@dataclass(frozen=True)
class NativeProjection:
    matrix: NativeV3Matrix
    signs: np.ndarray
    rows: np.ndarray | None = None
    columns: np.ndarray | None = None
    vocabulary: bool = False

    def __post_init__(self):
        shape = self.matrix.layout
        self.matrix.validate()
        if shape.group_size != 128 or shape.in_features % 128:
            raise ValueError("native GGUF v2 requires g128 and FWHT-128-aligned inputs")
        if (self.signs.dtype != np.int8 or self.signs.shape != (shape.in_features,)
                or not self.signs.flags.c_contiguous or not np.all(np.abs(self.signs.astype(int)) == 1)):
            raise ValueError("expected exact contiguous int8 FWHT signs")
        for permutation, size in ((self.rows, shape.out_features), (self.columns, shape.in_features)):
            if permutation is not None and (
                permutation.dtype != np.dtype("<i4") or permutation.shape != (size,)
                or not permutation.flags.c_contiguous
                or not np.array_equal(np.sort(permutation), np.arange(size))
            ):
                raise ValueError("invalid native permutation")
        if self.vocabulary and (shape.bits not in (6, 8) or shape.scale_bits != 16
                                or self.rows is not None or self.columns is not None):
            raise ValueError("unsupported shared vocabulary projection")

    def tensors(self, stem: str):
        """GGUF stores the vocabulary once; the model aliases its input and head."""
        result = {stem + ".rqv3": np.frombuffer(self.matrix.to_bytes(), dtype=np.int8),
                  stem + ".rqsign": self.signs}
        if self.rows is not None:
            result[stem + ".rqrow"] = self.rows
        if self.columns is not None:
            result[stem + ".rqcol"] = self.columns
        return result
