"""Native RotQuant-GGUF packing primitives.

The native llama.cpp integration deliberately stores RotQuant weights as raw
``I8`` tensors rather than claiming one of GGML's standard quantization types.
Each logical group is encoded as::

    fp16 scale | 64 bytes containing 128 4-bit codebook indices

The corresponding learned butterfly is stored as one float32 tensor containing
the sign vector followed by the flattened angle tensor.  A patched llama.cpp
runtime interprets these tensors through a custom matrix multiplication op;
stock llama.cpp fails closed because the normal ``*.weight`` tensor is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .codebooks import build_scalar_codebook
from .pack import PackedTensor, unpack_indices
from .quantize import QuantConfig, QuantizedWeight, Quantizer
from .rotate import ButterflyRotation, RandomizedHadamard, Rotation

FORMAT_NAME = "rotquant-gguf"
FORMAT_VERSION = 1
BITS = 4
GROUP_SIZE = 128
ROTATION_BLOCK = 128
SCALE_DTYPE = np.dtype("<f2")
BYTES_PER_GROUP = SCALE_DTYPE.itemsize + GROUP_SIZE * BITS // 8


@dataclass(frozen=True)
class NativeTensor:
    """Raw tensors and logical dimensions for one native RotQuant linear."""

    qdata: np.ndarray
    rotation: np.ndarray
    in_features: int
    out_features: int


def native_tied_tensor(weight: torch.Tensor, *, seed: int, scale: str,
                       chunk_rows: int = 1024) -> NativeTensor:
    """Quantize a large tied vocabulary matrix without a full fp32 copy."""
    if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
        raise ValueError(
            "tied embedding must be [vocab, hidden] with hidden % 128 == 0")
    if chunk_rows < 1:
        raise ValueError("tied embedding chunk_rows must be >= 1")
    in_features = int(weight.shape[1])
    rotation = RandomizedHadamard(
        in_features, block=ROTATION_BLOCK, seed=seed, device="cpu")
    config = QuantConfig(
        bits=4, codebook="gaussian", scale=scale,
        group_size=GROUP_SIZE, error_comp="none", seed=seed)
    chunks = []
    for start in range(0, weight.shape[0], chunk_rows):
        source = weight[start:start + chunk_rows].float()
        rotated = rotation.rotate_weight(source)
        qweight = Quantizer(config).quantize_weight(rotated)
        chunks.append(native_tensor(qweight, rotation).qdata)
    return NativeTensor(
        qdata=np.concatenate(chunks, axis=0),
        rotation=pack_rotation(rotation),
        in_features=in_features,
        out_features=int(weight.shape[0]),
    )


def gaussian_codebook_4bit() -> np.ndarray:
    """The exact v1 codebook embedded by the C++ reference kernel."""

    return (
        build_scalar_codebook("gaussian", 1 << BITS)
        .centroids.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )


def _validate_qweight(qweight: QuantizedWeight) -> None:
    if qweight.packed.bits != BITS:
        raise ValueError(f"native GGUF v1 requires {BITS}-bit codes")
    if qweight.group_size != GROUP_SIZE:
        raise ValueError(
            f"native GGUF v1 requires group_size={GROUP_SIZE}, got {qweight.group_size}"
        )
    if qweight.in_features % GROUP_SIZE:
        raise ValueError("native GGUF v1 requires input dimensions divisible by 128")
    if qweight.scales is None:
        raise ValueError("native GGUF v1 requires stored group scales")
    if qweight.scale_group_size not in (None, GROUP_SIZE):
        raise ValueError("native GGUF v1 does not support per-row scales")
    if qweight.residual_packed is not None or qweight.sketch is not None:
        raise ValueError("native GGUF v1 does not support residual or sketch weights")
    expected = gaussian_codebook_4bit()
    actual = qweight.codebook.centroids.detach().cpu().numpy().astype(np.float32)
    if qweight.codebook.name != "gaussian" or not np.array_equal(actual, expected):
        raise ValueError("native GGUF v1 requires the exact 4-bit Gaussian codebook")


def unpack_4bit_rows(
    packed: PackedTensor, *, out_features: int, in_features: int
) -> np.ndarray:
    """Return logical uint8 indices in ``[out_features, in_features]`` order."""

    if packed.bits != BITS or packed.numel != out_features * in_features:
        raise ValueError("packed 4-bit tensor does not match the logical matrix shape")
    return (
        unpack_indices(packed)
        .reshape(out_features, in_features)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
    )


def pack_qdata(
    indices: np.ndarray,
    scales: np.ndarray,
    *,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """Interleave fp16 scales and nibble-packed codes for the C++ kernel."""

    indices = np.asarray(indices, dtype=np.uint8)
    scales = np.asarray(scales, dtype=SCALE_DTYPE)
    if indices.ndim != 2:
        raise ValueError("indices must have shape [out_features, in_features]")
    out_features, in_features = indices.shape
    if group_size != GROUP_SIZE or in_features % group_size:
        raise ValueError("native GGUF v1 requires complete groups of 128")
    n_groups = in_features // group_size
    if scales.shape != (out_features, n_groups):
        raise ValueError(
            f"scale shape {scales.shape} does not match {(out_features, n_groups)}"
        )
    if indices.size and int(indices.max()) >= (1 << BITS):
        raise ValueError("4-bit code index out of range")

    grouped = indices.reshape(out_features, n_groups, group_size)
    codes = grouped[..., 0::2] | (grouped[..., 1::2] << BITS)
    scale_bytes = np.ascontiguousarray(scales, dtype=SCALE_DTYPE).view(np.uint8)
    scale_bytes = scale_bytes.reshape(out_features, n_groups, SCALE_DTYPE.itemsize)
    raw = np.concatenate((scale_bytes, codes), axis=-1)
    return raw.reshape(out_features, n_groups * BYTES_PER_GROUP).view(np.int8)


def unpack_qdata(
    qdata: np.ndarray,
    *,
    in_features: int,
    group_size: int = GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_qdata`, primarily for format conformance tests."""

    qdata = np.asarray(qdata).view(np.uint8)
    if qdata.ndim != 2 or in_features % group_size:
        raise ValueError("invalid native qdata shape")
    out_features = qdata.shape[0]
    n_groups = in_features // group_size
    if qdata.shape[1] != n_groups * BYTES_PER_GROUP:
        raise ValueError("qdata row byte count does not match the logical shape")
    blocks = qdata.reshape(out_features, n_groups, BYTES_PER_GROUP)
    scales = (
        blocks[..., : SCALE_DTYPE.itemsize]
        .copy()
        .reshape(out_features, n_groups * SCALE_DTYPE.itemsize)
        .view(SCALE_DTYPE)
        .reshape(out_features, n_groups)
    )
    codes = blocks[..., SCALE_DTYPE.itemsize :]
    indices = np.empty((out_features, n_groups, group_size), dtype=np.uint8)
    indices[..., 0::2] = codes & 0x0F
    indices[..., 1::2] = codes >> BITS
    return indices.reshape(out_features, in_features), scales


def pack_rotation(rotation: Rotation) -> np.ndarray:
    """Serialize signs and butterfly angles in the native v1 layout."""

    if isinstance(rotation, RandomizedHadamard):
        block = rotation.block
        n_blocks = rotation.dim // block
        n_stages = int(math.log2(block))
        theta = torch.full(
            (n_blocks, n_stages, block // 2), math.pi / 4, dtype=torch.float32
        )
        signs = rotation.signs
    elif isinstance(rotation, ButterflyRotation):
        block = rotation.block
        theta = rotation.theta
        signs = rotation.signs
    else:
        raise TypeError("native GGUF v1 supports FWHT and butterfly rotations only")
    if block != ROTATION_BLOCK:
        raise ValueError(
            f"native GGUF v1 requires rotation block={ROTATION_BLOCK}, got {block}"
        )
    return np.concatenate(
        (
            signs.detach().cpu().float().numpy().reshape(-1),
            theta.detach().cpu().float().numpy().reshape(-1),
        )
    ).astype(np.float32, copy=False)


def unpack_rotation(
    data: np.ndarray, *, in_features: int, block: int = ROTATION_BLOCK
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(signs, theta[n_blocks, stages, block/2])``."""

    data = np.asarray(data, dtype=np.float32).reshape(-1)
    n_blocks = in_features // block
    n_stages = int(math.log2(block))
    theta_count = n_blocks * n_stages * (block // 2)
    if data.size != in_features + theta_count:
        raise ValueError("rotation tensor length does not match the logical dimension")
    signs = data[:in_features]
    theta = data[in_features:].reshape(n_blocks, n_stages, block // 2)
    return signs, theta


def native_tensor(
    qweight: QuantizedWeight,
    rotation: Rotation,
    *,
    row_permutation: np.ndarray | None = None,
    column_permutation: np.ndarray | None = None,
) -> NativeTensor:
    """Convert one packed layer, optionally applying Qwen3.5 layout permutations."""

    _validate_qweight(qweight)
    indices = unpack_4bit_rows(
        qweight.packed,
        out_features=qweight.out_features,
        in_features=qweight.in_features,
    )
    scales = qweight.main_scales().detach().cpu().numpy().astype(
        SCALE_DTYPE, copy=False)
    rotation_data = pack_rotation(rotation)
    signs, theta = unpack_rotation(rotation_data, in_features=qweight.in_features)

    if row_permutation is not None:
        row_permutation = np.asarray(row_permutation, dtype=np.int64)
        indices = indices[row_permutation]
        scales = scales[row_permutation]

    if column_permutation is not None:
        column_permutation = np.asarray(column_permutation, dtype=np.int64)
        if column_permutation.shape != (qweight.in_features,):
            raise ValueError("column permutation has the wrong shape")
        grouped = column_permutation.reshape(-1, GROUP_SIZE)
        starts = grouped[:, 0]
        expected = starts[:, None] + np.arange(GROUP_SIZE, dtype=np.int64)
        if not np.array_equal(grouped, expected) or np.any(starts % GROUP_SIZE):
            raise ValueError(
                "column permutation must reorder complete 128-value blocks"
            )
        block_permutation = starts // GROUP_SIZE
        if not np.array_equal(
            np.sort(block_permutation), np.arange(block_permutation.size)
        ):
            raise ValueError("column permutation does not contain every block once")
        indices = indices[:, column_permutation]
        scales = scales[:, block_permutation]
        signs = signs[column_permutation]
        theta = theta[block_permutation]

    rotation_data = np.concatenate((signs, theta.reshape(-1))).astype(
        np.float32, copy=False
    )
    return NativeTensor(
        qdata=pack_qdata(indices, scales),
        rotation=rotation_data,
        in_features=qweight.in_features,
        out_features=qweight.out_features,
    )


def reference_linear(
    x: np.ndarray,
    native: NativeTensor,
    *,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Portable reference for the C++ native operator."""

    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != native.in_features:
        raise ValueError("activation dimension does not match native tensor")
    signs, theta = unpack_rotation(native.rotation, in_features=native.in_features)
    rotated = _reference_rotation(x, signs, theta, inverse=False)

    indices, scales = unpack_qdata(native.qdata, in_features=native.in_features)
    centroids = gaussian_codebook_4bit() if codebook is None else codebook
    weight = centroids[indices]
    weight *= np.repeat(scales.astype(np.float32), GROUP_SIZE, axis=1)
    return rotated @ weight.T


def _reference_rotation(x: np.ndarray, signs: np.ndarray, theta: np.ndarray,
                        *, inverse: bool) -> np.ndarray:
    original_shape = x.shape
    h = np.asarray(x, dtype=np.float32)
    if not inverse:
        h = h * signs
    h = h.reshape(-1, theta.shape[0], ROTATION_BLOCK)
    stages = (range(theta.shape[1] - 1, -1, -1)
              if inverse else range(theta.shape[1]))
    for stage in stages:
        step = 1 << stage
        groups = ROTATION_BLOCK // (2 * step)
        paired = h.reshape(-1, theta.shape[0], groups, 2, step)
        a = paired[..., 0, :]
        b = paired[..., 1, :]
        c = np.cos(theta[:, stage]).reshape(1, theta.shape[0], groups, step)
        s = np.sin(theta[:, stage]).reshape(1, theta.shape[0], groups, step)
        h = np.stack((c * a + s * b, s * a - c * b), axis=3)
        h = h.reshape(-1, theta.shape[0], ROTATION_BLOCK)
    h = h.reshape(*original_shape)
    return h * signs if inverse else h


def reference_embedding(
    token_ids: np.ndarray,
    native: NativeTensor,
    *,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Decode selected rows and inverse-rotate them into the model basis."""
    indices, scales = unpack_qdata(native.qdata, in_features=native.in_features)
    centroids = gaussian_codebook_4bit() if codebook is None else codebook
    weight = centroids[indices]
    weight *= np.repeat(scales.astype(np.float32), GROUP_SIZE, axis=1)
    selected = weight[np.asarray(token_ids, dtype=np.int64)]
    signs, theta = unpack_rotation(native.rotation, in_features=native.in_features)
    return _reference_rotation(selected, signs, theta, inverse=True)
