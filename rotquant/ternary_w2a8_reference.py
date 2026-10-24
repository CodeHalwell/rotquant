"""Independent NumPy oracle for the fused native W2A8 matrix operator."""
from __future__ import annotations

import numpy as np


def permutation_indices(permutation, width):
    if permutation is None:
        return np.arange(width)
    if permutation.dtype != np.int32 or permutation.shape != (width,):
        raise ValueError("native W2A8 permutations require complete int32 indices")
    if np.any(permutation < 0) or np.any(permutation >= width) or len(np.unique(permutation)) != width:
        raise ValueError("native W2A8 permutations must be bijective")
    return permutation


def reference_rotation(inputs, signs, columns):
    if inputs.dtype != np.float32 or inputs.ndim != 2 or not np.isfinite(inputs).all():
        raise ValueError("native W2A8 input must be finite token/column float32")
    if signs.dtype != np.int8 or signs.shape != (inputs.shape[1],) or not np.isin(signs, [-1, 1]).all():
        raise ValueError("native W2A8 signs must be int8 -1/+1")
    work = (inputs[:, columns].astype(np.float16).astype(np.float32) * signs).reshape(-1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        parts = work.reshape(-1, 128 // (2 * stride), 2, stride)
        lo, hi = parts[:, :, 0].copy(), parts[:, :, 1].copy()
        parts[:, :, 0], parts[:, :, 1] = lo + hi, lo - hi
    result = (work * np.float32(0.08838834764831845)).astype(np.float16).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("native W2A8 activation overflows FP16")
    return result.reshape(inputs.shape[0], -1, 128)


def group_symbols(matrix, row, group):
    first = row * matrix.layout.in_features + group * 128
    positions = np.arange(first, first + 128, dtype=np.int64)
    words = matrix.words.view(np.uint32)[positions // 16]
    codes = (words >> ((positions % 16) * 2).astype(np.uint32)) & np.uint32(3)
    if (codes == 3).any():
        raise ValueError("native W2A8 contains reserved ternary code 3")
    return codes.astype(np.int32) - 1


def native_w2a8_reference(matrix, signs, inputs, rows=None, columns=None):
    matrix.validate()
    spec = matrix.layout
    if (spec.bits, spec.group_size, spec.scale_bits) != (2, 128, 16) or spec.in_features % 128:
        raise ValueError("native W2A8 requires two-bit/group128/scale16 matrix")
    if not np.array_equal(matrix.codebook, [-1., 0., 1., 0.]) or not (matrix.scales > 0).all():
        raise ValueError("native W2A8 requires canonical legal ternary levels and positive scales")
    if inputs.shape[1] != spec.in_features:
        raise ValueError("native W2A8 input width differs from matrix")
    rows = permutation_indices(rows, spec.out_features)
    columns = permutation_indices(columns, spec.in_features)
    values = reference_rotation(inputs, signs, columns)
    maxima = np.max(np.abs(values), -1)
    activation_scales = np.where(maxima > 0, maxima / np.float32(127), np.float32(1))
    quantized = np.rint(values / activation_scales[..., None]).clip(-127, 127).astype(np.int32)
    output = np.zeros((inputs.shape[0], spec.out_features), dtype=np.float32)
    for target, row in enumerate(rows):
        for group in range(spec.in_features // 128):
            symbols = group_symbols(matrix, int(row), group)
            dot = np.sum(quantized[:, group] * symbols, -1, dtype=np.int32)
            scale = activation_scales[:, group] * np.float32(matrix.scales[row, group])
            output[:, target] = output[:, target] + dot.astype(np.float32) * scale
    return output.astype(np.float16).astype(np.float32)
