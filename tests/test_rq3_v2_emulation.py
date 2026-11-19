"""Lane-level emulation of the version-2 CUDA kernels against the reference kernel.

No GPU is needed. Both kernels are re-expressed in NumPy float32, one virtual
GPU lane per array element, following the CUDA source operation by operation:
the reference ``kernel_rq3`` (byte-wise code reads, 128 lanes, shared-memory
reduction tree and FWHT) and the warp kernels in ``rq3-v2.cuh`` (word loads,
funnel shifts, warp shuffles). NumPy float32 arithmetic rounds every operation
to nearest even, like CUDA built with ``--fmad=false --ftz=false``, and
``float16`` conversion matches ``__float2half_rn``.

What this establishes: the v2 indexing (code words, shifts, scale groups,
scale-metadata blocks, row permutations) reads the same weights as the
reference, and the v2 reduction and FWHT orders reproduce the reference's
results bit for bit. What it cannot establish: that the CUDA compiler or
hardware does what the source says. The GPU operator screen checks that.

The emulation is tied to one revision of ``rq3-v2.cuh``. If the kernel source
in the patch changes, ``test_emulated_source_revision`` fails until this file
is reviewed against it and the digest below is updated.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from rotquant.native_v3 import NativeV3Layout, NativeV3Matrix, decode_native_v3_rows

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "integrations/llama.cpp/rotquant-native-v2.patch"
CONTRACT = ROOT / "integrations/llama.cpp/rotquant-native-v2-files.json"
EMULATED_FILE = "ggml/src/ggml-cuda/rq3-v2.cuh"
EMULATED_SHA256 = "086f58f6e2438e3e5041245696f6e8c3031a42aa0a0950c8bfdd0c508c7ac1e6"

F32 = np.float32
FWHT_SCALE = F32(0.08838834764831845)


def f16(values):
    return np.asarray(values, dtype=F32).astype(np.float16).astype(F32)


def matrix(rows, cols, *, bits, scale_bits, group, block, seed):
    rng = np.random.default_rng(seed)
    layout = NativeV3Layout(bits, group, rows, cols, scale_bits, block if scale_bits == 8 else 0)
    words = rng.integers(0, 2**32, layout.word_count, dtype=np.uint64).astype(np.uint32)
    used = rows * cols * bits % 32
    if used:
        words[-1] &= np.uint32((1 << used) - 1)
    groups = cols // group
    scales = (rng.integers(0, 256, (rows, groups), dtype=np.uint8) if scale_bits == 8
              else rng.uniform(.01, .02, (rows, groups)).astype(np.float16))
    book = np.sort(rng.normal(0, 1, 1 << bits)).astype(np.float32)
    return NativeV3Matrix(
        layout, words.view(np.int32), scales,
        rng.uniform(.005, .015, layout.metadata_count).astype(np.float16),
        rng.uniform(.00001, .00005, layout.metadata_count).astype(np.float16), book)


class Blob:
    """The fields rq3_parse reads, from the serialised native-v3 bytes."""

    def __init__(self, m: NativeV3Matrix):
        data = m.to_bytes()
        header = np.frombuffer(data[:64], dtype=np.uint8)
        u32 = lambda o: int.from_bytes(header[o:o + 4].tobytes(), "little")
        u64 = lambda o: int.from_bytes(header[o:o + 8].tobytes(), "little")
        self.bits, self.group = u32(16), u32(20)
        self.rows, self.cols = u64(24), u64(32)
        self.sbits, self.block = u32(40), u32(44)
        self.group_shift = {32: 5, 64: 6}.get(self.group, 7)
        self.groups = self.cols >> self.group_shift
        payload = np.frombuffer(data, dtype=np.uint8)[64:]
        code_bytes = ((self.rows * self.cols * self.bits + 31) // 32) * 4
        self.code_bytes = payload[:code_bytes]
        self.words = self.code_bytes.view("<u4")
        cursor = code_bytes
        scale_count = self.rows * self.groups
        if self.sbits == 8:
            self.scales = payload[cursor:cursor + scale_count]
            cursor += scale_count
            meta = (scale_count + self.block - 1) // self.block
            self.offsets = payload[cursor:cursor + 2 * meta].view("<f2")
            cursor += 2 * meta
            self.steps = payload[cursor:cursor + 2 * meta].view("<f2")
            cursor += 2 * meta
        else:
            self.scales = payload[cursor:cursor + 2 * scale_count].view("<f2")
            cursor += 2 * scale_count
        self.book = payload[cursor:cursor + 4 * (1 << self.bits)].view("<f4")


# --------------------------------------------------------------------------- #
# Reference kernel_rq3 (128 lanes, rq3_weight, rq3_fwht, shared-memory tree)
# --------------------------------------------------------------------------- #
def reference_scale(b: Blob, index):
    index = np.asarray(index, dtype=np.int64)
    if b.sbits == 8:
        meta = index // b.block
        product = b.scales[index].astype(F32) * b.steps[meta].astype(F32)
        return b.offsets[meta].astype(F32) + product
    return b.scales[index].astype(F32)


def reference_weight(b: Blob, row, col):
    """rq3_weight: byte-wise code read, then centroid * scale in FP32."""
    bit = (np.int64(row) * b.cols + np.asarray(col, dtype=np.int64)) * b.bits
    shift = (bit % 8).astype(np.uint32)
    code = b.code_bytes[bit // 8].astype(np.uint32) >> shift
    spill = shift + b.bits > 8
    following = np.zeros_like(code)
    following[spill] = b.code_bytes[(bit // 8)[spill] + 1].astype(np.uint32) << (8 - shift[spill])
    code = (code | following) & np.uint32((1 << b.bits) - 1)
    scale = reference_scale(b, np.int64(row) * b.groups + (np.asarray(col) >> b.group_shift))
    return b.book[code].astype(F32) * scale


def reference_fwht(tile):
    lane = np.arange(128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        lo, hi = tile[lane & ~stride], tile[lane | stride]
        tile = np.where(lane & stride, lo - hi, lo + hi).astype(F32)
    return tile


def reference_tree(tile):
    tile = tile.copy()
    for stride in (64, 32, 16, 8, 4, 2, 1):
        tile[:stride] = tile[:stride] + tile[stride:2 * stride]
    return tile[0]


def reference_kernel(b: Blob, x, *, mode, permutation=None, signs=None, raw=False):
    """kernel_rq3 for RQ3_MATMUL (mode 0) and RQ3_HEAD (mode 1).

    ``raw=True`` returns the FP32 sums before the output's FP16 rounding; the
    rounding hides most summation-order differences, so order is tested raw.
    """
    tokens = x.shape[0]
    out = np.zeros((tokens, b.rows), dtype=F32)
    lane = np.arange(128)
    for token in range(tokens):
        for target in range(b.rows):
            row = int(permutation[target]) if permutation is not None else target
            total = np.zeros(128, dtype=F32)
            for start in range(0, b.cols, 128):
                col = start + lane
                w = reference_weight(b, row, col)
                if mode == 1:
                    w = (reference_fwht(w) * FWHT_SCALE) * signs[col].astype(F32)
                total = total + f16(w) * f16(x[token, col])
            out[token, target] = reference_tree(total)
    return out if raw else f16(out)


def reference_rotate(x, signs, columns=None):
    """kernel_rq3 RQ3_ROTATE, optionally with a column map."""
    out = np.zeros_like(x, dtype=F32)
    for token in range(x.shape[0]):
        for start in range(0, x.shape[1], 128):
            col = start + np.arange(128)
            source = columns[col] if columns is not None else col
            tile = f16(x[token, source]) * signs[col].astype(F32)
            out[token, col] = f16(reference_fwht(tile) * FWHT_SCALE)
    return out


# --------------------------------------------------------------------------- #
# Version-2 kernels (32 lanes x 4 parts, word loads, shuffles)
# --------------------------------------------------------------------------- #
LANES = np.arange(32)


def v2_block_shift(b: Blob):
    block = b.block
    return int(block).bit_length() - 1 if b.sbits == 8 and block and not block & (block - 1) else 64


def v2_scale(b: Blob, index):
    index = np.asarray(index, dtype=np.int64)
    if b.sbits == 8:
        shift = v2_block_shift(b)
        meta = index >> shift if shift < 64 else index // b.block
        product = b.scales[index].astype(F32) * b.steps[meta].astype(F32)
        return b.offsets[meta].astype(F32) + product
    return b.scales[index].astype(F32)


def v2_fetch(b: Blob, row, chunk):
    """rq3_v2_fetch: lane w < 4*bits holds code word w; lane j < 128/group a scale."""
    words_per_chunk = 4 * b.bits
    sub_shift = 7 - b.group_shift
    chunks = b.cols >> 7
    mine = np.zeros(32, dtype=np.uint32)
    scale = np.zeros(32, dtype=F32)
    if chunk < chunks:
        live = LANES < words_per_chunk
        mine[live] = b.words[(row * chunks + chunk) * words_per_chunk + LANES[live]]
        groups = LANES < (1 << sub_shift)
        scale[groups] = v2_scale(b, row * b.groups + (chunk << sub_shift) + LANES[groups])
    return mine, scale


def v2_code(mine, bits, c):
    """rq3_v2_code: two shuffles and a funnel shift."""
    bit = c * bits
    word, shift = bit >> 5, (bit & 31).astype(np.uint64)
    lo = mine[word].astype(np.uint64)
    hi = mine[(word + 1) & 31].astype(np.uint64)
    joined = (hi << np.uint64(32)) | lo
    return ((joined >> shift) & np.uint64((1 << bits) - 1)).astype(np.int64)


def v2_chunk_weights(b: Blob, row, chunk):
    mine, scale = v2_fetch(b, row, chunk)
    weights = np.zeros((4, 32), dtype=F32)
    for part in range(4):
        c = LANES + 32 * part
        lane_scale = scale[c >> b.group_shift]  # __shfl_sync(now.scale, c >> group_shift)
        weights[part] = b.book[v2_code(mine, b.bits, c)].astype(F32) * lane_scale
    return weights


def v2_reduce(sums):
    total = (sums[0] + sums[2]) + (sums[1] + sums[3])
    for stride in (16, 8, 4, 2, 1):
        other = np.concatenate([total[stride:], total[:stride]])  # shfl_down; tail unused
        update = LANES < stride
        total = np.where(update, total + other, total).astype(F32)
    return total[0]


def v2_matmul(b: Blob, x, *, permutation=None, raw=False):
    tokens = x.shape[0]
    out = np.zeros((tokens, b.rows), dtype=F32)
    for target in range(b.rows):
        row = int(permutation[target]) if permutation is not None else target
        sums = np.zeros((tokens, 4, 32), dtype=F32)
        for chunk in range(b.cols >> 7):
            weights = f16(v2_chunk_weights(b, row, chunk))
            for part in range(4):
                col = chunk * 128 + LANES + 32 * part
                for token in range(tokens):
                    sums[token, part] = sums[token, part] + weights[part] * f16(x[token, col])
        for token in range(tokens):
            out[token, target] = v2_reduce(sums[token])
    return out if raw else f16(out)


def v2_fwht(w):
    """rq3_v2_fwht on [4 parts, 32 lanes]."""
    for stride in (1, 2, 4, 8, 16):
        other = w[:, LANES ^ stride]  # __shfl_xor_sync
        w = np.where(LANES & stride, other - w, w + other).astype(F32)
    a0, a1 = w[0] + w[1], w[0] - w[1]
    a2, a3 = w[2] + w[3], w[2] - w[3]
    return np.stack([a0 + a2, a1 + a3, a0 - a2, a1 - a3]).astype(F32)


def v2_fused_rotation(x_row, signs, columns=None):
    """kernel_rq3_fused_v2's prologue: one warp per 128-column tile."""
    rotated = np.zeros(x_row.shape[0], dtype=F32)
    for tile in range(x_row.shape[0] // 128):
        cols = tile * 128 + LANES[None, :] + 32 * np.arange(4)[:, None]
        source = columns[cols] if columns is not None else cols
        w = f16(x_row[source]) * signs[cols].astype(F32)
        rotated[cols] = f16(v2_fwht(w) * FWHT_SCALE)
    return rotated


def v2_head(b: Blob, x, signs, *, raw=False):
    tokens = x.shape[0]
    out = np.zeros((tokens, b.rows), dtype=F32)
    for row in range(b.rows):
        sums = np.zeros((tokens, 4, 32), dtype=F32)
        for chunk in range(b.cols >> 7):
            w = v2_fwht(v2_chunk_weights(b, row, chunk))
            for part in range(4):
                col = chunk * 128 + LANES + 32 * part
                r = f16((w[part] * FWHT_SCALE) * signs[col].astype(F32))
                for token in range(tokens):
                    sums[token, part] = sums[token, part] + r * f16(x[token, col])
        for token in range(tokens):
            out[token, row] = v2_reduce(sums[token])
    return out if raw else f16(out)


def v2_unpack(b: Blob, permutation=None):
    rows = np.zeros((b.rows, b.cols), dtype=F32)
    for target in range(b.rows):
        source = int(permutation[target]) if permutation is not None else target
        for chunk in range(b.cols >> 7):
            weights = f16(v2_chunk_weights(b, source, chunk))
            for part in range(4):
                rows[target, chunk * 128 + LANES + 32 * part] = weights[part]
    return rows


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
CASES = [
    # bits, scale bits, group, metadata block, rows, cols
    *[(bits, 8, 128, 256, 3, 256) for bits in range(1, 9)],
    *[(bits, 16, 128, 0, 3, 256) for bits in range(1, 9)],
    (5, 8, 128, 7, 5, 384),      # block not a power of two
    (6, 8, 128, 2, 4, 256),      # smallest block
    (4, 8, 64, 256, 3, 256),     # two scale groups per chunk
    (4, 8, 32, 256, 3, 256),     # four scale groups per chunk
    (3, 16, 32, 0, 2, 384),
    (7, 8, 64, 3, 3, 128),
]


@pytest.mark.parametrize("bits,scale_bits,group,block,rows,cols", CASES)
def test_v2_reads_the_same_weights_as_the_reference(bits, scale_bits, group, block, rows, cols):
    m = matrix(rows, cols, bits=bits, scale_bits=scale_bits, group=group, block=block,
               seed=bits * 1000 + group + block)
    b = Blob(m)
    oracle = decode_native_v3_rows(m)
    for row in range(rows):
        for chunk in range(cols // 128):
            col = chunk * 128 + np.arange(128)
            reference = reference_weight(b, row, col)
            assert np.array_equal(reference, oracle[row, col])
            emulated = v2_chunk_weights(b, row, chunk).reshape(-1)
            assert np.array_equal(emulated, reference)
    permutation = np.random.default_rng(1).permutation(rows).astype(np.int32)
    expected = np.stack([f16(oracle[p]) for p in permutation])
    assert np.array_equal(v2_unpack(b, permutation), expected)


@pytest.mark.parametrize("bits,scale_bits,group,block,rows,cols", CASES[::3])
@pytest.mark.parametrize("tokens", [1, 3])
def test_v2_matmul_is_bitwise_equal_to_the_reference(bits, scale_bits, group, block, rows, cols, tokens):
    m = matrix(rows, cols, bits=bits, scale_bits=scale_bits, group=group, block=block, seed=7 + bits)
    b = Blob(m)
    rng = np.random.default_rng(bits + tokens)
    x = f16(rng.normal(0, 1.5, (tokens, cols)))
    for permutation in (None, rng.permutation(rows).astype(np.int32)):
        reference = reference_kernel(b, x, mode=0, permutation=permutation, raw=True)
        emulated = v2_matmul(b, x, permutation=permutation, raw=True)
        assert np.array_equal(emulated.view(np.uint32), reference.view(np.uint32))


@pytest.mark.parametrize("bits,scale_bits", [(6, 16), (8, 16), (6, 8), (8, 8)])
@pytest.mark.parametrize("tokens", [1, 2])
def test_v2_head_is_bitwise_equal_to_the_reference(bits, scale_bits, tokens):
    m = matrix(5, 384, bits=bits, scale_bits=scale_bits, group=128,
               block=256 if scale_bits == 8 else 0, seed=31 + bits)
    b = Blob(m)
    rng = np.random.default_rng(bits * tokens)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), 384)
    x = f16(rng.normal(0, 2, (tokens, 384)))
    reference = reference_kernel(b, x, mode=1, signs=signs, raw=True)
    emulated = v2_head(b, x, signs, raw=True)
    assert np.array_equal(emulated.view(np.uint32), reference.view(np.uint32))


def test_raw_sums_detect_a_wrong_reduction_order():
    """Guard the guard: a plausible but different order must change raw sums."""
    m = matrix(16, 1024, bits=5, scale_bits=8, group=128, block=256, seed=1)
    b = Blob(m)
    x = f16(np.random.default_rng(0).normal(0, 1.5, (2, 1024)))
    reference = reference_kernel(b, x, mode=0, raw=True)
    sums = np.zeros((2, 16, 4, 32), dtype=F32)
    for row in range(16):
        for chunk in range(8):
            weights = f16(v2_chunk_weights(b, row, chunk))
            for part in range(4):
                col = chunk * 128 + LANES + 32 * part
                for token in range(2):
                    sums[token, row, part] += weights[part] * f16(x[token, col])
    swapped = np.array([[((s[0] + s[1]) + (s[2] + s[3])).astype(F32) for s in token] for token in sums])
    wrong = np.zeros((2, 16), dtype=F32)
    for token in range(2):
        for row in range(16):
            total = swapped[token, row]
            for stride in (16, 8, 4, 2, 1):
                total = np.where(LANES < stride, total + np.roll(total, -stride), total).astype(F32)
            wrong[token, row] = total[0]
    assert not np.array_equal(wrong, reference)
    assert np.array_equal(v2_matmul(b, x, raw=True), reference)


@pytest.mark.parametrize("bits,scale_bits,group", [(5, 8, 128), (4, 16, 64), (8, 8, 32), (2, 16, 128)])
def test_fused_rotation_and_matmul_equal_the_two_ops(bits, scale_bits, group):
    m = matrix(6, 512, bits=bits, scale_bits=scale_bits, group=group,
               block=256 if scale_bits == 8 else 0, seed=61 + bits)
    b = Blob(m)
    rng = np.random.default_rng(bits)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), 512)
    x = rng.normal(0, 1.2, (1, 512)).astype(F32)   # raw: the fused op rounds it itself
    for columns in (None, rng.permutation(512).astype(np.int32)):
        for rows in (None, rng.permutation(6).astype(np.int32)):
            two_ops = reference_rotate(x, signs, columns)
            fused_input = v2_fused_rotation(x[0], signs, columns)
            assert np.array_equal(fused_input.view(np.uint32), two_ops[0].view(np.uint32))
            reference = reference_kernel(b, two_ops, mode=0, permutation=rows, raw=True)
            fused = v2_matmul(b, fused_input[None, :], permutation=rows, raw=True)
            assert np.array_equal(fused.view(np.uint32), reference.view(np.uint32))


def test_rotated_head_is_the_same_map_with_rounding_moved():
    """The rounding lane: within FP16 noise of the reference, not bitwise equal."""
    m = matrix(64, 512, bits=6, scale_bits=16, group=128, block=0, seed=5)
    b = Blob(m)
    rng = np.random.default_rng(3)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), 512)
    x = f16(rng.normal(0, 1, (2, 512)))
    reference = reference_kernel(b, x, mode=1, signs=signs)
    rotated = v2_matmul(b, reference_rotate(x, signs))
    # Unrounded float64 truth of the dense-equivalent head.
    dense = decode_native_v3_rows(m).astype(np.float64)
    hadamard = np.array([[1.0]])
    while hadamard.shape[0] < 128:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    tiles = dense.reshape(64, 4, 128) @ hadamard.T
    vocabulary = tiles.reshape(64, 512) * float(FWHT_SCALE) * signs
    truth = x.astype(np.float64) @ vocabulary.T
    assert not np.array_equal(rotated, reference)
    reference_error = np.linalg.norm(reference - truth)
    rotated_error = np.linalg.norm(rotated - truth)
    assert rotated_error < 2.0 * reference_error
    assert np.abs(rotated - reference).max() < 0.05 * np.abs(truth).max()


def test_emulated_source_revision():
    """Fail when rq3-v2.cuh changes, so this emulation is re-reviewed."""
    contract = json.loads(CONTRACT.read_text())
    assert contract["files_sha256"][EMULATED_FILE] == EMULATED_SHA256
    text = PATCH.read_text()
    match = re.search(r"^\+\+\+ b/" + re.escape(EMULATED_FILE) + r"$", text, re.MULTILINE)
    assert match, "rq3-v2.cuh missing from the integration patch"
    # The patch adds the file whole, so its added lines are the file.
    body = text[match.end() + 1:]
    hunk = body[:body.find("\ndiff --git")] if "\ndiff --git" in body else body
    lines = [line[1:] for line in hunk.splitlines() if line.startswith("+")]
    assert hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest() == EMULATED_SHA256
