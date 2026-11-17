"""The parts of the pipeline that are exactly lossless, and tools to measure them.

Quantisation to a few bits per weight is lossy by the rate-distortion theorem:
a unit-variance Gaussian coordinate coded at ``b`` bits cannot be reproduced
with mean squared error below ``4**-b``. Everything *around* the quantiser can
be exact, and this module provides those pieces:

* :func:`integer_butterfly` / :func:`integer_butterfly_inverse` -- a randomised
  block Hadamard (or any butterfly) rotation on integers that is a bijection.
  A floating-point rotation is orthogonal only up to round-off, so rotating and
  rotating back does not return the same bits; the lifting form does.
* :func:`floats_to_fixed_point` / :func:`fixed_point_to_floats` -- the exact
  integer view of an fp16/bf16/fp32 tensor that the integer rotation needs.
* :func:`code_entropy` -- the order-0 entropy of a packed code stream, i.e. the
  size an entropy coder can reach without changing a single code.
* :func:`rans_encode_rows` / :func:`rans_decode_rows` -- a table-driven rANS
  coder that actually reaches that size. Rows are independent streams, so a
  loader can decode them in parallel.
* :func:`compress_codes` / :func:`decompress_codes` -- the two applied to a
  :class:`~rotquant.pack.PackedTensor`, with a bit-exact round trip.

``docs/mathematical_review_2026-11-17.md`` records what these measure on real
weights. In short: entropy-coding the codes saves 2.5-5.4% of the code bytes at
three to six bits with no change to the model, whereas rotating before a
*lossless* coder makes the result larger, because a mixing rotation raises the
marginal entropy of every coordinate towards the Gaussian maximum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .pack import PackedTensor, pack_indices, unpack_indices

# --------------------------------------------------------------------------- #
# Exactly invertible integer rotation
# --------------------------------------------------------------------------- #
# Lifting multipliers are fixed-point integers so the transform is defined by
# integer arithmetic alone and gives the same bits on every platform.
_LIFT_BITS = 24
_LIFT_HALF = 1 << (_LIFT_BITS - 1)


def _input_limit(block: int) -> int:
    """Largest input magnitude whose intermediates keep ``P * value`` in int64.

    An orthonormal butterfly can grow one coordinate by ``sqrt(2)`` per stage,
    so by ``sqrt(block)`` overall; one further bit covers the shear overshoot
    inside a lifting step.
    """
    stages = int(math.log2(block))
    return 1 << (62 - _LIFT_BITS - (stages + 1) // 2 - 1)


def _lifting_multipliers(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-point ``(-tan(theta/2), sin(theta))`` for each butterfly angle."""
    angles = theta.detach().to(torch.float64)
    shear = torch.round(-torch.tan(angles / 2.0) * (1 << _LIFT_BITS)).to(torch.int64)
    sine = torch.round(torch.sin(angles) * (1 << _LIFT_BITS)).to(torch.int64)
    return shear, sine


def _scaled(multiplier: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``round(multiplier * value / 2**_LIFT_BITS)`` with floor rounding."""
    return (multiplier * value + _LIFT_HALF) >> _LIFT_BITS


def _stage_view(values: torch.Tensor, block: int, stage: int):
    step = 1 << stage
    groups = block // (2 * step)
    paired = values.reshape(-1, values.shape[-1] // block, groups, 2, step)
    return paired[..., 0, :], paired[..., 1, :], groups, step


def _check_integer_input(values: torch.Tensor, signs: torch.Tensor, block: int) -> None:
    if values.dtype != torch.int64:
        raise TypeError("the integer rotation operates on int64 tensors")
    dim = values.shape[-1]
    if block < 1 or block & (block - 1) or dim % block:
        raise ValueError("block must be a power of two dividing the last dimension")
    if signs.shape != (dim,):
        raise ValueError("signs must hold one +-1 entry per rotated coordinate")
    limit = _input_limit(block)
    if values.numel() and int(values.abs().max()) >= limit:
        raise ValueError(
            f"integer rotation inputs must stay below {limit} in magnitude "
            f"for block {block}"
        )


def integer_butterfly(
    values: torch.Tensor,
    signs: torch.Tensor,
    block: int = 128,
    theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate int64 rows by a butterfly that is exactly invertible.

    The real-valued transform is the one :class:`~rotquant.rotate.ButterflyRotation`
    applies: multiply by the ``+-1`` ``signs``, then ``log2(block)`` stages of
    two-coordinate reflections ``[[cos, sin], [sin, -cos]]``. ``theta`` has
    shape ``[dim // block, log2(block), block // 2]``; ``None`` means ``pi/4``
    everywhere, i.e. the randomised block Hadamard transform.

    Each reflection is a sign flip followed by a plane rotation, and a plane
    rotation is three shears. A shear ``x += round(p * y)`` changes one
    coordinate by a quantity computed from the other, so subtracting the same
    quantity undoes it exactly whatever the rounding. The result differs from
    the real-valued rotation by at most a few units per stage, and
    :func:`integer_butterfly_inverse` returns the input bit for bit.
    """
    _check_integer_input(values, signs, block)
    stages = int(math.log2(block))
    n_blocks = values.shape[-1] // block
    if theta is None:
        theta = torch.full((n_blocks, stages, block // 2), math.pi / 4)
    if tuple(theta.shape) != (n_blocks, stages, block // 2):
        raise ValueError("theta must have shape [dim // block, log2(block), block // 2]")
    shear, sine = _lifting_multipliers(theta)
    work = (values * signs.to(torch.int64)).reshape(-1, values.shape[-1]).clone()
    for stage in range(stages):
        a, b, groups, step = _stage_view(work, block, stage)
        p = shear[:, stage].reshape(1, n_blocks, groups, step)
        u = sine[:, stage].reshape(1, n_blocks, groups, step)
        b = -b
        a = a + _scaled(p, b)
        b = b + _scaled(u, a)
        a = a + _scaled(p, b)
        work = torch.stack((a, b), dim=3).reshape(-1, values.shape[-1])
    return work.reshape(values.shape)


def integer_butterfly_inverse(
    values: torch.Tensor,
    signs: torch.Tensor,
    block: int = 128,
    theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact inverse of :func:`integer_butterfly`."""
    if values.dtype != torch.int64:
        raise TypeError("the integer rotation operates on int64 tensors")
    dim = values.shape[-1]
    if block < 1 or block & (block - 1) or dim % block:
        raise ValueError("block must be a power of two dividing the last dimension")
    if signs.shape != (dim,):
        raise ValueError("signs must hold one +-1 entry per rotated coordinate")
    stages = int(math.log2(block))
    n_blocks = dim // block
    if theta is None:
        theta = torch.full((n_blocks, stages, block // 2), math.pi / 4)
    shear, sine = _lifting_multipliers(theta)
    work = values.reshape(-1, dim).clone()
    for stage in range(stages - 1, -1, -1):
        a, b, groups, step = _stage_view(work, block, stage)
        p = shear[:, stage].reshape(1, n_blocks, groups, step)
        u = sine[:, stage].reshape(1, n_blocks, groups, step)
        a = a - _scaled(p, b)
        b = b - _scaled(u, a)
        a = a - _scaled(p, b)
        b = -b
        work = torch.stack((a, b), dim=3).reshape(-1, dim)
    return (work * signs.to(torch.int64)).reshape(values.shape)


@dataclass(frozen=True)
class FixedPoint:
    """An exact integer view of a floating-point tensor.

    ``value == integers * 2.0 ** exponent`` holds for every element. Negative
    zeros are the only bit patterns the integers cannot carry, so their flat
    positions are kept separately.
    """

    integers: torch.Tensor
    exponent: int
    dtype: torch.dtype
    negative_zero: torch.Tensor


def floats_to_fixed_point(tensor: torch.Tensor) -> FixedPoint:
    """Express every element of a float tensor as ``integer * 2**exponent``.

    The exponent is the unit in the last place of the smallest non-zero
    magnitude present, so the conversion loses nothing. It fails closed when
    the tensor's dynamic range does not fit an exact integer.
    """
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("fixed-point conversion supports float16, bfloat16 and float32")
    if not torch.isfinite(tensor).all():
        raise ValueError("fixed-point conversion requires finite values")
    work = tensor.detach().to(torch.float64)
    nonzero = work[work != 0]
    if nonzero.numel() == 0:
        exponent = 0
    else:
        mantissa_bits = {torch.float16: 10, torch.bfloat16: 7, torch.float32: 23}[tensor.dtype]
        minimum_exponent = {torch.float16: -14, torch.bfloat16: -126, torch.float32: -126}[
            tensor.dtype
        ]
        _, exponents = torch.frexp(nonzero.abs().min())
        # frexp returns a mantissa in [0.5, 1); the IEEE exponent is one less.
        exponent = max(int(exponents) - 1, minimum_exponent) - mantissa_bits
    scaled = torch.ldexp(work, torch.tensor(-exponent))
    if scaled.numel() and float(scaled.abs().max()) >= 2.0 ** 53:
        raise ValueError("tensor dynamic range is too wide for an exact integer view")
    integers = scaled.to(torch.int64)
    if not torch.equal(integers.to(torch.float64), scaled):
        raise ValueError("tensor values are not exact multiples of the common unit")
    negative_zero = torch.nonzero(
        ((work == 0) & torch.signbit(work)).reshape(-1)
    ).reshape(-1)
    return FixedPoint(integers, exponent, tensor.dtype, negative_zero)


def fixed_point_to_floats(fixed: FixedPoint) -> torch.Tensor:
    """Inverse of :func:`floats_to_fixed_point`; returns the original bits."""
    values = torch.ldexp(
        fixed.integers.to(torch.float64), torch.tensor(fixed.exponent)
    ).to(fixed.dtype)
    if fixed.negative_zero.numel():
        flat = values.reshape(-1)
        flat[fixed.negative_zero] = -0.0
    return values


# --------------------------------------------------------------------------- #
# Entropy of a code stream
# --------------------------------------------------------------------------- #
def symbol_entropy(symbols: torch.Tensor | np.ndarray, alphabet: int) -> float:
    """Order-0 empirical entropy in bits per symbol."""
    values = (
        symbols.detach().cpu().numpy() if isinstance(symbols, torch.Tensor) else symbols
    ).reshape(-1)
    if values.size == 0:
        return 0.0
    counts = np.bincount(values.astype(np.int64), minlength=alphabet).astype(np.float64)
    probabilities = counts[counts > 0] / values.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def code_entropy(packed: PackedTensor) -> dict[str, float | int]:
    """Stored versus entropy-coded size of one packed code stream.

    ``entropy_bits`` is the size an ideal order-0 coder with one static
    histogram per tensor reaches; ``saving_fraction`` is the share of the
    fixed-rate code bits it removes. No code changes, so neither does the
    model.
    """
    indices = unpack_indices(packed)
    entropy = symbol_entropy(indices, 1 << packed.bits)
    return {
        "bits": packed.bits,
        "symbols": packed.numel,
        "entropy_bits_per_symbol": entropy,
        "fixed_bits": packed.numel * packed.bits,
        "entropy_bits": entropy * packed.numel,
        "saving_fraction": 1.0 - entropy / packed.bits if packed.bits else 0.0,
    }


# --------------------------------------------------------------------------- #
# rANS coder with independent row streams
# --------------------------------------------------------------------------- #
_PROB_BITS = 14
_PROB_TOTAL = 1 << _PROB_BITS
_STATE_LOW = 1 << 16


def _normalised_frequencies(symbols: np.ndarray, alphabet: int) -> np.ndarray:
    """Integer frequencies summing to ``_PROB_TOTAL``; present symbols get >= 1."""
    counts = np.bincount(symbols.reshape(-1), minlength=alphabet).astype(np.int64)
    if counts.sum() == 0:
        raise ValueError("cannot build a frequency table for an empty stream")
    present = counts > 0
    if int(present.sum()) > _PROB_TOTAL:
        raise ValueError("alphabet is too large for the probability resolution")
    scaled = counts * _PROB_TOTAL / counts.sum()
    frequencies = np.where(present, np.maximum(1, np.floor(scaled)), 0).astype(np.int64)
    # Hand the rounding remainder to (or take the excess from) the symbols
    # where one unit changes the code length least, i.e. the most frequent.
    order = np.argsort(-counts, kind="stable")
    order = order[counts[order] > 0]
    deficit = _PROB_TOTAL - int(frequencies.sum())
    position = 0
    while deficit != 0:
        symbol = order[position % order.size]
        if deficit > 0:
            frequencies[symbol] += 1
            deficit -= 1
        elif frequencies[symbol] > 1:
            frequencies[symbol] -= 1
            deficit += 1
        position += 1
    return frequencies


@dataclass(frozen=True)
class RansRows:
    """Entropy-coded rows: one frequency table, one 16-bit word stream per row."""

    frequencies: np.ndarray   # int64 [alphabet], sums to 2**14
    row_words: np.ndarray     # uint32 [rows], words in each row stream
    words: np.ndarray         # uint16, all row streams concatenated
    shape: tuple[int, int]

    @property
    def nbytes(self) -> int:
        """Serialised size: 16-bit table entries, 32-bit row lengths, words."""
        return self.frequencies.size * 2 + self.row_words.size * 4 + self.words.size * 2


def rans_encode_rows(symbols: np.ndarray, alphabet: int) -> RansRows:
    """Entropy-code a ``[rows, columns]`` integer array, each row on its own.

    Every row is a separate rANS stream with a 32-bit state and 16-bit
    renormalisation, sharing one static frequency table. The work is
    vectorised across rows; the loop runs once per column.
    """
    data = np.ascontiguousarray(symbols, dtype=np.int64)
    if data.ndim != 2:
        raise ValueError("symbols must be a [rows, columns] array")
    rows, columns = data.shape
    if data.size and (data.min() < 0 or data.max() >= alphabet):
        raise ValueError("symbol out of range for the alphabet")
    if data.size == 0:
        return RansRows(
            np.zeros(alphabet, dtype=np.int64), np.zeros(rows, dtype=np.uint32),
            np.zeros(0, dtype=np.uint16), (rows, columns),
        )
    frequencies = _normalised_frequencies(data, alphabet)
    cumulative = np.concatenate(([0], np.cumsum(frequencies)[:-1]))
    shift = np.uint64(32 - _PROB_BITS)
    row_words = np.zeros(rows, dtype=np.uint32)
    pieces: list[np.ndarray] = []
    # Bound the renormalisation buffer: a vocabulary matrix has 1e5+ rows.
    chunk_rows = max(1, min(rows, (1 << 26) // max(columns, 1)))
    for begin in range(0, rows, chunk_rows):
        block = data[begin:begin + chunk_rows]
        count_rows = block.shape[0]
        state = np.full(count_rows, _STATE_LOW, dtype=np.uint64)
        emitted = np.zeros((count_rows, columns), dtype=np.uint16)
        counts = np.zeros(count_rows, dtype=np.int64)
        row_index = np.arange(count_rows)
        for column in range(columns - 1, -1, -1):
            symbol = block[:, column]
            frequency = frequencies[symbol].astype(np.uint64)
            overflow = state >= (frequency << shift)
            if overflow.any():
                selected = row_index[overflow]
                emitted[selected, counts[selected]] = (
                    state[overflow] & np.uint64(0xFFFF)
                ).astype(np.uint16)
                counts[selected] += 1
                state[overflow] >>= np.uint64(16)
            state = (
                ((state // frequency) << np.uint64(_PROB_BITS))
                + (state % frequency)
                + cumulative[symbol].astype(np.uint64)
            )
        # A row stream is the final state (high word first) followed by the
        # renormalisation words in the order the decoder consumes them, which
        # is the reverse of the order they were produced.
        lengths = counts + 2
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        words = np.zeros(int(offsets[-1]), dtype=np.uint16)
        words[offsets[:-1]] = (state >> np.uint64(16)).astype(np.uint16)
        words[offsets[:-1] + 1] = (state & np.uint64(0xFFFF)).astype(np.uint16)
        for row in range(count_rows):
            count = int(counts[row])
            if count:
                start = int(offsets[row]) + 2
                words[start:start + count] = emitted[row, :count][::-1]
        row_words[begin:begin + count_rows] = lengths
        pieces.append(words)
    return RansRows(frequencies, row_words, np.concatenate(pieces), (rows, columns))


def rans_decode_rows(coded: RansRows) -> np.ndarray:
    """Inverse of :func:`rans_encode_rows`; returns the original int64 array."""
    rows, columns = coded.shape
    output = np.zeros((rows, columns), dtype=np.int64)
    if rows == 0 or columns == 0:
        return output
    frequencies = coded.frequencies.astype(np.uint64)
    if int(frequencies.sum()) != _PROB_TOTAL:
        raise ValueError("rANS frequency table does not sum to the probability total")
    cumulative = np.concatenate(([0], np.cumsum(coded.frequencies)[:-1])).astype(np.uint64)
    slot_symbol = np.repeat(np.arange(frequencies.size), coded.frequencies).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(coded.row_words.astype(np.int64))))
    if int(offsets[-1]) != coded.words.size or (coded.row_words < 2).any():
        raise ValueError("rANS row lengths do not match the word stream")
    words = coded.words.astype(np.uint64)
    state = (words[offsets[:-1]] << np.uint64(16)) | words[offsets[:-1] + 1]
    cursor = offsets[:-1] + 2
    limit = offsets[1:]
    mask = np.uint64(_PROB_TOTAL - 1)
    for column in range(columns):
        slot = state & mask
        symbol = slot_symbol[slot.astype(np.int64)]
        output[:, column] = symbol
        state = frequencies[symbol] * (state >> np.uint64(_PROB_BITS)) + slot - cumulative[symbol]
        refill = state < np.uint64(_STATE_LOW)
        if refill.any():
            if (cursor[refill] >= limit[refill]).any():
                raise ValueError("rANS stream ended before its row was decoded")
            state[refill] = (state[refill] << np.uint64(16)) | words[cursor[refill]]
            cursor[refill] += 1
    if not np.array_equal(cursor, limit) or (state != np.uint64(_STATE_LOW)).any():
        raise ValueError("rANS stream is corrupt: trailing data or wrong final state")
    return output


def compress_codes(packed: PackedTensor, rows: int | None = None) -> RansRows:
    """Entropy-code a packed code stream without changing any code.

    ``rows`` sets how many independent streams the codes are split into; it
    defaults to the leading dimension of a two-dimensional code tensor.
    """
    indices = unpack_indices(packed).detach().cpu().numpy().astype(np.int64)
    if rows is None:
        rows = packed.shape[0] if len(packed.shape) == 2 else 1
    if rows < 1 or packed.numel % rows:
        raise ValueError("rows must divide the number of codes")
    return rans_encode_rows(indices.reshape(rows, -1), 1 << packed.bits)


def decompress_codes(coded: RansRows, bits: int, shape: tuple[int, ...]) -> PackedTensor:
    """Rebuild the exact :class:`PackedTensor` that :func:`compress_codes` read."""
    indices = torch.from_numpy(rans_decode_rows(coded)).reshape(shape)
    return pack_indices(indices, bits)
