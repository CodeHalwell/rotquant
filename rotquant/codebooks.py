"""Quantisation codebooks.

Scalar grids (fully implemented and unit-tested on a unit Gaussian):

* :func:`lloyd_max_gaussian` -- MSE-optimal scalar grid for a unit Gaussian
  (the HIGGS / TurboQuant-MSE grid). Anchors the source-coding test.
* :func:`uniform_signed`     -- symmetric uniform signed grid.
* :func:`normal_float`       -- NormalFloat (NF) reference grid (bitsandbytes style).

Vector grids:

* :class:`E8LatticeCodebook` -- exact nearest-point quantiser for the E8 lattice
  (Conway & Sloane), the self-contained vector baseline.
* :class:`TrellisCodebook`   -- bridge to QTIP's trellis-coded quantiser; raises an
  informative error if the QTIP repo is not importable (we do not re-derive it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

SQRT2 = math.sqrt(2.0)


def turboquant_mse_bound(bits: float) -> float:
    """TurboQuant Theorem 1: theoretical MSE bound after randomised Hadamard rotation.

    After rotation the weight distribution is universal (concentrated Beta/Gaussian),
    so a single pre-computed Lloyd-Max codebook achieves:

        MSE ≤ (sqrt(3)·π/2) · 4^{-b}

    which is within ≈2.7× of the Shannon rate-distortion limit for a Gaussian source.
    ``bits`` can be fractional (e.g. 3.125 for 3-bit codes + 16-bit scale / 128 group).
    """
    return (math.sqrt(3) * math.pi / 2) * (4.0 ** -bits)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# --------------------------------------------------------------------------- #
# Scalar grids
# --------------------------------------------------------------------------- #
def lloyd_max_gaussian(levels: int, iters: int = 200, grid: int = 200_001,
                       span: float = 12.0) -> np.ndarray:
    """MSE-optimal scalar quantiser centroids for a unit (zero-mean, unit-var) Gaussian.

    Solved by Lloyd's algorithm on a dense pdf grid: alternately set each
    centroid to the conditional mean of its cell and each boundary to the
    midpoint of adjacent centroids. Returns ``levels`` sorted centroids.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    xs = np.linspace(-span, span, grid)
    dx = xs[1] - xs[0]
    pdf = _normal_pdf(xs) * dx
    # Symmetric initialisation across the support.
    centroids = np.linspace(-2.5, 2.5, levels) if levels > 1 else np.array([0.0])

    for _ in range(iters):
        # Boundaries are midpoints between adjacent centroids.
        bounds = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate(([-np.inf], bounds, [np.inf]))
        new_centroids = centroids.copy()
        idx = np.searchsorted(edges, xs, side="right") - 1
        for k in range(levels):
            mask = idx == k
            w = pdf[mask].sum()
            if w > 0:
                new_centroids[k] = (xs[mask] * pdf[mask]).sum() / w
        if np.allclose(new_centroids, centroids, atol=1e-10):
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.sort(centroids)


def quantizer_mse(centroids: np.ndarray, grid: int = 200_001,
                  span: float = 12.0) -> float:
    """Expected MSE of a scalar quantiser with given centroids on a unit Gaussian."""
    xs = np.linspace(-span, span, grid)
    dx = xs[1] - xs[0]
    pdf = _normal_pdf(xs) * dx
    c = np.sort(centroids)
    bounds = (c[:-1] + c[1:]) / 2.0
    edges = np.concatenate(([-np.inf], bounds, [np.inf]))
    idx = np.searchsorted(edges, xs, side="right") - 1
    q = c[idx]
    return float(((xs - q) ** 2 * pdf).sum())


def uniform_signed(levels: int, clip: float = 1.0) -> np.ndarray:
    """Symmetric uniform signed grid on ``[-clip, clip]`` with ``levels`` points."""
    if levels < 2:
        return np.array([0.0])
    return np.linspace(-clip, clip, levels)


def normal_float(levels: int, offset: float = 0.5) -> np.ndarray:
    """NormalFloat (NF) grid: equal-mass normal quantiles, normalised to [-1, 1].

    Mirrors the bitsandbytes NF construction: split the probability mass into
    ``levels`` quantiles (offset to avoid the infinite tails), map through the
    Gaussian inverse-CDF, and rescale so the extreme code is +-1.
    """
    from scipy.stats import norm

    if levels < 2:
        return np.array([0.0])
    # Equal-mass quantile midpoints with tail offsets.
    half = levels // 2
    if levels % 2 == 1:
        pos = norm.ppf(np.linspace(0.5, 1 - offset / levels, half + 1))[1:]
        neg = -pos[::-1]
        vals = np.concatenate([neg, [0.0], pos])
    else:
        pos = norm.ppf(np.linspace(0.5 + 0.5 / levels, 1 - offset / levels, half))
        neg = -pos[::-1]
        vals = np.concatenate([neg, pos])
    vals = np.sort(vals)
    m = np.max(np.abs(vals))
    return vals / m if m > 0 else vals


# --------------------------------------------------------------------------- #
# Codebook objects
# --------------------------------------------------------------------------- #
class ScalarCodebook:
    """Wraps a sorted set of centroids with nearest-centroid encode/decode."""

    def __init__(self, centroids, name: str = "scalar"):
        # Accepts an array-like (np.ndarray) or a torch.Tensor (e.g. the QJL grid).
        self.name = name
        self.centroids, _ = torch.sort(
            torch.as_tensor(centroids, dtype=torch.float32))
        self._bounds = (self.centroids[:-1] + self.centroids[1:]) / 2.0

    @property
    def levels(self) -> int:
        return self.centroids.numel()

    @property
    def code_bits(self) -> float:
        return math.log2(self.levels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer indices of the nearest centroid for each element."""
        bounds = self._bounds.to(x.device, x.dtype)
        return torch.bucketize(x, bounds)

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(idx.device)[idx]

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.encode(x)
        return self.decode(idx), idx

    def to(self, device) -> "ScalarCodebook":
        """Return a *new* codebook on ``device``.

        Non-mutating on purpose: codebooks are built once and shared across many
        :class:`QuantizedWeight` objects (see ``build_scalar_codebook``), so moving
        one in place would silently corrupt the others.
        """
        out = self.__class__.__new__(self.__class__)
        out.name = self.name
        out.centroids = self.centroids.to(device)
        out._bounds = self._bounds.to(device)
        return out


def build_scalar_codebook(kind: str, levels: int) -> ScalarCodebook:
    kind = kind.lower()
    if kind in ("gaussian", "lloyd", "lloyd_max", "mse"):
        return ScalarCodebook(lloyd_max_gaussian(levels), name="gaussian")
    if kind == "uniform":
        return ScalarCodebook(uniform_signed(levels), name="uniform")
    if kind in ("nf", "normalfloat", "normal_float"):
        return ScalarCodebook(normal_float(levels), name="nf")
    raise ValueError(f"unknown scalar codebook kind: {kind}")


# --------------------------------------------------------------------------- #
# Vector grids
# --------------------------------------------------------------------------- #
def _nearest_d8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of the D8 lattice (integer vectors with even coordinate sum)."""
    f = torch.round(x)
    s = f.sum(dim=-1)
    even = (s % 2) == 0
    # For odd-sum points, flip the coordinate with the largest rounding error.
    err = x - f
    j = torch.argmax(torch.abs(err), dim=-1, keepdim=True)  # (..., 1)
    err_j = err.gather(-1, j)                               # (..., 1)
    flip = torch.where(err_j >= 0, torch.ones_like(err_j), -torch.ones_like(err_j))
    g = f.clone()
    g.scatter_(-1, j, f.gather(-1, j) + flip)
    out = torch.where(even.unsqueeze(-1), f, g)
    return out


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Exact nearest point of the E8 lattice = D8 union (D8 + (1/2,...,1/2))."""
    a = _nearest_d8(x)
    half = 0.5 * torch.ones_like(x)
    b = _nearest_d8(x - half) + half
    da = ((x - a) ** 2).sum(dim=-1)
    db = ((x - b) ** 2).sum(dim=-1)
    return torch.where((da <= db).unsqueeze(-1), a, b)


class E8LatticeCodebook:
    """Vector quantiser snapping scaled blocks of 8 to the nearest E8 lattice point.

    A self-contained vector baseline (no external repo needed). The lattice scale
    controls the effective rate; pick it so the per-weight bit budget matches the
    scalar comparison in E6.
    """

    dim = 8

    def __init__(self, lattice_scale: float = 1.0, name: str = "e8"):
        self.lattice_scale = lattice_scale
        self.name = name

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        *lead, d = x.shape
        if d % self.dim != 0:
            raise ValueError(f"E8 needs a dim divisible by 8, got {d}")
        xb = x.reshape(*lead, d // self.dim, self.dim) / self.lattice_scale
        q = nearest_e8(xb) * self.lattice_scale
        return q.reshape(*lead, d)


class TrellisCodebook:
    """Bridge to QTIP's trellis-coded quantiser (we do not re-derive it).

    Importing succeeds only if the QTIP repo is on ``PYTHONPATH``; otherwise the
    constructor raises with the clone instructions from the spec.
    """

    def __init__(self, **kwargs):
        try:  # pragma: no cover - requires the external QTIP repo
            import qtip  # noqa: F401  type: ignore
        except Exception as exc:  # pragma: no cover
            raise NotImplementedError(
                "TrellisCodebook bridges to QTIP, which is a repo not a package. "
                "Clone it and add to PYTHONPATH:\n"
                "  git clone https://github.com/Cornell-RelaxML/qtip\n"
                f"(import failed: {exc})"
            )
        self.kwargs = kwargs
