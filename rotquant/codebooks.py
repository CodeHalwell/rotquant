"""Quantisation codebooks.

Scalar grids (fully implemented and unit-tested on a unit Gaussian):

* :func:`lloyd_max_gaussian` -- MSE-optimal scalar grid for a unit Gaussian
  (the HIGGS / TurboQuant-MSE grid). Anchors the source-coding test.
* :func:`lloyd_max_spherical` -- dimension-aware grid for a coordinate of a
  uniformly rotated, RMS-normalised vector (the finite-dimensional TurboQuant
  distribution, rather than its Gaussian limit).
* :func:`uniform_signed`     -- symmetric uniform signed grid.
* :func:`normal_float`       -- NormalFloat (NF) reference grid (bitsandbytes style).

Vector grids:

* :class:`E8LatticeCodebook` -- exact nearest-point primitive for the E8 lattice
  (Conway & Sloane). It is not a finite-rate encoder or a packed baseline.
* :class:`VectorCodebook` -- a finite-rate product-vector codebook trained on a
  deterministic Gaussian source for matched W1--W3 research trials.

Trellis-coded quantisation (QTIP) is deliberately not bridged here; integrating
it as a real packed baseline is tracked in the roadmap.
"""
from __future__ import annotations

import functools
import math

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


def _lloyd_max_grid(
    xs: np.ndarray,
    weights: np.ndarray,
    levels: int,
    *,
    initial: np.ndarray,
    iters: int,
) -> np.ndarray:
    """Solve a one-dimensional Lloyd-Max problem on a weighted grid."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    centroids = np.asarray(initial, dtype=np.float64).copy()
    if centroids.shape != (levels,):
        raise ValueError("initial centroids must contain exactly levels entries")
    for _ in range(iters):
        bounds = (centroids[:-1] + centroids[1:]) / 2.0
        indices = np.searchsorted(bounds, xs, side="right")
        new_centroids = centroids.copy()
        for index in range(levels):
            mask = indices == index
            mass = weights[mask].sum()
            if mass > 0:
                new_centroids[index] = (
                    xs[mask] * weights[mask]
                ).sum() / mass
        if np.allclose(new_centroids, centroids, atol=1e-10, rtol=0.0):
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.sort(centroids)


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


def spherical_coordinate_grid(
    dimension: int, grid: int = 200_001
) -> tuple[np.ndarray, np.ndarray]:
    """Return a grid and masses for an RMS-normalised spherical coordinate.

    If ``u`` is uniform on the unit sphere in ``dimension`` dimensions, then
    ``sqrt(dimension) * u[0]`` has unit variance and bounded support
    ``[-sqrt(dimension), sqrt(dimension)]``.  This is the exact marginal used by
    TurboQuant before taking its high-dimensional Gaussian approximation.

    Dimensions below three have discrete or endpoint-singular special cases and
    are not useful model rotation blocks, so they are rejected deliberately.
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 3:
        raise ValueError("spherical codebooks require an integer dimension >= 3")
    if grid < 3:
        raise ValueError("grid must be >= 3")
    radius = math.sqrt(dimension)
    xs = np.linspace(-radius, radius, grid, dtype=np.float64)
    unit = xs / radius
    exponent = (dimension - 3) / 2.0
    if exponent == 0:
        density = np.ones_like(xs)
    else:
        # The normalising constant cancels during Lloyd updates.  Computing the
        # bounded kernel directly is also stable for large model dimensions.
        density = np.maximum(1.0 - unit * unit, 0.0) ** exponent
    dx = xs[1] - xs[0]
    masses = density * dx
    masses /= masses.sum()
    return xs, masses


def lloyd_max_spherical(
    levels: int,
    dimension: int,
    iters: int = 200,
    grid: int = 200_001,
) -> np.ndarray:
    """Finite-dimensional TurboQuant scalar centroids with unit RMS.

    The result approaches :func:`lloyd_max_gaussian` as ``dimension`` grows but
    preserves the correct bounded tails at the relatively small block/head
    dimensions used by real model kernels.
    """
    xs, masses = spherical_coordinate_grid(dimension, grid=grid)
    gaussian = lloyd_max_gaussian(levels, iters=iters, grid=grid)
    initial = np.clip(gaussian, xs[0], xs[-1])
    return _lloyd_max_grid(
        xs, masses, levels, initial=initial, iters=iters
    )


def quantizer_mse_spherical(
    centroids: np.ndarray,
    dimension: int,
    grid: int = 200_001,
) -> float:
    """Expected MSE of ``centroids`` on the unit-RMS spherical marginal."""
    xs, masses = spherical_coordinate_grid(dimension, grid=grid)
    sorted_centroids = np.sort(np.asarray(centroids, dtype=np.float64))
    bounds = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2.0
    indices = np.searchsorted(bounds, xs, side="right")
    return float(((xs - sorted_centroids[indices]) ** 2 * masses).sum())


def lloyd_max_samples(samples, levels: int, iters: int = 200) -> np.ndarray:
    """Fit deterministic Lloyd-Max centroids to representative scalar samples.

    Callers should pass values in the same normalised, rotated domain consumed
    by the quantizer.  This is the inexpensive calibrated tier: the resulting
    centroids use the existing artifact/runtime format and require no per-weight
    metadata.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if isinstance(samples, torch.Tensor):
        values = samples.detach().float().cpu().numpy().reshape(-1)
    else:
        values = np.asarray(samples, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)].astype(np.float64, copy=False)
    if values.size < levels:
        raise ValueError("calibration requires at least levels finite samples")
    probabilities = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    centroids = np.quantile(values, probabilities)
    # Samples have equal mass.  Sorting once keeps the repeated cell reductions
    # deterministic across platforms and avoids a dense probability grid.
    values.sort()
    weights = np.ones_like(values)
    return _lloyd_max_grid(
        values, weights, levels, initial=centroids, iters=iters
    )


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

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.encode(x)
        return self.decode(idx), idx

    def to(self, device) -> ScalarCodebook:
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


class VectorCodebook:
    """Finite set of vector centroids with exact nearest-neighbour encoding.

    ``levels`` is deliberately a power of two so every vector index has a fixed
    packed width. A codebook with dimension ``d`` and ``2**(b*d)`` centroids
    therefore consumes exactly ``b`` code bits per scalar weight, before the
    same scale metadata charged to scalar RotQuant.
    """

    def __init__(self, centroids, *, name: str = "vector", chunk_size: int = 65536):
        values = torch.as_tensor(centroids, dtype=torch.float32)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
            raise ValueError(
                "vector centroids must have shape [levels >= 2, dimension >= 2]"
            )
        levels = int(values.shape[0])
        if levels & (levels - 1):
            raise ValueError("vector codebook levels must be a power of two")
        if chunk_size < 1:
            raise ValueError("vector codebook chunk_size must be positive")
        if not torch.isfinite(values).all():
            raise ValueError("vector centroids must be finite")
        self.name = name
        self.centroids = values.contiguous()
        self.chunk_size = int(chunk_size)

    @property
    def levels(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def code_bits(self) -> int:
        return int(math.log2(self.levels))

    @property
    def bits_per_weight(self) -> float:
        return self.code_bits / self.dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"vector width {x.shape[-1]} does not match codebook dim {self.dim}"
            )
        flat = x.reshape(-1, self.dim).float()
        centroids = self.centroids.to(flat.device)
        centroid_energy = centroids.square().sum(dim=1)
        indices = []
        for start in range(0, flat.shape[0], self.chunk_size):
            batch = flat[start:start + self.chunk_size]
            distances = (
                batch.square().sum(dim=1, keepdim=True)
                + centroid_energy.unsqueeze(0)
                - 2.0 * batch @ centroids.T
            )
            indices.append(distances.argmin(dim=1))
        return torch.cat(indices).reshape(x.shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(indices.device)[indices]

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.encode(x)
        return self.decode(indices), indices

    def to(self, device) -> VectorCodebook:
        return VectorCodebook(
            self.centroids.to(device), name=self.name, chunk_size=self.chunk_size
        )


def fit_vector_codebook(
    samples: torch.Tensor,
    levels: int,
    *,
    seed: int = 0,
    iters: int = 25,
    chunk_size: int = 65536,
    name: str = "vector_calibrated",
) -> VectorCodebook:
    """Fit a deterministic Lloyd/k-means vector codebook on CPU samples."""

    values = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("vector samples must have shape [samples, dimension >= 2]")
    if levels < 2 or levels & (levels - 1):
        raise ValueError("vector levels must be a power of two >= 2")
    if values.shape[0] < levels:
        raise ValueError("vector fitting requires at least one sample per centroid")
    if iters < 1:
        raise ValueError("vector fitting iterations must be positive")
    if not torch.isfinite(values).all():
        raise ValueError("vector fitting samples must be finite")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Deterministic k-means++ initialization avoids the severe empty-cell
    # behaviour of taking the first K Gaussian samples at W1/W2.
    first = int(torch.randint(values.shape[0], (1,), generator=generator).item())
    centroids = [values[first].clone()]
    closest = (values - centroids[0]).square().sum(dim=1)
    for _ in range(1, levels):
        total = closest.sum()
        if not torch.isfinite(total) or total <= 0:
            index = len(centroids) % values.shape[0]
        else:
            index = int(torch.multinomial(
                closest / total, 1, generator=generator
            ).item())
        centroid = values[index].clone()
        centroids.append(centroid)
        closest = torch.minimum(
            closest, (values - centroid).square().sum(dim=1)
        )
    centers = torch.stack(centroids)

    for _ in range(iters):
        codebook = VectorCodebook(
            centers, name=name, chunk_size=chunk_size
        )
        assignments = codebook.encode(values)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, values)
        counts = torch.bincount(assignments, minlength=levels)
        updated = centers.clone()
        occupied = counts > 0
        updated[occupied] = sums[occupied] / counts[occupied].unsqueeze(1)
        if torch.allclose(updated, centers, rtol=0.0, atol=1e-6):
            centers = updated
            break
        centers = updated
    return VectorCodebook(centers, name=name, chunk_size=chunk_size)


@functools.cache
def build_gaussian_vector_codebook(
    bits_per_weight: int,
    dimension: int = 2,
    *,
    seed: int = 0,
    samples: int = 16384,
    iters: int = 25,
) -> VectorCodebook:
    """Build a shared finite-rate codebook for rotated Gaussian coordinates."""

    if bits_per_weight < 1:
        raise ValueError("vector bits_per_weight must be positive")
    if dimension < 2:
        raise ValueError("vector dimension must be >= 2")
    code_bits = bits_per_weight * dimension
    if code_bits > 16:
        raise ValueError("vector indices cannot exceed the 16-bit packer limit")
    levels = 1 << code_bits
    if samples < levels:
        raise ValueError("vector training samples must be >= codebook levels")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
    training = torch.randn(samples, dimension, generator=generator)
    return fit_vector_codebook(
        training,
        levels,
        seed=seed + 130363,
        iters=iters,
        name=f"gaussian_vq_d{dimension}_w{bits_per_weight}",
    )


@functools.cache
def build_scalar_codebook(
    kind: str, levels: int, dimension: int | None = None
) -> ScalarCodebook:
    """Build (or return the cached) codebook for its complete specification.

    Cached because Lloyd-Max solves a 200k-point Lloyd iteration -- re-running it
    for every one of a model's hundreds of linears wastes minutes per run. The
    returned object is shared, which is safe because :meth:`ScalarCodebook.to`
    is non-mutating and nothing else writes to a built codebook.
    """
    kind = kind.lower()
    if kind in ("gaussian", "lloyd", "lloyd_max", "mse"):
        return ScalarCodebook(lloyd_max_gaussian(levels), name="gaussian")
    if kind in ("sphere", "spherical", "beta", "finite_beta"):
        if dimension is None:
            raise ValueError("spherical codebooks require a dimension")
        return ScalarCodebook(
            lloyd_max_spherical(levels, dimension),
            name=f"spherical_d{dimension}",
        )
    if kind == "uniform":
        return ScalarCodebook(uniform_signed(levels), name="uniform")
    if kind in ("nf", "normalfloat", "normal_float"):
        return ScalarCodebook(normal_float(levels), name="nf")
    raise ValueError(f"unknown scalar codebook kind: {kind}")


def fit_scalar_codebook(
    samples, levels: int, *, name: str = "calibrated", iters: int = 200
) -> ScalarCodebook:
    """Build a deployable scalar codebook from representative normalised values."""
    return ScalarCodebook(
        lloyd_max_samples(samples, levels, iters=iters), name=name
    )


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
    """Nearest-lattice quantiser snapping scaled blocks of 8 to an E8 point.

    This is a geometric primitive, not a finite-rate compression format: the E8
    lattice is infinite and this class does not encode or pack point indices.
    Consequently it must not be used as an equal-bits E6 baseline by itself.
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
