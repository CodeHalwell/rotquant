"""Quantisation codebooks.

Scalar grids (fully implemented and unit-tested on a unit Gaussian):

* :func:`lloyd_max_gaussian` -- MSE-optimal scalar grid for a unit Gaussian
  (the HIGGS / TurboQuant-MSE grid), solved to machine precision from the
  closed-form cell moments. Anchors the source-coding test.
  :func:`lloyd_max_gaussian_grid` keeps the earlier fixed-iteration grid solver
  for reproducing artifacts built before 2026-11-17.
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
    The right-hand side is the Panter-Dite high-resolution distortion of the
    optimal scalar quantiser; the exact Lloyd-Max distortion lies below it at
    every integer rate (0.3634, 0.1175, 0.0345, 0.00950 at one to four bits)
    and approaches it from below as the rate grows.

    ``bits`` counts code bits. A fractional value is accepted so the curve can
    be drawn against an effective rate, but scale metadata does not lower the
    distortion along this curve: 3-bit codes with a 16-bit scale per 128
    weights cost 3.125 bits and still have the 3-bit distortion.
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
    """Solve a one-dimensional Lloyd-Max problem on a sorted weighted grid.

    ``xs`` must be ascending. Cell sums come from two prefix sums, so one
    iteration costs O(levels log n) instead of O(levels * n). ``iters`` is a
    cap: the loop stops as soon as no centroid moves by more than 1e-10.
    A point lying exactly on a boundary belongs to the upper cell, as before.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    centroids = np.asarray(initial, dtype=np.float64).copy()
    if centroids.shape != (levels,):
        raise ValueError("initial centroids must contain exactly levels entries")
    xs = np.asarray(xs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mass_prefix = np.concatenate(([0.0], np.cumsum(weights)))
    moment_prefix = np.concatenate(([0.0], np.cumsum(xs * weights)))
    for _ in range(iters):
        bounds = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate(
            ([0], np.searchsorted(xs, bounds, side="left"), [xs.size])
        )
        mass = mass_prefix[edges[1:]] - mass_prefix[edges[:-1]]
        moment = moment_prefix[edges[1:]] - moment_prefix[edges[:-1]]
        occupied = mass > 0
        new_centroids = centroids.copy()
        new_centroids[occupied] = moment[occupied] / mass[occupied]
        if np.allclose(new_centroids, centroids, atol=1e-10, rtol=0.0):
            centroids = new_centroids
            break
        centroids = new_centroids
    return np.sort(centroids)


def _grid_quantizer_mse(
    xs: np.ndarray, weights: np.ndarray, centroids: np.ndarray
) -> float:
    """Weighted MSE of nearest-centroid quantisation on a sorted grid."""
    ordered = np.sort(np.asarray(centroids, dtype=np.float64))
    bounds = (ordered[:-1] + ordered[1:]) / 2.0
    indices = np.searchsorted(bounds, xs, side="right")
    total = float(np.sum(weights))
    return float(np.sum((xs - ordered[indices]) ** 2 * weights) / total)


# --------------------------------------------------------------------------- #
# Scalar grids
# --------------------------------------------------------------------------- #
def _gaussian_cells(
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form Voronoi cells of sorted ``centroids`` under N(0, 1).

    Returns ``(lower, upper, mass, first_moment)``. Each mass is evaluated on
    the side of the origin where the tail is small, so far-tail cells do not
    lose their value to cancellation between two numbers close to one.
    """
    from scipy.special import ndtr

    inner = (centroids[:-1] + centroids[1:]) / 2.0
    lower = np.concatenate(([-np.inf], inner))
    upper = np.concatenate((inner, [np.inf]))
    mass = np.where(
        lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower)
    )
    first_moment = _normal_pdf(lower) - _normal_pdf(upper)
    return lower, upper, mass, first_moment


def _lloyd_max_gaussian_exact(levels: int) -> np.ndarray:
    """Solve the Lloyd-Max stationarity equations for N(0, 1) by Newton.

    The unknowns are the centroids ``c``; the residual is ``c_i - E[Z | cell
    i]`` with boundaries at centroid midpoints. Its Jacobian is tridiagonal,
    so each Newton step is one banded solve. The start is the Panter-Dite
    compander (quantiles of N(0, 3)), which is already close at every rate,
    followed by a few plain Lloyd steps. A unit Gaussian is log-concave, so
    the stationary point is the unique global optimum (Fleischer, 1964).

    Plain Lloyd iteration converges only linearly, with a rate that approaches
    one as the level count grows: from a uniform start it needs about 2e5
    iterations at 256 levels, which is why a fixed 200-iteration loop is not a
    solver at five bits and above.
    """
    from scipy.linalg import solve_banded
    from scipy.special import ndtri

    centroids = math.sqrt(3.0) * ndtri((np.arange(levels) + 0.5) / levels)

    def lloyd_step(values: np.ndarray) -> np.ndarray:
        _, _, mass, first_moment = _gaussian_cells(values)
        return first_moment / mass

    for _ in range(50):
        centroids = lloyd_step(centroids)

    best, best_residual = centroids, np.inf
    for _ in range(100):
        _, upper, mass, first_moment = _gaussian_cells(centroids)
        means = first_moment / mass
        residual = centroids - means
        size = float(np.max(np.abs(residual)))
        if size < best_residual:
            best, best_residual = centroids, size
        if size < 1e-12:
            break
        inner = upper[:-1]
        density = _normal_pdf(inner)
        # d mean_i / d upper_i and d mean_(i+1) / d lower_(i+1); both
        # boundaries move at half the rate of either neighbouring centroid.
        d_upper = density * (inner - means[:-1]) / mass[:-1]
        d_lower = density * (means[1:] - inner) / mass[1:]
        diagonal = np.ones(levels)
        diagonal[:-1] -= 0.5 * d_upper
        diagonal[1:] -= 0.5 * d_lower
        banded = np.zeros((3, levels))
        banded[0, 1:] = -0.5 * d_upper
        banded[1] = diagonal
        banded[2, :-1] = -0.5 * d_lower
        step = solve_banded((1, 1), banded, -residual)
        if not np.all(np.isfinite(step)):
            break
        damping = 1.0
        trial = centroids + step
        while damping > 1e-6 and not np.all(np.diff(trial) > 0):
            damping *= 0.5
            trial = centroids + damping * step
        if not np.all(np.diff(trial) > 0):
            break
        centroids = trial

    centroids = best
    if best_residual > 1e-9:  # Newton stalled: finish on the fixed point map.
        for _ in range(200_000):
            updated = lloyd_step(centroids)
            if np.max(np.abs(updated - centroids)) < 1e-13:
                centroids = updated
                break
            centroids = updated
    # The optimum is antisymmetric; remove the last round-off asymmetry.
    return 0.5 * (centroids - centroids[::-1])


def lloyd_max_gaussian_grid(levels: int, iters: int = 200, grid: int = 200_001,
                            span: float = 12.0) -> np.ndarray:
    """The pre-2026-11-17 Gaussian solver, kept for reproducing old artifacts.

    It runs at most ``iters`` Lloyd steps on a discretised pdf from a uniform
    start in ``[-2.5, 2.5]``. That reaches the optimum only up to four bits;
    the grids it returns carry 0.07%, 0.62%, 5.3% and 27% excess MSE at five,
    six, seven and eight bits. New work should use :func:`lloyd_max_gaussian`.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    xs = np.linspace(-span, span, grid)
    dx = xs[1] - xs[0]
    pdf = _normal_pdf(xs) * dx
    centroids = np.linspace(-2.5, 2.5, levels) if levels > 1 else np.array([0.0])

    for _ in range(iters):
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


def lloyd_max_gaussian(levels: int, iters: int = 200, grid: int = 200_001,
                       span: float = 12.0) -> np.ndarray:
    """MSE-optimal scalar quantiser centroids for a unit (zero-mean, unit-var) Gaussian.

    Returns ``levels`` sorted centroids satisfying the Lloyd-Max conditions
    (boundaries at centroid midpoints, centroids at conditional means) to
    about 1e-12. ``iters``, ``grid`` and ``span`` belong to the earlier grid
    solver and are accepted only so existing call sites keep working; pass
    them to :func:`lloyd_max_gaussian_grid` to reproduce that solver.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if levels == 1:
        return np.array([0.0])
    return _lloyd_max_gaussian_exact(levels)


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
    iters: int = 20_000,
    grid: int = 200_001,
) -> np.ndarray:
    """Finite-dimensional TurboQuant scalar centroids with unit RMS.

    The result approaches :func:`lloyd_max_gaussian` as ``dimension`` grows but
    preserves the correct bounded tails at the relatively small block/head
    dimensions used by real model kernels.
    """
    xs, masses = spherical_coordinate_grid(dimension, grid=grid)
    gaussian = lloyd_max_gaussian(levels)
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


def lloyd_max_samples(samples, levels: int, iters: int = 2000) -> np.ndarray:
    """Fit deterministic Lloyd-Max centroids to representative scalar samples.

    Callers should pass values in the same normalised, rotated domain consumed
    by the quantizer.  This is the inexpensive calibrated tier: the resulting
    centroids use the existing artifact/runtime format and require no per-weight
    metadata.

    Lloyd's algorithm only finds a local optimum, and from equal-mass quantiles
    it approaches one slowly once there are 32 or more levels. Two starts are
    therefore refined and the one with the lower sample MSE is returned: the
    equal-mass quantiles, and the exact Gaussian Lloyd-Max grid placed at the
    sample mean and standard deviation, which is close whenever the samples
    are rotated weights. ``iters`` caps each refinement; both normally stop
    earlier on the 1e-10 movement tolerance.

    A fit cannot be better than its sample: with fewer than a few thousand
    samples per level the tail centroids follow the sample rather than the
    source. At six bits and above prefer the ``gaussian`` or ``spherical``
    grids unless the sample holds at least ``4096 * levels`` values.
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
    starts = [np.quantile(values, probabilities)]
    spread = float(values.std())
    if levels > 1 and spread > 0:
        starts.append(float(values.mean()) + spread * lloyd_max_gaussian(levels))
    # Samples have equal mass.  Sorting once keeps the repeated cell reductions
    # deterministic across platforms and avoids a dense probability grid.
    values.sort()
    weights = np.ones_like(values)
    best: np.ndarray | None = None
    best_mse = np.inf
    for initial in starts:
        candidate = _lloyd_max_grid(
            values, weights, levels, initial=initial, iters=iters
        )
        mse = _grid_quantizer_mse(values, weights, candidate)
        if mse < best_mse:
            best, best_mse = candidate, mse
    assert best is not None
    return best


def quantizer_mse(centroids: np.ndarray, grid: int = 200_001,
                  span: float = 12.0) -> float:
    """Expected MSE of a scalar quantiser with given centroids on a unit Gaussian.

    Evaluated in closed form from the Gaussian cell masses and first and
    second moments, so the result is exact for any centroid set. ``grid`` and
    ``span`` belonged to the earlier numerical integration and are ignored.
    """
    ordered = np.sort(np.asarray(centroids, dtype=np.float64))
    lower, upper, mass, first_moment = _gaussian_cells(ordered)
    with np.errstate(invalid="ignore"):
        lower_term = np.where(np.isinf(lower), 0.0, lower * _normal_pdf(lower))
        upper_term = np.where(np.isinf(upper), 0.0, upper * _normal_pdf(upper))
    second_moment = mass + lower_term - upper_term
    return float(
        (second_moment - 2.0 * ordered * first_moment + ordered**2 * mass).sum()
    )


def uniform_signed(levels: int, clip: float = 1.0) -> np.ndarray:
    """Symmetric uniform signed grid on ``[-clip, clip]`` with ``levels`` points."""
    if levels < 2:
        return np.array([0.0])
    return np.linspace(-clip, clip, levels)


def normal_float(levels: int, offset: float = 0.5) -> np.ndarray:
    """Symmetric NF-style grid from equal-mass Gaussian quantiles.

    This is a useful scalar control, but it is not the asymmetric bitsandbytes
    NF4 datatype (which includes an exact zero).  The historical ``nf`` config
    alias is retained for compatibility.
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
        # Bound the temporary distance matrix independently of codebook size.
        # This is essential for 16-bit vector indices (65,536 centroids).
        max_distance_values = 16_777_216
        row_chunk = min(
            self.chunk_size,
            max(1, max_distance_values // self.levels),
        )
        for start in range(0, flat.shape[0], row_chunk):
            batch = flat[start:start + row_chunk]
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


class FiniteE8Codebook(VectorCodebook):
    """Finite-rate, shaped E8 product codebook with a fixed packed index.

    The retained points are the lowest-energy non-zero points from both E8
    cosets, scaled for minimum MSE on unit-variance input (see
    :func:`build_finite_e8_codebook`). Encoding first uses the exact
    infinite-lattice nearest point. A sorted integer-key table maps in-support
    lattice points to their fixed index; rare overload points outside the
    shaping region fall back to exact nearest-neighbour search over the finite
    retained set. This makes the usual path O(d log K), while preserving exact
    finite-codebook semantics at the boundary.
    """

    def __init__(self, raw_points: torch.Tensor, *, normalization: float | None = None,
                 name: str = "e8p", chunk_size: int = 65536):
        raw = torch.as_tensor(raw_points, dtype=torch.float32)
        if raw.ndim != 2 or raw.shape[1] != 8:
            raise ValueError("finite E8 points must have shape [levels, 8]")
        norm = float(normalization or raw.square().mean().sqrt().item())
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("finite E8 normalization must be positive")
        self.normalization = norm
        self.raw_points = raw.contiguous()
        super().__init__(raw / norm, name=name, chunk_size=chunk_size)
        keys = self._point_keys(raw)
        self._sorted_keys, order = torch.sort(keys)
        self._sorted_indices = order.to(torch.int64)

    @staticmethod
    def _point_keys(points: torch.Tensor) -> torch.Tensor:
        doubled = torch.round(points * 2).to(torch.int64)
        if (doubled.abs() > 31).any():
            raise ValueError("finite E8 key codec supports doubled coordinates in [-31, 31]")
        powers = torch.tensor(
            [65 ** index for index in range(8)],
            dtype=torch.int64,
            device=points.device,
        )
        return ((doubled + 32) * powers).sum(dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 8:
            raise ValueError("finite E8 vectors must have width 8")
        flat = x.reshape(-1, 8).float()
        nearest = nearest_e8(flat * self.normalization)
        keys = self._point_keys(nearest)
        sorted_keys = self._sorted_keys.to(keys.device)
        sorted_indices = self._sorted_indices.to(keys.device)
        positions = torch.searchsorted(sorted_keys, keys)
        bounded = positions.clamp_max(sorted_keys.numel() - 1)
        hit = (positions < sorted_keys.numel()) & (sorted_keys[bounded] == keys)
        result = torch.empty_like(keys)
        result[hit] = sorted_indices[bounded[hit]]
        if (~hit).any():
            result[~hit] = super().encode(flat[~hit])
        return result.reshape(x.shape[:-1])

    def to(self, device) -> FiniteE8Codebook:
        return FiniteE8Codebook(
            self.raw_points.to(device),
            normalization=self.normalization,
            name=self.name,
            chunk_size=self.chunk_size,
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


def _e8_points_up_to(max_energy: int) -> torch.Tensor:
    """Every non-zero E8 point with squared norm at most ``max_energy``.

    E8 is ``D8`` (integer vectors with even coordinate sum) together with
    ``D8 + (1/2, ..., 1/2)``. Each coset is enumerated as pairs of 4-tuples so
    the full ``7**8`` box never has to be materialised. The coordinate ranges
    cover squared norm 12: an integer coordinate can reach 3 (``3^2 + 1 + 1 +
    1 = 12``) and a half-integer one 2.5 (``2.5^2 + 7/4 = 8``).
    """
    if max_energy > 12:
        raise ValueError("E8 enumeration is tabulated up to squared norm 12")

    def coset(axis: torch.Tensor) -> torch.Tensor:
        half_block = torch.cartesian_prod(*([axis] * 4))
        energy = half_block.square().sum(dim=1)
        keep = energy <= max_energy
        half_block, energy = half_block[keep], energy[keep]
        total = half_block.sum(dim=1)
        pair_energy = energy[:, None] + energy[None, :]
        pair_sum = total[:, None] + total[None, :]
        valid = (pair_energy <= max_energy) & (torch.remainder(pair_sum, 2) == 0)
        left, right = valid.nonzero(as_tuple=True)
        return torch.cat([half_block[left], half_block[right]], dim=1)

    integer = coset(torch.arange(-3.0, 4.0))
    shifted = coset(torch.arange(-3.0, 3.0) + 0.5)
    points = torch.cat([integer, shifted], dim=0)
    return points[points.square().sum(dim=1) > 0].contiguous()


# Scale of the unit-RMS codebook that minimises MSE on a unit Gaussian, found
# by a scan on 400,000 samples (see tests/test_vector_quantization.py). Unit
# RMS itself is not optimal: reconstruction points of an MSE quantiser carry
# less energy than the source at one bit, and slightly more than the retained
# shells suggest at two bits because 40% of Gaussian blocks overload them.
_E8_GAUSSIAN_SCALE = {1: 0.84, 2: 1.04}


@functools.cache
def build_finite_e8_codebook(bits_per_weight: int = 2) -> FiniteE8Codebook:
    """Build a deterministic 8D E8P-style codebook at an exact finite rate.

    The ``2**(8 * bits)`` retained points are the lowest-energy non-zero E8
    points: every complete shell that fits, then a subset of the next shell
    made of ``+p``/``-p`` pairs. The set is therefore closed under negation
    and has zero mean. The origin is left out because, together with complete
    shells, it would leave an odd number of slots for the last shell, and that
    cannot be filled symmetrically. The pairs of the partial shell are ordered
    by an integer hash of the point, which spreads them over the shell without
    depending on a random number generator's version.

    Before 2026-11-17 the candidates came from the box ``[-2, 2]^8`` and its
    ``+1/2`` shift ``[-1.5, 2.5]^8``. That box misses lattice points of squared
    norm 8 and 10 (512 and 3,808 of them at two bits) and is not symmetric, so
    the old codebook had a non-zero mean and 2.9% more Gaussian MSE.
    """

    if bits_per_weight not in (1, 2):
        raise ValueError("finite E8 supports 1 or 2 bits per weight")
    levels = 1 << (bits_per_weight * 8)
    candidates = _e8_points_up_to(12)
    energy = candidates.square().sum(dim=1)
    chosen: list[torch.Tensor] = []
    count = 0
    for shell_energy in torch.unique(energy).tolist():
        shell = candidates[energy == shell_energy]
        if count + shell.shape[0] <= levels:
            chosen.append(shell)
            count += shell.shape[0]
            continue
        needed = levels - count
        if needed % 2:
            raise RuntimeError("finite E8 shell fill must be even")
        if needed:
            first_nonzero = (shell != 0).to(torch.int64).argmax(dim=1)
            leading = shell.gather(1, first_nonzero[:, None])[:, 0]
            representatives = shell[leading > 0]
            doubled = torch.round(representatives * 2).to(torch.int64) + 8
            powers = torch.tensor(
                [17**index for index in range(8)], dtype=torch.int64
            )
            key = (doubled * powers).sum(dim=1)
            mixed = (key * 2654435761) % (1 << 32)
            order = torch.argsort(mixed * (1 << 34) + key)
            half = representatives[order[: needed // 2]]
            chosen.append(torch.cat([half, -half], dim=0))
        count = levels
        break
    if count != levels:
        raise RuntimeError("finite E8 enumeration holds too few points")
    points = torch.cat(chosen, dim=0).contiguous()
    unit_rms = float(points.square().mean().sqrt().item())
    return FiniteE8Codebook(
        points,
        normalization=unit_rms / _E8_GAUSSIAN_SCALE[bits_per_weight],
        name=f"e8p_d8_w{bits_per_weight}",
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
    if kind in ("gaussian_legacy", "gaussian_grid"):
        return ScalarCodebook(
            lloyd_max_gaussian_grid(levels), name="gaussian_legacy"
        )
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
