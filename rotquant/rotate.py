"""Rotation primitives for weight-only and activation-aware quantisation.

The math invariance we rely on everywhere is

    y = x W^T = (x R^T)(W R^T)^T            (nn.Linear convention, W is [out, in])

i.e. rotating the *input* dimension of the weight by ``R`` and the activation by
the same ``R`` leaves the linear map unchanged, because ``R^T R = I``. Both
``rotate_activation`` and ``rotate_weight`` therefore multiply by ``R^T`` on the
last (input) dimension.

Implemented rotations:

* :class:`RandomizedHadamard` -- a fixed random sign flip followed by the
  (block-wise) fast Walsh-Hadamard transform. This is the QuaRot / QuIP# primitive.
* :class:`DenseOrthogonal`   -- a dense random orthogonal matrix from the QR of a
  Gaussian (the E1 "dense" comparison).
* :class:`LearnedRotation`   -- an orthogonal matrix parametrised on the Stiefel
  manifold via the Cayley transform, trainable for the W4A4 / E1 ablation.
* :class:`Identity`          -- the "none" baseline.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:  # the fast CUDA kernel QuaRot/QuIP# use; optional, we fall back to pure torch
    from fast_hadamard_transform import hadamard_transform as _fht_cuda
except Exception:  # pragma: no cover - kernel only present with CUDA build
    _fht_cuda = None


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Fast Walsh-Hadamard transform along the last dimension.

    Uses the ``fast_hadamard_transform`` CUDA kernel when available, otherwise a
    pure-PyTorch iterative FWHT (works on CPU, used by the correctness tests).
    With ``normalize=True`` the transform is orthonormal (``H / sqrt(d)``) and is
    its own inverse.
    """
    d = x.shape[-1]
    if not _is_pow2(d):
        raise ValueError(f"FWHT length must be a power of two, got {d}")

    if _fht_cuda is not None and x.is_cuda:
        # The kernel applies the unnormalised H; scale to match our convention.
        out = _fht_cuda(x.contiguous())
        return out / math.sqrt(d) if normalize else out

    orig_shape = x.shape
    h = x.reshape(-1, d).clone()
    step = 1
    while step < d:
        h = h.view(-1, d // (2 * step), 2, step)
        a = h[:, :, 0, :]
        b = h[:, :, 1, :]
        h = torch.stack([a + b, a - b], dim=2).view(-1, d)
        step *= 2
    if normalize:
        h = h / math.sqrt(d)
    return h.view(orig_shape)


class Rotation(nn.Module):
    """Base class. ``dim`` is the input dimension being rotated."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x @ R^T`` along the last dim."""
        raise NotImplementedError

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Return ``weight @ R^T`` (rotates the input dim of an ``[out, in]`` weight)."""
        raise NotImplementedError

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x @ R`` -- the inverse, fused into dequant by the patcher."""
        raise NotImplementedError

    def as_matrix(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Materialise the ``[dim, dim]`` rotation matrix (tests / dense use)."""
        eye = torch.eye(self.dim, device=device, dtype=dtype)
        # rows of R are R^T applied to basis vectors... we want R itself.
        # rotate_activation(e_i) = e_i @ R^T = (R^T)_i = i-th row of R^T = i-th col of R.
        cols = self.rotate_activation(eye)
        return cols.transpose(-1, -2).contiguous()


class Identity(Rotation):
    """The "none" baseline -- no rotation."""

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RandomizedHadamard(Rotation):
    """Randomised (block) Hadamard transform: ``R = blockdiag(H/sqrt(b) @ diag(s))``.

    A fixed random sign vector ``s`` is applied first, then a block-wise FWHT.
    ``block`` must be a power of two dividing ``dim`` (128 default, 256 if divisible).
    """

    def __init__(self, dim: int, block: int = 128, seed: Optional[int] = None,
                 device=None, dtype=torch.float32):
        super().__init__(dim)
        if dim % block != 0:
            # fall back to the largest power-of-two block dividing dim
            block = self._largest_pow2_divisor(dim)
        if not _is_pow2(block):
            raise ValueError(f"Hadamard block must be a power of two, got {block}")
        self.block = block
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) * 2 - 1
        self.register_buffer("signs", signs.to(device=device, dtype=dtype))

    @staticmethod
    def _largest_pow2_divisor(n: int) -> int:
        b = 1
        while n % (b * 2) == 0:
            b *= 2
        return b

    def _blocked_fwht(self, t: torch.Tensor) -> torch.Tensor:
        *lead, d = t.shape
        nb = d // self.block
        tb = t.reshape(*lead, nb, self.block)
        tb = fwht(tb, normalize=True)
        return tb.reshape(*lead, d)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        # x @ R^T = FWHT_norm(x * s)
        return self._blocked_fwht(x * self.signs.to(x.dtype))

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self._blocked_fwht(weight * self.signs.to(weight.dtype))

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        # x @ R = FWHT_norm(x) * s
        return self._blocked_fwht(x) * self.signs.to(x.dtype)


class DenseOrthogonal(Rotation):
    """Dense random orthogonal rotation from the QR of a Gaussian matrix."""

    def __init__(self, dim: int, seed: Optional[int] = None, device=None,
                 dtype=torch.float32):
        super().__init__(dim)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        # Make the decomposition unique / sign-stable.
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        self.register_buffer("R", q.to(device=device, dtype=dtype))

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R.to(x.dtype).T

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight @ self.R.to(weight.dtype).T

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R.to(x.dtype)


class LearnedRotation(Rotation):
    """Orthogonal rotation parametrised on the Stiefel manifold via the Cayley map.

    ``R = (I - A)(I + A)^{-1}`` with ``A`` skew-symmetric is always orthogonal, so
    gradient descent on the free (lower-triangular) parameters of ``A`` stays on
    the manifold. Used for the E1 learned-rotation ablation (it should only pull
    ahead once activations are also quantised, e.g. W4A4).
    """

    def __init__(self, dim: int, seed: Optional[int] = None, device=None,
                 dtype=torch.float32):
        super().__init__(dim)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        # Free parameters = strictly-lower-triangular entries of the skew matrix.
        self._tril_idx = torch.tril_indices(dim, dim, offset=-1)
        n = self._tril_idx.shape[1]
        init = 1e-3 * torch.randn(n, generator=gen, dtype=torch.float32)
        self.theta = nn.Parameter(init.to(device=device, dtype=torch.float32))
        self._dtype = dtype

    def _skew(self) -> torch.Tensor:
        a = torch.zeros(self.dim, self.dim, device=self.theta.device,
                        dtype=torch.float32)
        i, j = self._tril_idx
        a[i, j] = self.theta
        a = a - a.T
        return a

    def matrix(self) -> torch.Tensor:
        a = self._skew()
        eye = torch.eye(self.dim, device=a.device, dtype=a.dtype)
        r = torch.linalg.solve(eye + a, eye - a)
        return r

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        r = self.matrix().to(x.dtype)
        return x @ r.T

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        r = self.matrix().to(weight.dtype)
        return weight @ r.T

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        r = self.matrix().to(x.dtype)
        return x @ r


def build_rotation(kind: str, dim: int, *, block: int = 128,
                   seed: Optional[int] = None, device=None,
                   dtype=torch.float32) -> Rotation:
    kind = (kind or "none").lower()
    if kind in ("none", "identity"):
        return Identity(dim)
    if kind in ("fwht", "hadamard", "randomized_hadamard", "rht"):
        return RandomizedHadamard(dim, block=block, seed=seed, device=device, dtype=dtype)
    if kind in ("dense", "dense_qr", "orthogonal"):
        return DenseOrthogonal(dim, seed=seed, device=device, dtype=dtype)
    if kind in ("learned", "cayley", "stiefel"):
        return LearnedRotation(dim, seed=seed, device=device, dtype=dtype)
    raise ValueError(f"unknown rotation kind: {kind}")
