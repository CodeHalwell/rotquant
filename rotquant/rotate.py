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
* :class:`ButterflyRotation`  -- a trainable, exactly-orthogonal butterfly
  initialised to the same randomised Hadamard transform.
* :class:`DenseOrthogonal`   -- a dense random orthogonal matrix from the QR of a
  Gaussian (the E1 "dense" comparison).
* :class:`LearnedRotation`   -- an orthogonal matrix parametrised on the Stiefel
  manifold via the Cayley transform, trainable for the W4A4 / E1 ablation.
* :class:`Identity`          -- the "none" baseline.
"""
from __future__ import annotations

import math
import os

import torch
from torch import nn

try:  # the fast CUDA kernel QuaRot/QuIP# use; optional, we fall back to pure torch
    from fast_hadamard_transform import hadamard_transform as _fht_cuda

    _fht_import_error: Exception | None = None
except Exception as exc:  # pragma: no cover - kernel only present with CUDA build
    _fht_cuda = None
    _fht_import_error = exc

_slow_fwht_warned = False


def _fast_hadamard_disabled() -> bool:
    """Return whether callers explicitly disabled the optional CUDA extension."""
    return os.environ.get("ROTQUANT_DISABLE_FAST_HADAMARD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _warn_slow_cuda_fwht() -> None:
    """Warn once when a CUDA tensor takes the pure-torch FWHT path.

    A missing (or broken) ``fast-hadamard-transform`` install is otherwise
    invisible: results stay correct but every rotation runs orders of
    magnitude slower on GPU. Correct-but-degraded must never be silent.
    """
    global _slow_fwht_warned
    if _slow_fwht_warned:
        return
    _slow_fwht_warned = True
    from .utils import get_logger

    if _fast_hadamard_disabled():
        detail = " (disabled by ROTQUANT_DISABLE_FAST_HADAMARD)"
    else:
        detail = f" (import failed: {_fht_import_error!r})" if _fht_import_error else ""
    get_logger(__name__).warning(
        "fast-hadamard-transform CUDA kernel unavailable%s; using the pure-torch "
        "FWHT, which is prohibitively slow at model dimensions on GPU. Install "
        "it with: uv pip install fast-hadamard-transform --no-build-isolation",
        detail,
    )


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

    if x.is_cuda:
        if _fht_cuda is not None and not _fast_hadamard_disabled():
            # The kernel applies the unnormalised H; scale to match our convention.
            out = _fht_cuda(x.contiguous())
            return out / math.sqrt(d) if normalize else out
        _warn_slow_cuda_fwht()

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

    def __init__(self, dim: int, block: int = 128, seed: int | None = None,
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

    def _signs(self, ref: torch.Tensor) -> torch.Tensor:
        # Follow the input's device as well as dtype: during patching the weight
        # lives on the GPU before the rotation module has been moved anywhere.
        return self.signs.to(device=ref.device, dtype=ref.dtype)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        # x @ R^T = FWHT_norm(x * s)
        return self._blocked_fwht(x * self._signs(x))

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self._blocked_fwht(weight * self._signs(weight))

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        # x @ R = FWHT_norm(x) * s
        return self._blocked_fwht(x) * self._signs(x)


class ButterflyRotation(Rotation):
    """Trainable structured rotation initialised exactly as randomised FWHT.

    Each Hadamard stage is replaced by independent two-coordinate orthogonal
    transforms

        [[cos(theta),  sin(theta)],
         [sin(theta), -cos(theta)]].

    At ``theta = pi/4`` these are the normalised Hadamard butterflies, so a new
    instance is numerically equivalent to :class:`RandomizedHadamard` with the
    same block and seed. Training changes ``d/2 * log2(block)`` angles while
    preserving exact orthogonality and O(d log(block)) application cost. The
    fixed random signs retain the useful FWHT starting point.
    """

    def __init__(self, dim: int, block: int = 128, seed: int | None = None,
                 device=None, dtype=torch.float32):
        super().__init__(dim)
        if dim % block != 0:
            block = RandomizedHadamard._largest_pow2_divisor(dim)
        if not _is_pow2(block):
            raise ValueError(f"Butterfly block must be a power of two, got {block}")
        self.block = block
        self.n_blocks = dim // block
        self.n_stages = int(math.log2(block))

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), generator=gen,
                              dtype=torch.float32) * 2 - 1
        self.register_buffer("signs", signs.to(device=device, dtype=dtype))
        angles = torch.full((self.n_blocks, self.n_stages, block // 2),
                            math.pi / 4, dtype=torch.float32, device=device)
        self.theta = nn.Parameter(angles)
        self.register_buffer("_cached_cos", None, persistent=False)
        self.register_buffer("_cached_sin", None, persistent=False)
        self._cached_theta_version: int | None = None

    def _invalidate_cache(self) -> None:
        self._cached_cos = None
        self._cached_sin = None
        self._cached_theta_version = None

    def train(self, mode: bool = True):
        if self.training != mode:
            self._invalidate_cache()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self._invalidate_cache()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _trig(self):
        if (not self.training
                and self._cached_cos is not None
                and self._cached_cos.device == self.theta.device
                and self._cached_theta_version == self.theta._version):
            return self._cached_cos, self._cached_sin
        cos, sin = self.theta.cos(), self.theta.sin()
        if not self.training:
            self._cached_cos = cos.detach()
            self._cached_sin = sin.detach()
            self._cached_theta_version = self.theta._version
            return self._cached_cos, self._cached_sin
        return cos, sin

    def _apply_stages(self, x: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        original_shape = x.shape
        h = x.reshape(-1, self.n_blocks, self.block)
        cos, sin = self._trig()
        stages = range(self.n_stages - 1, -1, -1) if inverse \
            else range(self.n_stages)
        for stage in stages:
            step = 1 << stage
            groups = self.block // (2 * step)
            paired = h.reshape(-1, self.n_blocks, groups, 2, step)
            a, b = paired[:, :, :, 0, :], paired[:, :, :, 1, :]
            c = cos[:, stage].reshape(1, self.n_blocks, groups, step).to(
                device=x.device, dtype=x.dtype)
            s = sin[:, stage].reshape(1, self.n_blocks, groups, step).to(
                device=x.device, dtype=x.dtype)
            h = torch.stack((c * a + s * b, s * a - c * b), dim=3)
            h = h.reshape(-1, self.n_blocks, self.block)
        return h.reshape(original_shape)

    def _signs(self, ref: torch.Tensor) -> torch.Tensor:
        return self.signs.to(device=ref.device, dtype=ref.dtype)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        # R^T = D_s B_0 ... B_k; at initialisation this is D_s H.
        return self._apply_stages(x * self._signs(x), inverse=False)

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.rotate_activation(weight)

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        # R = B_k ... B_0 D_s. Each butterfly stage is its own inverse.
        return self._apply_stages(x, inverse=True) * self._signs(x)


class DenseOrthogonal(Rotation):
    """Dense random orthogonal rotation from the QR of a Gaussian matrix."""

    def __init__(self, dim: int, seed: int | None = None, device=None,
                 dtype=torch.float32):
        super().__init__(dim)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        # Sample on CPU for seed reproducibility, but run the O(d^3) QR on the
        # target device -- CPU QR at 11008 dims costs minutes per layer.
        a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        if device is not None:
            a = a.to(device)
        q, r = torch.linalg.qr(a)
        # Make the decomposition unique / sign-stable.
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        self.register_buffer("R", q.to(device=device, dtype=dtype))

    def _r(self, ref: torch.Tensor) -> torch.Tensor:
        return self.R.to(device=ref.device, dtype=ref.dtype)

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._r(x).T

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight @ self._r(weight).T

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._r(x)


class LearnedRotation(Rotation):
    """Orthogonal rotation parametrised on the Stiefel manifold via the Cayley map.

    ``R = (I - A)(I + A)^{-1}`` with ``A`` skew-symmetric is always orthogonal, so
    gradient descent on the free (lower-triangular) parameters of ``A`` stays on
    the manifold. Used for the E1 learned-rotation ablation (it should only pull
    ahead once activations are also quantised, e.g. W4A4).
    """

    def __init__(self, dim: int, seed: int | None = None, device=None,
                 dtype=torch.float32):
        super().__init__(dim)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        # Free parameters = strictly-lower-triangular entries of the skew matrix.
        # Registered (non-persistent) so device moves carry the indices along
        # with theta -- indexing a CUDA tensor with CPU indices is an error.
        self.register_buffer("_tril_idx",
                             torch.tril_indices(dim, dim, offset=-1, device=device),
                             persistent=False)
        n = self._tril_idx.shape[1]
        init = 1e-3 * torch.randn(n, generator=gen, dtype=torch.float32)
        self.theta = nn.Parameter(init.to(device=device, dtype=torch.float32))
        self._dtype = dtype
        # Non-persistent buffer: follows the module across .to() / device moves
        # (PyTorch's _apply propagates registered buffers) but is not saved to
        # state_dict so it never carries stale values across checkpoints.
        self.register_buffer("_cached_R", None, persistent=False)
        self._cached_theta_version: int | None = None

    def train(self, mode: bool = True):
        if self.training != mode:
            # Only invalidate when the mode actually changes; a repeated .eval()
            # call otherwise evicts the cache and forces an O(d^3) recompute.
            self._cached_R = None
            self._cached_theta_version = None
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        # A cached matrix converted independently from theta can have the wrong
        # precision after .half()/.float(), even when it remains on the same device.
        self._cached_R = None
        self._cached_theta_version = None
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self._cached_R = None
        self._cached_theta_version = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _skew(self) -> torch.Tensor:
        a = torch.zeros(self.dim, self.dim, device=self.theta.device,
                        dtype=torch.float32)
        i, j = self._tril_idx
        a[i, j] = self.theta
        a = a - a.T
        return a

    def matrix(self) -> torch.Tensor:
        # In eval mode theta is frozen, so cache the (expensive O(d^3)) solve and
        # reuse it across forward passes instead of recomputing per token.
        # Also guard against a stale cache after load_state_dict() by checking
        # that the cached tensor is on the same device as theta.
        if (not self.training
                and self._cached_R is not None
                and self._cached_R.device == self.theta.device
                and self._cached_theta_version == self.theta._version):
            return self._cached_R
        a = self._skew()
        eye = torch.eye(self.dim, device=a.device, dtype=a.dtype)
        r = torch.linalg.solve(eye + a, eye - a)
        if not self.training:
            self._cached_R = r.detach()
            self._cached_theta_version = self.theta._version
            return self._cached_R
        return r

    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        r = self.matrix().to(device=x.device, dtype=x.dtype)
        return x @ r.T

    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        r = self.matrix().to(device=weight.device, dtype=weight.dtype)
        return weight @ r.T

    def inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        r = self.matrix().to(device=x.device, dtype=x.dtype)
        return x @ r


def build_rotation(kind: str, dim: int, *, block: int = 128,
                   seed: int | None = None, device=None,
                   dtype=torch.float32) -> Rotation:
    kind = (kind or "none").lower()
    if kind in ("none", "identity"):
        return Identity(dim)
    if kind in ("fwht", "hadamard", "randomized_hadamard", "rht"):
        return RandomizedHadamard(dim, block=block, seed=seed, device=device, dtype=dtype)
    if kind in ("butterfly", "learned_butterfly", "structured"):
        return ButterflyRotation(dim, block=block, seed=seed, device=device, dtype=dtype)
    if kind in ("dense", "dense_qr", "orthogonal"):
        return DenseOrthogonal(dim, seed=seed, device=device, dtype=dtype)
    if kind in ("learned", "cayley", "stiefel"):
        return LearnedRotation(dim, seed=seed, device=device, dtype=dtype)
    raise ValueError(f"unknown rotation kind: {kind}")
