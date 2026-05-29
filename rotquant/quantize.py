"""The single ``Quantizer`` API.

Pluggable along four axes, matching the experiment matrix:

* **codebook**       -- ``gaussian`` | ``uniform`` | ``nf`` (see :mod:`codebooks`)
* **scale strategy** -- ``rms`` | ``mse_search`` | ``turboquant``
* **group size**     -- per-group scales along the input dimension
* **error comp**     -- ``none`` | ``gptq`` | ``residual`` | ``qjl`` | ``turboquant``

``scale="turboquant"`` skips per-group scale metadata entirely: after a randomised
Hadamard pre-rotation the weight distribution is universal (concentrated Gaussian),
so the Lloyd-Max codebook centroids are applied directly without rescaling.  This
saves ``scale_bits / group_size`` bits/weight overhead.

``error_comp="turboquant"`` applies the TurboQuant Stage-2 QJL correction: a
1-bit random-projection sketch ``sign(r @ G)`` of the quantisation residual is
stored at pack time and used at inference to cancel the inner-product bias that
would otherwise remain from the unscaled codebook rounding.

``qjl`` (the *old* stochastic residual) is kept as the null hypothesis for E3 --
it loses to deterministic residual and TurboQuant QJL at equal bits.

Rotation is applied to the weight *before* quantisation by the patcher; this
module quantises an (already-rotated) ``[out, in]`` weight matrix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .codebooks import ScalarCodebook, build_scalar_codebook
from .pack import PackedTensor, pack_indices, unpack_indices
from .utils import BitBudget, get_logger

logger = get_logger()


def _generate_sketch_matrix(in_features: int, k: int, seed: int, device) -> torch.Tensor:
    """Random Gaussian sketch matrix ``G`` of shape ``[in_features, k]``.

    Deterministically seeded so the same matrix is reconstructed at inference
    from ``sketch_seed`` without storing ``G``.  Columns are normalised by
    ``1/sqrt(k)`` so that ``||Gx||^2 ≈ ||x||^2`` in expectation (JL property).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + 7919)
    G = torch.randn(in_features, k, generator=gen, dtype=torch.float32) / math.sqrt(k)
    return G.to(device)


@dataclass
class QuantConfig:
    bits: int = 3
    codebook: str = "gaussian"          # gaussian | uniform | nf
    scale: str = "rms"                  # rms | mse_search | turboquant
    group_size: int = 128
    error_comp: str = "none"            # none | gptq | residual | qjl | turboquant
    residual_bits: int = 1              # bits for residual / qjl pass
    residual_codebook: str = "gaussian"
    percdamp: float = 0.01              # Hessian damping (1% of mean diagonal)
    mse_search_grid: int = 41           # candidate scales for mse_search
    mse_search_lo: float = 0.5
    mse_search_hi: float = 1.5
    seed: int = 0
    scale_bits: float = 16.0
    sketch_k: int = 64                  # QJL projection dimension (error_comp="turboquant")


@dataclass
class QuantizedWeight:
    packed: PackedTensor
    scales: Optional[torch.Tensor]       # [out, n_groups]; [out, 1] for TurboQuant per-row
    codebook: ScalarCodebook
    group_size: int
    out_features: int
    in_features: int
    residual_packed: Optional[PackedTensor] = None
    residual_scales: Optional[torch.Tensor] = None
    residual_codebook: Optional[ScalarCodebook] = None
    # TurboQuant Stage-2 QJL sketch for inner-product bias correction (error_comp="turboquant")
    sketch: Optional[PackedTensor] = None
    sketch_row_norms: Optional[torch.Tensor] = None  # [out_features] fp16
    sketch_k: int = 0
    sketch_seed: int = 0
    # Effective group size for scale metadata (None → same as group_size).
    # TurboQuant uses scale_group_size=in_features (one scale per output row), which
    # gives (scale_bits / in_features) bpw overhead instead of (scale_bits / group_size).
    scale_group_size: Optional[int] = None

    def dequantize(self) -> torch.Tensor:
        idx = unpack_indices(self.packed).reshape(self.out_features, self.in_features)
        centroids = self.codebook.centroids.to(idx.device)
        q = centroids[idx]
        if self.scales is not None:
            sgs = self.scale_group_size if self.scale_group_size is not None else self.group_size
            w = q * _expand_scales(self.scales, sgs, self.in_features)
        else:
            w = q
        if self.residual_packed is not None:
            ridx = unpack_indices(self.residual_packed).reshape(
                self.out_features, self.in_features)
            rc = self.residual_codebook.centroids.to(ridx.device)[ridx]
            rs = _expand_scales(self.residual_scales, self.group_size, self.in_features)
            w = w + rc * rs
        return w

    def bit_budget(self) -> BitBudget:
        extra_code_bits = 0.0
        extra_scale_bits = 0.0
        if self.residual_packed is not None:
            extra_code_bits = self.residual_packed.bits
            extra_scale_bits = self.scale_bits_residual
        sgs = self.scale_group_size if self.scale_group_size is not None else self.group_size
        if self.scales is None:
            main_scale = 0.0
        else:
            # Amortise scale cost over the code group.  For per-row TurboQuant scales
            # sgs = in_features, so the per-code-group cost is scale_bits * group / in_features.
            main_scale = self.scale_bits_main * self.group_size / sgs
        # Sketch overhead: sketch_k 1-bit projections + 1 fp16 row norm per output row,
        # amortised over (out * in_features) weights → (sketch_k + 16) * group / in_features
        # bits per code group.
        sketch_overhead = 0.0
        if self.sketch is not None:
            sketch_overhead = (self.sketch_k + 16) * self.group_size / self.in_features
        return BitBudget(levels=2 ** self.packed.bits, group_size=self.group_size,
                         scale_bits=(main_scale + extra_scale_bits
                                     + extra_code_bits * self.group_size
                                     + sketch_overhead))

    # bookkeeping for accounting
    scale_bits_main: float = 16.0
    scale_bits_residual: float = 16.0


def _expand_scales(scales: torch.Tensor, group_size: int, in_features: int) -> torch.Tensor:
    """[out, n_groups] -> [out, in_features] by repeating each group scale."""
    out, ng = scales.shape
    rep = scales.repeat_interleave(group_size, dim=1)
    if rep.shape[1] < in_features:  # last partial group
        rep = torch.cat([rep, rep[:, -1:].expand(out, in_features - rep.shape[1])], dim=1)
    return rep[:, :in_features]


def _group_scales_rms(w: torch.Tensor, group_size: int) -> torch.Tensor:
    out, inf = w.shape
    ng = (inf + group_size - 1) // group_size
    pad = ng * group_size - inf
    wp = torch.nn.functional.pad(w, (0, pad))
    wg = wp.reshape(out, ng, group_size)
    rms = wg.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
    return rms  # [out, ng]


def _quantize_groups(w: torch.Tensor, scales: Optional[torch.Tensor],
                     codebook: ScalarCodebook,
                     group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (dequantized weight, integer indices).

    When ``scales`` is ``None`` (TurboQuant mode) the codebook is applied directly
    to ``w`` without any normalisation -- the Hadamard rotation has already made
    the distribution universal.
    """
    if scales is None:
        q, idx = codebook.quantize(w)
        return q, idx
    out, inf = w.shape
    sc = _expand_scales(scales, group_size, inf)
    normed = w / sc
    q, idx = codebook.quantize(normed)
    return q * sc, idx


class Quantizer:
    def __init__(self, config: QuantConfig):
        self.cfg = config
        self.codebook = build_scalar_codebook(config.codebook, 2 ** config.bits)

    # ------------------------------------------------------------------ #
    # scale selection
    # ------------------------------------------------------------------ #
    def _select_scales(self, w: torch.Tensor) -> Optional[torch.Tensor]:
        if self.cfg.scale == "turboquant":
            # Per-row RMS: one scale per output neuron, amortised over in_features weights.
            # After Hadamard rotation the distribution *shape* is universal (Gaussian) but
            # the *scale* varies per layer; a per-row scale is necessary for correctness.
            # Overhead: scale_bits/in_features bpw vs scale_bits/group_size for per-group.
            return _group_scales_rms(w, w.shape[1])  # [out, 1]
        rms = _group_scales_rms(w, self.cfg.group_size)
        if self.cfg.scale == "rms":
            return rms
        if self.cfg.scale == "mse_search":
            return self._mse_search_scales(w, rms)
        raise ValueError(f"unknown scale strategy: {self.cfg.scale}")

    def _mse_search_scales(self, w: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        """Data-free per-group scale search minimising quantisation MSE (E4)."""
        out, inf = w.shape
        gs = self.cfg.group_size
        ng = rms.shape[1]
        pad = ng * gs - inf
        wg = torch.nn.functional.pad(w, (0, pad)).reshape(out, ng, gs)
        cand = torch.linspace(self.cfg.mse_search_lo, self.cfg.mse_search_hi,
                              self.cfg.mse_search_grid, device=w.device)
        best_scales = rms.clone()
        best_err = torch.full_like(rms, float("inf"))
        centroids = self.codebook.centroids.to(w.device)
        bounds = (centroids[:-1] + centroids[1:]) / 2.0
        for c in cand:
            sc = (rms * c).unsqueeze(-1).clamp_min(1e-12)   # [out, ng, 1]
            normed = wg / sc
            idx = torch.bucketize(normed, bounds)
            q = centroids[idx] * sc
            err = (wg - q).pow(2).sum(dim=-1)               # [out, ng]
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scales = torch.where(better, (rms * c), best_scales)
        return best_scales

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def quantize_weight(self, weight: torch.Tensor,
                        H: Optional[torch.Tensor] = None) -> QuantizedWeight:
        w = weight.detach().to(torch.float32)
        out, inf = w.shape
        scales = self._select_scales(w)

        if self.cfg.error_comp == "gptq":
            if self.cfg.scale == "turboquant":
                raise ValueError(
                    "GPTQ requires per-group scales with group_size < in_features; "
                    "set scale='rms' or 'mse_search' when error_comp='gptq'."
                )
            q_w, idx = self._gptq(w, scales, H)
        else:
            q_w, idx = _quantize_groups(w, scales, self.codebook, self.cfg.group_size)

        packed = pack_indices(idx.reshape(-1), self.cfg.bits)
        # TurboQuant uses one scale per output row; pass scale_group_size=in_features
        # so dequantize() and bit_budget() use the right expansion factor.
        scale_group_size = inf if self.cfg.scale == "turboquant" else None
        qw = QuantizedWeight(
            packed=packed, scales=scales, codebook=self.codebook,
            group_size=self.cfg.group_size, out_features=out, in_features=inf,
            scale_group_size=scale_group_size,
            scale_bits_main=self.cfg.scale_bits,
        )

        if self.cfg.error_comp in ("residual", "qjl"):
            self._add_residual(qw, w, q_w)
        elif self.cfg.error_comp == "turboquant":
            self._turboquant_sketch(qw, w - q_w)
        return qw

    # ------------------------------------------------------------------ #
    # GPTQ error feedback
    # ------------------------------------------------------------------ #
    def _gptq(self, w: torch.Tensor, scales: torch.Tensor,
              H: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out, inf = w.shape
        if H is None:
            logger.warning(
                "GPTQ requested without a Hessian; falling back to H=I, which is "
                "exactly plain rounding (no error feedback). Did you forget "
                "calibration?")
            H = torch.eye(inf, device=w.device, dtype=torch.float32)
        H = H.to(torch.float32).clone()
        gs = self.cfg.group_size

        # Dead columns -> identity diagonal so Cholesky stays well-posed.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0

        Hinv = self._stable_hinv(H)

        W = w.clone()
        Q = torch.zeros_like(W)
        Idx = torch.zeros_like(W, dtype=torch.int64)
        centroids = self.codebook.centroids.to(w.device)
        bounds = (centroids[:-1] + centroids[1:]) / 2.0

        for i in range(inf):
            d = Hinv[i, i]
            col = W[:, i]
            sc = scales[:, i // gs].clamp_min(1e-12)
            idx = torch.bucketize(col / sc, bounds)
            q = centroids[idx] * sc
            Q[:, i] = q
            Idx[:, i] = idx
            err = (col - q) / d
            if i + 1 < inf:
                W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
        return Q, Idx

    def _stable_hinv(self, H: torch.Tensor) -> torch.Tensor:
        """Upper-triangular Cholesky factor of H^{-1}, with auto-increasing damping."""
        inf = H.shape[0]
        mean_diag = torch.diag(H).mean().clamp_min(1e-8)
        damp = self.cfg.percdamp
        for _ in range(8):
            Hd = H.clone()
            Hd[range(inf), range(inf)] += damp * mean_diag
            try:
                L = torch.linalg.cholesky(Hd)
                Hinv = torch.cholesky_inverse(L)
                Hinv = torch.linalg.cholesky(Hinv, upper=True)
                return Hinv
            except torch.linalg.LinAlgError:
                logger.warning("Cholesky failed; increasing damping %.4f -> %.4f",
                               damp, damp * 10)
                damp *= 10
        raise RuntimeError("GPTQ Cholesky failed even after increasing damping")

    # ------------------------------------------------------------------ #
    # residual passes
    # ------------------------------------------------------------------ #
    def _add_residual(self, qw: QuantizedWeight, w: torch.Tensor,
                      q_w: torch.Tensor) -> None:
        r = w - q_w
        rscales = _group_scales_rms(r, self.cfg.group_size)
        if self.cfg.error_comp == "residual":
            rcb = build_scalar_codebook(self.cfg.residual_codebook,
                                        2 ** self.cfg.residual_bits)
            _, ridx = _quantize_groups(r, rscales, rcb, self.cfg.group_size)
        else:  # qjl: stochastic 1-bit residual (the deliberate loser)
            rcb, ridx = self._qjl_residual(r, rscales)
        qw.residual_packed = pack_indices(ridx.reshape(-1), self.cfg.residual_bits)
        qw.residual_scales = rscales
        qw.residual_codebook = rcb
        qw.scale_bits_residual = self.cfg.scale_bits

    def _qjl_residual(self, r: torch.Tensor,
                      rscales: torch.Tensor) -> Tuple[ScalarCodebook, torch.Tensor]:
        """Stochastic 1-bit residual. Two levels at +-1; rounding is *stochastic*,
        which injects variance the deterministic pass avoids -> it loses at equal bits.
        """
        cb = ScalarCodebook(torch.tensor([-1.0, 1.0]), name="qjl1bit")
        sc = _expand_scales(rscales, self.cfg.group_size, r.shape[1])
        normed = (r / sc).clamp(-1, 1)
        p_pos = (normed + 1.0) / 2.0  # P(round to +1)
        gen = torch.Generator(device=r.device).manual_seed(self.cfg.seed)
        u = torch.rand(normed.shape, generator=gen, device=r.device)
        idx = (u < p_pos).to(torch.int64)  # 1 -> +1, 0 -> -1
        return cb, idx

    def _turboquant_sketch(self, qw: QuantizedWeight, r: torch.Tensor) -> None:
        """TurboQuant Stage-2: store a QJL sketch of the quantisation residual.

        At pack time: compute ``sign(r @ G)`` where ``G`` is a random Gaussian
        ``[in, k]`` matrix.  Stores the 1-bit packed sketch and the per-row L2
        norm of ``r`` so the forward pass can apply an unbiased inner-product
        correction ``(sign(xr @ G) @ sketch^T) * (row_norms / k)``.
        """
        k = self.cfg.sketch_k
        G = _generate_sketch_matrix(r.shape[1], k, self.cfg.seed, r.device)
        proj = r @ G  # [out, k]
        sketch_bits = (proj > 0).to(torch.int64)
        qw.sketch = pack_indices(sketch_bits.reshape(-1), 1)
        qw.sketch_row_norms = r.norm(dim=1).half()
        qw.sketch_k = k
        qw.sketch_seed = self.cfg.seed
