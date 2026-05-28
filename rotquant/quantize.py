"""The single ``Quantizer`` API.

Pluggable along four axes, matching the experiment matrix:

* **codebook**       -- ``gaussian`` | ``uniform`` | ``nf`` (see :mod:`codebooks`)
* **scale strategy** -- ``rms`` | ``mse_search`` (data-free scale search, E4)
* **group size**     -- per-group scales along the input dimension
* **error comp**     -- ``none`` | ``gptq`` | ``residual`` | ``qjl``

``qjl`` (a stochastic 1-bit Johnson-Lindenstrauss-style residual) is implemented
**only so it can be shown to lose** against the deterministic ``residual`` pass at
equal bits -- it is the null hypothesis for finding 2.

Rotation is applied to the weight *before* quantisation by the patcher; this
module quantises an (already-rotated) ``[out, in]`` weight matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .codebooks import ScalarCodebook, build_scalar_codebook
from .pack import PackedTensor, pack_indices, unpack_indices
from .utils import BitBudget, get_logger

logger = get_logger()


@dataclass
class QuantConfig:
    bits: int = 3
    codebook: str = "gaussian"          # gaussian | uniform | nf
    scale: str = "rms"                  # rms | mse_search
    group_size: int = 128
    error_comp: str = "none"            # none | gptq | residual | qjl
    residual_bits: int = 1              # bits for residual / qjl pass
    residual_codebook: str = "gaussian"
    percdamp: float = 0.01              # Hessian damping (1% of mean diagonal)
    mse_search_grid: int = 41           # candidate scales for mse_search
    mse_search_lo: float = 0.5
    mse_search_hi: float = 1.5
    seed: int = 0
    scale_bits: float = 16.0


@dataclass
class QuantizedWeight:
    packed: PackedTensor
    scales: torch.Tensor                # [out, n_groups]
    codebook: ScalarCodebook
    group_size: int
    out_features: int
    in_features: int
    residual_packed: Optional[PackedTensor] = None
    residual_scales: Optional[torch.Tensor] = None
    residual_codebook: Optional[ScalarCodebook] = None

    def dequantize(self) -> torch.Tensor:
        idx = unpack_indices(self.packed).reshape(self.out_features, self.in_features)
        centroids = self.codebook.centroids.to(idx.device)
        q = centroids[idx]
        scales = _expand_scales(self.scales, self.group_size, self.in_features)
        w = q * scales
        if self.residual_packed is not None:
            ridx = unpack_indices(self.residual_packed).reshape(
                self.out_features, self.in_features)
            rc = self.residual_codebook.centroids.to(ridx.device)[ridx]
            rs = _expand_scales(self.residual_scales, self.group_size, self.in_features)
            w = w + rc * rs
        return w

    def bit_budget(self) -> BitBudget:
        extra = 0.0
        if self.residual_packed is not None:
            extra = self.residual_packed.bits + self.scale_bits_residual
        return BitBudget(levels=2 ** self.packed.bits, group_size=self.group_size,
                         scale_bits=self.scale_bits_main + extra * self.group_size)

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


def _quantize_groups(w: torch.Tensor, scales: torch.Tensor, codebook: ScalarCodebook,
                     group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (dequantized weight, integer indices) using fixed per-group scales."""
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
    def _select_scales(self, w: torch.Tensor) -> torch.Tensor:
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
            q_w, idx = self._gptq(w, scales, H)
        else:
            q_w, idx = _quantize_groups(w, scales, self.codebook, self.cfg.group_size)

        packed = pack_indices(idx.reshape(-1), self.cfg.bits)
        qw = QuantizedWeight(
            packed=packed, scales=scales, codebook=self.codebook,
            group_size=self.cfg.group_size, out_features=out, in_features=inf,
            scale_bits_main=self.cfg.scale_bits,
        )

        if self.cfg.error_comp in ("residual", "qjl"):
            self._add_residual(qw, w, q_w)
        return qw

    # ------------------------------------------------------------------ #
    # GPTQ error feedback
    # ------------------------------------------------------------------ #
    def _gptq(self, w: torch.Tensor, scales: torch.Tensor,
              H: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out, inf = w.shape
        if H is None:
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
            except Exception:
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
