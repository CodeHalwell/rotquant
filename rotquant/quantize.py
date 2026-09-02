"""The single ``Quantizer`` API.

Pluggable along four axes, matching the experiment matrix:

* **codebook**       -- scalar grids plus research-only finite-rate ``vector``
* **scale strategy** -- ``rms`` | ``mse_search`` | ``turboquant``
* **group size**     -- per-group scales along the input dimension
* **error comp**     -- ``none`` | ``gptq`` | ``residual`` | ``qjl`` | ``turboquant``

``scale="turboquant"`` replaces per-group scales with one fp16 RMS scale per output
row. After a randomised Hadamard pre-rotation the weight distribution shape is close
to universal (concentrated Gaussian), while the row scale retains the layer's actual
magnitude. This reduces overhead from ``scale_bits / group_size`` to
``scale_bits / in_features`` bits/weight.

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

import hashlib
import math
import random
from dataclasses import dataclass

import torch

from .codebooks import (
    ScalarCodebook,
    VectorCodebook,
    build_finite_e8_codebook,
    build_gaussian_vector_codebook,
    build_scalar_codebook,
    fit_scalar_codebook,
)
from .pack import PackedTensor, pack_indices, unpack_indices
from .utils import BitBudget, get_logger

logger = get_logger(__name__)


def _weight_digest(w: torch.Tensor) -> str:
    """Exact content digest of a weight matrix (shape + float32 bytes)."""
    data = w.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256(str(tuple(data.shape)).encode())
    digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _calibration_sample_indices(
    total_values: int,
    sample_count: int,
    *,
    device: torch.device | str,
    seed: int = 0,
) -> torch.Tensor:
    """Return a seeded uniform sample without replacement in O(sample_count).

    Floyd's algorithm avoids both the column-periodic aliasing of evenly spaced
    indices and the O(total_values) temporary allocated by ``randperm``.  The
    returned indices are sorted so the subsequent device gather is coalesced;
    sorting does not change the sampled set used by the Lloyd fit.
    """

    if total_values < 2:
        raise ValueError("total_values must be at least 2")
    if not 2 <= sample_count <= total_values:
        raise ValueError("sample_count must be in [2, total_values]")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    selected: set[int] = set()
    for upper in range(total_values - sample_count, total_values):
        candidate = rng.randrange(upper + 1)
        selected.add(upper if candidate in selected else candidate)
    indices = torch.tensor(sorted(selected), dtype=torch.int64)
    if indices.numel() != sample_count:  # Floyd's invariant; defensive guard.
        raise RuntimeError("bounded calibration sampler returned the wrong size")
    return indices.to(device=device)


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
    gptq_block: int = 128               # lazy-update block width for GPTQ
    gptq_actorder: bool = True          # process columns by descending Hessian diagonal
    gptq_recompute_scales: bool = True  # refit each stored group from updated weights
    mse_search_grid: int = 41           # candidate scales for mse_search
    mse_search_lo: float = 0.5
    mse_search_hi: float = 1.5
    seed: int = 0
    scale_bits: float = 16.0
    scale_quant_group_size: int = 256  # second-level block for 8-bit scales
    sketch_k: int = 64                  # QJL projection dimension (error_comp="turboquant")
    codebook_dim: int | None = None      # spherical marginal dimension; defaults to group_size
    bias_correction: str = "none"       # none | length | mean | length_mean
    vector_dim: int = 2                  # product-vector width for codebook="vector"
    vector_samples: int = 16384          # deterministic Gaussian training samples
    vector_iters: int = 25               # Lloyd/k-means refinement iterations
    calibrated_samples: int = 65536      # per-matrix normalized scalar samples
    calibrated_iters: int = 100          # scalar Lloyd refinement iterations

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int) or not 1 <= self.bits <= 16:
            raise ValueError("bits must be an integer in [1, 16]")
        if not isinstance(self.residual_bits, int) or not 1 <= self.residual_bits <= 16:
            raise ValueError("residual_bits must be an integer in [1, 16]")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.codebook.lower() not in {
            "gaussian", "lloyd", "lloyd_max", "mse",
            "sphere", "spherical", "beta", "finite_beta",
            "uniform", "nf", "normalfloat", "normal_float",
            "calibrated", "empirical", "weight_calibrated",
            "vector", "vector_kmeans", "product_vector", "pq",
            "e8", "e8p", "finite_e8", "e8_product",
        }:
            raise ValueError(f"unknown codebook kind: {self.codebook}")
        if self.codebook_dim is not None and (
            isinstance(self.codebook_dim, bool)
            or not isinstance(self.codebook_dim, int)
            or self.codebook_dim < 3
        ):
            raise ValueError("codebook_dim must be an integer >= 3")
        if self.scale not in {"rms", "mse_search", "turboquant"}:
            raise ValueError(f"unknown scale strategy: {self.scale}")
        if self.error_comp not in {"none", "gptq", "residual", "qjl", "turboquant"}:
            raise ValueError(f"unknown error compensation strategy: {self.error_comp}")
        if self.error_comp == "qjl" and self.residual_bits != 1:
            raise ValueError("error_comp='qjl' requires residual_bits=1")
        if self.error_comp == "gptq" and self.scale == "turboquant":
            raise ValueError("error_comp='gptq' requires scale='rms' or 'mse_search'")
        if self.error_comp == "residual":
            # Validate lazily-used residual codebook names at config construction,
            # before an expensive model has been loaded or calibrated.
            valid = {
                "gaussian", "lloyd", "lloyd_max", "mse",
                "uniform", "nf", "normalfloat", "normal_float",
            }
            if self.residual_codebook.lower() not in valid:
                raise ValueError(
                    f"unknown residual scalar codebook kind: {self.residual_codebook}")
        if self.scale_bits not in (8.0, 16.0, 32.0):
            raise ValueError(
                "scale_bits must be 8, 16, or 32")
        if (not isinstance(self.scale_quant_group_size, int)
                or isinstance(self.scale_quant_group_size, bool)
                or self.scale_quant_group_size < 2):
            raise ValueError("scale_quant_group_size must be an integer >= 2")
        if self.percdamp < 0:
            raise ValueError("percdamp must be >= 0")
        if self.gptq_block < 1:
            raise ValueError("gptq_block must be >= 1")
        if not isinstance(self.gptq_actorder, bool):
            raise TypeError("gptq_actorder must be boolean")
        if not isinstance(self.gptq_recompute_scales, bool):
            raise TypeError("gptq_recompute_scales must be boolean")
        if self.mse_search_grid < 1:
            raise ValueError("mse_search_grid must be >= 1")
        if self.mse_search_lo <= 0 or self.mse_search_hi < self.mse_search_lo:
            raise ValueError("mse_search bounds must satisfy 0 < lo <= hi")
        if self.error_comp == "turboquant" and self.sketch_k < 1:
            raise ValueError("sketch_k must be >= 1 for TurboQuant correction")
        if self.bias_correction not in {"none", "length", "mean", "length_mean"}:
            raise ValueError(
                "bias_correction must be one of "
                "{'none', 'length', 'mean', 'length_mean'}")
        e8 = self.codebook.lower() in {
            "e8", "e8p", "finite_e8", "e8_product"
        }
        vector = self.codebook.lower() in {
            "vector", "vector_kmeans", "product_vector", "pq",
            "e8", "e8p", "finite_e8", "e8_product",
        }
        if e8:
            self.vector_dim = 8
        if not isinstance(self.vector_dim, int) or isinstance(self.vector_dim, bool):
            raise TypeError("vector_dim must be an integer")
        if vector and self.vector_dim < 2:
            raise ValueError("vector codebooks require vector_dim >= 2")
        if vector and self.bits * self.vector_dim > 16:
            raise ValueError("vector index width cannot exceed 16 packed bits")
        if vector and self.group_size % self.vector_dim:
            raise ValueError("vector_dim must divide group_size")
        if vector and self.error_comp != "none":
            raise ValueError(
                "vector codebooks currently require error_comp='none'"
            )
        if self.vector_samples < 2 or self.vector_iters < 1:
            raise ValueError("vector_samples and vector_iters must be positive")
        if self.calibrated_samples < 2 or self.calibrated_iters < 1:
            raise ValueError(
                "calibrated_samples and calibrated_iters must be positive"
            )


@dataclass
class QuantizedWeight:
    packed: PackedTensor
    scales: torch.Tensor | None       # [out, n_groups]; [out, 1] for TurboQuant per-row
    codebook: ScalarCodebook | VectorCodebook
    group_size: int
    out_features: int
    in_features: int
    residual_packed: PackedTensor | None = None
    residual_scales: torch.Tensor | None = None
    residual_codebook: ScalarCodebook | None = None
    # TurboQuant Stage-2 QJL sketch for inner-product bias correction (error_comp="turboquant")
    sketch: PackedTensor | None = None
    sketch_row_norms: torch.Tensor | None = None  # [out_features] fp16
    sketch_k: int = 0
    sketch_seed: int = 0
    # Effective group size for scale metadata (None → same as group_size).
    # TurboQuant uses scale_group_size=in_features (one scale per output row), which
    # gives (scale_bits / in_features) bpw overhead instead of (scale_bits / group_size).
    scale_group_size: int | None = None
    # For scale_bits=8, ``scales`` stores uint8 codes. Each consecutive metadata
    # block has one fp16 offset and fp16 step (double quantization).
    scale_offsets: torch.Tensor | None = None
    scale_steps: torch.Tensor | None = None
    scale_quant_group_size: int = 256
    residual_scale_offsets: torch.Tensor | None = None
    residual_scale_steps: torch.Tensor | None = None

    def main_scales(self) -> torch.Tensor | None:
        return _decode_storage_scales(
            self.scales,
            self.scale_bits_main,
            self.scale_offsets,
            self.scale_steps,
            self.scale_quant_group_size,
        )

    def residual_scale_values(self) -> torch.Tensor | None:
        return _decode_storage_scales(
            self.residual_scales,
            self.scale_bits_residual,
            self.residual_scale_offsets,
            self.residual_scale_steps,
            self.scale_quant_group_size,
        )

    def dequantize(self) -> torch.Tensor:
        idx = unpack_indices(self.packed)
        if isinstance(self.codebook, VectorCodebook):
            q = self.codebook.decode(idx).reshape(
                self.out_features, self.in_features
            )
        else:
            idx = idx.reshape(self.out_features, self.in_features)
            centroids = self.codebook.centroids.to(idx.device)
            q = centroids[idx]
        scales = self.main_scales()
        if scales is not None:
            sgs = self.scale_group_size if self.scale_group_size is not None else self.group_size
            w = q * _expand_scales(scales, sgs, self.in_features)
        else:
            w = q
        if self.residual_packed is not None:
            ridx = unpack_indices(self.residual_packed).reshape(
                self.out_features, self.in_features)
            rc = self.residual_codebook.centroids.to(ridx.device)[ridx]
            rs = _expand_scales(
                self.residual_scale_values(), self.group_size, self.in_features)
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
        stored_bits = self.packed.data.numel() * self.packed.data.element_size() * 8
        if self.scales is not None:
            stored_bits += self.scales.numel() * self.scales.element_size() * 8
            if self.scale_offsets is not None:
                stored_bits += self.scale_offsets.numel() * 16
                stored_bits += self.scale_steps.numel() * 16
        if self.residual_packed is not None:
            stored_bits += (self.residual_packed.data.numel()
                            * self.residual_packed.data.element_size() * 8)
            stored_bits += (self.residual_scales.numel()
                            * self.residual_scales.element_size() * 8)
            if self.residual_scale_offsets is not None:
                stored_bits += self.residual_scale_offsets.numel() * 16
                stored_bits += self.residual_scale_steps.numel() * 16
        if self.sketch is not None:
            stored_bits += self.sketch.data.numel() * self.sketch.data.element_size() * 8
            stored_bits += (self.sketch_row_norms.numel()
                            * self.sketch_row_norms.element_size() * 8)
        return BitBudget(
            levels=2 ** self.packed.bits,
            group_size=self.group_size,
            scale_bits=(main_scale + extra_scale_bits
                        + extra_code_bits * self.group_size
                        + sketch_overhead),
            stored_bits=float(stored_bits),
            stored_weights=self.out_features * self.in_features,
        )

    # bookkeeping for accounting
    scale_bits_main: float = 16.0
    scale_bits_residual: float = 16.0


def _expand_scales(scales: torch.Tensor, group_size: int, in_features: int) -> torch.Tensor:
    """[out, n_groups] -> [out, in_features] by repeating each group scale."""
    out, _ng = scales.shape
    rep = scales.repeat_interleave(group_size, dim=1)
    if rep.shape[1] < in_features:  # last partial group
        rep = torch.cat([rep, rep[:, -1:].expand(out, in_features - rep.shape[1])], dim=1)
    return rep[:, :in_features]


def _group_counts(in_features: int, group_size: int, device) -> torch.Tensor:
    """Number of *real* (non-padded) weights in each group, shape [ng]."""
    ng = (in_features + group_size - 1) // group_size
    counts = torch.full((ng,), group_size, dtype=torch.float32, device=device)
    rem = in_features - (ng - 1) * group_size
    counts[-1] = rem
    return counts


def _storage_scales(scales: torch.Tensor | None,
                    scale_bits: float,
                    scale_quant_group_size: int = 256) -> torch.Tensor | None:
    """Round scales to the precision the bit accounting claims.

    The protocol charges ``scale_bits`` (default 16) per scale, so the stored
    tensor must actually be fp16 -- and quantisation must run against the
    *rounded* values so pack-time indices and dequant agree. Floored at the
    smallest normal fp16 so the fp32 ``1e-12`` clamp on all-zero groups does not
    underflow to zero and divide out to NaN. The only other supported format is
    fp32 (``scale_bits=32``). For 8-bit storage this returns the values after a
    real blockwise uint8 encode/decode round trip, so code assignment agrees
    exactly with deployment. The codes and second-level metadata are retained
    later by :func:`_encode_storage_scales`.
    """
    if scales is None or scale_bits == 32.0:
        return scales
    if scale_bits == 16.0:
        return scales.to(torch.float16).clamp_min(
            torch.finfo(torch.float16).smallest_normal)
    if scale_bits == 8.0:
        stored, offsets, steps = _encode_storage_scales(
            scales, scale_bits, scale_quant_group_size)
        return _decode_storage_scales(
            stored, scale_bits, offsets, steps, scale_quant_group_size)
    raise ValueError("scale_bits must be 8, 16, or 32")


def _encode_storage_scales(
    scales: torch.Tensor | None,
    scale_bits: float,
    group_size: int = 256,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Encode scale metadata at its actual retained precision.

    The 8-bit path is QLoRA-style double quantization: flattened primary scales
    are affine-quantized in bounded blocks, while each block stores one fp16
    offset and one fp16 step. The exact metadata overhead is included in packed
    byte accounting rather than hidden in a nominal bits/weight claim.
    """

    if scales is None:
        return None, None, None
    if scale_bits == 32.0:
        return scales.float(), None, None
    if scale_bits == 16.0:
        return (scales.to(torch.float16).clamp_min(
            torch.finfo(torch.float16).smallest_normal), None, None)
    if scale_bits != 8.0:
        raise ValueError("scale_bits must be 8, 16, or 32")
    if group_size < 2:
        raise ValueError("scale quantization group size must be >= 2")
    flat = scales.detach().float().reshape(-1)
    if flat.numel() == 0:
        return torch.empty_like(scales, dtype=torch.uint8), None, None
    blocks = (flat.numel() + group_size - 1) // group_size
    pad = blocks * group_size - flat.numel()
    padded = torch.cat([flat, flat[-1:].expand(pad)]) if pad else flat
    grouped = padded.reshape(blocks, group_size)
    minimum = grouped.amin(dim=1)
    maximum = grouped.amax(dim=1)
    offsets = minimum.to(torch.float16).clamp_min(
        torch.finfo(torch.float16).smallest_normal)
    steps = ((maximum - offsets.float()).clamp_min(0.0) / 255.0).to(
        torch.float16)
    # A constant block has no range; a zero step is an exact compact encoding.
    # A range too small for even a subnormal fp16 step is treated the same way.
    constant = (maximum <= offsets.float()) | (steps.float() == 0)
    steps[constant] = 0
    # Divide by exactly the fp16 step that decoding multiplies by.  Subnormal
    # fp16 steps are legitimate for narrow-range blocks (a range below 0.0156
    # gives a step below the smallest normal).  Clamping the divisor to the
    # smallest normal, as an earlier version did, encoded such blocks with a
    # larger step than decoding used and pulled every scale toward the block
    # minimum by the subnormal ratio.
    divisor = torch.where(constant, torch.ones_like(steps.float()), steps.float())
    codes = torch.round(
        (grouped - offsets.float().unsqueeze(1)) / divisor.unsqueeze(1)
    ).clamp_(0, 255).to(torch.uint8)
    codes[constant] = 0
    return codes.reshape(-1)[:flat.numel()].reshape(scales.shape), offsets, steps


def _decode_storage_scales(
    stored: torch.Tensor | None,
    scale_bits: float,
    offsets: torch.Tensor | None,
    steps: torch.Tensor | None,
    group_size: int = 256,
) -> torch.Tensor | None:
    """Decode retained scale metadata to fp32 values used by dequantization."""

    if stored is None or scale_bits != 8.0:
        return stored
    if offsets is None or steps is None:
        raise ValueError("8-bit scales require offset and step metadata")
    flat = stored.reshape(-1).float()
    block_ids = torch.arange(flat.numel(), device=flat.device) // group_size
    return (
        offsets.to(device=flat.device, dtype=torch.float32)[block_ids]
        + flat * steps.to(device=flat.device, dtype=torch.float32)[block_ids]
    ).reshape(stored.shape)


def _encoded_storage_scales(
    scales: torch.Tensor | None,
    scale_bits: float,
    scale_quant_group_size: int = 256,
) -> tuple[torch.Tensor | None, torch.Tensor | None,
           torch.Tensor | None, torch.Tensor | None]:
    """Encode scales once and return ``(decoded, stored, offsets, steps)``.

    Codes must be assigned against ``decoded`` and the artifact must retain
    exactly this ``stored``/``offsets``/``steps`` triple.  Re-encoding the
    decoded values later is not guaranteed to reproduce the same affine grid
    (a block whose extreme codes are not 0 and 255 gets a new step), which would
    silently desynchronise the stored scales from the packed codes.
    """

    if scales is None:
        return None, None, None, None
    stored, offsets, steps = _encode_storage_scales(
        scales, scale_bits, scale_quant_group_size)
    decoded = _decode_storage_scales(
        stored, scale_bits, offsets, steps, scale_quant_group_size)
    return decoded, stored, offsets, steps


def _group_scales_rms(w: torch.Tensor, group_size: int) -> torch.Tensor:
    out, inf = w.shape
    ng = (inf + group_size - 1) // group_size
    pad = ng * group_size - inf
    wp = torch.nn.functional.pad(w, (0, pad))
    wg = wp.reshape(out, ng, group_size)
    # Divide by the number of real elements per group, not group_size: averaging
    # over the zero padding would shrink the last group's scale by
    # sqrt(real/group_size) and clip its weights.
    counts = _group_counts(inf, group_size, w.device)
    rms = (wg.pow(2).sum(dim=-1) / counts).clamp_min(1e-12).sqrt()
    return rms  # [out, ng]


def _quantize_groups(w: torch.Tensor, scales: torch.Tensor | None,
                     codebook: ScalarCodebook,
                     group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dequantized weight, integer indices).

    When ``scales`` is ``None`` the codebook is applied directly to ``w``
    without any normalisation. Scale selection, including TurboQuant's per-row
    scale, happens before this helper is called.
    """
    if scales is None:
        q, idx = codebook.quantize(w)
        return q, idx
    _out, inf = w.shape
    sc = _expand_scales(scales, group_size, inf)
    normed = w / sc
    q, idx = codebook.quantize(normed)
    return q * sc, idx


def _quantize_vector_groups(
    w: torch.Tensor,
    scales: torch.Tensor,
    codebook: VectorCodebook,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return vector-codebook reconstruction and one index per vector."""

    out, in_features = w.shape
    if in_features % codebook.dim:
        raise ValueError(
            f"vector_dim={codebook.dim} must divide in_features={in_features}"
        )
    if group_size % codebook.dim:
        raise ValueError("vector_dim must divide group_size")
    expanded = _expand_scales(scales, group_size, in_features)
    normalized = (w / expanded).reshape(out, in_features // codebook.dim, codebook.dim)
    vectors, indices = codebook.quantize(normalized)
    return vectors.reshape_as(w) * expanded, indices


class Quantizer:
    def __init__(
        self,
        config: QuantConfig,
        *,
        codebook: ScalarCodebook | VectorCodebook | None = None,
    ):
        self.cfg = config
        vector = config.codebook.lower() in {
            "vector", "vector_kmeans", "product_vector", "pq",
            "e8", "e8p", "finite_e8", "e8_product",
        }
        e8 = config.codebook.lower() in {
            "e8", "e8p", "finite_e8", "e8_product"
        }
        spherical = config.codebook.lower() in {
            "sphere", "spherical", "beta", "finite_beta"}
        self._auto_spherical_dimension = (
            spherical
            and config.codebook_dim is None
            and config.scale == "turboquant"
            and codebook is None
        )
        self._spherical_dimension: int | None = None
        dimension = (
            config.codebook_dim or config.group_size
            if spherical and not self._auto_spherical_dimension
            else None
        )
        calibrated = config.codebook.lower() in {
            "calibrated", "empirical", "weight_calibrated"
        }
        # A calibrated grid is fitted to the matrix being quantized. When we fit
        # it ourselves we must refit for each new matrix; silently reusing the
        # first matrix's grid would corrupt every later layer. An explicitly
        # supplied codebook is the caller's contract and is never refitted.
        self._auto_calibrated = calibrated and codebook is None
        self._calibrated_key: str | None = None
        self.codebook = codebook or (
            build_finite_e8_codebook(config.bits)
            if e8
            else build_gaussian_vector_codebook(
                config.bits,
                config.vector_dim,
                seed=config.seed,
                samples=config.vector_samples,
                iters=config.vector_iters,
            )
            if vector
            else None
            if calibrated or self._auto_spherical_dimension
            else build_scalar_codebook(config.codebook, 2 ** config.bits, dimension)
        )
        expected_levels = 2 ** (
            config.bits * config.vector_dim if vector else config.bits
        )
        if self.codebook is not None and self.codebook.levels != expected_levels:
            raise ValueError(
                "codebook override has the wrong number of centroids")
        if vector and not isinstance(self.codebook, VectorCodebook):
            raise TypeError("vector quantization requires a VectorCodebook override")
        if vector and self.codebook.dim != config.vector_dim:
            raise ValueError(
                "vector codebook dimension does not match config.vector_dim"
            )
        if not vector and self.codebook is not None and not isinstance(
            self.codebook, ScalarCodebook
        ):
            raise TypeError("scalar quantization requires a ScalarCodebook override")

    def _fit_calibrated_codebook(self, weight: torch.Tensor) -> None:
        """Fit one deployable scalar grid in the normalized rotated domain."""

        rms = _group_scales_rms(weight, self.cfg.group_size)
        normalized = (
            weight / _expand_scales(rms, self.cfg.group_size, weight.shape[1])
        ).reshape(-1)
        if normalized.numel() > self.cfg.calibrated_samples:
            indices = _calibration_sample_indices(
                normalized.numel(),
                self.cfg.calibrated_samples,
                device=normalized.device,
                seed=self.cfg.seed,
            )
            normalized = normalized[indices]
        self.codebook = fit_scalar_codebook(
            normalized,
            2 ** self.cfg.bits,
            name=f"calibrated_w{self.cfg.bits}",
            iters=self.cfg.calibrated_iters,
        )

    def _ensure_dimension_dependent_codebook(self, in_features: int) -> None:
        """Build a spherical grid for the actual normalization dimension."""

        if not self._auto_spherical_dimension:
            return
        if self.codebook is not None and self._spherical_dimension == in_features:
            return
        self.codebook = build_scalar_codebook(
            self.cfg.codebook, 2 ** self.cfg.bits, in_features
        )
        self._spherical_dimension = in_features

    # ------------------------------------------------------------------ #
    # scale selection
    # ------------------------------------------------------------------ #
    def select_scales(self, w: torch.Tensor) -> torch.Tensor | None:
        """Deployable per-group (or per-row) scales this config selects for ``w``.

        Public because calibration/training modules must reproduce exactly the
        scales the packer would choose. Returns None for scale-free profiles.
        """
        self._ensure_dimension_dependent_codebook(w.shape[1])
        if self.codebook is None:
            raise RuntimeError(
                "scale selection requires a codebook; calibrated quantizers fit "
                "one inside quantize_weight before scales are selected"
            )
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
        if isinstance(self.codebook, VectorCodebook):
            return self._vector_mse_search_scales(w, rms)
        out, inf = w.shape
        gs = self.cfg.group_size
        ng = rms.shape[1]
        pad = ng * gs - inf
        wg = torch.nn.functional.pad(w, (0, pad)).reshape(out, ng, gs)
        # Padded slots must not vote: their (0 - q)^2 error varies with the
        # candidate scale and would bias the search for the last partial group.
        valid = torch.ones(ng, gs, dtype=torch.bool, device=w.device)
        if pad:
            valid[-1, gs - pad:] = False
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
            err = ((wg - q).pow(2) * valid).sum(dim=-1)     # [out, ng]
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scales = torch.where(better, (rms * c), best_scales)
        return best_scales

    def _vector_mse_search_scales(
        self, w: torch.Tensor, rms: torch.Tensor
    ) -> torch.Tensor:
        """MSE scale search using complete vector assignments."""

        if not isinstance(self.codebook, VectorCodebook):
            raise TypeError("vector scale search requires a VectorCodebook")
        out, in_features = w.shape
        group_size = self.cfg.group_size
        groups = rms.shape[1]
        if in_features % group_size or group_size % self.codebook.dim:
            raise ValueError(
                "vector MSE search currently requires complete aligned groups"
            )
        grouped = w.reshape(out, groups, group_size)
        candidates = torch.linspace(
            self.cfg.mse_search_lo,
            self.cfg.mse_search_hi,
            self.cfg.mse_search_grid,
            device=w.device,
        )
        best_scales = rms.clone()
        best_error = torch.full_like(rms, float("inf"))
        for multiplier in candidates:
            scales = (rms * multiplier).unsqueeze(-1).clamp_min(1e-12)
            normalized = (grouped / scales).reshape(
                out, groups, group_size // self.codebook.dim, self.codebook.dim
            )
            reconstructed, _ = self.codebook.quantize(normalized)
            reconstructed = reconstructed.reshape_as(grouped) * scales
            error = (grouped - reconstructed).square().sum(dim=-1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scales = torch.where(better, rms * multiplier, best_scales)
        return best_scales

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def quantize_weight(self, weight: torch.Tensor,
                        H: torch.Tensor | None = None,
                        scales_override: torch.Tensor | None = None,
                        ) -> QuantizedWeight:
        """Quantize ``weight``, optionally using explicit deployable scales.

        ``scales_override`` is primarily used by calibration-time learned
        clipping. It replaces scale selection, but still goes through the same
        storage-precision rounding and exact index assignment as an ordinary
        packed weight.
        """
        w = weight.detach().to(torch.float32)
        out, inf = w.shape
        self._ensure_dimension_dependent_codebook(inf)
        if self._auto_calibrated:
            # Exact content digest: summary statistics (sums, norms) collide
            # for distinct matrices (e.g. any permutation), which would silently
            # reuse a grid fitted to a different layer. Calibration already
            # costs a Lloyd fit, so hashing the bytes once is negligible.
            key = _weight_digest(w)
            if self.codebook is None or key != self._calibrated_key:
                self._fit_calibrated_codebook(w)
                self._calibrated_key = key
        if self.codebook is None:
            raise RuntimeError("quantizer has no codebook; construction should have set one")
        if scales_override is None:
            selected_scales = self.select_scales(w)
        else:
            expected_groups = (1 if self.cfg.scale == "turboquant"
                               else (inf + self.cfg.group_size - 1)
                               // self.cfg.group_size)
            if tuple(scales_override.shape) != (out, expected_groups):
                raise ValueError(
                    "scales_override must have shape "
                    f"{(out, expected_groups)}, got {tuple(scales_override.shape)}")
            selected_scales = scales_override.detach().to(
                device=w.device, dtype=torch.float32)
            if (not torch.isfinite(selected_scales).all()
                    or (selected_scales <= 0).any()):
                raise ValueError("scales_override must be finite and positive")
        # The encoded triple is retained verbatim: every code below is assigned
        # against ``scales`` (the decoded values) so that pack-time assignment
        # and deployed dequantization agree exactly on every path.
        scales, stored_scales, scale_offsets, scale_steps = _encoded_storage_scales(
            selected_scales,
            self.cfg.scale_bits,
            self.cfg.scale_quant_group_size,
        )

        vector = isinstance(self.codebook, VectorCodebook)
        if vector:
            q_w, idx = _quantize_vector_groups(
                w, scales, self.codebook, self.cfg.group_size
            )
        elif self.cfg.error_comp == "gptq":
            if self.cfg.scale == "turboquant":
                raise ValueError(
                    "GPTQ requires per-group scales with group_size < in_features; "
                    "set scale='rms' or 'mse_search' when error_comp='gptq'."
                )
            scale_grid = (
                (stored_scales, scale_offsets, scale_steps)
                if self.cfg.scale_bits == 8.0 else None
            )
            q_w, idx, scales, encoded = self._gptq_with_scale_grid(
                w, scales, H, scale_grid)
            stored_scales, scale_offsets, scale_steps = encoded
        else:
            q_w, idx = _quantize_groups(w, scales, self.codebook, self.cfg.group_size)

        if self.cfg.bias_correction in {"length", "length_mean"}:
            # Scalar quantisation shortens reconstructed directions.  The
            # self-dot multiplier ||w||^2/<w,q(w)> is TurboVec's length
            # renormalisation expressed in our already-scaled domain.
            # Fold it into every scale in the row: codes and storage size remain
            # unchanged, and existing runtimes need no new metadata or branch.
            energy = w.square().sum(dim=1)
            alignment = (w * q_w).sum(dim=1)
            usable = (energy > 0) & (alignment > torch.finfo(w.dtype).eps * energy)
            correction = torch.ones_like(energy)
            correction[usable] = energy[usable] / alignment[usable]
            (corrected_scales, stored_scales,
             scale_offsets, scale_steps) = _encoded_storage_scales(
                scales.float() * correction.unsqueeze(1),
                self.cfg.scale_bits,
                self.cfg.scale_quant_group_size,
            )
            expanded = _expand_scales(
                corrected_scales,
                inf if self.cfg.scale == "turboquant" else self.cfg.group_size,
                inf,
            )
            if vector:
                q_w = self.codebook.decode(idx).reshape_as(w) * expanded
            else:
                q_w = self.codebook.centroids.to(w.device)[idx] * expanded
            scales = corrected_scales

        packed = pack_indices(
            idx if vector else idx.reshape(-1),
            self.cfg.bits * self.codebook.dim if vector else self.cfg.bits,
        )
        # TurboQuant uses one scale per output row; pass scale_group_size=in_features
        # so dequantize() and bit_budget() use the right expansion factor.
        scale_group_size = inf if self.cfg.scale == "turboquant" else None
        qw = QuantizedWeight(
            packed=packed, scales=stored_scales, codebook=self.codebook,
            group_size=self.cfg.group_size, out_features=out, in_features=inf,
            scale_group_size=scale_group_size,
            scale_offsets=scale_offsets,
            scale_steps=scale_steps,
            scale_quant_group_size=self.cfg.scale_quant_group_size,
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
              H: torch.Tensor | None,
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blocked GPTQ returning ``(dequantized, indices, decoded scales)``."""
        q_w, idx, decoded, _encoded = self._gptq_with_scale_grid(w, scales, H, None)
        return q_w, idx, decoded

    def _gptq_with_scale_grid(
        self, w: torch.Tensor, scales: torch.Tensor,
        H: torch.Tensor | None,
        scale_grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        """Blocked GPTQ (lazy batch updates, as in the reference implementation).

        Columns inside a ``gptq_block``-wide block are processed sequentially with
        immediate rank-1 updates *within the block only*; the propagation to all
        later columns is deferred and applied once per block as a single matmul
        ``W[:, i2:] -= Err @ Hinv[i1:i2, i2:]``. Mathematically identical to the
        column-at-a-time loop (the update is linear in the errors), but turns
        O(in_features) full-width rank-1 kernels into O(in_features/block)
        matmuls -- the difference between minutes and hours on a 7B model.

        With 8-bit scales, ``scale_grid`` carries the retained encoding of the
        initial scale selection: the uint8 codes and the blockwise affine grid
        (offsets, steps).  The codes are used as they are rather than being
        re-derived from decoded values, and lazily refit group scales are
        snapped onto the same frozen grid, so the value each code is assigned
        against is exactly the value the artifact stores; encoding refit scales
        with their own per-group blocks and re-encoding the whole matrix
        afterwards would place them on two different grids.

        Returns ``(dequantized, indices, decoded scales, (stored, offsets,
        steps))``.
        """
        out, inf = w.shape
        if H is None:
            logger.warning(
                "GPTQ requested without a Hessian; falling back to H=I, which is "
                "exactly plain rounding (no error feedback). Did you forget "
                "calibration?")
            H = torch.eye(inf, device=w.device, dtype=torch.float32)
        H = H.to(torch.float32).clone()
        gs = self.cfg.group_size
        bs = max(1, int(self.cfg.gptq_block))

        # Dead columns -> identity diagonal so Cholesky stays well-posed.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0

        permutation = (
            torch.argsort(torch.diag(H), descending=True, stable=True)
            if self.cfg.gptq_actorder
            else torch.arange(inf, device=H.device)
        )
        H = H[permutation][:, permutation]
        Hinv = self._stable_hinv(H)

        W = w[:, permutation].clone()
        Q = torch.zeros_like(W)
        Idx = torch.zeros_like(W, dtype=torch.int64)
        working_scales = scales.float().clone()
        original_groups = torch.div(permutation, gs, rounding_mode="floor")
        scale_ready = torch.zeros(
            working_scales.shape[1], dtype=torch.bool, device=W.device
        )
        centroids = self.codebook.centroids.to(w.device)
        bounds = (centroids[:-1] + centroids[1:]) / 2.0

        grid_offsets = grid_steps = grid_codes = safe_steps = None
        if scale_grid is not None:
            if self.cfg.scale_bits != 8.0:
                raise ValueError("a scale grid is only meaningful for 8-bit scales")
            codes, offsets, steps = scale_grid
            n_groups = working_scales.shape[1]
            if tuple(codes.shape) != (out, n_groups):
                raise ValueError("scale grid codes must match the scale matrix shape")
            block_ids = (
                torch.arange(out * n_groups, device=W.device)
                // self.cfg.scale_quant_group_size
            ).reshape(out, n_groups)
            grid_offsets = offsets.to(device=W.device, dtype=torch.float32)[block_ids]
            grid_steps = steps.to(device=W.device, dtype=torch.float32)[block_ids]
            # Divide by the exact (possibly subnormal) fp16 step that decoding
            # multiplies by; a zero step marks a constant block.
            safe_steps = torch.where(
                grid_steps == 0, torch.ones_like(grid_steps), grid_steps)
            grid_codes = codes.to(device=W.device, dtype=torch.float32)
            working_scales = grid_offsets + grid_codes * grid_steps

        for i1 in range(0, inf, bs):
            i2 = min(i1 + bs, inf)
            Err = torch.zeros(out, i2 - i1, device=W.device, dtype=W.dtype)
            for i in range(i1, i2):
                d = Hinv[i, i]
                col = W[:, i]
                group = int(original_groups[i].item())
                if self.cfg.gptq_recompute_scales and not scale_ready[group]:
                    group_positions = torch.nonzero(
                        original_groups == group, as_tuple=False
                    ).flatten()
                    group_weight = W[:, group_positions]
                    group_rms = _group_scales_rms(
                        group_weight, group_weight.shape[1]
                    )
                    selected = (
                        group_rms
                        if self.cfg.scale == "rms"
                        else self._mse_search_scales(group_weight, group_rms)
                    )
                    if grid_codes is not None:
                        code = torch.round(
                            (selected[:, 0] - grid_offsets[:, group])
                            / safe_steps[:, group]
                        ).clamp_(0, 255)
                        code[grid_steps[:, group] == 0] = 0
                        grid_codes[:, group] = code
                        working_scales[:, group] = (
                            grid_offsets[:, group] + code * grid_steps[:, group]
                        )
                    else:
                        working_scales[:, group] = _storage_scales(
                            selected,
                            self.cfg.scale_bits,
                            self.cfg.scale_quant_group_size,
                        )[:, 0]
                    scale_ready[group] = True
                sc = working_scales[:, group].clamp_min(1e-12)
                idx = torch.bucketize(col / sc, bounds)
                q = centroids[idx] * sc
                Q[:, i] = q
                Idx[:, i] = idx
                err = (col - q) / d
                Err[:, i - i1] = err
                if i + 1 < i2:
                    W[:, i + 1:i2] -= err.unsqueeze(1) * Hinv[i, i + 1:i2].unsqueeze(0)
            if i2 < inf:
                W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(inf, device=permutation.device)
        if grid_codes is not None:
            _initial_codes, offsets, steps = scale_grid
            encoded = (grid_codes.to(torch.uint8), offsets, steps)
            decoded = working_scales
        else:
            decoded, stored, offsets, steps = _encoded_storage_scales(
                working_scales,
                self.cfg.scale_bits,
                self.cfg.scale_quant_group_size,
            )
            encoded = (stored, offsets, steps)
        return Q[:, inverse], Idx[:, inverse], decoded, encoded

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
                               damp, damp * 10 if damp > 0 else 1e-6)
                damp = damp * 10 if damp > 0 else 1e-6
        raise RuntimeError("GPTQ Cholesky failed even after increasing damping")

    # ------------------------------------------------------------------ #
    # residual passes
    # ------------------------------------------------------------------ #
    def _add_residual(self, qw: QuantizedWeight, w: torch.Tensor,
                      q_w: torch.Tensor) -> None:
        r = w - q_w
        (rscales, stored_rscales,
         residual_offsets, residual_steps) = _encoded_storage_scales(
            _group_scales_rms(r, self.cfg.group_size),
            self.cfg.scale_bits,
            self.cfg.scale_quant_group_size,
        )
        if self.cfg.error_comp == "residual":
            rcb = build_scalar_codebook(self.cfg.residual_codebook,
                                        2 ** self.cfg.residual_bits)
            _, ridx = _quantize_groups(r, rscales, rcb, self.cfg.group_size)
        else:  # qjl: stochastic 1-bit residual (the deliberate loser)
            rcb, ridx = self._qjl_residual(r, rscales)
        qw.residual_packed = pack_indices(ridx.reshape(-1), self.cfg.residual_bits)
        qw.residual_scales = stored_rscales
        qw.residual_scale_offsets = residual_offsets
        qw.residual_scale_steps = residual_steps
        qw.residual_codebook = rcb
        qw.scale_bits_residual = self.cfg.scale_bits

    def _qjl_residual(self, r: torch.Tensor,
                      rscales: torch.Tensor) -> tuple[ScalarCodebook, torch.Tensor]:
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
        norm of ``r``.  Following QJL, only the *stored* side is sign-quantised;
        the activation side keeps its full projection ``xr @ G`` at inference,
        giving the exactly unbiased inner-product estimator

            sqrt(pi/2)/sqrt(k) * ||r_i|| * (xr @ G) . sign(r_i @ G)

        (E[sign(g.a)(g.b)] = sqrt(2/pi) (a.b)/||a|| for Gaussian ``g``, at any
        angle -- unlike the sign-sign variant, whose pi/2-scaled estimate is only
        first-order correct near orthogonality).
        """
        k = self.cfg.sketch_k
        G = _generate_sketch_matrix(r.shape[1], k, self.cfg.seed, r.device)
        proj = r @ G  # [out, k]
        sketch_bits = (proj > 0).to(torch.int64)
        qw.sketch = pack_indices(sketch_bits.reshape(-1), 1)
        qw.sketch_row_norms = r.norm(dim=1).half()
        qw.sketch_k = k
        qw.sketch_seed = self.cfg.seed
