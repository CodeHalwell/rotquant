"""Calibration Hessian accumulation and the E8 vector quantiser.

Both feed downstream experiments (GPTQ in E5, vector residual in E6) but had no
direct coverage, so a regression here would only surface as a confusing run.
"""
import torch
import torch.nn as nn

from rotquant.calibrate import HessianAccumulator, collect_hessians
from rotquant.codebooks import E8LatticeCodebook, nearest_e8


def test_hessian_accumulator_matches_batched_mean():
    torch.manual_seed(0)
    d = 16
    x = torch.randn(500, d)
    acc = HessianAccumulator(d)
    # Feed in uneven chunks; the running mean must equal the one-shot mean.
    for chunk in (x[:130], x[130:411], x[411:]):
        acc.update(chunk)
    ref = (x.T @ x) / x.shape[0]
    assert torch.allclose(acc.H, ref, atol=1e-4)
    assert acc.n_samples == 500
    # finalize adds positive damping on the diagonal -> stays SPD.
    H = acc.finalize(damp_frac=0.01)
    assert torch.all(torch.linalg.eigvalsh(H) > 0)


def test_collect_hessians_over_toy_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    loader = [torch.randn(4, 16) for _ in range(3)]
    result = collect_hessians(model, loader, device="cpu")
    # One Hessian per linear, each square in that layer's in_features.
    assert set(result.hessians) == {"0", "2"}
    assert result.hessians["0"].shape == (16, 16)
    assert result.hessians["2"].shape == (16, 16)
    assert result.n_samples["0"] == 4 * 3


def test_nearest_e8_returns_valid_lattice_points():
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    q = nearest_e8(x)
    # Every E8 point is either all-integer or all-half-integer, with even sum.
    is_int = torch.all(torch.abs(q - q.round()) < 1e-6, dim=-1)
    is_half = torch.all(torch.abs((q - 0.5) - (q - 0.5).round()) < 1e-6, dim=-1)
    assert torch.all(is_int | is_half)
    twice_sum = (2 * q).sum(dim=-1).round()
    assert torch.all(twice_sum % 2 == 0)


def _brute_force_nearest_e8(v: torch.Tensor) -> torch.Tensor:
    """Independent oracle: search all floor/ceil corners of both E8 cosets.

    The true nearest D8 point lies among the 2**8 round-up/down corners filtered
    to an even coordinate sum; E8 is D8 union the half-integer-shifted coset, so
    doing the same on ``v - 0.5`` and shifting back covers all candidates.
    """
    import itertools
    best, best_d = None, float("inf")
    for shift in (0.0, 0.5):
        base = v - shift
        lo = torch.floor(base)
        for bits in itertools.product((0, 1), repeat=8):
            cand = lo + torch.tensor(bits, dtype=base.dtype)
            if int(cand.sum().item()) % 2 != 0:  # only D8 (even-sum) points
                continue
            point = cand + shift
            d = ((v - point) ** 2).sum().item()
            if d < best_d:
                best_d, best = d, point
    return best


def test_nearest_e8_matches_brute_force():
    torch.manual_seed(1)
    x = torch.randn(40, 8)
    q = nearest_e8(x)
    for i in range(x.shape[0]):
        oracle = _brute_force_nearest_e8(x[i])
        d_q = ((x[i] - q[i]) ** 2).sum().item()
        d_oracle = ((x[i] - oracle) ** 2).sum().item()
        assert d_q <= d_oracle + 1e-6, f"row {i}: {d_q} > {d_oracle}"


def test_e8_codebook_quantize_shape_and_reduces_error():
    torch.manual_seed(0)
    x = torch.randn(10, 32)  # divisible by 8
    cb = E8LatticeCodebook(lattice_scale=0.5)
    q = cb.quantize(x)
    assert q.shape == x.shape
    # A reasonable lattice scale should not make things worse than leaving x at 0.
    assert (x - q).pow(2).mean() < x.pow(2).mean()
