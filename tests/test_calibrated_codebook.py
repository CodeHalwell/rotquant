"""Auto-calibrated codebook reuse semantics.

The quantizer must refit its calibrated grid for every distinct weight matrix
(silently reusing another layer's grid corrupts results), while byte-identical
repeat calls keep the fitted grid. The permutation case is the adversarial
one: a permuted matrix shares shape, sum, and abs-sum with the original, so a
summary-statistic reuse key cannot distinguish them — only exact content can.
"""
import torch

from rotquant.quantize import (
    QuantConfig,
    Quantizer,
    _calibration_sample_indices,
)


def _cfg() -> QuantConfig:
    return QuantConfig(bits=3, codebook="calibrated", group_size=32,
                       calibrated_samples=512, calibrated_iters=20)


def test_large_calibration_sampling_never_rounds_past_matrix_end():
    total_values = 2_560 * 9_216  # Qwen3.5-4B MLP projection size (> 2**24)
    indices = _calibration_sample_indices(
        total_values, 65_536, device="cpu"
    )

    assert indices.dtype == torch.int64
    assert indices[0].item() == 0
    assert indices[-1].item() == total_values - 1
    assert indices.min().item() >= 0
    assert indices.max().item() < total_values
    assert torch.all(indices[1:] > indices[:-1])


def test_calibrated_refits_for_permuted_matrix():
    torch.manual_seed(0)
    w1 = torch.randn(8, 64)
    permutation = torch.randperm(w1.numel())
    w2 = w1.reshape(-1)[permutation].reshape(w1.shape)
    assert not torch.equal(w1, w2)
    # Same shape / sum / abs-sum: defeats any summary-statistic reuse key.
    assert torch.isclose(w1.sum(), w2.sum())
    assert torch.isclose(w1.abs().sum(), w2.abs().sum())

    reused = Quantizer(_cfg())
    q1 = reused.quantize_weight(w1)
    q2 = reused.quantize_weight(w2)
    fresh = Quantizer(_cfg()).quantize_weight(w2)

    # The second matrix must get its own freshly fitted grid...
    assert torch.equal(q2.codebook.centroids, fresh.codebook.centroids)
    assert torch.equal(q2.dequantize(), fresh.dequantize())
    # ...which differs from the first matrix's grid (different sampled values).
    assert not torch.equal(q2.codebook.centroids, q1.codebook.centroids)


def test_calibrated_reuses_grid_for_identical_matrix():
    torch.manual_seed(1)
    w = torch.randn(8, 64)
    quantizer = Quantizer(_cfg())
    quantizer.quantize_weight(w)
    first_grid = quantizer.codebook
    quantizer.quantize_weight(w.clone())
    assert quantizer.codebook is first_grid  # byte-identical input: no refit


def test_explicit_codebook_is_never_refitted():
    from rotquant.codebooks import fit_scalar_codebook

    torch.manual_seed(2)
    supplied = fit_scalar_codebook(torch.randn(4096), 8, name="calibrated_w3")
    quantizer = Quantizer(_cfg(), codebook=supplied)
    quantizer.quantize_weight(torch.randn(8, 64))
    quantizer.quantize_weight(torch.randn(8, 64))
    assert quantizer.codebook is supplied
