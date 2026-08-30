"""Source-coding anchors: scalar Lloyd-Max on a unit Gaussian must hit the known
MSE values, and the Shannon bound 2^(-2R) must come out right.

This validates the codebook code and anchors the whole scalar-ceiling argument.
"""
import math

import numpy as np
import torch

from rotquant.codebooks import (
    build_scalar_codebook,
    fit_scalar_codebook,
    lloyd_max_gaussian,
    lloyd_max_spherical,
    quantizer_mse,
    quantizer_mse_spherical,
)
from rotquant.utils import BitBudget


def test_lloyd_max_gaussian_mse_anchors():
    c2 = lloyd_max_gaussian(4)   # 2-bit
    c3 = lloyd_max_gaussian(8)   # 3-bit
    mse2 = quantizer_mse(c2)
    mse3 = quantizer_mse(c3)
    assert abs(mse2 - 0.1175) < 1e-3, f"2-bit MSE {mse2}"
    assert abs(mse3 - 0.0345) < 1e-3, f"3-bit MSE {mse3}"


def test_shannon_bound_values():
    # rate-distortion bound for a unit Gaussian: D(R) = 2^(-2R)
    assert abs(2 ** (-2 * 2) - 0.0625) < 1e-9
    assert abs(2 ** (-2 * 3) - 0.015625) < 1e-9


def test_scalar_ceiling_is_about_2x_bound_at_3bit():
    """Scalar Lloyd-Max sits well above the rate-distortion bound (the ceiling)."""
    mse3 = quantizer_mse(lloyd_max_gaussian(8))
    bound3 = 2 ** (-2 * 3)
    ratio = mse3 / bound3
    assert 2.0 < ratio < 2.4, f"3-bit scalar/bound ratio {ratio}"


def test_spherical_codebook_matches_uniform_sphere_in_three_dimensions():
    # A coordinate of a uniform point on S^2 is uniform on [-1, 1].  RMS
    # normalisation multiplies it by sqrt(3), whose optimal four centroids are
    # the midpoints of four equal-width cells.
    actual = lloyd_max_spherical(4, dimension=3, grid=100_001)
    expected = math.sqrt(3) * np.array([-0.75, -0.25, 0.25, 0.75])
    assert np.allclose(actual, expected, atol=5e-5)


def test_spherical_codebook_beats_gaussian_on_finite_dimension_source():
    dimension = 8
    spherical = lloyd_max_spherical(4, dimension=dimension)
    gaussian = lloyd_max_gaussian(4)
    assert quantizer_mse_spherical(
        spherical, dimension) < quantizer_mse_spherical(gaussian, dimension)


def test_spherical_codebook_cache_includes_dimension():
    d8 = build_scalar_codebook("spherical", 4, 8)
    assert d8 is build_scalar_codebook("spherical", 4, 8)
    assert d8 is not build_scalar_codebook("spherical", 4, 128)
    assert d8.name == "spherical_d8"


def test_empirical_codebook_is_deployable_and_fits_samples():
    generator = torch.Generator().manual_seed(4)
    samples = torch.cat([
        torch.randn(5_000, generator=generator) * 0.2 - 1.5,
        torch.randn(5_000, generator=generator) * 0.2 + 1.5,
    ])
    calibrated = fit_scalar_codebook(samples, 4)
    gaussian = build_scalar_codebook("gaussian", 4)
    calibrated_error = (samples - calibrated.quantize(samples)[0]).square().mean()
    gaussian_error = (samples - gaussian.quantize(samples)[0]).square().mean()
    assert calibrated.levels == 4
    assert calibrated_error < gaussian_error


def test_bits_per_weight_accounting():
    # 3-bit codes, group 128, 16-bit scale -> 3 + 16/128 = 3.125 bpw
    bb = BitBudget(levels=8, group_size=128, scale_bits=16.0)
    assert abs(bb.bits_per_weight - 3.125) < 1e-9
    bb.assert_matches(3.125)

    # A "3-bit" claim that's secretly 8-bit storage must fail the assertion.
    bad = BitBudget(levels=256, group_size=128, scale_bits=16.0)
    try:
        bad.assert_matches(3.0)
        raised = False
    except AssertionError:
        raised = True
    assert raised
