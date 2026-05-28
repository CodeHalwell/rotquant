"""Source-coding anchors: scalar Lloyd-Max on a unit Gaussian must hit the known
MSE values, and the Shannon bound 2^(-2R) must come out right.

This validates the codebook code and anchors the whole scalar-ceiling argument.
"""
import math

import numpy as np

from rotquant.codebooks import lloyd_max_gaussian, quantizer_mse
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
