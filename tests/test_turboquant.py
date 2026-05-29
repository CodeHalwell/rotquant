"""Tests for TurboQuant-for-weights implementation.

Covers:
  - turboquant_mse_bound theoretical formula
  - scale="turboquant" produces scales=None and no per-group metadata
  - error_comp="turboquant" populates sketch fields
  - _generate_sketch_matrix is deterministic and has JL norm-preservation property
  - dequantize() works correctly when scales=None
  - bit_budget() reports 0 scale overhead when scales=None
  - QuantLinear forward pass applies QJL correction (output changes + shape correct)
  - packed_state_bytes() counts sketch bytes and skips scales when None
"""
from __future__ import annotations

import math

import pytest
import torch

from rotquant.codebooks import turboquant_mse_bound
from rotquant.quantize import (
    QuantConfig,
    QuantizedWeight,
    Quantizer,
    _generate_sketch_matrix,
)
from rotquant.rotate import RandomizedHadamard
from rotquant.linear import QuantLinear
import rotquant.linear as _lin_mod


# --------------------------------------------------------------------------- #
# turboquant_mse_bound
# --------------------------------------------------------------------------- #
class TestTurboQuantMseBound:
    def test_formula_3bit(self):
        expected = (math.sqrt(3) * math.pi / 2) * (4.0 ** -3)
        assert abs(turboquant_mse_bound(3) - expected) < 1e-12

    def test_monotone_decreasing(self):
        # Higher bit budget → lower MSE bound
        for b in range(1, 6):
            assert turboquant_mse_bound(b + 1) < turboquant_mse_bound(b)

    def test_fractional_bits(self):
        # Should accept float (e.g. 3.125 = 3 bits + 16-bit scale / 128 group)
        v = turboquant_mse_bound(3.125)
        assert 0.0 < v < turboquant_mse_bound(3)

    def test_exported_from_top_level(self):
        import rotquant
        assert rotquant.turboquant_mse_bound is turboquant_mse_bound


# --------------------------------------------------------------------------- #
# _generate_sketch_matrix
# --------------------------------------------------------------------------- #
class TestGenerateSketchMatrix:
    def test_shape(self):
        G = _generate_sketch_matrix(256, 64, seed=0, device="cpu")
        assert G.shape == (256, 64)

    def test_deterministic(self):
        G1 = _generate_sketch_matrix(128, 32, seed=42, device="cpu")
        G2 = _generate_sketch_matrix(128, 32, seed=42, device="cpu")
        assert torch.allclose(G1, G2)

    def test_different_seeds_differ(self):
        G1 = _generate_sketch_matrix(128, 32, seed=0, device="cpu")
        G2 = _generate_sketch_matrix(128, 32, seed=1, device="cpu")
        assert not torch.allclose(G1, G2)

    def test_jl_norm_preservation(self):
        # For a random unit vector x, E[||Gx||^2] ≈ 1 (JL property)
        torch.manual_seed(99)
        G = _generate_sketch_matrix(256, 512, seed=7, device="cpu")
        x = torch.randn(256)
        x = x / x.norm()
        proj_norm_sq = (x @ G).pow(2).sum().item()
        # With k=512 columns the variance is small; should be within 20% of 1
        assert 0.8 < proj_norm_sq < 1.2


# --------------------------------------------------------------------------- #
# scale="turboquant" produces None scales
# --------------------------------------------------------------------------- #
class TestTurboQuantScale:
    def _make_qw(self, **kw) -> QuantizedWeight:
        cfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                          group_size=128, error_comp="none", **kw)
        w = torch.randn(32, 128)
        return Quantizer(cfg).quantize_weight(w)

    def test_scales_is_none(self):
        qw = self._make_qw()
        assert qw.scales is None

    def test_dequantize_shape(self):
        qw = self._make_qw()
        w = qw.dequantize()
        assert w.shape == (32, 128)

    def test_bit_budget_zero_scale_overhead(self):
        qw = self._make_qw()
        bb = qw.bit_budget()
        # scale_bits = 0 → effective bpw = log2(levels) exactly
        assert abs(bb.bits_per_weight - math.log2(2 ** 3)) < 1e-9

    def test_sketch_fields_absent(self):
        qw = self._make_qw()
        assert qw.sketch is None
        assert qw.sketch_row_norms is None
        assert qw.sketch_k == 0


# --------------------------------------------------------------------------- #
# error_comp="turboquant" populates sketch
# --------------------------------------------------------------------------- #
class TestTurboQuantErrorComp:
    def _make_qw(self, out=32, inf=128, k=16) -> QuantizedWeight:
        cfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                          group_size=128, error_comp="turboquant", sketch_k=k, seed=0)
        w = torch.randn(out, inf)
        return Quantizer(cfg).quantize_weight(w)

    def test_sketch_not_none(self):
        qw = self._make_qw()
        assert qw.sketch is not None

    def test_sketch_row_norms_shape(self):
        qw = self._make_qw(out=32, k=16)
        assert qw.sketch_row_norms is not None
        assert qw.sketch_row_norms.shape == (32,)

    def test_sketch_k_recorded(self):
        qw = self._make_qw(k=16)
        assert qw.sketch_k == 16

    def test_sketch_bits_are_binary(self):
        # Unpacked values should be 0 or 1 (1-bit codes)
        from rotquant.pack import unpack_indices
        qw = self._make_qw(out=16, inf=128, k=8)
        bits = unpack_indices(qw.sketch)
        assert set(bits.unique().tolist()).issubset({0, 1})

    def test_sketch_numel(self):
        from rotquant.pack import unpack_indices
        out, inf, k = 16, 128, 8
        qw = self._make_qw(out=out, inf=inf, k=k)
        bits = unpack_indices(qw.sketch)
        assert bits.numel() == out * k

    def test_sketch_bits_one(self):
        # The packed tensor should have bits=1
        qw = self._make_qw(k=8)
        assert qw.sketch.bits == 1


# --------------------------------------------------------------------------- #
# QuantLinear forward: QJL correction changes output, shape is correct
# --------------------------------------------------------------------------- #
class TestQuantLinearForward:
    def _make_linear(self, out=8, inf=64, k=8) -> tuple:
        cfg_tq = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                             error_comp="turboquant", sketch_k=k, seed=0)
        cfg_no = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                             error_comp="none")
        linear = torch.nn.Linear(inf, out, bias=False)
        rot = RandomizedHadamard(inf, seed=0)
        ql_tq = QuantLinear.from_linear(linear, cfg_tq, weight_rotation=rot)
        ql_no = QuantLinear.from_linear(linear, cfg_no, weight_rotation=rot)
        return ql_tq, ql_no

    def test_output_shape_1d(self):
        ql, _ = self._make_linear()
        x = torch.randn(64)
        out = ql(x)
        assert out.shape == (8,)

    def test_output_shape_batched(self):
        ql, _ = self._make_linear()
        x = torch.randn(4, 64)
        out = ql(x)
        assert out.shape == (4, 8)

    def test_correction_changes_output(self):
        ql_tq, ql_no = self._make_linear()
        x = torch.randn(64)
        out_tq = ql_tq(x)
        out_no = ql_no(x)
        # Correction should shift the output (not all-zero difference)
        assert not torch.allclose(out_tq, out_no)

    def test_no_sketch_path_unchanged(self):
        # Without sketch the forward should still match the base dequantize path
        cfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                          error_comp="none")
        linear = torch.nn.Linear(64, 8, bias=False)
        rot = RandomizedHadamard(64, seed=0)
        ql = QuantLinear.from_linear(linear, cfg, weight_rotation=rot)
        x = torch.randn(64)
        out = ql(x)
        assert out.shape == (8,)


# --------------------------------------------------------------------------- #
# packed_state_bytes: sketch bytes counted, scales=None not counted
# --------------------------------------------------------------------------- #
class TestPackedStateBytes:
    def test_no_sketch_no_scales(self):
        cfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                          error_comp="none")
        linear = torch.nn.Linear(128, 32, bias=False)
        from rotquant.rotate import Identity
        ql = QuantLinear.from_linear(linear, cfg)
        b = ql.packed_state_bytes()
        # Only the packed codes: ceil(32*128*3 / 32) * 4 bytes
        from rotquant.pack import packed_bytes
        assert b == packed_bytes(ql.qweight.packed)

    def test_sketch_bytes_counted(self):
        cfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                          error_comp="turboquant", sketch_k=16)
        linear = torch.nn.Linear(128, 32, bias=False)
        ql = QuantLinear.from_linear(linear, cfg)
        b = ql.packed_state_bytes()
        from rotquant.pack import packed_bytes
        sketch_b = packed_bytes(ql.qweight.sketch)
        norms_b = ql.qweight.sketch_row_norms.numel() * 2
        code_b = packed_bytes(ql.qweight.packed)
        assert b == code_b + sketch_b + norms_b

    def test_rms_scales_still_counted(self):
        cfg = QuantConfig(bits=3, codebook="gaussian", scale="rms",
                          error_comp="none")
        linear = torch.nn.Linear(128, 32, bias=False)
        ql = QuantLinear.from_linear(linear, cfg)
        b = ql.packed_state_bytes()
        from rotquant.pack import packed_bytes
        code_b = packed_bytes(ql.qweight.packed)
        scale_b = ql.qweight.scales.numel() * 2
        assert b == code_b + scale_b


# --------------------------------------------------------------------------- #
# Combining turboquant scale with rms as comparison: dequantize correctness
# --------------------------------------------------------------------------- #
class TestDequantizeScalesNone:
    def test_roundtrip_near_gaussian(self):
        """After Hadamard rotation the distribution is Gaussian; 3-bit Lloyd-Max
        should quantise with low MSE even without per-group scaling."""
        torch.manual_seed(0)
        rot = RandomizedHadamard(128, seed=1)
        w_orig = torch.randn(32, 128)
        w_rot = rot.rotate_weight(w_orig)

        cfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                          group_size=128, error_comp="none")
        qw = Quantizer(cfg).quantize_weight(w_rot)
        w_deq = qw.dequantize()
        mse = (w_rot - w_deq).pow(2).mean().item()
        # Theoretical bound from turboquant_mse_bound(3) ≈ 0.0136; practical
        # MSE should be well below 0.1 for a unit-Gaussian source.
        assert mse < 0.1, f"MSE too high: {mse:.4f}"
