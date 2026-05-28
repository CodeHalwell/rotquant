"""E1 foundation: rotating the activation then matmul must equal dequant-then-matmul.

We had this backwards once (it cost a whole run), so this is a guard test.
"""
import torch

from rotquant.rotate import build_rotation
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.linear import QuantLinear
import torch.nn as nn


def _max_rel(a, b):
    return ((a - b).abs() / (b.abs().max() + 1e-9)).max().item()


def test_rotation_orthogonal_and_invariant():
    torch.manual_seed(0)
    d = 256
    for kind in ["none", "fwht", "dense", "learned"]:
        R = build_rotation(kind, d, block=128, seed=1)
        M = R.as_matrix(dtype=torch.float64)
        ortho = (M @ M.T - torch.eye(d, dtype=torch.float64)).abs().max().item()
        assert ortho < 1e-4, f"{kind} not orthogonal: {ortho}"

        x = torch.randn(8, d)
        W = torch.randn(17, d)
        y = x @ W.T
        yr = R.rotate_activation(x) @ R.rotate_weight(W).T
        assert _max_rel(yr, y) < 1e-3, f"{kind} invariance broken: {_max_rel(yr, y)}"

        # Fused inverse round-trips.
        rt = R.inverse_activation(R.rotate_activation(x))
        assert _max_rel(rt, x) < 1e-3


def test_quantlinear_forward_matches_dequant_matmul():
    """forward_quant (rotate activation) == dequant-then-matmul to ~1e-3."""
    torch.manual_seed(0)
    d_in, d_out = 256, 64
    lin = nn.Linear(d_in, d_out, bias=True)
    cfg = QuantConfig(bits=4, codebook="gaussian", scale="mse_search", group_size=128)
    R = build_rotation("fwht", d_in, block=128, seed=3)
    ql = QuantLinear.from_linear(lin, cfg, weight_rotation=R, act_rotation=R)

    x = torch.randn(5, d_in)
    out_forward = ql(x)

    # Reference: rotate activation, then matmul with the (rotated) dequantised weight.
    W_deq = ql.qweight.dequantize()
    ref = R.rotate_activation(x) @ W_deq.T + lin.bias
    assert _max_rel(out_forward, ref) < 1e-3
