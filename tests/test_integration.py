"""CPU integration test: patch a toy model, check the consistency trap signal,
footprint, and that QJL loses to a deterministic residual at equal bits.
"""
import copy

import torch
import torch.nn as nn

from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig, Quantizer
from eval.layer_mse import layer_output_mse


class _Toy(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _patch(fp, mode):
    q = copy.deepcopy(fp)
    qcfg = QuantConfig(bits=3, codebook="gaussian", scale="mse_search",
                       group_size=128, error_comp="none")
    patch_model(q, PatchConfig(quant=qcfg, rotation="fwht", mode=mode, seed=0))
    return q


def test_consistency_trap_drift():
    torch.manual_seed(0)
    fp = _Toy().eval()
    x = torch.randn(4, 32, 256)

    consistent = layer_output_mse(fp, _patch(fp, "consistent"), x)
    mismatched = layer_output_mse(fp, _patch(fp, "mismatched"), x)

    # Consistent quantisation keeps high cosine; mismatched basis collapses it.
    assert min(consistent.cosine.values()) > 0.8
    assert max(mismatched.mse.values()) > 10 * max(consistent.mse.values())


def test_packed_smaller_than_fp16():
    torch.manual_seed(0)
    fp = _Toy().eval()
    q = _patch(fp, "consistent")
    fp_bytes = sum(p.numel() * 2 for p in fp.parameters())  # fp16
    packed = q.fc1.packed_state_bytes() + q.fc2.packed_state_bytes()
    assert packed < fp_bytes


def test_qjl_loses_to_residual_at_equal_bits():
    torch.manual_seed(0)
    W = torch.randn(32, 256)
    common = dict(bits=3, group_size=128, residual_bits=1)
    res = Quantizer(QuantConfig(error_comp="residual", **common)).quantize_weight(W).dequantize()
    qjl = Quantizer(QuantConfig(error_comp="qjl", **common)).quantize_weight(W).dequantize()
    assert (W - res).pow(2).mean() < (W - qjl).pow(2).mean()
