"""GPTQ with H = I must reduce *exactly* to plain rounding.

If it doesn't, the error-feedback indexing is wrong.
"""
import torch

from rotquant.quantize import QuantConfig, Quantizer


def _deq(qw):
    return qw.dequantize()


def test_gptq_identity_equals_rounding():
    torch.manual_seed(0)
    W = torch.randn(48, 256)
    common = dict(bits=3, codebook="gaussian", scale="rms", group_size=128)

    gptq = Quantizer(QuantConfig(error_comp="gptq", **common))
    plain = Quantizer(QuantConfig(error_comp="none", **common))

    q_gptq = _deq(gptq.quantize_weight(W, H=torch.eye(256)))
    q_plain = _deq(plain.quantize_weight(W))

    assert torch.equal(q_gptq, q_plain), \
        f"GPTQ(H=I) != rounding, max diff {(q_gptq - q_plain).abs().max()}"


def test_gptq_reduces_error_with_real_hessian():
    """With a non-trivial Hessian, GPTQ should not increase weighted error."""
    torch.manual_seed(0)
    d = 128
    W = torch.randn(32, d)
    X = torch.randn(512, d)
    H = X.T @ X

    common = dict(bits=3, codebook="gaussian", scale="rms", group_size=64)
    gptq = Quantizer(QuantConfig(error_comp="gptq", **common)).quantize_weight(W, H=H)
    plain = Quantizer(QuantConfig(error_comp="none", **common)).quantize_weight(W)

    def proxy_err(qw):
        e = W - qw.dequantize()
        return (e @ H * e).sum().item()

    assert proxy_err(gptq) <= proxy_err(plain) * 1.001
