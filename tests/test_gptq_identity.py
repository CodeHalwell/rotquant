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
    common = {"bits": 3, "codebook": "gaussian", "scale": "rms", "group_size": 128}

    gptq = Quantizer(QuantConfig(error_comp="gptq", **common))
    plain = Quantizer(QuantConfig(error_comp="none", **common))

    q_gptq = _deq(gptq.quantize_weight(W, H=torch.eye(256)))
    q_plain = _deq(plain.quantize_weight(W))

    assert torch.equal(q_gptq, q_plain), \
        f"GPTQ(H=I) != rounding, max diff {(q_gptq - q_plain).abs().max()}"


def test_gptq_blocked_matches_column_at_a_time():
    """The lazy-batch blocked update must reproduce the naive column-at-a-time
    GPTQ (same maths, reorganised): compare against an inline reference."""
    torch.manual_seed(1)
    out, d = 16, 96
    W = torch.randn(out, d)
    X = torch.randn(256, d)
    H = X.T @ X

    cfg = QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=32,
                      gptq_block=16, gptq_actorder=False,
                      gptq_recompute_scales=False)  # several blocks over d=96
    qz = Quantizer(cfg)
    scales = qz.select_scales(W)
    Q_blocked, _, output_scales = qz._gptq(W.clone(), scales, H.clone())
    assert torch.equal(output_scales, scales.half())

    # Naive reference: identical setup, full-width rank-1 update per column.
    Hd = H.to(torch.float32).clone()
    Hinv = qz._stable_hinv(Hd)
    Wc = W.clone()
    Q_ref = torch.zeros_like(Wc)
    centroids = qz.codebook.centroids
    bounds = (centroids[:-1] + centroids[1:]) / 2.0
    for i in range(d):
        sc = scales[:, i // cfg.group_size].clamp_min(1e-12)
        idx = torch.bucketize(Wc[:, i] / sc, bounds)
        q = centroids[idx] * sc
        Q_ref[:, i] = q
        err = (Wc[:, i] - q) / Hinv[i, i]
        if i + 1 < d:
            Wc[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)

    assert torch.allclose(Q_blocked, Q_ref, atol=1e-4), \
        f"blocked GPTQ diverges from reference, max diff {(Q_blocked - Q_ref).abs().max()}"


def test_gptq_reduces_error_with_real_hessian():
    """With a non-trivial Hessian, GPTQ should not increase weighted error."""
    torch.manual_seed(0)
    d = 128
    W = torch.randn(32, d)
    X = torch.randn(512, d)
    H = X.T @ X

    common = {"bits": 3, "codebook": "gaussian", "scale": "rms", "group_size": 64}
    gptq = Quantizer(QuantConfig(error_comp="gptq", **common)).quantize_weight(W, H=H)
    plain = Quantizer(QuantConfig(error_comp="none", **common)).quantize_weight(W)

    def proxy_err(qw):
        e = W - qw.dequantize()
        return (e @ H * e).sum().item()

    assert proxy_err(gptq) <= proxy_err(plain) * 1.001


def test_gptq_actorder_and_refitted_scales_preserve_packed_layout():
    torch.manual_seed(11)
    out, d = 8, 64
    weight = torch.randn(out, d)
    activations = torch.randn(256, d) * torch.linspace(0.2, 3.0, d)
    hessian = activations.T @ activations
    quantized = Quantizer(QuantConfig(
        bits=3,
        codebook="gaussian",
        scale="mse_search",
        group_size=16,
        error_comp="gptq",
        gptq_actorder=True,
        gptq_recompute_scales=True,
    )).quantize_weight(weight, H=hessian)
    assert quantized.scales.shape == (out, 4)
    assert quantized.dequantize().shape == weight.shape
