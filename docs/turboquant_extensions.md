# TurboQuant/TurboVec extension experiments

RotQuant now exposes three compatible scalar-codebook tiers:

- `gaussian`: the existing high-dimensional Lloyd-Max default;
- `spherical`: the finite-dimensional, unit-RMS spherical marginal selected by
  `codebook_dim`;
- calibrated: `fit_scalar_codebook` fitted from representative values and
  supplied through `Quantizer(..., codebook=...)`.

`bias_correction="length"` additionally enforces
`<w, reconstructed_w> / ||w||^2 ~= 1` per row. The correction is folded into
existing scales after code assignment, so it changes neither codes nor stored
bits. It targets inner-product shrinkage rather than reconstruction MSE and is
therefore not enabled by default.

Run the deterministic screening protocol with:

```bash
uv run python scripts/benchmark_quantizer_variants.py \
  --bits 1 2 3 4 --dimension 128 --rows 512 --probes 256 \
  --output results/quantizer_variants.json
```

The initial seed-2026 screening at dimension 128 found:

| Bits | Profile | Self-dot ratio | Weight NMSE | Random-probe NMSE |
|---:|---|---:|---:|---:|
| 2 | Gaussian | 0.8851 | 0.11673 | 0.11682 |
| 2 | Spherical | 0.8838 | 0.11672 | 0.11681 |
| 2 | Gaussian + length | 1.0000 | 0.13161 | 0.13163 |
| 4 | Gaussian | 0.9911 | 0.00935 | 0.00940 |
| 4 | Spherical | 0.9897 | 0.00936 | 0.00940 |
| 4 | Gaussian + length | 1.0000 | 0.00927 | 0.00931 |

These are screening numbers, not model-quality claims. At realistic 128-wide
blocks the analytic spherical grid is extremely close to Gaussian. Length
correction exactly removes shrinkage but worsens isotropic-probe error at 1–3
bits in this protocol; at 4 bits it gives a small improvement. Consequently the
release default remains `gaussian` plus no correction. The new profiles should
advance only if held-out layer outputs, teacher KL, perplexity, and KV attention
logits improve at the same exact rate.
