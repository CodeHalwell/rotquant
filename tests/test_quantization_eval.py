import pytest
import torch

from rotquant.eval.quantization import compare_quantizers
from rotquant.quantize import QuantConfig


def test_compare_quantizers_reports_rate_reconstruction_and_probe_metrics():
    generator = torch.Generator().manual_seed(21)
    weight = torch.randn(16, 64, generator=generator)
    probes = torch.randn(8, 64, generator=generator)
    common = {"bits": 3, "group_size": 64, "scale": "rms"}
    results = compare_quantizers(
        weight,
        {
            "gaussian": QuantConfig(codebook="gaussian", **common),
            "spherical_length": QuantConfig(
                codebook="spherical", codebook_dim=64,
                bias_correction="length", **common),
        },
        probes=probes,
    )
    assert results["gaussian"]["effective_bpw"] == (
        results["spherical_length"]["effective_bpw"])
    assert results["gaussian"]["weight_nmse"] > 0
    assert results["gaussian"]["probe_output_nmse"] > 0
    assert abs(results["spherical_length"]["global_self_dot_ratio"] - 1) < 1e-3


def test_compare_quantizers_rejects_unmatched_rates():
    weight = torch.randn(4, 32)
    with pytest.raises(ValueError, match="not budget matched"):
        compare_quantizers(
            weight,
            {
                "two": QuantConfig(bits=2, group_size=32),
                "three": QuantConfig(bits=3, group_size=32),
            },
        )
