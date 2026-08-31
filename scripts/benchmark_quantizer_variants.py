"""Screen Gaussian, finite-dimensional, and length-corrected scalar profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rotquant.eval.quantization import compare_quantizers
from rotquant.quantize import QuantConfig
from rotquant.utils import write_result


def benchmark_variants(
    *,
    bits: tuple[int, ...],
    dimension: int,
    rows: int,
    probes: int,
    seed: int,
) -> dict[str, object]:
    if dimension < 3 or rows < 1 or probes < 1:
        raise ValueError("dimension >= 3 and positive rows/probes are required")
    if not bits or any(value < 1 or value > 8 for value in bits):
        raise ValueError("bits must contain values in [1, 8]")
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, dimension, generator=generator)
    probe_values = torch.randn(probes, dimension, generator=generator)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for width in bits:
        common = {"bits": width, "group_size": dimension, "scale": "rms"}
        candidates = {
            "gaussian": QuantConfig(
                codebook="gaussian", bias_correction="none", **common),
            "spherical": QuantConfig(
                codebook="spherical",
                codebook_dim=dimension,
                bias_correction="none",
                **common,
            ),
            "gaussian_length": QuantConfig(
                codebook="gaussian", bias_correction="length", **common),
            "spherical_length": QuantConfig(
                codebook="spherical",
                codebook_dim=dimension,
                bias_correction="length",
                **common,
            ),
        }
        results[str(width)] = compare_quantizers(
            weight, candidates, probes=probe_values)
    return {
        "schema_version": 1,
        "protocol": {
            "source": "deterministic Gaussian rows with per-row RMS scaling",
            "bits": list(bits),
            "dimension": dimension,
            "rows": rows,
            "probes": probes,
            "seed": seed,
            "budget_matched": True,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--probes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = benchmark_variants(
        bits=tuple(args.bits),
        dimension=args.dimension,
        rows=args.rows,
        probes=args.probes,
        seed=args.seed,
    )
    if args.output is not None:
        write_result(str(args.output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
