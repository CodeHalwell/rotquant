#!/usr/bin/env python3
"""Benchmark the exact native v2 reference path across 1--8-bit profiles.

This is a correctness and regression baseline, not a production performance
claim. Fused CPU, Metal, CUDA, or Mojo kernels should consume the same encoded
matrices and report speedups relative to these named operations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rotquant.native import (
    NativeEncodedMatrix,
    NativeLayout,
    pack_native_blocks,
    reference_dequantize,
    unpack_native_blocks,
)
from rotquant.runtime import run_kernel, runtime_capabilities


def _milliseconds(function, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        function()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - start) * 1000.0 / iterations


def benchmark_case(
    *,
    bits: int,
    out_features: int,
    in_features: int,
    group_size: int,
    batch: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    if min(out_features, in_features, group_size, batch, iterations) < 1:
        raise ValueError("matrix, group, batch and iteration sizes must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    layout = NativeLayout(bits=bits, group_size=group_size)
    rng = np.random.default_rng(seed + bits)
    indices = rng.integers(
        0,
        1 << bits,
        size=(out_features, in_features),
        dtype=np.uint16,
    )
    groups = layout.groups_for(in_features)
    scales = rng.uniform(0.01, 0.25, size=(out_features, groups)).astype(
        np.float16
    )
    codebook = np.linspace(-2.5, 2.5, 1 << bits, dtype=np.float32)
    values = rng.normal(size=(batch, in_features)).astype(np.float32)
    qdata = pack_native_blocks(indices, scales, layout)
    encoded = NativeEncodedMatrix(
        qdata=qdata,
        codebook=codebook,
        layout=layout,
        in_features=in_features,
        out_features=out_features,
    )

    pack_ms = _milliseconds(
        lambda: pack_native_blocks(indices, scales, layout), iterations, warmup
    )
    unpack_ms = _milliseconds(
        lambda: unpack_native_blocks(
            encoded.qdata, in_features=in_features, layout=layout
        ),
        iterations,
        warmup,
    )
    dequantize_ms = _milliseconds(
        lambda: run_kernel("dequantize", encoded), iterations, warmup
    )
    matmul_ms = _milliseconds(
        lambda: run_kernel("matmul", encoded, values), iterations, warmup
    )

    dense = reference_dequantize(encoded)
    expected = values @ dense.T
    actual = run_kernel("matmul", encoded, values)
    max_abs_error = float(np.max(np.abs(actual - expected)))
    dense_matmul_ms = _milliseconds(
        lambda: values @ dense.T, iterations, warmup
    )
    weights = out_features * in_features
    return {
        "bits": bits,
        "group_size": group_size,
        "out_features": out_features,
        "in_features": in_features,
        "batch": batch,
        "qdata_bytes": encoded.qdata.nbytes,
        "codebook_bytes": encoded.codebook.nbytes,
        "qdata_bits_per_weight": encoded.qdata.nbytes * 8 / weights,
        "persistent_bits_per_weight": encoded.persistent_bytes * 8 / weights,
        "pack_ms": pack_ms,
        "unpack_ms": unpack_ms,
        "dequantize_ms": dequantize_ms,
        "reference_matmul_ms": matmul_ms,
        "dense_oracle_matmul_ms": dense_matmul_ms,
        "dense_oracle_weight_bytes": dense.nbytes,
        "max_abs_error": max_abs_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [
        benchmark_case(
            bits=bits,
            out_features=args.out_features,
            in_features=args.in_features,
            group_size=args.group_size,
            batch=args.batch,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )
        for bits in args.bits
    ]
    report = {
        "schema_version": 1,
        "benchmark": "rotquant-native-v2-reference",
        "claim_boundary": (
            "NumPy correctness baseline; not a fused-runtime performance claim"
        ),
        "runtime_capabilities": runtime_capabilities(),
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
