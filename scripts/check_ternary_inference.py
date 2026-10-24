"""Bounded operator, graph-pruning and compact-sampling conformance, never quality."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from rotquant.native_v3 import encode_native_v3
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.ternary import TernaryConfig, TernaryLinear
from scripts.check_rq3_model import runtime_identity
from scripts.rq3_test_runtime import NativeTests
from scripts.ternary_inference import (
    DispatchReceipt,
    InferenceProfile,
    expected_model_dispatch,
    require_dispatch,
)
from scripts.w4_research_measure import ChunkedNativeModel


def matrix_fixture(bits, width):
    generator = torch.Generator().manual_seed(1703)
    weight = torch.randn(5, width, generator=generator) * .02
    if bits == 2:
        layer = torch.nn.Linear(width, 5, bias=False)
        layer.weight.data.copy_(weight)
        quantized = TernaryLinear.from_linear(layer, TernaryConfig(seed=17)).to_quant_linear().qweight
    else:
        quantized = Quantizer(QuantConfig(bits=bits, group_size=128, scale="rms",
                                         scale_bits=8 if bits == 4 else 16)).quantize_weight(weight)
    return encode_native_v3(quantized)


def sampling_check(runtime, backend):
    results = []
    for values in ([0, 3, 3, 1], [-3, -1, -1], [-0., 0.], [5]):
        token = runtime.greedy_operator(backend, values)
        if token != int(np.argmax(values)):
            raise ValueError("compact sampler violated lowest-index argmax")
        results.append(token)
    for value in (np.nan, np.inf, -np.inf):
        try:
            runtime.greedy_operator(backend, [1, value, 9])
        except RuntimeError as error:
            if "finite" not in str(error):
                raise
        else:
            raise ValueError("non-finite logits escaped the native sampler")
    return {"argmax_tokens": results, "nonfinite_rejected": 3, "passed": True}


def graph_check(runtime, fixture, backend):
    with runtime.model(fixture, backend, 128) as native:
        legacy = ChunkedNativeModel(native)
        optimized = ChunkedNativeModel(native, graph_prefill=True, compact_greedy=True)
        prompt = list(range(3, 36))
        expected = legacy.evaluate(prompt, reset=True)
        before = native.inference_diagnostics
        native.evaluate(prompt[:16], reset=True, selected=0)
        after = native.inference_diagnostics
        if after["head_ops"] != before["head_ops"] or after["full_logit_rows"] != before["full_logit_rows"]:
            raise ValueError("no-logits prefill executed a head or copied vocabulary logits")
        actual = optimized.evaluate(prompt, reset=True)
        np.testing.assert_array_equal(actual, expected)
        expected_trace, actual_trace = [], []
        logits = legacy.evaluate(prompt, reset=True)
        for _ in range(4):
            token = int(logits.argmax())
            expected_trace.append(token)
            logits = legacy.evaluate([token])
        compact_before = native.inference_diagnostics
        token = optimized.greedy(prompt, reset=True)
        for _ in range(4):
            actual_trace.append(token)
            token = optimized.greedy([token])
        if native.inference_diagnostics["full_logit_rows"] != compact_before["full_logit_rows"]:
            raise ValueError("compact generation copied full vocabulary logits")
        if actual_trace != expected_trace:
            raise ValueError("compact cached generation diverged")
        result = {"head_pruned": True, "exact_prefill_logits": True, "compact_no_vocab_readback": True,
                  "cached_trace": actual_trace, "diagnostics": native.inference_diagnostics,
                  "packed_custom_ops": native.custom_ops, "passed": True}
        np.testing.assert_array_equal(legacy.evaluate(prompt, reset=True), expected)
        return result


def dispatch_expectations(kernel):
    profile = InferenceProfile(kernel, arithmetic="w2a8" if kernel == "ternary-w2a8" else "w2a16")
    return expected_model_dispatch(profile, "ternary_operator_fixture")


def operator_case(runtime, backend, kernel, matrix, signs, inputs, mode, rows=None, columns=None):
    os.environ["ROTQUANT_RQ3_KERNEL"] = "reference"
    reference = runtime.operator(backend, matrix, signs, inputs, mode=mode, rows=rows, columns=columns)
    os.environ["ROTQUANT_RQ3_KERNEL"] = kernel
    actual = runtime.operator(backend, matrix, signs, inputs, mode=mode, rows=rows, columns=columns)
    if kernel == "ternary-w2a8" and matrix.layout.bits == 2:
        from rotquant.ternary_w2a8 import native_w2a8_reference
        reference = native_w2a8_reference(matrix, signs, inputs, rows=rows, columns=columns)
    np.testing.assert_allclose(actual, reference, rtol=.002, atol=.001)
    if ("gemm" not in kernel and kernel != "tensorcore-fused") or mode == 1:
        np.testing.assert_array_equal(actual, reference)
    return {"tokens": len(inputs), "max_abs": float(np.abs(actual - reference).max()),
            "exact": bool(np.array_equal(actual, reference)), "passed": True}


def operator_fixture_rows(runtime, backend, kernel, bits, width, generator):
    matrix = matrix_fixture(bits, width)
    signs = generator.choice(np.array([-1, 1], dtype=np.int8), width)
    rows = []
    for count in (1, 7, 16, 17):
        inputs = generator.normal(0, .3, (count, width)).astype(np.float16).astype(np.float32)
        permuted = count == 7 and bits != 6
        row_map = np.ascontiguousarray(generator.permutation(5), dtype=np.int32) if permuted else None
        col_map = np.ascontiguousarray(generator.permutation(width), dtype=np.int32) if permuted else None
        row = operator_case(runtime, backend, kernel, matrix, signs, inputs, 1 if bits == 6 else 0,
                            rows=row_map, columns=col_map)
        row["permuted"] = permuted
        rows.append({"bits": bits, "scale_bits": 8 if bits == 4 else 16, "width": width, **row})
    return rows


def operator_check(runtime, backend, kernel, *, production_widths=False):
    generator = np.random.default_rng(1707)
    rows = []
    dispatch = DispatchReceipt(runtime.library) if backend.startswith("CUDA") else None
    if dispatch:
        dispatch.snapshot(reset=True)
    for bits in (2, 4, 6):
        for width in ((2560, 9216) if production_widths else (256,)):
            rows.extend(operator_fixture_rows(runtime, backend, kernel, bits, width, generator))
    receipt = dispatch.snapshot() if dispatch else None
    if receipt:
        require_dispatch(receipt, dispatch_expectations(kernel))
    return {"rows": rows, "dispatch": receipt,
            "cuda_kernel_executed": backend.startswith("CUDA"),
            "cpu_execution": None if backend.startswith("CUDA") else ("W2A8-oracle" if kernel == "ternary-w2a8" else "reference"),
            "dispatch_binding": dispatch.binding if dispatch else None, "passed": True}


def check(library, fixture, backend, kernel="reference"):
    profile = InferenceProfile(kernel, graph_prefill=True, compact_greedy=True,
                               arithmetic="w2a8" if kernel == "ternary-w2a8" else "w2a16")
    old = os.environ.get("ROTQUANT_RQ3_KERNEL")
    try:
        profile.apply()
        runtime = NativeTests(library)
        return {"protocol": "rotquant-inference-conformance-v2", "passed": True,
                "profile": profile.identity(), "backend": backend,
                "runtime_files": runtime_identity(library),
                "operators": operator_check(runtime, backend, kernel, production_widths=backend.startswith("CUDA")),
                "sampling": sampling_check(runtime, backend), "graph": graph_check(runtime, fixture, backend),
                "boundary": "Synthetic operators and two-layer hybrid graph; no 4B quality or speed claim"}
    finally:
        if old is None:
            os.environ.pop("ROTQUANT_RQ3_KERNEL", None)
        else:
            os.environ["ROTQUANT_RQ3_KERNEL"] = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--kernel", default="reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = check(args.library, args.model, args.backend, args.kernel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
