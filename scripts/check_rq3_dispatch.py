"""Cheap CUDA dispatch gate before retained-model work; not a timing result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rotquant.native_v3 import encode_native_v3
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import RandomizedHadamard
from scripts.check_rq3_model import runtime_identity
from scripts.native_cuda_diagnostics import CudaDiagnostics, difference
from scripts.native_gpu_workflow import write_json
from scripts.native_performance_study import kernel_environment
from scripts.rq3_test_runtime import NativeTests


class DispatchProbeError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def require_dispatch(delta, kernel, tokens):
    """Check new calls, not cumulative counters left by earlier operations."""
    decode = delta["decode_host_dispatches"]
    tiled = delta["tiled_host_dispatches"]
    expected_decode = kernel == "decode4" and tokens == 1
    expected_tiled = kernel in {"tiled4", "decode4"} and tokens >= 4
    if (delta["operators"]["matrix"]["host_dispatches"] <= 0
            or (decode <= 0 if expected_decode else decode != 0)
            or (tiled <= 0 if expected_tiled else tiled != 0)):
        raise ValueError(f"Wrong or missing {kernel} dispatch for {tokens} tokens: {delta}")


def check(library, kernel):
    if kernel not in {"tiled4", "decode4"}:
        raise ValueError("dispatch probe requires tiled4/decode4")
    report = {"protocol": "rq3-cuda-dispatch-probe-v1", "passed": False, "backend": "CUDA0",
              "settings": {"rq3_kernel": kernel}, "cases": [],
              "boundary": "Tiny same-data reference/candidate calls with fresh counter deltas; not retained-model parity or speed."}
    try:
        runtime = NativeTests(library)
        report["runtime_files"] = runtime_identity(library)
        diagnostics = CudaDiagnostics(library)
        report["binding"] = diagnostics.binding
        generator = torch.Generator().manual_seed(7811)
        quantized = Quantizer(QuantConfig(bits=5, group_size=128, scale="rms", scale_bits=8,
                            scale_quant_group_size=256)).quantize_weight(torch.randn(5, 256, generator=generator) * .02)
        matrix = encode_native_v3(quantized)
        signs = RandomizedHadamard(256, block=128, seed=2701).signs.numpy().astype(np.int8)
        for tokens in (1, 4):
            inputs = torch.randn(tokens, 256, generator=generator).half().float().numpy()
            outputs = []
            for selected in ("reference", kernel):
                with kernel_environment(selected):
                    before = diagnostics.snapshot()
                    outputs.append(runtime.operator("CUDA0", matrix, signs, inputs))
                    delta = difference(before, diagnostics.snapshot())
                row = {"kernel": selected, "tokens": tokens, "delta": delta, "passed": False}
                report["cases"].append(row)
                require_dispatch(delta, selected, tokens)
                row["passed"] = True
                print("Dispatch probe:", json.dumps(row), flush=True)
            np.testing.assert_array_equal(outputs[0], outputs[1])
        report["exact_reference"] = True
        report["passed"] = True
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise DispatchProbeError(report["error"], report) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--kernel", choices=("tiled4", "decode4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        report = check(args.library, args.kernel)
    except DispatchProbeError as error:
        write_json(args.output, error.evidence)
        raise
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
