"""Bounded packed-operator conformance against canonical Torch arithmetic."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rotquant.native_v3 import decode_native_v3_rows, encode_native_v3
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import RandomizedHadamard
from scripts.check_rq3_model import execution_settings, runtime_identity
from scripts.native_hashing import digest
from scripts.rq3_test_runtime import NativeTests


def check_decode_tails(runtime, backend, generator):
    """One-token matmul across row tails, formats and longer reduction widths."""
    from scripts.native_performance_study import kernel_environment
    results = []
    shapes = [(rows, 256) for rows in (1, 2, 3, 4, 5, 136, 137)]
    shapes += [(5, width) for width in (512, 2048, 4096, 11008)]
    formats = [(bits, sbits) for bits in range(1, 9) for sbits in (8, 16)]
    for index, (rows, width) in enumerate(shapes + [(5, 512)] * len(formats)):
        bits, sbits = (5, 8) if index < len(shapes) else formats[index - len(shapes)]
        q = Quantizer(QuantConfig(bits=bits, group_size=128, scale="rms", scale_bits=sbits,
                                 scale_quant_group_size=256)).quantize_weight(
            torch.randn(rows, width, generator=generator) * 0.02)
        native = encode_native_v3(q)
        rotation = RandomizedHadamard(width, block=128, seed=2701)
        signs = rotation.signs.numpy().astype(np.int8)
        w = torch.from_numpy(decode_native_v3_rows(native)).half()
        x = torch.randn(1, width, generator=generator).half()
        for permuted in (False, True):
            rp = np.random.default_rng(13).permutation(rows).astype(np.int32) if permuted else None
            cp = np.random.default_rng(17).permutation(width).astype(np.int32) if permuted else None
            expected = F.linear(rotation.rotate_activation(x[:, cp] if permuted else x),
                                w[rp] if permuted else w).float().numpy()
            actual = runtime.operator(backend, native, signs, x.float().numpy(), rows=rp, columns=cp)
            with kernel_environment():
                reference = runtime.operator(backend, native, signs, x.float().numpy(), rows=rp, columns=cp)
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_allclose(actual, expected, rtol=0.002, atol=0.001)
            result = {"bits": bits, "scale_bits": sbits, "rows": rows, "width": width,
                      "tokens": 1, "permuted": permuted, "exact_reference": True,
                      "max_abs": float(np.max(np.abs(actual - expected))), "passed": True}
            print(json.dumps(result), flush=True)
            results.append(result)
    return results


def check(library, backend):
    runtime = NativeTests(library)
    dispatch = None
    if backend == "CUDA0" and os.environ.get("ROTQUANT_RQ3_KERNEL") in ("tiled4", "decode4"):
        from scripts.check_rq3_dispatch import check as check_dispatch
        dispatch = check_dispatch(library, os.environ["ROTQUANT_RQ3_KERNEL"])
    generator = torch.Generator().manual_seed(3701)
    reports = []
    for bits, sbits in ((5, 8), (6, 16), (8, 16)):
        for width in (256, 512):
            q = Quantizer(QuantConfig(bits=bits, group_size=128, scale="rms", scale_bits=sbits,
                                     scale_quant_group_size=256)).quantize_weight(
                torch.randn(137, width, generator=generator) * 0.02)
            native = encode_native_v3(q)
            rotation = RandomizedHadamard(width, block=128, seed=2701)
            signs = rotation.signs.numpy().astype(np.int8)
            weights = torch.from_numpy(decode_native_v3_rows(native))
            vocabulary = rotation.inverse_activation(weights).half()
            variant = os.environ.get("ROTQUANT_RQ3_KERNEL", "reference")
            candidate = variant in ("tiled4", "decode4")
            for tokens in ((1, 2, 3, 4, 5, 7, 33) if candidate else (1, 7, 33)):
                x = torch.randn(tokens, width, generator=generator).half()
                rows = np.random.default_rng(13).permutation(137).astype(np.int32) if bits == 5 else None
                columns = np.random.default_rng(17).permutation(width).astype(np.int32) if bits == 5 else None
                if bits == 5:
                    expected = F.linear(rotation.rotate_activation(x[:, columns]), weights[rows].half()).float().numpy()
                    actual = runtime.operator(backend, native, signs, x.float().numpy(), rows=rows, columns=columns)
                else:
                    expected = F.linear(x, vocabulary).float().numpy()
                    actual = runtime.operator(backend, native, signs, x.float().numpy(), mode=1)
                    ids = np.arange(tokens, dtype=np.int32) % 137
                    embeddings = runtime.operator(backend, native, signs, ids, mode=2)
                    np.testing.assert_allclose(embeddings, vocabulary[ids].float().numpy(), rtol=0, atol=0.000125)
                # FP32 parallel reductions may cross an FP16 rounding boundary.
                np.testing.assert_allclose(actual, expected, rtol=0.002, atol=0.001)
                if candidate and bits == 5:
                    # Same data and arithmetic: additionally require exact
                    # equivalence to the original native kernel, including tails.
                    os.environ["ROTQUANT_RQ3_KERNEL"] = "reference"
                    try:
                        reference = runtime.operator(backend, native, signs, x.float().numpy(), rows=rows, columns=columns)
                    finally:
                        os.environ["ROTQUANT_RQ3_KERNEL"] = variant
                    np.testing.assert_array_equal(actual, reference)
                report = {"bits": bits, "scale_bits": sbits, "width": width, "tokens": tokens,
                          "max_abs": float(np.max(np.abs(actual - expected))), "passed": True}
                print(json.dumps(report), flush=True)
                reports.append(report)
    extra = check_decode_tails(runtime, backend, generator) if os.environ.get("ROTQUANT_RQ3_KERNEL") == "decode4" else []
    diagnostics = None
    if backend == "CUDA0" and os.environ.get("ROTQUANT_RQ3_KERNEL") == "decode4":
        from scripts.native_cuda_diagnostics import CudaDiagnostics
        diagnostics = CudaDiagnostics(library).snapshot()
        if not diagnostics["decode_host_dispatches"] or not diagnostics["tiled_host_dispatches"]:
            raise ValueError("Candidate decode/prefill dispatch missing")
    return {"protocol": "rq3-packed-operator-check-v1", "backend": backend,
            "library_sha256": digest(library), "runtime_files": runtime_identity(library),
            "settings": execution_settings(), "cases": reports, "decode_cases": extra,
            "native_diagnostics": diagnostics, "passed": True,
            "dispatch_probe": dispatch,
            "scope": "synthetic operators; not retained-model quality or serving performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = check(args.library, args.backend)
    except BaseException as error:
        if args.output:
            from scripts.native_gpu_workflow import write_json
            write_json(args.output, {"protocol": "rq3-packed-operator-check-v1", "passed": False,
                "backend": args.backend, "settings": execution_settings(),
                "error": f"{type(error).__name__}: {error}",
                "dispatch_probe": getattr(error, "evidence", None),
                "boundary": "Failed gate. Completed numerical cases, if any, remain in the persistent log."})
        raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
