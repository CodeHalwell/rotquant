"""Bounded, local CPU format conformance; never launches a model or paid run."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rotquant.native_v3 import encode_native_v3
from rotquant.native_v3_ffi import NativeV3Runtime
from rotquant.quantize import QuantConfig, Quantizer


def check_library(library):
    torch.set_num_threads(1)
    runtime = NativeV3Runtime(library)
    rows = []
    for bits in range(1, 9):
        for scale_bits in (8, 16):
            print(f"check scalar CPU matrix W{bits}/scale{scale_bits}", flush=True)
            weight = torch.randn(7, 137, generator=torch.Generator().manual_seed(9))
            weight *= torch.arange(1, 8)[:, None]
            q = Quantizer(QuantConfig(bits=bits, scale_bits=scale_bits, group_size=128,
                                     scale_quant_group_size=4, scale="rms")).quantize_weight(weight)
            encoded = encode_native_v3(q)
            handle = runtime.prepare(encoded)
            expected = q.dequantize().numpy()
            decoded = handle.dequantize_rows()
            np.testing.assert_array_equal(decoded, expected)
            x = np.random.default_rng(9).normal(size=(2, 137)).astype(np.float32)
            actual, oracle = handle.matmul(x), x @ expected.T
            np.testing.assert_allclose(actual, oracle, atol=2e-4, rtol=2e-5)
            rows.append({"bits": bits, "scale_bits": scale_bits,
                         "exact_decode": True, "matmul_max_abs_error": float(np.max(abs(actual - oracle))),
                         "matmul_atol": 2e-4, "matmul_rtol": 2e-5,
                         "packed_bytes": handle.persistent_bytes})
    return {
        "protocol": "rotquant-native-v3-local-conformance-v1",
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "library_sha256": hashlib.sha256(runtime.path.read_bytes()).hexdigest(),
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "backend": "scalar_cpu_matrix_reference", "cases": rows, "matrix_conformance_passed": True,
        "paid_model_run_ready": False,
        "blockers": [
            "W5/scale8 and W6/W8 shared-vocabulary GGUF model exporter/operators not implemented",
            "full-model CPU/Metal/CUDA parity not measured for this recipe",
            "GPU allocation/transfer audit and warm prefill/decode speed preflight not measured",
            "no runtime-bound task manifest or user-approved session cost forecast",
        ],
        "boundary": "Synthetic CPU matrix checks only; not task accuracy, model equivalence or a speed claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional local JSON report; refuses overwrite")
    parser.add_argument("--require-model-runtime", action="store_true",
                        help="Exit 2 unless full-model paid-run readiness is established (currently blocked)")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = check_library(args.library)
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(serialized + "\n")
    print(serialized, flush=True)
    if args.require_model_runtime and not report["paid_model_run_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
