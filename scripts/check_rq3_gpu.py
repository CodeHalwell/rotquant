"""Bounded packed-operator conformance against canonical Torch arithmetic."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts.rq3_test_runtime import NativeTests


def check(library, backend):
    runtime = NativeTests(library)
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
            for tokens in (1, 7, 33):
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
                report = {"bits": bits, "scale_bits": sbits, "width": width, "tokens": tokens,
                          "max_abs": float(np.max(np.abs(actual - expected))), "passed": True}
                print(json.dumps(report), flush=True)
                reports.append(report)
    with Path(library).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"protocol": "rq3-packed-operator-check-v1", "backend": backend,
            "library_sha256": digest, "runtime_files": runtime_identity(library),
            "settings": execution_settings(), "cases": reports, "passed": True,
            "scope": "synthetic operators; not retained-model quality or serving performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    result = check(args.library, args.backend)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
