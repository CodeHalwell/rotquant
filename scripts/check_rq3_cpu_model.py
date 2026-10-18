"""Run the patched CPU graph after building and loading its real binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.check_rq3_model import runtime_identity
from scripts.rq3_test_runtime import NativeTests


def check(library, fixture):
    runtime = NativeTests(library)
    with runtime.model(fixture, "CPU", 128) as model:
        prompt = np.arange(3, 20, dtype=np.int32)
        first = model.evaluate(prompt, reset=True)
        repeated = model.evaluate(prompt, reset=True)
        if not np.isfinite(first).all():
            raise ValueError("CPU prefill produced non-finite logits")
        np.testing.assert_array_equal(first, repeated)
        for _ in range(4):
            first = model.evaluate([int(first.argmax())])
            if not np.isfinite(first).all():
                raise ValueError("CPU graph produced non-finite logits")
        if not model.custom_ops:
            raise ValueError("CPU graph did not execute packed operators")
        return {"passed": True, "runtime_files": runtime_identity(library),
                "custom_ops": model.custom_ops, "model": str(fixture),
                "boundary": "Tiny synthetic CPU graph; no GPU or model-quality claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.library, args.model), indent=2))
