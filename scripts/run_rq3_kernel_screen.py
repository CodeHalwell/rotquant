"""Resident synthetic-graph screen against decode4, not model throughput.

Allocation, static uploads and output readback are outside native timing.
Rotation + packed matmul + graph launch/sync are inside. CUDA graphs and event
profiling are disabled. This is a shortlist heuristic, never kernel promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rotquant.native_v3 import NativeV3Layout, NativeV3Matrix
from scripts.check_rq3_dispatch import require_dispatch
from scripts.check_rq3_model import runtime_identity
from scripts.native_cuda_diagnostics import CudaDiagnostics, difference
from scripts.native_gpu_workflow import write_json
from scripts.native_kernel_candidates import W5_TILES
from scripts.native_performance_study import kernel_environment
from scripts.rq3_test_runtime import NativeTests

# Square, expanding and contracting projections; synthetic distributions,
# not claimed to be a frequency-weighted replay of every Qwen layer.
SHAPES = ((2560, 2560), (9216, 2560), (2560, 9216))
TOKEN_COUNTS = (1, 128)
ROUNDS, REPEATS, ITERATIONS = 2, 3, 10
MIN_CASE_RATIO, MIN_PHASE_RATIO = .95, 1.05


def fixture(rows, width, seed=8419, *, bits=5, scale_bits=8, block=256):
    """Generate compact random codes directly: no model or dense quantization."""
    rng = np.random.default_rng(seed)
    layout = NativeV3Layout(bits, 128, rows, width, scale_bits, block if scale_bits == 8 else 0)
    words = rng.integers(0, 2**32, layout.word_count, dtype=np.uint32).view(np.int32)
    scales = rng.integers(0, 256, (rows, width // 128), dtype=np.uint8) if scale_bits == 8 else (
        rng.uniform(.01, .02, (rows, width // 128)).astype(np.float16))
    return NativeV3Matrix(layout, words, scales,
        rng.uniform(.005, .015, layout.metadata_count).astype(np.float16),
        rng.uniform(.00001, .00005, layout.metadata_count).astype(np.float16),
        np.linspace(-2.5, 2.5, 2**bits, dtype=np.float32))


def decision(rows):
    """Require both paired rounds and every declared shape; fail closed."""
    expected = {(r, c, t, turn) for r, c in SHAPES for t in TOKEN_COUNTS for turn in range(ROUNDS)}
    keys = [(x["rows"], x["width"], x["tokens"], x["round"]) for x in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete/duplicate kernel screen cases")
    ratios = {t: [] for t in TOKEN_COUNTS}
    for row in rows:
        if row.get("exact_decode4") is not True:
            raise ValueError("Screen parity failed")
        for key in ("decode4_seconds", "candidate_seconds"):
            samples = row[key]
            if len(samples) != REPEATS or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("Invalid/incomplete screen samples")
        # Same fixed graph and iteration count, so total-time ratio is valid.
        ratio = sum(row["decode4_seconds"]) / sum(row["candidate_seconds"])
        ratios[row["tokens"]].append(ratio)
    means = {"decode" if t == 1 else "prefill": math.exp(sum(map(math.log, values)) / len(values))
             for t, values in ratios.items()}
    worst = min(value for values in ratios.values() for value in values)
    eligible = worst >= MIN_CASE_RATIO and max(means.values()) >= MIN_PHASE_RATIO
    return {"eligible": eligible, "worst_case_ratio": worst, "geomean_ratios": means,
            "score": max(means.values()),
            "thresholds": {"minimum_each_case_ratio": MIN_CASE_RATIO, "minimum_one_phase_geomean_ratio": MIN_PHASE_RATIO},
            "boundary": "Synthetic screening ratios, not model speedups or statistical significance. Full-model gates/timings are still required."}


def check(library, candidate, output):
    if candidate not in W5_TILES:
        raise ValueError("Unknown W5 candidate")
    if output.exists():
        raise FileExistsError(output)
    report = {"protocol": "rq3-w5-kernel-screen-v1", "passed": False, "status": "running",
              "candidate": candidate, "baseline": "decode4", "rows": [],
              "settings": {"graphs_disabled": True, "event_profiling": False,
                           "repetitions": REPEATS, "iterations": ITERATIONS, "rounds": ROUNDS},
              "boundary": __doc__}
    write_json(output, report)
    try:
        runtime = NativeTests(library)
        counters = CudaDiagnostics(library)
        report["runtime_files"] = runtime_identity(library)
        for index, (rows, width) in enumerate(SHAPES):
            matrix = fixture(rows, width, seed=8419 + index)
            matrix_hash = hashlib.sha256(matrix.to_bytes()).hexdigest()
            rng = np.random.default_rng(2819 + index)
            signs = rng.choice(np.array([-1, 1], dtype=np.int8), width)
            for tokens in TOKEN_COUNTS:
                inputs = rng.normal(0, .5, (tokens, width)).astype(np.float16).astype(np.float32)
                input_hash = hashlib.sha256(inputs.tobytes() + signs.tobytes()).hexdigest()
                for turn in range(ROUNDS):
                    # Reverse order to expose order/thermal/cache sensitivity.
                    order = ("decode4", candidate) if turn == 0 else (candidate, "decode4")
                    values, dispatch = {}, {}
                    for selected in order:
                        print(f"SCREEN {candidate}: {rows}x{width} tokens={tokens} round={turn+1}/{ROUNDS} kernel={selected}", flush=True)
                        with kernel_environment(selected, graphs_disabled=True):
                            before = counters.snapshot()
                            values[selected] = runtime.operator("CUDA0", matrix, signs, inputs,
                                benchmark=(REPEATS, ITERATIONS))
                            delta = difference(before, counters.snapshot())
                            require_dispatch(delta, selected, tokens)
                            dispatch[selected] = delta
                    np.testing.assert_array_equal(values[candidate][0], values["decode4"][0])
                    row = {"rows": rows, "width": width, "tokens": tokens, "round": turn,
                           "order": order, "matrix_sha256": matrix_hash, "inputs_sha256": input_hash,
                           "decode4_seconds": values["decode4"][1], "candidate_seconds": values[candidate][1],
                           "exact_decode4": True, "dispatch": dispatch, "binding": counters.binding}
                    report["rows"].append(row)
                    write_json(output, report)
                    print("SCREEN pair:", json.dumps(row), flush=True)
        report.update(decision=decision(report["rows"]), passed=True, status="passed")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(W5_TILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.library, args.candidate, args.output)), flush=True)


if __name__ == "__main__":
    main()
