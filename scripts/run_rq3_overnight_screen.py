"""Isolated, fail-closed numerical and speed screens for the overnight registry."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rotquant.native_v3 import decode_native_v3_rows
from rotquant.rotate import RandomizedHadamard
from scripts.check_rq3_model import runtime_identity
from scripts.native_gpu_workflow import write_json
from scripts.native_overnight_candidates import (
    CANDIDATES,
    CONTROL,
    OPERATOR_ATOL,
    OPERATOR_RTOL,
    REGISTRY_VERSION,
    ExperimentDiagnostics,
    experimental_dispatch,
    requested_scratch,
)
from scripts.native_performance_study import kernel_environment
from scripts.rq3_test_runtime import NativeTests
from scripts.run_rq3_kernel_screen import fixture


class NumericalRejection(Exception):
    pass


def parity(actual, reference, exact):
    if not np.isfinite(actual).all():
        raise NumericalRejection("Non-finite candidate output")
    try:
        if exact:
            np.testing.assert_array_equal(actual, reference)
        else:
            np.testing.assert_allclose(actual, reference, rtol=OPERATOR_RTOL, atol=OPERATOR_ATOL)
    except AssertionError as error:
        raise NumericalRejection(str(error)) from error
    return {"exact": bool(np.array_equal(actual, reference)),
            "max_abs_error": float(np.abs(actual-reference).max())}


def canonical(matrix, rotation, inputs, mode, rows=None, columns=None):
    weights = torch.from_numpy(decode_native_v3_rows(matrix))
    if mode == 2:
        return rotation.inverse_activation(weights).half()[inputs].float().numpy()
    x = torch.from_numpy(inputs).half()
    if mode == 1:
        w = rotation.inverse_activation(weights).half()
    else:
        x = rotation.rotate_activation(x[:, columns] if columns is not None else x)
        w = weights[rows].half() if rows is not None else weights.half()
    return F.linear(x, w).float().numpy()


def operator_cases():
    for bits in range(1, 9):
        for scale_bits in (8, 16):
            for tokens in (1, 2, 3, 4, 7, 17):
                yield (bits, scale_bits, 5 if tokens % 2 else 137, 256, tokens,
                       7 if bits % 2 else 256)
    for rows in (1, 2, 3, 4, 5, 127, 129):
        yield 5, 8, rows, 512, 4, 256
    for width in (2560, 9216, 11008):
        yield 5, 8, 5, width, 7, 256


def check_operators(runtime, diagnostics, candidate, report, persist):
    rng = np.random.default_rng(9483)
    for index, (bits, sbits, rows, width, tokens, block) in enumerate(operator_cases()):
        matrix = fixture(rows, width, 9483+index, bits=bits, scale_bits=sbits, block=block)
        rotation = RandomizedHadamard(width, block=128, seed=2701)
        signs = rotation.signs.numpy().astype(np.int8)
        x = rng.normal(0, (0.05, 0.5, 2)[index % 3], (tokens, width)).astype(np.float16).astype(np.float32)
        for mode in (0, 1, 2):
            rp = rng.permutation(rows).astype(np.int32) if mode == 0 and index % 2 else None
            cp = rng.permutation(width).astype(np.int32) if rp is not None else None
            inputs = np.arange(tokens, dtype=np.int32) % rows if mode == 2 else x
            expected = canonical(matrix, rotation, inputs, mode, rp, cp)
            with kernel_environment("reference", graphs_disabled=True):
                reference = runtime.operator("CUDA0", matrix, signs, inputs, mode, rp, cp)
            # A broken shared reference is fatal, NOT a rejected candidate.
            np.testing.assert_allclose(reference, expected, rtol=OPERATOR_RTOL, atol=OPERATOR_ATOL)
            with kernel_environment(candidate, graphs_disabled=True):
                before = diagnostics.snapshot(candidate)
                actual = runtime.operator("CUDA0", matrix, signs, inputs, mode, rp, cp)
                dispatch = diagnostics.require(before, candidate, mode=mode, tokens=tokens)
            exact = CANDIDATES[candidate]["lane"] == "exact" or not experimental_dispatch(candidate, mode, tokens)
            result = parity(actual, reference, exact)
            parity(actual, expected, False)
            report["rows"].append({"bits": bits, "scale_bits": sbits, "block": block,
                "outputs": rows, "width": width, "tokens": tokens, "mode": mode,
                "permuted": rp is not None, "dispatch": dispatch, **result})
        persist()
        print(f"OPERATORS {candidate}: {index+1}/{len(list(operator_cases()))} fixtures; {len(report['rows'])} cases passed", flush=True)


def summarize_pairs(rows):
    if not rows or len(rows) % 2:
        raise ValueError("Incomplete paired screen")
    groups = {}
    for row in rows:
        for samples in (row["control_seconds"], row["candidate_seconds"]):
            if len(samples) != 3 or any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError("Invalid screen measurements")
        key = (row["outputs"], row["width"], row["tokens"], row["mode"])
        groups.setdefault(key, []).append(row)
    if any(sorted(r["round"] for r in pair) != [0, 1] for pair in groups.values()):
        raise ValueError("Missing counterbalanced pair")
    ratios = [sum(r["control_seconds"])/sum(r["candidate_seconds"]) for r in rows]
    mean = math.exp(sum(map(math.log, ratios))/len(ratios))
    return {"eligible": min(ratios) >= .95 and mean >= 1.05,
            "score": mean, "worst_ratio": min(ratios),
            "thresholds": {"minimum_pair_ratio": .95, "minimum_geomean_ratio": 1.05},
            "boundary": "Warm synthetic shape screening; not a full-model speedup or significance test."}


def check_speed(runtime, diagnostics, candidate, report, persist):
    head = CANDIDATES[candidate]["head"] and not CANDIDATES[candidate]["rows"]
    # Checkpoint-representative backbone dimensions, NOT frequency weighted.
    # The head has the correct width but bounded row count; full vocabulary is
    # assessed by retained execution, not extrapolated from this warm fixture.
    shapes = ((4096, 2560),) if head else ((2560, 2560), (9216, 2560), (2560, 9216))
    counts = (1, 16) if head else (16, 128)
    for index, (rows, width) in enumerate(shapes):
        matrix = fixture(rows, width, 2903+index, bits=6 if head else 5, scale_bits=16 if head else 8)
        rng = np.random.default_rng(2293+index)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), width)
        for tokens in counts:
            requested_scratch(candidate, rows, width, tokens)
            x = rng.normal(0, .5, (tokens, width)).astype(np.float16).astype(np.float32)
            for turn, order in enumerate(((CONTROL, candidate), (candidate, CONTROL))):
                values, dispatch = {}, {}
                for kernel in order:
                    print(f"SPEED {candidate}: {rows}x{width}, tokens={tokens}, pair={turn}, kernel={kernel}", flush=True)
                    with kernel_environment(kernel, graphs_disabled=True):
                        before = diagnostics.snapshot(candidate)
                        values[kernel] = runtime.operator("CUDA0", matrix, signs, x, mode=int(head), benchmark=(3, 3))
                        if kernel == candidate:
                            dispatch = diagnostics.require(before, candidate, mode=int(head), tokens=tokens)
                metrics = parity(values[candidate][0], values[CONTROL][0], CANDIDATES[candidate]["lane"] == "exact")
                report["rows"].append({"outputs": rows, "width": width, "tokens": tokens, "mode": int(head),
                    "round": turn, "order": order, "control_seconds": values[CONTROL][1],
                    "candidate_seconds": values[candidate][1], "iterations": 3, "dispatch": dispatch,
                    "matrix_sha256": hashlib.sha256(matrix.to_bytes()).hexdigest(),
                    "input_sha256": hashlib.sha256(x.tobytes()+signs.tobytes()).hexdigest(), **metrics})
                persist()
    report["decision"] = summarize_pairs(report["rows"])


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"protocol": REGISTRY_VERSION, "candidate": args.candidate, "baseline": CONTROL,
        "phase": args.phase, "passed": False, "status": "running", "rows": [],
        "contract": CANDIDATES[args.candidate], "tolerances": {"rtol": OPERATOR_RTOL, "atol": OPERATOR_ATOL}}
    persist = lambda: write_json(args.output, report)
    persist()
    try:
        torch.set_num_threads(2)
        runtime, diagnostics = NativeTests(args.library), ExperimentDiagnostics(args.library)
        report["runtime_files"] = runtime_identity(args.library)
        (check_operators if args.phase == "operators" else check_speed)(runtime, diagnostics, args.candidate, report, persist)
        report.update(passed=True, status="passed")
    except NumericalRejection as error:
        report.update(status="numerical_rejected", error=str(error))
        return 2
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--phase", choices=("operators", "speed"), required=True)
    sys.exit(run(parser.parse_args()))
