"""Extended common-input numerical stress probes; explicitly NOT task accuracy."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_rq3_model import execution_settings, numerical_metrics, runtime_identity
from scripts.native_gpu_workflow import write_json
from scripts.native_hashing import digest
from scripts.native_overnight_candidates import CANDIDATES, CONTROL, ExperimentDiagnostics
from scripts.rq3_test_runtime import NativeTests
from scripts.run_rq3_retained_gpu import verified_evidence

THRESHOLDS = {"max_abs_error": .125, "mean_abs_error": .005, "mean_kl": 1e-4, "top1_agreement": .99}
CONTEXTS = (128, 512, 2048)
DECODE = 16


def guards(metrics, trace, expected_trace):
    return {**{k: (metrics[k] >= v if k == "top1_agreement" else metrics[k] <= v)
               for k, v in THRESHOLDS.items()}, "exact_generation": trace == expected_trace}


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "report.json"
    kernel = os.environ.get("ROTQUANT_RQ3_KERNEL", "reference")
    if kernel not in (CONTROL, *CANDIDATES):
        raise ValueError("Unknown fidelity kernel")
    probes, _ = verified_evidence(args.source_arm, args.exported)
    identity = {"runtime_files": runtime_identity(args.library), "export_sha256": digest(args.exported / "export.json"),
                "probe_sha256": digest(probes), "contexts": CONTEXTS, "decode_steps": DECODE}
    # JSON roundtrip makes tuple/list comparisons stable across reference files.
    identity = json.loads(json.dumps(identity))
    report = {"protocol": "rq3-overnight-fidelity-v1", "passed": False, "identity": identity,
        "settings": execution_settings(), "rows": [], "thresholds": THRESHOLDS,
        "boundary": "Repeated saved token prefixes at longer lengths, 16 selected logits each and 16 greedy steps. Quantized tile8 is the reference, NOT FP16 teacher/task accuracy or a realistic long-context benchmark."}
    persist = lambda: write_json(report_path, report)
    persist()
    try:
        expected, original = {}, None
        if args.reference:
            original = json.loads(args.reference.read_text())
            if (original.get("passed") is not True or original.get("identity") != identity
                    or original.get("settings", {}).get("rq3_kernel") != CONTROL
                    or original.get("logits_sha256") != digest(Path(original["logits_path"]))):
                raise ValueError("Reference fidelity receipt mismatch")
            expected = load_file(original["logits_path"])
            report["reference_sha256"] = digest(args.reference)
        elif kernel != CONTROL:
            raise ValueError("Candidate requires frozen control logits")
        saved = load_file(str(probes))
        keys = sorted(k for k in saved if k.endswith(".input.input_ids"))
        if not 1 <= len(keys) <= 8:
            raise ValueError("Expected 1..8 saved prefixes")
        runtime, logits = NativeTests(args.library), {}
        diag = ExperimentDiagnostics(args.library) if kernel in CANDIDATES else None
        before = diag.snapshot(kernel) if diag else None
        os.environ["ROTQUANT_REQUIRE_GPU"] = "1"
        with runtime.model(args.exported / "model.gguf", "CUDA0", 2304) as model:
            for key in keys:
                ids = saved[key]
                if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 128:
                    raise ValueError("Invalid saved input shape")
                for context in CONTEXTS:
                    label = f"{key}-{context}"
                    prompt = np.resize(ids[0], context)
                    out = model.evaluate(prompt, reset=True, selected=16)
                    selected = out.copy()
                    trace = []
                    for _ in range(DECODE):
                        token = int(out[-1].argmax()) if out.ndim == 2 else int(out.argmax())
                        trace.append(token)
                        if model.is_eog(token):
                            break
                        out = model.evaluate([token])
                    row = {"id": label, "input_sha256": __import__("hashlib").sha256(prompt.tobytes()).hexdigest(),
                           "positions": 16, "trace": trace}
                    if original:
                        target = next(r for r in original["rows"] if r["id"] == label)
                        if target["input_sha256"] != row["input_sha256"]:
                            raise ValueError("Input mismatch")
                        row["metrics"] = numerical_metrics(selected, expected[label])
                        row["guards"] = guards(row["metrics"], trace, target["trace"])
                        row["passed"] = all(row["guards"].values())
                    else:
                        if not np.isfinite(selected).all():
                            raise ValueError("Non-finite reference logits")
                        logits[label] = selected
                        row["passed"] = True
                    report["rows"].append(row)
                    persist()
                    print(f"FIDELITY {kernel}: {label}, passed={row['passed']}", flush=True)
            if model.custom_ops <= 0:
                raise ValueError("No native model execution")
        if diag:
            report["experimental_dispatch"] = diag.require(before, kernel)
        if not original:
            args.logits_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(logits, str(args.logits_path))
            report.update(logits_path=str(args.logits_path), logits_sha256=digest(args.logits_path))
        report["passed"] = all(r["passed"] for r in report["rows"])
        report["status"] = "passed" if report["passed"] else "numerical_rejected"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "source-arm", "export", "output-dir"):
        parser.add_argument("--" + name, dest="exported" if name == "export" else name.replace("-", "_"), type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--logits-path", type=Path)
    args = parser.parse_args()
    if not args.reference and not args.logits_path:
        parser.error("Control capture requires --logits-path")
    sys.exit(run(args))
