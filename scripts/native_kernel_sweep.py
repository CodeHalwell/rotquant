"""Cheap fresh operator gates and a bounded shortlist before retained-model work."""
from __future__ import annotations

import json
import time

from scripts.native_gpu_workflow import digest, write_json
from scripts.native_kernel_candidates import W5_TILES
from scripts.native_performance_study import kernel_environment


def run_screen(workflow, command, library, runtime, candidates, maximum):
    from scripts.run_rq3_kernel_screen import decision

    result = {"protocol": "rq3-w5-kernel-shortlist-v1", "completed": False, "baseline": "decode4",
              "started_utc_epoch": time.time(),
              "maximum_finalists": maximum, "runtime_files": runtime, "candidates": [], "finalists": [],
              "boundary": "A synthetic speed-screen shortlist, not a kernel promotion or measured full-model speedup."}
    destination = workflow.root / "kernel-shortlist.json"
    write_json(destination, result)
    for kernel in ("decode4", *candidates):
        def operators(stage, kernel=kernel):
            path = stage.directory / "report.json"
            with kernel_environment(kernel):
                command(stage, "check_rq3_gpu.py", "--library", library, "--backend", "CUDA0", "--output", path)
            record = json.loads(path.read_text())
            if (record.get("passed") is not True or record.get("runtime_files") != runtime
                    or record.get("settings", {}).get("rq3_kernel") != kernel):
                raise ValueError("Wrong/failed screen operator gate")
            return record, [path]
        workflow.stage(f"screen-operators-{kernel}", operators,
                       signature={"runtime": runtime, "kernel": kernel}, minutes=6, reuse=False)
        if kernel == "decode4":
            continue
        def microbench(stage, kernel=kernel):
            path = stage.directory / "report.json"
            command(stage, "run_rq3_kernel_screen.py", "--library", library, "--candidate", kernel, "--output", path)
            record = json.loads(path.read_text())
            if (record.get("passed") is not True or record.get("runtime_files") != runtime
                    or record.get("candidate") != kernel or record.get("baseline") != "decode4"):
                raise ValueError("Wrong/failed microbenchmark receipt")
            checked = decision(record["rows"])
            if checked != record.get("decision"):
                raise ValueError("Microbenchmark decision does not match its raw timings")
            return {"kernel": kernel, "report_path": str(path), "report_sha256": digest(path), **checked}, [path]
        measured = workflow.stage(f"screen-{kernel}", microbench,
                                  signature={"runtime": runtime, "kernel": kernel}, minutes=6, reuse=False)
        result["candidates"].append(measured)
        write_json(destination, result)
    ranked = sorted((x for x in result["candidates"] if x["eligible"]), key=lambda x: (-x["score"], x["kernel"]))
    result.update(completed=True, finalists=[x["kernel"] for x in ranked[:maximum]])
    result["outcome"] = "full_model_validation_required" if result["finalists"] else "no_promising_candidate"
    write_json(destination, result)
    print("KERNEL SHORTLIST:", json.dumps(result), flush=True)
    return result["finalists"]


def validate_candidates(candidates, maximum):
    if not candidates or len(set(candidates)) != len(candidates) or set(candidates) - set(W5_TILES):
        raise ValueError("Choose unique known W5 kernel candidates")
    if type(maximum) is not int or not 1 <= maximum <= 2:
        raise ValueError("Choose one or two full-model finalists")
