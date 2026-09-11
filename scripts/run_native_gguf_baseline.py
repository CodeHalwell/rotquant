"""Pinned conventional GGUF controls through the SAME native model test bridge.

No llama-cpp-python build. This is a fixed-token text-throughput comparison,
not an accuracy evaluation or a quality/size-matched compression claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_rq3_model import execution_settings
from scripts.native_gpu_workflow import digest, write_json
from scripts.rq3_test_runtime import NativeTests
from scripts.run_rq3_performance_pilot import measure, reference_tokens, summarize, verify_inputs
from scripts.run_rq3_retained_gpu import ProcessVRAM

# Same immutable releases as the project's earlier KL comparison, without its
# heavy experiment-runner imports or its different llama.cpp Python engine.
REPO = "unsloth/Qwen3.5-4B-GGUF"
REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
RELEASES = {
    "bf16": {"name": "Qwen3.5-4B-BF16.gguf", "bytes": 8424393632,
             "sha256": "9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a"},
    "ud_q4": {"name": "Qwen3.5-4B-UD-Q4_K_XL.gguf", "bytes": 2912109728,
              "sha256": "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7"},
}


def verify_file(path, release):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != release["bytes"] or digest(path) != release["sha256"]:
        raise ValueError("Conventional GGUF does not match its pinned release")


def prepare(args):
    from huggingface_hub import hf_hub_download

    release = RELEASES[args.baseline]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = args.artifact_dir / release["name"]
    if not path.exists():
        if shutil.disk_usage(args.artifact_dir).free < 2 * release["bytes"] + 256 * 1024**2:
            raise ValueError("Insufficient free disk for conventional GGUF download/staging")
        path = Path(hf_hub_download(repo_id=REPO, revision=REVISION,
                                   filename=release["name"], local_dir=args.artifact_dir))
    verify_file(path, release)
    record = {"protocol": "rq3-pinned-gguf-control-v1", "passed": True,
              "baseline": args.baseline, "repo": REPO, "revision": REVISION,
              "path": str(path.resolve()), "release": release}
    write_json(args.output, record)


def token_identity(path, llama_dir):
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    field = reader.get_field("tokenizer.ggml.tokens")
    if field is None:
        raise ValueError("GGUF tokenizer vocabulary missing")
    # Length-prefix raw token strings to preserve an unambiguous ID -> bytes map.
    value = hashlib.sha256()
    for index in field.data:
        token = reader.fields["tokenizer.ggml.tokens"].parts[index].tobytes()
        value.update(len(token).to_bytes(8, "little"))
        value.update(token)
    architecture = reader.get_field("general.architecture").contents()
    blocks = reader.get_field(f"{architecture}.block_count").contents()
    return {"vocabulary_sha256": value.hexdigest(), "vocabulary_size": len(field.data),
            "architecture": architecture, "blocks": blocks}


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = {"protocol": "rq3-native-gguf-speed-control-v1", "passed": False, "status": "running",
              "measurement_kind": "throughput", "baseline": args.baseline,
              "settings": execution_settings(), "rows": [], "summary": None,
              "boundary": __doc__}
    memory = ProcessVRAM()
    path = args.output_dir / "report.json"
    def persist(rows=None, current=None):
        if rows is not None:
            report.update(rows=rows, in_progress=current, summary=summarize(rows))
        report["memory"] = memory.report()
        write_json(path, report)
    persist()
    try:
        if report["settings"]["rq3_profile"] or report["settings"]["rq3_kernel"] != "reference":
            raise ValueError("Conventional controls require the uninstrumented reference settings")
        gate, identity = verify_inputs(args.library, args.exported, args.parity_report, args.probes)
        if gate["backend"] != "CUDA0":
            raise ValueError("Production conventional comparison requires CUDA0")
        reference = json.loads(args.reference_report.read_text())
        fixed = reference_tokens(reference)
        if (reference["runtime_files"] != identity or reference["settings"] != report["settings"]
                or reference["export_sha256"] != gate["export_sha256"]
                or reference["probe_sha256"] != gate["probe_sha256"]
                or reference["parity_report_sha256"] != digest(args.parity_report)):
            raise ValueError("Reference model/runtime/settings/evidence mismatch")
        receipt = json.loads(args.baseline_receipt.read_text())
        if (receipt.get("passed") is not True or receipt.get("baseline") != args.baseline
                or receipt.get("release") != RELEASES[args.baseline]
                or receipt.get("repo") != REPO or receipt.get("revision") != REVISION):
            raise ValueError("Baseline provenance mismatch")
        model_path = Path(receipt["path"])
        verify_file(model_path, RELEASES[args.baseline])
        tokenizer = token_identity(model_path, args.llama_dir)
        if tokenizer != token_identity(args.exported / "model.gguf", args.llama_dir):
            raise ValueError("Baseline and RotQuant have different token-ID maps or architectures")
        controls, context = reference["controls"], reference["context"]
        ids = load_file(str(args.probes))["p0.input.input_ids"][0].numpy()
        prompt_hash = hashlib.sha256(np.resize(ids, context).astype("<i4").tobytes()).hexdigest()
        if prompt_hash != reference["prompt_ids_sha256"]:
            raise ValueError("Reference input hash mismatch")
        report.update(context=context, context_capacity=reference["context_capacity"], controls=controls,
                      runtime_files=identity, backend=gate["backend"], cpu_fallback_forbidden=True,
                      prompt_ids_sha256=prompt_hash, token_identity=tokenizer,
                      text_model_bytes=RELEASES[args.baseline]["bytes"],
                      reference_report_sha256=digest(args.reference_report),
                      baseline_receipt_sha256=digest(args.baseline_receipt),
                      model_sha256=RELEASES[args.baseline]["sha256"])
        os.environ["ROTQUANT_REQUIRE_GPU"] = "1"
        runtime = NativeTests(args.library)
        with memory:
            start = time.perf_counter()
            with runtime.model(model_path, gate["backend"], report["context_capacity"]) as model:
                report["load_seconds"] = time.perf_counter() - start
                if model.vocab != tokenizer["vocabulary_size"]:
                    raise ValueError("Runtime vocabulary mismatch")
                measure(model, ids, context=context, decode=controls["decode_steps"],
                        repetitions=controls["repetitions"], minimum_tps=controls["min_decode_tps"],
                        maximum_vram=controls["max_process_vram_mib"], memory=memory,
                        persist=persist, forced_decode=fixed)
                if model.custom_ops != 0:
                    raise ValueError("Conventional control unexpectedly executed RotQuant custom operators")
        if not memory.samples or memory.report()["sampled_peak_process_vram_mib"] > controls["max_process_vram_mib"]:
            raise ValueError("Conventional control did not pass the process VRAM guard")
        report.update(passed=True, status="passed")
        persist()
    except BaseException as error:
        report.update(passed=False, status="stopped", error=f"{type(error).__name__}: {error}")
        persist()
        raise
    print(json.dumps({"baseline": args.baseline, "summary": report["summary"], "memory": report["memory"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--baseline", choices=tuple(RELEASES), required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path)
    for name in ("library", "export", "parity-report", "probes", "reference-report", "baseline-receipt", "llama-dir", "output-dir"):
        parser.add_argument("--" + name, dest="exported" if name == "export" else name.replace("-", "_"), type=Path)
    args = parser.parse_args()
    required = ("artifact_dir", "output") if args.phase == "prepare" else (
        "library", "exported", "parity_report", "probes", "reference_report", "baseline_receipt", "llama_dir", "output_dir")
    if any(getattr(args, name) is None for name in required):
        parser.error(f"{args.phase} requires: {', '.join(required)}")
    def terminate(signum, frame):
        raise KeyboardInterrupt("Conventional control interrupted; partial results preserved")
    signal.signal(signal.SIGTERM, terminate)
    (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
