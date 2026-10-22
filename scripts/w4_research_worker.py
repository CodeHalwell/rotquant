"""One model per process for the resumable W4 research workflow."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rotquant.checkpoint import load_packed_model, verify_checkpoint
from rotquant.utils import set_seed
from scripts.check_rq3_model import numerical_metrics, runtime_identity
from scripts.native_gpu_workflow import artifacts, write_json
from scripts.w4_research_data import freeze, validate_manifest
from scripts.w4_research_model import load_source, prepare_arm
from scripts.w4_research_plan import Arm


class Worker:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(args.config.read_text())
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(args.data.read_text()) if args.data else None
        if self.data is not None:
            validate_manifest(self.data)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        set_seed(self.config["seed"])
        os.environ["ROTQUANT_TOKEN_CACHE_DIR"] = str(args.root / "token_cache")

    def run(self):
        result = getattr(self, self.args.phase)()
        write_json(self.output / "report.json", {"status": "complete", **result})

    def freeze(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"], revision=self.config["model_revision"])
        manifest = freeze(self.config, tokenizer, self.output)
        return {"data_path": str(self.output / "data.json"), "fingerprint": manifest["fingerprint"]}

    def teacher(self):
        from scripts.w4_research_measure import HFTeacher
        model, tokenizer, processor, _ = load_source(self.config)
        source = self.output / "source"
        model.save_pretrained(source, safe_serialization=True)
        tokenizer.save_pretrained(source)
        processor.save_pretrained(source)
        backend = HFTeacher(model, self.config["device"])
        directory = self.output / "logits"
        directory.mkdir()
        for part in ("selection", "test", "long_test"):
            for item in self.data["documents"][part]:
                print(f"TEACHER {item['id']}", flush=True)
                logits = backend.predict(item["input_ids"])
                if not np.isfinite(logits).all():
                    raise ValueError("non-finite FP16 teacher logits")
                np.save(directory / f"{item['id']}.npy", logits.astype(np.float16))
        self.save_probes(backend)
        from scripts import public_task_suite
        scorer = public_task_suite.prepare_scorers(self.output / "scorer")
        return {"source": str(source), "teacher_dir": str(directory),
                "probes": str(self.output / "probes.npz"), "scorer": scorer,
                "scorer_dir": str(self.output / "scorer"),
                "artifact_bytes": sum(p.stat().st_size for p in source.rglob("*") if p.is_file())}

    def save_probes(self, backend):
        probes = {item["id"]: backend.predict(item["input_ids"])[-16:]
                  for item in self.data["documents"]["selection"][:2]}
        np.savez(self.output / "probes.npz", **probes)

    def selection(self, model, tokenizer, device, destination):
        from scripts.w4_research_measure import HFTeacher, evaluate_quality
        backend = HFTeacher(model, device)
        report = evaluate_quality(destination / "selection", backend,
                                  self.data["documents"]["selection"], self.args.teacher_dir)
        self.save_probes(backend)
        return report["summary"]

    def prepare(self):
        arm = Arm(**json.loads(self.args.arm.read_text()))
        result = prepare_arm(arm, self.config, self.data, self.args.root,
                             self.output, self.args.identity, self.selection)
        return {**result, "checkpoint": str(self.output / "checkpoint"),
                "probes": str(self.output / "probes.npz")}

    def reload(self):
        from scripts.w4_research_measure import HFTeacher
        verify_checkpoint(self.args.checkpoint)
        model = load_packed_model(self.args.checkpoint, device=self.config["device"],
                                  dtype=torch.float16, fallback=True)
        backend = HFTeacher(model, self.config["device"])
        return {"parity": self.parity(lambda ids: backend.predict(ids)[-16:], reload=True)}

    def parity(self, predict, *, reload=False):
        rows = []
        with np.load(self.args.probes) as expected:
            for item in self.data["documents"]["selection"][:2]:
                metrics = numerical_metrics(predict(item["input_ids"]), expected[item["id"]])
                limits = (.002, .0001, 1e-6, 1.) if reload else (.125, .005, 1e-4, .99)
                passed = all(metrics[key] <= limit for key, limit in zip(
                    ("max_abs_error", "mean_abs_error", "mean_kl"), limits[:3]))
                passed = passed and metrics["top1_agreement"] >= limits[3]
                rows.append({"id": item["id"], "metrics": metrics, "passed": passed})
        write_json(self.output / "parity.json", {"passed": all(r["passed"] for r in rows), "rows": rows})
        if not rows or not all(r["passed"] for r in rows):
            raise ValueError("deployed model failed parity; quality/speed not accepted")
        return rows

    def export(self):
        from scripts.export_rotquant_gguf_v2 import export_checkpoint
        output = self.output / "native"
        report = export_checkpoint(self.args.checkpoint, output, self.args.llama_dir)
        return {"native": str(output), "artifact_bytes": report["complete_model_payload_bytes"]}

    def export_fp16(self):
        from scripts.w4_research_export import export_fp16
        output = self.output / "native"
        report = export_fp16(self.args.source, output, self.args.llama_dir, self.args.quant_export)
        return {"native": str(output), "artifact_bytes": report["complete_model_payload_bytes"]}

    def native(self):
        from transformers import AutoTokenizer

        from scripts import public_task_suite
        from scripts.rq3_test_runtime import NativeTests
        from scripts.run_qwen35_fresh_eval import stop_ids_for_source
        from scripts.w4_research_measure import NativeBackend
        tokenizer = AutoTokenizer.from_pretrained(self.args.source)
        scorer = public_task_suite.IFEvalScorer(self.args.scorer_dir)
        os.environ["ROTQUANT_REQUIRE_GPU"] = "1"
        os.environ["ROTQUANT_RQ3_KERNEL"] = "reference"
        os.environ.pop("ROTQUANT_RQ3_PROFILE", None)
        os.environ.pop("GGML_CUDA_DISABLE_GRAPHS", None)
        study = self.config["research"]
        capacity = max(study["long_context"] + max(64, study["speed_decode_steps"]), 4160)
        runtime = NativeTests(self.args.library)
        with runtime.model(self.args.native / "model.gguf", "CUDA0", capacity) as model:
            backend = NativeBackend(model, stop_ids_for_source(tokenizer))
            self.parity(lambda ids: backend.model.evaluate(ids[:-1], reset=True, selected=16))
            result = self.native_results(backend.model, backend, tokenizer, scorer)
            if bool(model.custom_ops) != (self.args.label != "fp16"):
                raise ValueError("native dispatch does not match the expected model format")
        return {**result, "label": self.args.label, "runtime": runtime_identity(self.args.library),
                "native_artifacts": artifacts(self.args.native.glob("*.gguf")),
                "data_fingerprint": self.data["fingerprint"], "backend": "CUDA0/reference/chunk16"}

    def native_results(self, model, backend, tokenizer, scorer):
        from scripts.w4_research_measure import (
            evaluate_public_tasks,
            evaluate_quality,
            native_speeds,
        )
        quality = {}
        for part in ("test", "long_test"):
            quality[part] = evaluate_quality(self.output / part, backend,
                                            self.data["documents"][part], self.args.teacher_dir)
        tasks = evaluate_public_tasks(self.output / "tasks", backend,
                                      self.data["public_items"], tokenizer, ifeval=scorer)
        speed = native_speeds(model, self.data["speed_inputs"],
                              self.config["research"]["speed_repetitions"], self.output / "speed.json")
        return {"quality": quality, "tasks": tasks, "speed": speed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "teacher", "prepare", "reload", "export", "export_fp16", "native"))
    for name in ("config", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("data", "arm", "teacher-dir", "checkpoint", "probes", "llama-dir", "source",
                 "quant-export", "native", "library", "scorer-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--label")
    Worker(parser.parse_args()).run()


if __name__ == "__main__":
    main()
