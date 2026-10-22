"""Isolated preparation, recovery and native measurement for the ternary study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_gpu_workflow import artifacts, write_json
from scripts.ternary_research_plan import Arm
from scripts.w4_research_worker import Worker


class TernaryWorker(Worker):
    def prepare(self):
        from scripts.ternary_research_model import prepare_arm
        arm = Arm(**json.loads(self.args.arm.read_text()))
        result = prepare_arm(arm, self.config, self.data, self.args.root, self.output,
                             self.args.identity, self.selection, parent=self.args.parent)
        return {**result, "checkpoint": str(self.output / "checkpoint"),
                "probes": str(self.output / "probes.npz")}

    def export(self):
        return self.export_sizes(super().export())

    def export_fp16(self):
        return self.export_sizes(super().export_fp16())

    def native_results(self, model, backend, tokenizer, scorer):
        # Reached only after native parity, or Bonsai's tokenizer/reset gate.
        # Persist the binding before measuring so a later timeout can retain
        # completed quality/task reports without accepting unrelated files.
        bonsai = self.args.phase == "bonsai"
        label = "bonsai2_4b" if bonsai else self.args.label
        if not bonsai and bool(model.custom_ops) != (label != "fp16"):
            raise ValueError("native dispatch does not match the expected model format")
        write_json(self.output / "measurement-context.json", {
            "protocol": "ternary-measurement-context-v1", "identity": self.args.identity,
            "data_fingerprint": self.data["fingerprint"], "label": label,
            "backend": "CUDA0/Prism/chunk16" if bonsai else "CUDA0/reference/chunk16",
            "prefill_chunk_tokens": model.chunk_tokens})
        return super().native_results(model, backend, tokenizer, scorer)

    @staticmethod
    def export_sizes(result):
        return {**result, "language_payload_bytes": (Path(result["native"]) / "model.gguf").stat().st_size,
                "scope": "language GGUF plus retained source auxiliary payload"}

    def bonsai(self):
        from transformers import AutoTokenizer

        from scripts import public_task_suite
        from scripts.run_qwen35_fresh_eval import stop_ids_for_source
        from scripts.ternary_research_bonsai import (
            BonsaiRuntime,
            smoke_model,
            validate_tokenizer,
            verify_assets,
        )
        from scripts.w4_research_measure import NativeBackend

        receipt = verify_assets(self.args.assets, self.config["bonsai"]["packing"])
        mapping = validate_tokenizer(receipt["model"], self.args.reference_gguf)
        tokenizer = AutoTokenizer.from_pretrained(self.args.source)
        scorer = public_task_suite.IFEvalScorer(self.args.scorer_dir)
        capacity = max(self.config["research"]["long_context"] +
                       max(64, self.config["research"]["speed_decode_steps"]), 4160)
        runtime = BonsaiRuntime(self.args.library)
        with runtime.model(receipt["model"], "CUDA0", capacity) as model:
            backend = NativeBackend(model, stop_ids_for_source(tokenizer))
            smoke = smoke_model(backend.model, self.data["documents"]["selection"][0]["input_ids"])
            if model.vocab != mapping["vocabulary_size"]:
                raise ValueError("Bonsai runtime/tokenizer vocabulary size mismatch")
            if any(not model.is_eog(token) for token in backend.stop_ids):
                raise ValueError("Bonsai and source stop-token semantics disagree")
            result = self.native_results(backend.model, backend, tokenizer, scorer)
        return {**result, "label": "bonsai2_4b", "backend": "CUDA0/Prism/chunk16",
                "artifacts": receipt, "tokenizer_parity": mapping, "smoke": smoke,
                "runtime": artifacts([self.args.library]), "data_fingerprint": self.data["fingerprint"],
                "dispatch": "All layers requested on CUDA; Prism operation fallback is not audited by RQ3 counters"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "teacher", "prepare", "reload", "export", "export_fp16", "native", "bonsai"))
    for name in ("config", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--identity", required=True)
    for name in ("data", "arm", "teacher-dir", "checkpoint", "probes", "llama-dir", "source",
                 "quant-export", "library", "native", "scorer-dir", "parent", "assets", "reference-gguf"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    try:
        TernaryWorker(args).run()
    except Exception as error:
        write_json(args.output / "report.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
