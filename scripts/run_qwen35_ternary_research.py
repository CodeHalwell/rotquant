"""Compare ternary block QAT/recovery with Bonsai, FP16 and W4/W5/W6."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_gpu_workflow import (
    Workflow,
    artifacts,
    artifacts_match,
    fingerprint,
    write_json,
)
from scripts.run_qwen35_w4_research import ResearchStudy, read
from scripts.ternary_research_plan import (
    DEFAULT_CONFIG,
    LABELS,
    arm_record,
    arms,
    bind_run,
    validate_config,
)


class TernaryStudy(ResearchStudy):
    def __init__(self, args, config):
        from scripts.run_qwen35_vocabulary_budget import runtime_identity
        self.args, self.config = args, config
        self.root, self.work = args.output_dir.resolve(), args.work_dir.resolve()
        self.identity = bind_run(self.root, config, runtime_identity())
        self.config_path = self.root / "config.json"
        write_json(self.config_path, config)
        self.workflow = Workflow(self.root, ROOT, {"identity": self.identity}, args.budget_minutes,
                                 overnight=args.budget_minutes > 180)
        self.data, self.teacher, self.runtime = None, None, None
        self.prepared, self.exported, self.measured, self.failures = {}, {}, {}, {}

    def worker_options(self, **options):
        defaults = {"config": self.config_path, "root": self.root, "identity": self.identity}
        if self.data:
            defaults["data"] = self.data["data_path"]
        defaults.update(options)
        return defaults

    def worker(self, name, phase, *, minutes=90, **options):
        defaults = self.worker_options(**options)

        def action(stage):
            arguments = [str(value) for key, val in defaults.items()
                         for value in ("--" + key.replace("_", "-"), val)]
            try:
                self.command(stage, "ternary_research_worker.py", phase, "--output", stage.directory, *arguments)
            except BaseException as error:
                if phase in {"native", "bonsai"}:
                    self.preserve_partial(stage.directory, "bonsai2_4b" if phase == "bonsai" else options["label"], error)
                raise
            return read(stage.directory / "report.json"), self.stage_files(stage.directory)

        result = self.workflow.stage(name, action,
                                     signature={k: str(v) for k, v in defaults.items()}, minutes=minutes)
        if phase in {"native", "bonsai"}:
            self.failures.pop("bonsai2_4b" if phase == "bonsai" else options["label"], None)
        return result

    def preserve_partial(self, directory, label, error, *, row=None):
        from scripts.ternary_research_results import partial_measurement
        expected = {"protocol": "ternary-measurement-context-v1", "identity": self.identity,
                    "data_fingerprint": self.data["fingerprint"], "label": label,
                    "backend": "CUDA0/Prism" if label == "bonsai2_4b" else "CUDA0/reference"}
        try:
            partial = partial_measurement(directory, expected, f"{type(error).__name__}: {error}")
            if partial is not None:
                result, files = partial
                record = self.workflow.state["attempts"][-1] if row is None else row
                record["partial"] = {"value": result, "artifacts": artifacts(files)}
                self.measured[label] = result
        except (OSError, ValueError):
            # Preserve the original stage error if a killed worker left a
            # partially written file, or storage itself has failed.
            return

    def request_record(self, name, signature):
        previous = [row for row in self.workflow.state["attempts"] if row["name"] == name]
        if not previous:
            return None
        row = previous[-1]
        return row if row["signature"] == fingerprint(signature) else None

    def saved(self, name, signature, *, partial=False):
        row = self.request_record(name, signature)
        if row is None:
            return None
        if row["status"] == "passed":
            record = row
        elif partial and row["status"] in {"failed", "interrupted"}:
            record = row.get("partial", {})
        else:
            return None
        if record.get("artifacts") and artifacts_match(record["artifacts"]):
            return record["value"]
        return None

    def saved_worker(self, name, *, partial=False, **options):
        signature = {key: str(value) for key, value in self.worker_options(**options).items()}
        return self.saved(name, signature, partial=partial)

    def native_options(self, label, probes):
        return {"native": self.exported[label]["native"], "probes": probes,
                "library": self.runtime["library"], "source": self.teacher["source"],
                "teacher_dir": self.teacher["teacher_dir"], "scorer_dir": self.teacher["scorer_dir"],
                "label": label}

    def restore_measurement(self, name, **options):
        signature = {key: str(value) for key, value in self.worker_options(**options).items()}
        row = self.request_record("measure-" + name, signature)
        if row and row["status"] == "interrupted" and "partial" not in row:
            # An unclean kernel exit cannot write the controller's partial
            # receipt. Recover only its ledger-bound attempt and deployment
            # context; subsequent resumes require the resulting file hashes.
            directory = Path(row["directory"])
            expected_parent = self.root / "stages" / ("measure-" + name)
            if directory.parent == expected_parent and directory.name.startswith("attempt-"):
                self.preserve_partial(directory, name, RuntimeError("measurement process interrupted"), row=row)
        result = self.saved_worker("measure-" + name, partial=True, **options)
        if result:
            self.measured[name] = result
            if result.get("status") == "failed":
                self.failures[name] = result["error"]

    def hydrate(self):
        """Reconstruct only the current dependency chain from verified receipts."""
        self.data = self.saved_worker("freeze")
        self.runtime = self.saved("runtime", self.identity)
        if not self.data or not self.runtime:
            return
        self.teacher = self.saved_worker("teacher")
        if not self.teacher:
            return
        for arm in arms():
            self.hydrate_arm(arm)
        if "w5_v6" not in self.exported:
            return
        fp16 = self.saved_worker("export-fp16", source=self.teacher["source"],
                                 quant_export=self.exported["w5_v6"]["native"], llama_dir=self.runtime["source"])
        if fp16:
            self.exported["fp16"] = fp16
            self.restore_measurement("fp16", **self.native_options("fp16", self.teacher["probes"]))
            self.hydrate_bonsai()

    def hydrate_arm(self, arm):
        if arm.parent and arm.parent not in self.prepared:
            return
        options = {"parent": self.prepared[arm.parent]["checkpoint"]} if arm.parent else {}
        prepared = self.saved_worker("prepare-" + arm.name, arm=self.root / "arms" / f"{arm.name}.json",
                                     teacher_dir=self.teacher["teacher_dir"], **options)
        if not prepared or not self.saved_worker("reload-" + arm.name, checkpoint=prepared["checkpoint"],
                                                  probes=prepared["probes"]):
            return
        self.prepared[arm.name] = prepared
        exported = self.saved_worker("export-" + arm.name, checkpoint=prepared["checkpoint"],
                                     llama_dir=self.runtime["source"])
        if exported:
            self.exported[arm.name] = exported
            self.restore_measurement(arm.name, **self.native_options(arm.name, prepared["probes"]))

    def bonsai_options(self, runtime):
        return {"assets": self.work / "bonsai2_4b" / "artifacts", "library": runtime["library"],
                "reference_gguf": Path(self.exported["fp16"]["native"]) / "model.gguf",
                "source": self.teacher["source"], "teacher_dir": self.teacher["teacher_dir"],
                "scorer_dir": self.teacher["scorer_dir"]}

    def hydrate_bonsai(self):
        receipt = self.saved("bonsai-assets", self.config["bonsai"])
        if receipt:
            self.exported["bonsai2_4b"] = {"artifact_bytes": receipt["complete_model_payload_bytes"],
                                         "language_payload_bytes": receipt["language_model_bytes"], "scope": receipt["scope"]}
            runtime = self.saved("bonsai-runtime", self.identity)
            if runtime:
                self.restore_measurement("bonsai2_4b", **self.bonsai_options(runtime))

    def preflight(self):
        super().preflight()

        def action(stage):
            for bits in (2, 6):
                model = stage.directory / f"b{bits}.gguf"
                self.command(stage, "make_rq3_model_fixture.py", "--output", model,
                             "--llama-dir", self.runtime["source"], "--backbone-bits", bits)
                self.command(stage, "check_rq3_model.py", "--library", self.runtime["library"],
                             "--model", model, "--backend", "CUDA0", "--output", model.with_suffix(".json"))
            self.command(stage, "check_rq3_conversion.py", "--library", self.runtime["library"],
                         "--llama-dir", self.runtime["source"], "--backend", "CUDA0",
                         "--output-dir", stage.directory / "conversion",
                         "--report", stage.directory / "conversion.json", "--backbone-bits", "2")
            return {"passed": True}, self.stage_files(stage.directory)
        self.workflow.stage("ternary-native-preflight", action, signature=self.runtime, minutes=10, reuse=False)

    def prepare(self, arm):
        path = self.root / "arms" / f"{arm.name}.json"
        write_json(path, arm_record(arm))
        options = {"parent": self.prepared[arm.parent]["checkpoint"]} if arm.parent else {}
        prepared = self.worker("prepare-" + arm.name, "prepare", arm=path,
                               teacher_dir=self.teacher["teacher_dir"], minutes=180, **options)
        self.worker("reload-" + arm.name, "reload", checkpoint=prepared["checkpoint"],
                    probes=prepared["probes"], minutes=20)
        self.prepared[arm.name] = prepared
        self.exported[arm.name] = self.worker(
            "export-" + arm.name, "export", checkpoint=prepared["checkpoint"],
            llama_dir=self.runtime["source"], minutes=30)
        self.summarize()

    def bonsai(self):
        asset_dir = self.work / "bonsai2_4b" / "artifacts"
        build_dir = self.work / self.identity[:12] / "bonsai-build"

        def acquire(stage):
            self.command(stage, "ternary_research_bonsai.py", "acquire", "--output-dir", asset_dir,
                         "--packing", self.config["bonsai"]["packing"], "--report", stage.directory / "report.json")
            receipt = read(stage.directory / "report.json")
            return receipt, [stage.directory / "report.json", Path(receipt["model"]), Path(receipt["vision"])]

        receipt = self.workflow.stage("bonsai-assets", acquire, signature=self.config["bonsai"], minutes=30)
        self.exported["bonsai2_4b"] = {"artifact_bytes": receipt["complete_model_payload_bytes"],
                                     "language_payload_bytes": receipt["language_model_bytes"], "scope": receipt["scope"]}
        self.summarize()

        def build(stage):
            self.command(stage, "ternary_research_bonsai.py", "build", "--source-dir",
                         self.work / "bonsai-prism-source", "--build-dir", build_dir,
                         "--backend", "CUDA", "--jobs", "4", "--report", stage.directory / "report.json")
            result = read(stage.directory / "report.json")
            files = [p for p in build_dir.rglob("*") if p.is_file() and (".so" in p.name or p.suffix == ".dylib")]
            return result, [stage.directory / "report.json", build_dir / "build-receipt.json", *files]

        runtime = self.workflow.stage("bonsai-runtime", build, signature=self.identity, minutes=60)
        self.measured["bonsai2_4b"] = self.worker(
            "measure-bonsai2_4b", "bonsai", **self.bonsai_options(runtime), minutes=90)
        self.summarize()

    def run(self):
        active = None
        with self.workflow:
            try:
                from scripts.ternary_research_results import complete_measurement
                self.hydrate()
                if all(complete_measurement(self.measured.get(label, {})) for label in LABELS):
                    print("RESUME complete study: all measurement artifacts and dependencies verified", flush=True)
                    return
                self.data = self.data or self.worker("freeze", "freeze", minutes=30)
                self.build_runtime()
                self.teacher = self.teacher or self.worker("teacher", "teacher", minutes=45)
                active = "w5_v6"
                self.prepare(arms()[0])
                active = "fp16"
                self.exported["fp16"] = self.worker(
                    "export-fp16", "export_fp16", source=self.teacher["source"],
                    quant_export=self.exported["w5_v6"]["native"], llama_dir=self.runtime["source"], minutes=30)
                self.measure("fp16", self.exported["fp16"], self.teacher["probes"])
                active = "w5_v6"
                self.measure(active, self.exported[active], self.prepared[active]["probes"])
                active = "bonsai2_4b"
                self.bonsai()
                for arm in arms()[1:]:
                    active = arm.name
                    self.prepare(arm)
                    self.measure(active, self.exported[active], self.prepared[active]["probes"])
            except BaseException as error:
                self.failures[active or "setup"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                self.summarize()

    def summarize(self):
        from scripts.ternary_research_results import comparison
        result = comparison(self.prepared, self.exported, self.measured, self.teacher, self.failures)
        result["declared_task_margin"] = self.config["research"]["task_margin"]
        result["error"] = self.failures or None
        result["selection"] = {"basis": "independent selection documents; all predeclared arms measured",
                               "scores": {k: v["selection"] for k, v in self.prepared.items()}}
        write_json(self.root / "comparison.json", result)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/rotquant-ternary-work"))
    parser.add_argument("--budget-minutes", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = validate_config(yaml.safe_load(args.config.read_text()))
    if not 1 <= args.budget_minutes <= 480:
        parser.error("budget must be 1..480 minutes")
    if args.dry_run:
        print(json.dumps({"labels": LABELS, "arms": [arm_record(a) for a in arms()],
                          "config": config, "budget_minutes": args.budget_minutes}, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("the comparison requires CUDA; use --dry-run locally")
    TernaryStudy(args, config).run()


if __name__ == "__main__":
    main()
