"""Compare W4 research recipes with fresh FP16 and W5/W6 controls in Colab."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_gpu_workflow import Workflow, write_json
from scripts.w4_research_plan import (
    DEFAULT_CONFIG,
    arm_record,
    bind_run,
    choose_candidate,
    initial_arms,
    validate_config,
)


def read(path):
    return json.loads(Path(path).read_text())


class ResearchStudy:
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
        self.prepared, self.exported, self.measured = {}, {}, {}

    def command(self, stage, script, *arguments):
        stage.command([sys.executable, "-u", ROOT / "scripts" / script, *arguments], script)

    def worker(self, name, phase, *, minutes=90, **options):
        defaults = {"config": self.config_path, "root": self.root, "identity": self.identity}
        if self.data:
            defaults["data"] = self.data["data_path"]
        defaults.update(options)

        def action(stage):
            arguments = [str(value) for key, val in defaults.items()
                         for value in ("--" + key.replace("_", "-"), val)]
            self.command(stage, "w4_research_worker.py", phase, "--output", stage.directory, *arguments)
            return read(stage.directory / "report.json"), self.stage_files(stage.directory)

        signature = {key: str(value) for key, value in defaults.items()}
        return self.workflow.stage(name, action, signature=signature, minutes=minutes)

    @staticmethod
    def stage_files(directory):
        return [p for p in directory.rglob("*") if p.is_file() and "logs" not in p.parts
                and p.suffix not in {".pt", ".pyc"}]

    def build_runtime(self):
        source, build = self.work / self.identity[:12] / "llama", self.work / self.identity[:12] / "build"

        def action(stage):
            self.command(stage, "build_rq3_runtime.py", "--source-dir", source, "--build-dir", build,
                         "--backend", "CUDA", "--jobs", "4")
            report = read(build / "build-receipt.json")
            libraries = [p for p in (build / "bin").iterdir() if p.is_file()]
            return {"library": report["library"], "source": str(source)}, [build / "build-receipt.json", *libraries]

        self.runtime = self.workflow.stage("runtime", action, signature=self.identity, minutes=60)
        self.preflight()

    def preflight(self):
        def action(stage):
            library, source = self.runtime["library"], self.runtime["source"]
            self.command(stage, "check_rq3_gpu.py", "--library", library, "--backend", "CUDA0",
                         "--output", stage.directory / "operators.json")
            for bits, group in ((5, 128), (4, 128), (4, 64), (4, 32)):
                model = stage.directory / f"b{bits}-g{group}.gguf"
                self.command(stage, "make_rq3_model_fixture.py", "--output", model, "--llama-dir", source,
                             "--backbone-bits", bits, "--group-size", group)
                self.command(stage, "check_rq3_model.py", "--library", library, "--model", model,
                             "--backend", "CUDA0", "--output", model.with_suffix(".json"))
            self.command(stage, "check_rq3_conversion.py", "--library", library, "--llama-dir", source,
                         "--backend", "CUDA0", "--output-dir", stage.directory / "conversion",
                         "--report", stage.directory / "conversion.json",
                         "--backbone-bits", "4", "--group-size", "32")
            return {"passed": True}, self.stage_files(stage.directory)

        self.workflow.stage("native-preflight", action, signature=self.runtime, minutes=15, reuse=False)

    def prepare(self, arm):
        path = self.root / "arms" / f"{arm.name}.json"
        write_json(path, arm_record(arm))
        prepared = self.worker("prepare-" + arm.name, "prepare", arm=path,
                               teacher_dir=self.teacher["teacher_dir"], minutes=150)
        self.worker("reload-" + arm.name, "reload", checkpoint=prepared["checkpoint"],
                    probes=prepared["probes"], minutes=20)
        self.prepared[arm.name] = prepared
        self.exported[arm.name] = self.worker(
            "export-" + arm.name, "export", checkpoint=prepared["checkpoint"],
            llama_dir=self.runtime["source"], minutes=30)
        self.summarize()

    def research(self):
        arms = initial_arms(self.config)
        for arm in arms:
            if arm.name not in self.prepared:
                self.prepare(arm)
        selected = choose_candidate(list(self.prepared.values()))
        base = next(arm for arm in arms if arm.name == selected["arm"]["name"])
        coded = replace(base, name="w4_codes", code_refit_passes=self.config["research"]["code_refit_passes"])
        self.prepare(coded)
        self.prepare(replace(base, name="w4_recovered", recovery=True))
        self.prepare(replace(coded, name="w4_codes_recovered", recovery=True))
        selection = {"basis": "independent selection documents; no final tasks/test results",
                     "selected": choose_candidate(list(self.prepared.values()))["arm"]["name"],
                     "scores": {k: v["selection"] for k, v in self.prepared.items()}}
        write_json(self.root / "selection.json", selection)

    def measure(self, label, exported, probes):
        measured = self.worker(
            "measure-" + label, "native", native=exported["native"], probes=probes,
            library=self.runtime["library"], source=self.teacher["source"],
            teacher_dir=self.teacher["teacher_dir"], scorer_dir=self.teacher["scorer_dir"],
            label=label, minutes=90)
        self.measured[label] = measured
        self.summarize()

    def run(self):
        with self.workflow:
            try:
                self.data = self.worker("freeze", "freeze", minutes=30)
                self.build_runtime()
                self.teacher = self.worker("teacher", "teacher", minutes=45)
                self.prepare(initial_arms(self.config)[0])
                fp16 = self.worker("export-fp16", "export_fp16", source=self.teacher["source"],
                                   quant_export=self.exported["w5_v6"]["native"],
                                   llama_dir=self.runtime["source"], minutes=30)
                self.exported["fp16"] = fp16
                self.measure("fp16", fp16, self.teacher["probes"])
                self.measure("w5_v6", self.exported["w5_v6"], self.prepared["w5_v6"]["probes"])
                self.research()
                for label, prepared in self.prepared.items():
                    if label != "w5_v6":
                        self.measure(label, self.exported[label], prepared["probes"])
            finally:
                self.summarize()

    def summarize(self):
        from scripts.w4_research_results import comparison
        result = comparison(self.prepared, self.exported, self.measured, self.teacher,
                            expected=["fp16", *[a.name for a in initial_arms(self.config)],
                                      "w4_codes", "w4_recovered", "w4_codes_recovered"])
        selection_path = self.root / "selection.json"
        result["selection"] = read(selection_path) if selection_path.exists() else None
        result["declared_task_margin"] = self.config["research"]["task_margin"]
        write_json(self.root / "comparison.json", result)


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/rotquant-w4-work"))
    parser.add_argument("--budget-minutes", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = validate_config(yaml.safe_load(args.config.read_text()))
    if not 1 <= args.budget_minutes <= 480:
        parser.error("budget must be 1..480 minutes")
    if args.dry_run:
        print(json.dumps({"arms": [arm_record(a) for a in initial_arms(config)],
                          "followups": ["selected W4 + H-aware code refinement", "selected W4 + scale-only recovery",
                                        "selected W4 + code refinement + scale-only recovery"],
                          "controls": ["fresh FP16", "fresh W5/W6 current recipe"],
                          "budget_minutes": args.budget_minutes, "config": config}, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("the full research experiment requires CUDA; use --dry-run locally")
    ResearchStudy(args, config).run()


if __name__ == "__main__":
    main()
