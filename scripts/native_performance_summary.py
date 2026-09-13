"""Bounded notebook readout. Never fold instrumented/partial results into speedups."""
from __future__ import annotations

import json
from pathlib import Path


def latest_reports(root, *, since_epoch=None):
    current = None
    if since_epoch is not None:
        workflow = root / "workflow.json"
        if not workflow.exists():
            return
        current = {(a["name"], Path(a["directory"]).name)
                   for a in json.loads(workflow.read_text())["attempts"]
                   if a["started_utc_epoch"] >= since_epoch}
    for stage in sorted((root / "stages").glob("*")):
        attempts = sorted(stage.glob("attempt-*"))
        if not attempts:
            continue
        if current is not None and (stage.name, attempts[-1].name) not in current:
            continue
        # Only the newest attempt, even when a prior attempt succeeded.
        for kind in ("pilot", "profile"):
            path = attempts[-1] / kind / "report.json"
            if path.exists():
                yield stage.name, json.loads(path.read_text())


def tables(root, *, since_epoch=None):
    def num(x):
        return "—" if x is None else f"{x:.2f}"
    timing = ["| Model / kernel | Prompt tokens | Status | Reps | Prefill tok/s | Decode tok/s | Peak MiB |",
              "| :-- | --: | :-- | --: | --: | --: | --: |"]
    profiles = ["| Diagnostic | Phase | Operator | Mean GPU ms / measured repetition |",
                "| :-- | :-- | :-- | --: |"]
    for name, report in latest_reports(root, since_epoch=since_epoch):
        if report.get("measurement_kind") == "diagnostic":
            measured = [r for r in report.get("rows", []) if r.get("completed") and not r["warmup"]]
            if measured:
                for phase in ("prefill", "decode"):
                    for operator in ("rotation", "matrix", "vocabulary_head", "embedding"):
                        ms = sum(r[phase + "_profile"]["operators"][operator]["milliseconds"] for r in measured) / len(measured)
                        profiles.append(f"| {name} ({report['status']}; {len(measured)} reps) | {phase} | {operator} | {num(ms)} |")
            continue
        metrics = report.get("summary") or {}
        timing.append(f"| {name} | {report.get('context', '—')} | {report.get('status', 'not completed')} | "
                      f"{metrics.get('measured_repetitions', 0)} | {num(metrics.get('prefill_tokens_per_second'))} | "
                      f"{num(metrics.get('decode_tokens_per_second'))} | "
                      f"{num(report.get('memory', {}).get('sampled_peak_process_vram_mib'))} |")
    return "\n".join(timing), "\n".join(profiles)
