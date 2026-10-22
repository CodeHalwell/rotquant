"""Readout of measured complete artifacts, quality and native serving rates."""
from __future__ import annotations


def comparison(prepared, exported, measured, teacher, *, expected):
    rows = []
    for label in expected:
        checkpoint = teacher if label == "fp16" else prepared.get(label)
        native, result = exported.get(label), measured.get(label)
        rows.append(comparison_row(label, checkpoint, native, result))
    return {"complete": set(expected) <= measured.keys(), "rows": rows,
            "paired_tasks": paired_comparisons(measured),
            "scope": "Fresh within-run comparison. Native FP16/W5/W4 use the same frozen tokens and runtime. Text quality only; retained auxiliary tensors are counted. Screening subsets do not establish no-loss performance."}


def comparison_row(label, checkpoint, native, result):
    return {"label": label, "status": "measured" if result else "pending",
            "checkpoint_bytes": (checkpoint or {}).get("artifact_bytes"),
            "native_payload_bytes": (native or {}).get("artifact_bytes"),
            "selection": (checkpoint or {}).get("selection"),
            **{key: (result or {}).get(key) for key in ("quality", "tasks", "speed")}}


def paired_comparisons(measured):
    from scripts.w4_research_measure import paired_task_deltas
    comparisons = {}
    for baseline in ("fp16", "w5_v6"):
        if baseline not in measured:
            continue
        for label, report in measured.items():
            if label != baseline:
                comparisons[f"{label}_vs_{baseline}"] = paired_task_deltas(
                    report["tasks"]["rows"], measured[baseline]["tasks"]["rows"])
    return comparisons
