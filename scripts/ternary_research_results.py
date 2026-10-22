"""Comparable measured results, including verified incomplete measurement stages."""
import json
from pathlib import Path

from scripts.ternary_research_plan import LABELS
from scripts.w4_research_results import comparison_row, paired_comparisons


def complete_measurement(result):
    """A receipt key alone does not establish that all measurements finished."""
    return (result.get("status") == "complete"
            and all((result.get("quality") or {}).get(part, {}).get("summary")
                    for part in ("test", "long_test"))
            and bool((result.get("tasks") or {}).get("rows"))
            and bool((result.get("tasks") or {}).get("summary"))
            and (result.get("speed") or {}).get("completed") is True
            and bool((result.get("speed") or {}).get("summary")))


def _read_object(path):
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(), parse_constant=lambda value: None)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def partial_measurement(directory, expected, error):
    """Read only this attempt's completed subreports after its deployment gate.

    The controller subsequently hashes these exact files into the failed stage
    receipt. Resumes must verify those hashes before accepting this value.
    """
    directory = Path(directory)
    context_path = directory / "measurement-context.json"
    context = _read_object(context_path)
    if not context or any(context.get(key) != value for key, value in expected.items()):
        return None
    result = {**expected, "status": "failed", "error": str(error),
              "backend": context["backend"], "quality": {}}
    files = [context_path]
    for part in ("test", "long_test"):
        path = directory / part / "quality.json"
        value = _read_object(path)
        if value and value.get("rows") and value.get("summary"):
            result["quality"][part] = value
            files.append(path)
    for name, relative in (("tasks", "tasks/tasks.json"), ("speed", "speed.json")):
        path = directory / relative
        value = _read_object(path)
        if value and isinstance(value.get("rows"), list):
            result[name] = value
            files.append(path)
    return (result, files) if len(files) > 1 else None


def comparison(prepared, exported, measured, teacher, failures=None):
    rows = []
    failures = failures or {}
    for label in LABELS:
        checkpoint = teacher if label == "fp16" else prepared.get(label)
        native, result = exported.get(label, {}), measured.get(label, {})
        row = comparison_row(label, checkpoint, native, result)
        row.update(language_payload_bytes=native.get("language_payload_bytes"),
                   backend=result.get("backend"), precision_scope=(checkpoint or {}).get("precision_scope"),
                   training_metrics=(checkpoint or {}).get("metrics"),
                   error=failures.get(label), artifact_scope=native.get("scope"))
        if label == "fp16":
            row["precision_scope"] = "FP16 source; original auxiliary tensors"
        if label == "bonsai2_4b":
            row["precision_scope"] = "Bonsai ternary backbone/vocabulary; FP32 exceptions; BF16/F32 vision sidecar"
        if label in failures or result.get("status") == "failed":
            row["status"] = "failed"
            row["error"] = failures.get(label) or result.get("error")
        elif result and not complete_measurement(result):
            row["status"] = "pending"
        rows.append(row)
    tasks_available = {label: result for label, result in measured.items()
                       if (result.get("tasks") or {}).get("rows")}
    return {"complete": all(complete_measurement(measured.get(label, {})) for label in LABELS) and not failures,
            "rows": rows, "paired_tasks": paired_comparisons(tasks_available),
            "scope": "Fresh frozen-input text comparison. Complete payloads include retained auxiliary/vision files. Bonsai uses its pinned Prism runtime; RotQuant and FP16 share the RotQuant runtime. Greedy non-thinking evaluation is not the vendor's thinking benchmark. Pilot task subsets do not establish non-inferiority."}
