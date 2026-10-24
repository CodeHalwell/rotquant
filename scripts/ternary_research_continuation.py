"""Explicit, hash-verified child continuation of the interrupted ternary pilot.

Original measurements retain their source/runtime identity. Bonsai loader and
continuation plumbing may differ; a parity worker correction additionally needs
an exact-source diagnostic receipt. Configuration and other numerical code must
match. Parent files are always read-only inputs.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from scripts.native_gpu_workflow import artifacts_match, fingerprint, write_json
from scripts.native_hashing import digest
from scripts.ternary_research_plan import PROTOCOL, arm_record, arms
from scripts.ternary_research_results import complete_measurement

MODULE = "scripts/ternary_research_continuation.py"
SOURCE_CHANGES = frozenset({MODULE, "scripts/ternary_research_bonsai.py",
                          "scripts/run_qwen35_ternary_research.py",
                          "scripts/build_qwen35_ternary_research_notebook.py"})
IMPORTED_LABELS = ("fp16", "w5_v6")
PARITY_WORKER = "scripts/w4_research_worker.py"


class HashVerifier:
    """Deduplicate hashes only within this process while file metadata agrees."""
    def __init__(self):
        self.cache = {}

    def __call__(self, files):
        try:
            return all(self.matches(Path(name), expected) for name, expected in files.items())
        except OSError:
            return False

    def matches(self, path, expected):
        if not path.is_file() or path.is_symlink():
            return False
        before = path.stat()
        key = self.key(path, before)
        if before.st_size != expected["bytes"]:
            return False
        if key not in self.cache:
            self.cache[key] = digest(path)
        return self.key(path, path.stat()) == key and self.cache[key] == expected["sha256"]

    @staticmethod
    def key(path, stat):
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def read(path):
    return json.loads(Path(path).read_text())


def source_changes(before, after, *, parity_correction=None):
    changes = {name: {"old": before.get(name), "new": after.get(name)}
               for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    for name, pair in changes.items():
        if name == PARITY_WORKER and parity_correction == {name: pair}:
            continue
        if name not in SOURCE_CHANGES or pair["new"] is None or (pair["old"] is None and name != MODULE):
            raise ValueError(f"continuation does not permit source change: {name}")
    return changes


def passed_record(state, name, signature):
    rows = [row for row in state["attempts"] if row["name"] == name]
    if not rows or rows[-1]["status"] != "passed" or rows[-1]["signature"] != fingerprint(signature):
        raise ValueError(f"continuation requires the latest passed original request: {name}")
    return rows[-1]


def verify_record(root, row, *, verify=artifacts_match):
    """Require every recorded payload, its report and its original stage path."""
    directory = Path(row["directory"])
    expected = root / "stages" / row["name"]
    if directory.parent != expected or not directory.name.startswith("attempt-"):
        raise ValueError(f"parent stage path is not bound to its root: {row['name']}")
    files = row.get("artifacts", {})
    if not files or str(directory / "report.json") not in files:
        raise ValueError(f"parent stage has no complete artifact receipt: {row['name']}")
    for name in files:
        path = Path(name)
        if not path.is_relative_to(directory) or path.resolve() != path:
            raise ValueError(f"parent artifact escaped its original stage: {name}")
    verify_payload_paths(directory, row["value"], files)
    print(f"CONTINUATION verify {row['name']}: {len(files)} files", flush=True)
    if not verify(files) or read(directory / "report.json") != row["value"]:
        raise ValueError(f"parent artifacts missing or changed: {row['name']}")


def verify_payload_paths(directory, value, files):
    for key in ("data_path", "source", "teacher_dir", "scorer_dir", "probes", "checkpoint", "native"):
        if key not in value:
            continue
        path = Path(value[key])
        if not path.is_relative_to(directory) or not path.exists() or path.resolve() != path:
            raise ValueError(f"parent referenced payload is missing or escaped its stage: {path}")
        payloads = payload_files(path)
        if not payloads or any(str(p) not in files for p in payloads):
            raise ValueError(f"parent referenced payload is not fully hashed: {path}")


def payload_files(path):
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file() and p.suffix != ".pyc"]
    return [path]


def original_requests(root, identity, state, *, verify=artifacts_match):
    """Reconstruct the original dependency chain, without requiring local DLLs."""
    defaults = {"config": root / "config.json", "root": root, "identity": identity}
    records = {}

    def worker(name, **options):
        signature = {key: str(value) for key, value in {**defaults, **options}.items()}
        row = passed_record(state, name, signature)
        verify_record(root, row, verify=verify)
        records[name] = row
        return row["value"]

    data = worker("freeze")
    defaults["data"] = data["data_path"]
    runtime = passed_record(state, "runtime", identity)
    teacher = worker("teacher")
    arm_path = root / "arms" / "w5_v6.json"
    if read(arm_path) != arm_record(arms()[0]):
        raise ValueError("parent W5 arm definition changed")
    prepared = worker("prepare-w5_v6", arm=arm_path, teacher_dir=teacher["teacher_dir"])
    worker("reload-w5_v6", checkpoint=prepared["checkpoint"], probes=prepared["probes"])
    native = runtime["value"]
    w5 = worker("export-w5_v6", checkpoint=prepared["checkpoint"], llama_dir=native["source"])
    fp16 = worker("export-fp16", source=teacher["source"], quant_export=w5["native"], llama_dir=native["source"])
    for label, exported, probes in (("fp16", fp16, teacher["probes"]), ("w5_v6", w5, prepared["probes"])):
        result = worker("measure-" + label, native=exported["native"], probes=probes,
                        library=native["library"], source=teacher["source"],
                        teacher_dir=teacher["teacher_dir"], scorer_dir=teacher["scorer_dir"], label=label)
        verify_measurement(result, label, data["fingerprint"])
    if records["measure-fp16"]["value"]["runtime"] != records["measure-w5_v6"]["value"]["runtime"]:
        raise ValueError("parent FP16/W5 controls used different native runtimes")
    return records, runtime


def verify_measurement(result, label, data_identity):
    if (not complete_measurement(result) or result.get("label") != label
            or result.get("data_fingerprint") != data_identity
            or result.get("backend") != "CUDA0/reference/chunk16"
            or result["speed"].get("prefill_chunk_tokens") != 16 or not result.get("runtime")):
        raise ValueError(f"parent measurement is incomplete or has a different execution binding: {label}")


def validate_parent(parent, child, current):
    if parent == child or parent.is_relative_to(child) or child.is_relative_to(parent):
        raise ValueError("continuation requires a separate child root outside the parent")
    previous, state = read(parent / "identity.json"), read(parent / "workflow.json")
    if previous["protocol"] != PROTOCOL or current["protocol"] != PROTOCOL:
        raise ValueError("continuation protocol differs")
    if previous["config"] != current["config"] or read(parent / "config.json") != current["config"]:
        raise ValueError("continuation configuration differs")
    if previous["runtime"] != current["runtime"]:
        raise ValueError("continuation requires matching hardware/software to reuse measured speed")
    identity = fingerprint(previous)
    if (state["controls"]["identity"] != identity or state["status"] not in {"failed", "passed"}
            or any(row["status"] == "running" for row in state["attempts"])):
        raise ValueError("parent ledger is unbound or still running")
    active = state["active_seconds"]
    if not math.isfinite(active) or active < 0:
        raise ValueError("invalid parent execution allowance")
    return previous, state, identity


def correction_receipt(path, previous, current, identity, verify):
    if path is None:
        return None
    path = Path(path).resolve()
    receipt = read(path)
    expected = {PARITY_WORKER: {"old": previous["source"].get(PARITY_WORKER),
                               "new": current["source"].get(PARITY_WORKER)}}
    if (receipt.get("protocol") != "ternary-parity-correction-v1"
            or receipt.get("parent_identity") != identity or receipt.get("source_changes") != expected
            or expected[PARITY_WORKER]["old"] == expected[PARITY_WORKER]["new"]
            or receipt.get("validation", {}).get("passed") is not True):
        raise ValueError("parity correction is not bound to this validated source change")
    diagnostics = receipt.get("diagnostics", {})
    if not diagnostics or not verify(diagnostics):
        raise ValueError("parity correction diagnostic artifacts are missing or changed")
    report_path = receipt["validation"].get("report")
    if report_path not in diagnostics:
        raise ValueError("parity correction lacks its hashed validation report")
    report = read(report_path)
    if (report.get("passed") is not True or report.get("parent_identity") != identity
            or report.get("source_changes") != expected):
        raise ValueError("parity correction validation report differs from its source binding")
    return {"path": str(path), "sha256": digest(path), **receipt}


def verify_stage(root, row, verify, *, external=False):
    directory = Path(row["directory"])
    if directory.parent != root / "stages" / row["name"] or not directory.name.startswith("attempt-"):
        raise ValueError("imported stage is outside its original root")
    files = row.get("artifacts", {})
    if not files or (not external and any(not Path(p).is_relative_to(directory) for p in files)):
        raise ValueError("imported stage artifacts are unbound")
    if not verify(files):
        raise ValueError(f"imported stage artifacts missing or changed: {row['name']}")
    report = directory / "report.json"
    if str(report) in files and read(report) != row["value"]:
        raise ValueError("imported stage report differs from its ledger")


def inherited_records(parent, state, identity, verify):
    """Reconstruct each historical request and recheck every retained payload."""
    receipt_path = parent / "continuation.json"
    receipt = read(receipt_path)
    protocols = {f"ternary-continuation-v{version}" for version in (1, 2, 3)}
    if (receipt.get("protocol") not in protocols or receipt["child_identity"] != identity
            or state["controls"].get("continuation") != fingerprint(receipt)
            or state["controls"]["active_budget_minutes"] != receipt["budget_minutes"]):
        raise ValueError("parent continuation receipt is not bound to its ledger")
    ancestor = Path(receipt["parent_root"])
    validate_lineage_source(ancestor, parent, receipt, verify)
    ancestor_state = read(ancestor / "workflow.json")
    extra = {}
    if (ancestor / "continuation.json").exists():
        records, runtime, extra = next_generation(ancestor, ancestor_state, receipt["parent_identity"], verify)
    else:
        records, runtime = original_requests(ancestor, receipt["parent_identity"], ancestor_state, verify=verify)
    if (records != receipt["imports"] or runtime != receipt["original_runtime"]
            or any(receipt.get(key) != value for key, value in extra.items())):
        raise ValueError("ancestor stages differ from the retained continuation receipt")
    if digest(ancestor / "workflow.json") != receipt["parent_workflow_sha256"]:
        raise ValueError("ancestor ledger changed during verification")
    origin = {"root": str(ancestor), "identity": receipt["parent_identity"], "imported": True}
    origins = extra.get("origins", {name: origin for name in records})
    return records, origins, {"path": str(receipt_path), "sha256": digest(receipt_path), "receipt": receipt}


def validate_lineage_source(ancestor, parent, receipt, verify):
    if (digest(ancestor / "identity.json") != receipt["parent_identity_sha256"]
            or digest(ancestor / "workflow.json") != receipt["parent_workflow_sha256"]):
        raise ValueError("ancestor identity or ledger changed")
    current = read(parent / "identity.json")
    previous, state, identity = validate_parent(ancestor, parent, current)
    correction = receipt.get("parity_correction")
    if correction:
        checked = correction_receipt(correction["path"], previous, current, identity, verify)
        if checked != correction:
            raise ValueError("historical parity correction changed")
    changes = source_changes(previous["source"], current["source"],
                             parity_correction=correction["source_changes"] if correction else None)
    if identity != receipt["parent_identity"] or changes != receipt["source_changes"]:
        raise ValueError("ancestor source identity differs from continuation")
    validate_lineage_budget(receipt, state, previous)


def validate_lineage_budget(receipt, state, previous):
    if (receipt["parent_active_seconds"] != state["active_seconds"]
            or receipt["parent_budget_minutes"] != state["controls"]["active_budget_minutes"]
            or receipt["parent_wall_deadline_epoch"] != state.get("wall_deadline_epoch")
            or receipt["runtime_identity"] != previous["runtime"]):
        raise ValueError("historical continuation allowance or runtime differs")
    cap = math.floor(state["controls"]["active_budget_minutes"] - state["active_seconds"] / 60)
    if receipt.get("fresh_budget"):
        if receipt["protocol"] != "ternary-continuation-v3" or receipt.get("budget_mode") != "fresh":
            raise ValueError("fresh allowance lacks its explicit continuation binding")
        cap = 480
    if not 1 <= receipt["budget_minutes"] <= min(cap, 480):
        raise ValueError("historical continuation exceeded its execution allowance")


def verify_native_source(runtime):
    from scripts.build_rq3_runtime import INTEGRATION, REVISION, ROOT, git, verify_patched_source
    contract = read(INTEGRATION / "rotquant-native-v2-files.json")
    source = Path(runtime["value"]["source"])
    build = Path(runtime["value"]["library"]).parent.parent / "build-receipt.json"
    if str(build) not in runtime["artifacts"]:
        raise ValueError("runtime lacks its original build receipt")
    receipt = read(build)
    if (receipt.get("backend") != "CUDA" or receipt["base_revision"] != REVISION
            or receipt["patch_sha256"] != contract["patch_sha256"]
            or git(source, "rev-parse", "HEAD") != REVISION):
        raise ValueError("live runtime source/build differs from the pinned CUDA runtime")
    if any(digest(ROOT / name) != sha for name, sha in receipt["external_sources"].items()):
        raise ValueError("live runtime external source changed")
    changed = set(git(source, "diff", "HEAD", "--name-only").splitlines())
    changed.update(git(source, "ls-files", "--others", "--exclude-standard").splitlines())
    if changed != set(contract["files_sha256"]):
        raise ValueError("live runtime checkout has unrelated changes")
    verify_patched_source(source, contract["files_sha256"])


def next_generation(parent, state, identity, verify):
    records, origins, lineage = inherited_records(parent, state, identity, verify)
    defaults = {"config": parent / "config.json", "root": parent, "identity": identity,
                "data": records["freeze"]["value"]["data_path"]}
    origin = {"root": str(parent), "identity": identity, "imported": True}

    def stage(name, signature, *, external=False):
        row = passed_record(state, name, signature)
        verify_stage(parent, row, verify, external=external)
        return row

    def worker(name, **options):
        signature = {key: str(value) for key, value in {**defaults, **options}.items()}
        row = passed_record(state, name, signature)
        verify_record(parent, row, verify=verify)
        records[name], origins[name] = row, origin
        return row["value"]

    runtime, preflights = inherited_runtime(state, identity, lineage["receipt"], stage)
    if "measure-bonsai2_4b" not in records:
        import_bonsai(parent, state, identity, records, origins, origin, worker, stage)
    import_native_arms(parent, state, records, runtime, worker)
    prepared = [arm.name for arm in arms() if "export-" + arm.name in records]
    measured = [*IMPORTED_LABELS, "bonsai2_4b"]
    measured.extend(arm.name for arm in arms() if arm.name not in measured and "measure-" + arm.name in records)
    return records, runtime, {"origins": origins, "lineage": lineage, "reused_runtime": runtime,
                              "reused_preflights": preflights, "prepared_labels": prepared,
                              "imported_labels": measured}


def inherited_runtime(state, identity, receipt, stage):
    names = {row["name"] for row in state["attempts"]}
    runtime = stage("runtime", identity, external=True) if "runtime" in names else receipt["reused_runtime"]
    verify_native_source(runtime)
    preflights = {}
    for name in ("native-preflight", "ternary-native-preflight"):
        preflights[name] = (stage(name, runtime["value"]) if name in names
                            else receipt["reused_preflights"][name])
    return runtime, preflights


def import_native_arms(parent, state, records, runtime, worker):
    names = {row["name"] for row in state["attempts"]}
    teacher = records["teacher"]["value"]
    for arm in arms():
        if "export-" + arm.name not in records and "export-" + arm.name in names:
            import_prepared_arm(parent, arm, records, runtime, teacher, worker)
        measure = "measure-" + arm.name
        rows = [row for row in state["attempts"] if row["name"] == measure]
        if rows and rows[-1]["status"] == "passed":
            prepared = records["prepare-" + arm.name]["value"]
            result = worker(measure, native=records["export-" + arm.name]["value"]["native"],
                            probes=prepared["probes"], library=runtime["value"]["library"],
                            source=teacher["source"], teacher_dir=teacher["teacher_dir"],
                            scorer_dir=teacher["scorer_dir"], label=arm.name)
            verify_measurement(result, arm.name, records["freeze"]["value"]["fingerprint"])


def import_prepared_arm(parent, arm, records, runtime, teacher, worker):
    path = parent / "arms" / f"{arm.name}.json"
    if read(path) != arm_record(arm):
        raise ValueError(f"parent {arm.name} arm definition changed")
    options = {"parent": records["prepare-" + arm.parent]["value"]["checkpoint"]} if arm.parent else {}
    prepared = worker("prepare-" + arm.name, arm=path, teacher_dir=teacher["teacher_dir"], **options)
    worker("reload-" + arm.name, checkpoint=prepared["checkpoint"], probes=prepared["probes"])
    worker("export-" + arm.name, checkpoint=prepared["checkpoint"], llama_dir=runtime["value"]["source"])


def import_bonsai(parent, state, identity, records, origins, origin, worker, stage):
    config = read(parent / "config.json")
    assets = stage("bonsai-assets", config["bonsai"], external=True)
    runtime = stage("bonsai-runtime", identity, external=True)
    for row in (assets, runtime):
        records[row["name"]], origins[row["name"]] = row, origin
    teacher = records["teacher"]["value"]
    result = worker("measure-bonsai2_4b", assets=Path(assets["value"]["model"]).parent,
                    library=runtime["value"]["library"],
                    reference_gguf=Path(records["export-fp16"]["value"]["native"]) / "model.gguf",
                    source=teacher["source"], teacher_dir=teacher["teacher_dir"], scorer_dir=teacher["scorer_dir"])
    if (not complete_measurement(result) or result.get("label") != "bonsai2_4b"
            or result.get("data_fingerprint") != records["freeze"]["value"]["fingerprint"]
            or result.get("backend") != "CUDA0/Prism/chunk16" or result["speed"].get("prefill_chunk_tokens") != 16):
        raise ValueError("parent Bonsai measurement execution binding differs")


def continuation(parent, child, current, requested_minutes, *, parity_correction=None, fresh_budget=False):
    """Validate before writing a child receipt; repeated calls never renew time."""
    parent, child = Path(parent).resolve(), Path(child).resolve()
    previous, state, identity = validate_parent(parent, child, current)
    verify = HashVerifier()
    correction = correction_receipt(parity_correction, previous, current, identity, verify)
    changes = source_changes(previous["source"], current["source"],
                             parity_correction=correction["source_changes"] if correction else None)
    remaining = math.floor(state["controls"]["active_budget_minutes"] - state["active_seconds"] / 60)
    budget = requested_minutes if fresh_budget else min(requested_minutes, remaining)
    if not 1 <= requested_minutes <= 480:
        raise ValueError("continuation budget must be 1..480 minutes")
    if budget < 1:
        raise ValueError("parent active execution allowance is exhausted")
    hashes = {"parent_identity_sha256": digest(parent / "identity.json"),
              "parent_workflow_sha256": digest(parent / "workflow.json")}
    extra = {}
    if (parent / "continuation.json").exists():
        records, runtime, extra = next_generation(parent, state, identity, verify)
    else:
        records, runtime = original_requests(parent, identity, state, verify=verify)
    if hashes["parent_workflow_sha256"] != digest(parent / "workflow.json"):
        raise ValueError("parent ledger changed during continuation verification")
    receipt = {"protocol": "ternary-continuation-v1", "parent_root": str(parent),
               "parent_identity": identity, **hashes, "child_identity": fingerprint(current),
               "source_changes": changes, "budget_minutes": budget,
               "parent_active_seconds": state["active_seconds"],
               "parent_budget_minutes": state["controls"]["active_budget_minutes"],
               "parent_wall_deadline_epoch": state.get("wall_deadline_epoch"),
               "runtime_identity": previous["runtime"], "original_runtime": runtime, "imports": records, **extra}
    if extra:
        receipt["protocol"] = "ternary-continuation-v2"
    if fresh_budget or (extra and extra["lineage"]["receipt"]["protocol"] != "ternary-continuation-v1"):
        receipt.update(protocol="ternary-continuation-v3", fresh_budget=fresh_budget,
                       budget_mode="fresh" if fresh_budget else "remaining")
    if correction:
        receipt["parity_correction"] = correction
    save_continuation(child, receipt)
    print(f"CONTINUATION verified {receipt.get('imported_labels', IMPORTED_LABELS)}; child allowance {budget} minutes", flush=True)
    return receipt


def save_continuation(child, receipt):
    path = child / "continuation.json"
    if path.exists():
        if read(path) != receipt:
            raise ValueError("continuation provenance changed; existing child remains unchanged")
    elif (child / "workflow.json").exists():
        raise ValueError("cannot import controls into an existing non-continuation run")
    else:
        write_json(path, receipt)


def hydrate_imports(study, receipt):
    """Restore measured rows before attempting any missing local runtime build."""
    values = {name: copy.deepcopy(row["value"]) for name, row in receipt["imports"].items()}
    study.data, study.teacher = values["freeze"], values["teacher"]
    for label in receipt.get("prepared_labels", ["w5_v6"]):
        study.prepared[label] = values["prepare-" + label]
        study.exported[label] = values["export-" + label]
    study.exported["fp16"] = values["export-fp16"]
    for label in receipt.get("imported_labels", IMPORTED_LABELS):
        study.measured[label] = values["measure-" + label]
    if "bonsai-assets" in values:
        assets = values["bonsai-assets"]
        study.exported["bonsai2_4b"] = {"artifact_bytes": assets["complete_model_payload_bytes"],
                                       "language_payload_bytes": assets["language_model_bytes"], "scope": assets["scope"]}
    if "reused_runtime" in receipt:
        study.runtime = copy.deepcopy(receipt["reused_runtime"]["value"])


def provenance(receipt):
    result = {key: value for key, value in receipt.items()
              if key not in {"imports", "original_runtime", "reused_runtime", "reused_preflights", "lineage"}}
    if "lineage" in receipt:
        result["lineage"] = {key: value for key, value in receipt["lineage"].items() if key != "receipt"}
    return result


def runtime_provenance(study):
    """Label a rebuilt runtime separately; never rewrite original timing hashes."""
    result = {"hardware_software_match": True, "native_source_unchanged": True,
              "binary_policy": "Original controls retain their binary hashes; rebuilt runtime has separate provenance."}
    rows = study.workflow.state["attempts"]
    runtimes = [row for row in rows if row["name"] == "runtime" and row["status"] == "passed"]
    if "reused_runtime" in study.continuation:
        runtimes = [study.continuation["reused_runtime"], *runtimes]
        rows = [*study.continuation["reused_preflights"].values(), *rows]
        result["runtime_reused"] = True
    if not runtimes or not study.runtime:
        return result
    row = runtimes[-1]
    library = Path(study.runtime["library"])
    original = study.continuation["imports"]["measure-fp16"]["value"]["runtime"].get(library.name)
    rebuilt = row["artifacts"].get(str(library.resolve()), {}).get("sha256")
    result.update(original_library_sha256=original, rebuilt_library_sha256=rebuilt,
                  same_library_binary=bool(original and original == rebuilt),
                  build_artifacts=row["artifacts"],
                  preflight={name: latest_passed(rows, name, study.runtime)
                             for name in ("native-preflight", "ternary-native-preflight")})
    return result


def latest_passed(rows, name, signature):
    matching = [row for row in rows if row["name"] == name]
    return bool(matching and matching[-1]["status"] == "passed"
                and matching[-1]["signature"] == fingerprint(signature))
