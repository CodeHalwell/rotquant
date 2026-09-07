#!/usr/bin/env python3
"""Export two W5 vocabulary finalists and validate fresh-process packed reload.

This is an artifact-conformance experiment, not independent recipe confirmation
or a provider benchmark. Completed stages and source Hessians resume by identity.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rotquant.checkpoint import MANIFEST_NAME, load_packed_model, save_packed_checkpoint
from rotquant.eval.promotion import QUALITY_GUARDS, quality_guards
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig
from rotquant.quantize import QuantConfig
from rotquant.utils import enable_default_logging, environment_record, set_seed, write_result
from rotquant.validation import (
    PROTOTYPE_PARITY,
    RELOAD_PARITY,
    audit_artifact,
    capture_probes,
    compare_probes,
    packed_residency,
    probe_inputs,
    save_probes,
)
from rotquant.vocabulary import (
    VocabularyConfig,
    file_digest,
    install_packed_vocabulary,
    quantize_vocabulary,
    tensor_digest,
    tied_modules,
    vocabulary_prototype,
)
from scripts import run_experiment as experiment
from scripts.assess_qwen35_vocabulary_budget import row_for
from scripts.run_qwen35_next_stage import _Heartbeat, _timestamp
from scripts.run_qwen35_vocabulary_budget import DEFAULT_CONFIG as SCREEN_CONFIG
from scripts.run_qwen35_vocabulary_budget import (
    digest,
    resolved_config,
    resource_ledger,
    runtime_identity,
    source_identity,
)

PROTOCOL = "qwen35-packed-vocabulary-validation-v1"
DEFAULT_CONFIG = ROOT / "configs/qwen35_4b_packed_validation_cuda.yaml"
ARMS = ("b5_v6", "b5_v8")
REVALIDATION_PROTOCOL = "qwen35-packed-checkpoint-revalidation-v1"


def load_config(path):
    settings = experiment.load_config(str(path))
    base = Path(settings.get("screen_config", SCREEN_CONFIG))
    config = resolved_config(base if base.is_absolute() else ROOT / base)
    config["packed_validation"] = settings["packed_validation"]
    config["quant"]["bits"] = 5
    checks = config["packed_validation"]
    for field in ("probe_prompts", "probe_prompt_tokens", "probe_logit_positions", "probe_new_tokens"):
        if type(checks.get(field)) is not int or checks[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if checks.get("projection_mode") != "dense_equivalent":
        raise ValueError("this protocol requires the dense-equivalent tiled vocabulary head")
    if not config.get("model_revision") or config.get("dtype") != "float16":
        raise ValueError("this protocol requires a pinned source model and FP16 execution")
    return config


def identity_for(config, seed, arm):
    prompt_files = {suite["prompt_file"]: file_digest(Path(suite["prompt_file"]))
                    for key in ("logit_fidelity_suites", "trajectory_suites")
                    for suite in config.get("eval", {}).get(key, {}).values()
                    if suite.get("prompt_file")}
    return json.loads(json.dumps({"protocol": PROTOCOL, "config": config, "seed": seed, "arm": arm,
            "source": source_identity(), "runtime": runtime_identity(),
            "prompt_files": prompt_files, "prototype_parity": PROTOTYPE_PARITY,
            "reload_parity": RELOAD_PARITY, "quality_guards": QUALITY_GUARDS}))


def save_record(path, payload):
    write_result(str(path), payload)
    write_result(str(path.with_suffix(".sha256")), {"sha256": file_digest(path)})


def read_record(path, identity=None):
    if not path.exists():
        return None
    checksum = path.with_suffix(".sha256")
    if not checksum.exists() or json.loads(checksum.read_text()).get("sha256") != file_digest(path):
        raise ValueError(f"record checksum mismatch: {path}")
    result = json.loads(path.read_text())
    if identity is not None and result.get("identity") != identity:
        raise ValueError(f"record identity mismatch; use a fresh run directory: {path}")
    return result


@contextmanager
def writer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another runner owns {path.parent}") from exc
        yield


def progress(root, phase, **details):
    event = {"protocol": PROTOCOL, "phase": phase, "pid": os.getpid(),
             "updated_at": _timestamp(), **details}
    write_result(str(root / "progress.json"), event)
    print(json.dumps(event), flush=True)


def release_models():
    experiment._LOGIT_REFERENCE_CACHE.clear()
    experiment._TRAJECTORY_REFERENCE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probe_settings(config):
    settings = config["packed_validation"]
    return {"prompt_tokens": settings["probe_prompt_tokens"],
            "logit_positions": settings["probe_logit_positions"],
            "new_tokens": settings["probe_new_tokens"]}


@contextmanager
def packed_context(model, owner, device, dtype):
    """Restore the source aliases for the next prototype; export excludes them."""
    from rotquant._internal import get_parent

    embedding, head = tied_modules(model)
    owner.to(device=device, dtype=dtype)
    aliases = install_packed_vocabulary(model, owner)
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.fallback = False
            module._fp_cache = None
    try:
        yield
    finally:
        for key, original in (("embedding", embedding), ("head", head)):
            parent, attribute = get_parent(model, aliases[key])
            setattr(parent, attribute, original)
        owner.cpu()


def verify_prepared(root, prepared, identity):
    if prepared["identity"] != identity or prepared["status"] != "prepared":
        raise ValueError("preparation identity/status mismatch")
    for name, expected in prepared["probe_files"].items():
        if Path(name).name != name or not (root / name).is_file() or file_digest(root / name) != expected:
            raise ValueError("prepared probe checksum mismatch")
    artifact = root / "checkpoint"
    ledger = audit_artifact(artifact, expected_manifest_sha256=prepared["export"]["manifest_sha256"])
    manifest = json.loads((artifact / MANIFEST_NAME).read_text())
    if manifest.get("deployment", {}).get("experiment_identity") != digest(identity):
        raise ValueError("artifact is not bound to this experiment")
    intent = read_record(root / "preparation.json", identity)
    if intent is None or manifest["deployment"].get("preparation_sha256") != file_digest(root / "preparation.json"):
        raise ValueError("artifact preparation evidence is missing or changed")
    expected_core = {key: value for key, value in intent.items() if key != "status"}
    actual_core = {key: value for key, value in prepared.items() if key not in {"status", "export"}}
    if expected_core != json.loads(json.dumps(actual_core)) or not prepared["prototype_parity"]["passed"]:
        raise ValueError("prepared result differs from artifact-bound evidence")
    if ledger["measured_artifact_bytes"] != prepared["export"]["artifact_bytes"]:
        raise ValueError("exported byte count changed")
    return ledger


def recover_prepared(root, identity):
    """Finish a published export using its manifest-bound, prewritten evidence.

    An incomplete or unrelated checkpoint is never overwritten or relabelled.
    """
    prepared = read_record(root / "prepared.json", identity)
    if prepared is None and (root / "checkpoint").exists():
        intent = read_record(root / "preparation.json", identity)
        if intent is None or intent.get("status") != "preparing":
            raise ValueError(f"orphan checkpoint lacks preparation evidence at {root}; preserve it and use a new root")
        ledger = audit_artifact(root / "checkpoint")
        manifest_path = root / "checkpoint" / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        prepared = {**intent, "status": "prepared", "export": {
            "path": str(root / "checkpoint"), "format": manifest["format"],
            "format_version": manifest["format_version"],
            "quantized_modules": len(manifest["quantized_modules"]),
            "artifact_bytes": ledger["measured_artifact_bytes"],
            "manifest_sha256": file_digest(manifest_path),
            "identity": digest(identity), "fallback_cache_serialized": False}}
        verify_prepared(root, prepared, identity)
        save_record(root / "prepared.json", prepared)
        print(f"recovered published checkpoint {root.name} from bound preparation evidence", flush=True)
    return prepared


def revalidation_paths(root, source_root):
    """Keep every recovery write outside the original evidence tree."""
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    if root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError("revalidation output and source directories must not overlap")
    if not source_root.is_dir():
        raise ValueError("revalidation source directory does not exist")
    return root, source_root


def comparable_identity(identity):
    """Only implementation revision and checkout location may change here.

    Model/config, data content, runtime and acceptance thresholds must match.
    Prompt paths are relocatable only when their recorded hashes match.
    """
    value = copy.deepcopy(identity)
    source = value.pop("source", None)
    if not isinstance(source, dict) or not all(source.get(k) for k in ("revision", "source_fingerprint")):
        raise ValueError("preparation lacks implementation provenance")
    files = value.pop("prompt_files")
    used = set()
    for key in ("logit_fidelity_suites", "trajectory_suites"):
        for suite in value["config"].get("eval", {}).get(key, {}).values():
            path = suite.get("prompt_file")
            if path:
                used.add(path)
                suite["prompt_file"] = "sha256:" + files[path]
    if used != set(files):
        raise ValueError("prompt file provenance is incomplete or unexpected")
    return value


def validation_input(config, root, arm, seed, revalidate_from=None):
    """Read and bind old artifacts; never copy, relabel or rewrite their evidence."""
    identity = identity_for(config, seed, arm)
    source_root = root
    if revalidate_from is not None:
        _, source_root = revalidation_paths(root, revalidate_from)
    source_arm = source_root / f"{arm}_s{seed}"
    prepared = read_record(source_arm / "prepared.json",
                           identity if revalidate_from is None else None)
    if prepared is None:
        raise ValueError(f"prepared checkpoint is missing at {source_arm}; no quantization will be launched")
    preparation_identity = prepared["identity"]
    if revalidate_from is not None and comparable_identity(preparation_identity) != comparable_identity(identity):
        raise ValueError("revalidation requires unchanged model/config, runtime, data and thresholds")
    ledger = verify_prepared(source_arm, prepared, preparation_identity)
    if revalidate_from is not None:
        identity["revalidation"] = {
            "protocol": REVALIDATION_PROTOCOL, "source_arm_dir": str(source_arm.resolve()),
            "preparation_identity_sha256": digest(preparation_identity),
            "prepared_sha256": file_digest(source_arm / "prepared.json"),
            "manifest_sha256": prepared["export"]["manifest_sha256"],
        }
    return source_arm, prepared, ledger, identity


def prepare_seed(config, root, seed, heartbeat_seconds):
    pending = []
    for arm in ARMS:
        arm_root = root / f"{arm}_s{seed}"
        identity = identity_for(config, seed, arm)
        progress(root, "verify_preparation", arm=arm, seed=seed)
        with _Heartbeat(f"verify-export/{arm}/seed-{seed}", seconds=heartbeat_seconds):
            saved = recover_prepared(arm_root, identity)
            if saved is not None:
                verify_prepared(arm_root, saved, identity)
        if saved is not None:
            print(f"resume preparation {arm}/seed-{seed}: verified checkpoint and probes", flush=True)
        else:
            pending.append((arm, arm_root, identity))
    if not pending:
        return
    device, dtype = experiment.resolve_device_dtype(config)
    set_seed(seed)
    current = copy.deepcopy(config)
    source = pending[0][2]["source"]
    current.update(seed=seed, stage_code_revision=source["revision"],
                   stage_source_fingerprint=source["source_fingerprint"],
                   stage_runtime_fingerprint=digest(runtime_identity()),
                   hessian_cache_dir=str(root / "hessians"))
    with _Heartbeat(f"prepare-W5/seed-{seed}", seconds=heartbeat_seconds):
        progress(root, "source_load", seed=seed)
        model, tokenizer, loader = experiment.load_hf_model(
            config["model"], dtype, device, config["model_loader"], config["model_revision"])
        model.eval()
        processor = None
        if config["model_loader"] == "multimodal_lm":
            # Fail closed: do not quietly omit processor files from the counted bundle.
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(config["model"], revision=config["model_revision"])
        embedding, head = tied_modules(model)
        teacher_digest = tensor_digest(embedding.weight)
        qcfg = QuantConfig(**current["quant"], seed=seed)
        pcfg = PatchConfig(quant=qcfg, **current["patch"], seed=seed)
        metrics = {"data_manifest": {}, "teacher_source_vocab_digest": teacher_digest,
                   "teacher_captured_before_mutation": True}
        write_result(str(root / f"resources_s{seed}.json"), resource_ledger(model, current, pcfg, root))
        progress(root, "source_hessians", seed=seed)
        calibration = experiment._prepare_calibration(current, model, tokenizer, device, qcfg, pcfg, metrics)
        progress(root, "source_teacher_capture", seed=seed)
        references = experiment._capture_references(current, current["eval"], model, tokenizer,
                                                    device, config["model"], metrics)
        experiment.validate_internal_data_disjointness(metrics["data_manifest"])
        prompts = [reference.inputs for reference in references.logit_references[
            :config["packed_validation"]["probe_prompts"]]]
        owners = {}
        for arm, _, _ in pending:
            bits = int(arm.rsplit("v", 1)[1])
            progress(root, "vocabulary_quantization", seed=seed, arm=arm)
            owners[arm] = quantize_vocabulary(
                embedding.weight, VocabularyConfig(bits=bits, seed=seed, **config["vocabulary"]),
                cache_dir=root / "vocabulary_cache" / source["source_fingerprint"] / digest(runtime_identity()),
                progress=lambda event: print(json.dumps(event), flush=True))
            owners[arm].projection_mode = config["packed_validation"]["projection_mode"]
        if tensor_digest(embedding.weight) != teacher_digest:
            raise ValueError("teacher vocabulary changed during quantization")
        del embedding, head
        progress(root, "W5_backbone_quantization", seed=seed)
        experiment._apply_quantization(current, model, pcfg, calibration, metrics)
        for arm, arm_root, identity in pending:
            arm_root.mkdir(parents=True, exist_ok=True)
            # Preserve an identity marker even if the first evaluation is interrupted.
            marker = read_record(arm_root / "identity.json")
            if marker is not None and marker != identity:
                raise ValueError("partial preparation identity mismatch; use a new run root")
            save_record(arm_root / "identity.json", identity)
            progress(root, "dense_prototype_quality", seed=seed, arm=arm)
            arm_metrics = copy.deepcopy(metrics)
            arm_metrics.update(experiment.footprint_metrics(model, {}))
            with vocabulary_prototype(model, owners[arm]) as record:
                arm_metrics["vocabulary"] = record
                experiment._run_evaluations(current, current["eval"], model, tokenizer, device,
                                            seed, references, arm_metrics)
                if arm_metrics.get("evaluation_halted") or not all(quality_guards(
                        row_for({"arm": arm, "seed": seed, "metrics": arm_metrics}),
                        row_for({"arm": arm, "seed": seed, "metrics": arm_metrics})).values()):
                    save_record(arm_root / "failed_prototype.json", {"metrics": arm_metrics, "identity": identity})
                    raise ValueError(f"{arm} prototype failed finite/completeness guards")
                dense_probes = capture_probes(model, prompts, device, **probe_settings(config))
            progress(root, "packed_preexport_probes", seed=seed, arm=arm)
            with packed_context(model, owners[arm], device, dtype):
                residency = packed_residency(model)
                packed_probes = capture_probes(model, prompts, device, **probe_settings(config))
                parity = compare_probes(packed_probes, dense_probes)
                save_probes(arm_root / "dense_probes.safetensors", dense_probes)
                save_probes(arm_root / "packed_probes.safetensors", packed_probes)
                save_record(arm_root / "prototype_parity.json", {"identity": identity, **parity})
                if not parity["passed"]:
                    raise ValueError(f"{arm} packed/prototype parity failed; inspect prototype_parity.json")
                intent = {"protocol": PROTOCOL, "status": "preparing", "arm": arm, "seed": seed,
                          "identity": identity, "metrics": arm_metrics, "prototype_parity": parity,
                          "residency": residency, "probe_files": {
                              name: file_digest(arm_root / name)
                              for name in ("dense_probes.safetensors", "packed_probes.safetensors")}}
                save_record(arm_root / "preparation.json", intent)
                progress(root, "export", seed=seed, arm=arm)
                exported = save_packed_checkpoint(
                    model, arm_root / "checkpoint", base_model=config["model"],
                    base_model_revision=config["model_revision"], model_loader=loader,
                    tokenizer=tokenizer, processor=processor,
                    deployment_metadata={"experiment_identity": digest(identity),
                                         "preparation_sha256": file_digest(arm_root / "preparation.json"),
                                         "protocol": PROTOCOL, "arm": arm, "seed": seed,
                                         "runtime": "tiled-reference-no-dense-cache"})
            prepared = {**intent, "status": "prepared", "export": exported}
            verify_prepared(arm_root, prepared, identity)
            save_record(arm_root / "prepared.json", prepared)
            print(f"prepared {arm}/seed-{seed}: {exported['artifact_bytes'] / 1e9:.3f} measured GB", flush=True)
        del model, tokenizer, processor, references, owners, calibration


def paired_quality(actual, expected, arm, seed):
    """Quality gates only make sense on exactly matching evaluated inputs."""
    for path in (("logit_fidelity",), ("logit_fidelity_suites", "diverse"),
                 ("trajectory_suites", "diverse")):
        a, b = actual, expected
        for key in path:
            a, b = a.get(key, {}), b.get(key, {})
        if not a.get("input_hashes") or a["input_hashes"] != b.get("input_hashes"):
            raise ValueError(f"quality pairing mismatch: {path}")
        pairs = []
        for suite in (a, b):
            records = suite.get("prompt_metrics", [])
            if (not records or len(records) != len(suite["input_hashes"])
                    or len(set(suite["input_hashes"])) != len(records)
                    or [r.get("input_hash") for r in records] != suite["input_hashes"]
                    or any(type(r.get("tokens")) is not int or r["tokens"] < 1 for r in records)):
                raise ValueError("quality pairing requires unique per-prompt records and token counts")
            pairs.append([(r["input_hash"], r["tokens"]) for r in records])
        if pairs[0] != pairs[1]:
            raise ValueError("quality pairing denominators differ")
    for key in ("ppl_wikitext2", "ppl_c4"):
        a = actual.get("data_manifest", {}).get(key, {}).get("window_hashes")
        b = expected.get("data_manifest", {}).get(key, {}).get("window_hashes")
        if not a or a != b:
            raise ValueError("PPL evaluated windows differ")
    candidate = row_for({"arm": arm, "seed": seed, "metrics": actual})
    baseline = row_for({"arm": arm, "seed": seed, "metrics": expected})
    guards = quality_guards(candidate, baseline)
    return {"guards": guards, "thresholds": QUALITY_GUARDS,
            "passed": all(guards.values()) and not actual.get("evaluation_halted", False),
            "paired_inputs": True}


def validate_worker(config, root, arm, seed, heartbeat_seconds=60, *, revalidate_from=None):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    arm_root = root / f"{arm}_s{seed}"
    if revalidate_from is not None:
        revalidation_paths(root, revalidate_from)
        os.environ["ROTQUANT_TOKEN_CACHE_DIR"] = str(root / "token_cache")
    progress(arm_root, "verify_artifact_and_evidence", arm=arm, seed=seed)
    with _Heartbeat(f"verify-reload/{arm}/seed-{seed}", seconds=heartbeat_seconds):
        source_arm, prepared, ledger, identity = validation_input(
            config, root, arm, seed, revalidate_from)
        existing = read_record(arm_root / "validation.json", identity)
    if existing is not None:
        if existing.get("manifest_sha256") != prepared["export"]["manifest_sha256"]:
            raise ValueError("validation is bound to a different checkpoint")
        print(f"resume validation {arm}/seed-{seed}: passed={existing['passed']}", flush=True)
        return existing
    device, dtype = experiment.resolve_device_dtype(config)
    set_seed(seed)
    started = time.monotonic()
    with _Heartbeat(f"reload/{arm}/seed-{seed}", seconds=heartbeat_seconds):
        progress(arm_root, "capture_fresh_source_teacher", arm=arm, seed=seed)
        source, tokenizer, _ = experiment.load_hf_model(
            config["model"], dtype, device, config["model_loader"], config["model_revision"])
        if tensor_digest(source.get_input_embeddings().weight) != prepared["metrics"]["teacher_source_vocab_digest"]:
            raise ValueError("fresh source teacher differs from preparation")
        metrics = {"data_manifest": copy.deepcopy(prepared["metrics"]["data_manifest"])}
        references = experiment._capture_references(config, config["eval"], source, tokenizer,
                                                    device, config["model"], metrics)
        del source, tokenizer
        release_models()  # references remain CPU-owned, with no teacher on the GPU.
        progress(arm_root, "fresh_packed_load", arm=arm, seed=seed)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = load_packed_model(source_arm / "checkpoint", device=device, dtype=dtype, fallback=False)
        tokenizer = AutoTokenizer.from_pretrained(source_arm / "checkpoint", local_files_only=True)
        residency_before = packed_residency(model)
        if residency_before["vocabulary_fingerprint"] != prepared["residency"]["vocabulary_fingerprint"]:
            raise ValueError("reloaded packed vocabulary fingerprint differs")
        load_peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
        expected = load_file(str(source_arm / "packed_probes.safetensors"))
        progress(arm_root, "reload_numerical_probes", arm=arm, seed=seed)
        actual = capture_probes(model, probe_inputs(expected), device, **probe_settings(config))
        reload_parity = compare_probes(actual, expected, reload=True)
        prototype_parity = compare_probes(actual, load_file(str(source_arm / "dense_probes.safetensors")))
        # Persist/report the useful numbers immediately, not just passed=False
        # after the worker exits. Thresholds are unchanged on a recovery run.
        save_record(arm_root / "reload_probe_report.json", {
            "identity": identity, "reload_parity": reload_parity, "prototype_parity": prototype_parity})
        print(json.dumps({"arm": arm, "seed": seed, "reload_parity": reload_parity,
                          "prototype_parity": prototype_parity}), flush=True)
        result = {"protocol": PROTOCOL, "status": "complete", "identity": identity,
                  "preparation_identity": prepared["identity"],
                  "arm": arm, "seed": seed, "worker_pid": os.getpid(),
                  "manifest_sha256": prepared["export"]["manifest_sha256"],
                  "artifact": ledger, "reload_parity": reload_parity,
                  "prototype_parity": prototype_parity, "residency_before": residency_before,
                  "peak_allocated_load_bytes": load_peak, "provider_competitive": False}
        if reload_parity["passed"] and prototype_parity["passed"]:
            progress(arm_root, "packed_full_quality", arm=arm, seed=seed)
            metrics.update(experiment.footprint_metrics(model, {}))
            experiment._run_evaluations(config, config["eval"], model, tokenizer, device,
                                        seed, references, metrics)
            experiment.validate_internal_data_disjointness(metrics["data_manifest"])
            result["quality"] = paired_quality(metrics, prepared["metrics"], arm, seed)
            result["metrics"] = metrics
        else:
            result["quality"] = {"passed": False, "skipped": "numerical parity failed"}
        result["residency_after"] = packed_residency(model)
        target = config["target_artifact_bytes"]
        size = ledger["measured_artifact_bytes"]
        result["size"] = {"target_bytes": target, "measured_bytes": size,
                          "delta_bytes": size - target, "within_budget": size <= target,
                          "within_one_percent_pairing": abs(size / target - 1) <= 0.01,
                          "no_more_than_one_percent_over": size <= target * 1.01}
        result["passed"] = all((reload_parity["passed"], prototype_parity["passed"],
                                result["quality"]["passed"], size <= target * 1.01))
        result["seconds"] = time.monotonic() - started
        save_record(arm_root / "validation.json", result)
        progress(arm_root, "complete", passed=result["passed"], arm=arm, seed=seed)
        return result


def summarize(root, config, seeds, *, revalidate_from=None):
    if revalidate_from is not None:
        root, revalidate_from = revalidation_paths(root, revalidate_from)
    rows = []
    missing = []
    for seed in seeds:
        for arm in ARMS:
            arm_root = root / f"{arm}_s{seed}"
            row = read_record(arm_root / "validation.json")
            if row is None:
                missing.append(f"{arm}_s{seed}")
                continue
            _, prepared, _, identity = validation_input(config, root, arm, seed, revalidate_from)
            if row.get("identity") != identity:
                raise ValueError("validation record identity mismatch")
            if row["manifest_sha256"] != prepared["export"]["manifest_sha256"]:
                raise ValueError("validation artifact identity mismatch")
            rows.append({"arm": arm, "seed": seed, "passed": row["passed"],
                         "measured_artifact_bytes": row["artifact"]["measured_artifact_bytes"],
                         "size": row["size"], "reload_parity": row["reload_parity"]["passed"],
                         "prototype_parity": row["prototype_parity"]["passed"],
                         "quality": row["quality"]["passed"],
                         **({k: v for k, v in row_for(row).items()
                             if k not in {"arm", "seed", "measured_artifact_bytes"}}
                            if row.get("metrics") else {})})
    result = {"protocol": PROTOCOL, "complete": not missing, "missing": missing, "rows": rows,
              "revalidate_from": str(Path(revalidate_from).resolve()) if revalidate_from is not None else None,
              "seeds": list(seeds), "artifact_validated_arms": [arm for arm in ARMS
                  if not missing and all(r["passed"] for r in rows if r["arm"] == arm)],
              "provider_competitive": False, "independent_confirmation": False,
              "interpretation": "Packed artifact conformance on reused development inputs. No provider, speed, or independent-validation win is claimed."}
    write_result(str(root / "summary.json"), result)
    return result


def run(config_path, output_dir, *, seeds=(0,), heartbeat_seconds=60, prepare_only=False,
        revalidate_from=None):
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    if revalidate_from is not None:
        if prepare_only:
            raise ValueError("revalidation cannot be combined with preparation")
        root, revalidate_from = revalidation_paths(root, revalidate_from)
    root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root / ".runner.lock"):
        marker = {"protocol": PROTOCOL, "config": config, "source": source_identity(),
                  "runtime": runtime_identity()}
        if revalidate_from is not None:
            marker["revalidate_from"] = str(revalidate_from)
        old = read_record(root / "run_identity.json")
        if old is not None and old != marker:
            raise ValueError("run identity changed; keep old results and use a new output directory")
        save_record(root / "run_identity.json", marker)
        write_result(str(root / "environment.json"), environment_record())
        # Never write caches into the original run even if a notebook inherited
        # that environment variable from the earlier experiment.
        os.environ["ROTQUANT_TOKEN_CACHE_DIR"] = str(root / "token_cache")
        try:
            for seed in seeds:
                release_models()
                if revalidate_from is None:
                    prepare_seed(config, root, seed, heartbeat_seconds)
                else:
                    print(f"checkpoint-only revalidation from {revalidate_from}; no calibration/quantization", flush=True)
                    with _Heartbeat(f"verify-revalidation/seed-{seed}", seconds=heartbeat_seconds):
                        for arm in ARMS:
                            validation_input(config, root, arm, seed, revalidate_from)
                release_models()
                if not prepare_only:
                    for arm in ARMS:
                        progress(root, "fresh_process_validation", arm=arm, seed=seed)
                        try:
                            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                                       "--config", str(Path(config_path).resolve()),
                                       "--output-dir", str(root), "--seed", str(seed),
                                       "--worker-arm", arm, "--heartbeat-seconds", str(heartbeat_seconds)]
                            if revalidate_from is not None:
                                command.extend(["--revalidate-from", str(revalidate_from)])
                            subprocess.run(command, cwd=ROOT, check=True)
                        finally:
                            # A failed worker must still appear as failed/missing,
                            # not vanish from a success-only partial summary.
                            summarize(root, config, seeds, revalidate_from=revalidate_from)
            result = summarize(root, config, seeds, revalidate_from=revalidate_from)
            progress(root, "prepared" if prepare_only else "complete", summary=result)
            return result
        except BaseException as exc:
            progress(root, "failed", error_type=type(exc).__name__, error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--heartbeat-seconds", type=float, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--assess-only", action="store_true")
    parser.add_argument("--revalidate-from", type=Path,
                        help="read original checkpoints/probes and write a new validation-only run; never requantize")
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    seeds = tuple(args.seed or [0])
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("seeds must be unique nonnegative integers")
    if args.heartbeat_seconds < 0:
        parser.error("heartbeat interval must be nonnegative")
    if args.revalidate_from is not None:
        if args.prepare_only:
            parser.error("--revalidate-from cannot be combined with --prepare-only")
        args.output_dir, args.revalidate_from = revalidation_paths(args.output_dir, args.revalidate_from)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "arms": ARMS, "seeds": seeds,
              "backbone_quantizations_per_seed": 0 if args.revalidate_from else 1,
              "revalidate_from": str(args.revalidate_from) if args.revalidate_from else None,
              "source": source_identity(),
              "target_bytes": config["target_artifact_bytes"],
              "checks": config["packed_validation"], "provider_benchmark": False,
              "storage_note": (
                  "Read existing artifacts/probes in place. No Hessians, quantization or re-export; new logs/results/token cache and source downloads still need space."
                  if args.revalidate_from else
                  "Two ~3.5 GB artifacts plus ~17 GB Hessians, source downloads, caches and staging space; reserve at least 45 GB of persistent free space.")}, indent=2))
        return
    enable_default_logging()
    if args.assess_only:
        result = summarize(args.output_dir, config, seeds, revalidate_from=args.revalidate_from)
        print(json.dumps(result, indent=2))
        if not result["complete"]:
            raise SystemExit("Validation incomplete")
    elif args.worker_arm:
        if len(seeds) != 1:
            parser.error("worker requires exactly one seed")
        arm_root = args.output_dir / f"{args.worker_arm}_s{seeds[0]}"
        with writer_lock(arm_root / ".worker.lock"):
            result = validate_worker(config, args.output_dir, args.worker_arm, seeds[0],
                                     args.heartbeat_seconds, revalidate_from=args.revalidate_from)
        if not result["passed"]:
            raise SystemExit("Packed validation failed; inspect validation.json before spending more GPU time")
    else:
        run(args.config, args.output_dir, seeds=seeds, heartbeat_seconds=args.heartbeat_seconds,
            prepare_only=args.prepare_only, revalidate_from=args.revalidate_from)


if __name__ == "__main__":
    main()
