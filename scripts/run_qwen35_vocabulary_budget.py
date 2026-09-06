#!/usr/bin/env python3
"""Resumable nine-arm vocabulary/backbone factorial, without teacher mutation.

Only the registered development screen is runnable here. Recovery, rebudgeting
and external promotion require the screen decision and their own next protocol.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rotquant.patch import PatchConfig
from rotquant.quantize import QuantConfig
from rotquant.utils import enable_default_logging, environment_record, set_seed, write_result
from rotquant.vocabulary import (
    VocabularyConfig,
    file_digest,
    quantize_vocabulary,
    tensor_digest,
    tied_modules,
    vocabulary_prototype,
)
from scripts import run_experiment as experiment
from scripts.run_qwen35_next_stage import _code_revision, _gpu_snapshot, _Heartbeat, _timestamp

PROTOCOL = "qwen35-vocabulary-budget-screen-v1"
DEFAULT_CONFIG = ROOT / "configs/qwen35_4b_vocabulary_budget_cuda.yaml"
ARMS = tuple((backbone, vocabulary) for backbone in (16, 4, 5) for vocabulary in (16, 8, 6))


def arm_name(backbone, vocabulary):
    return f"b{backbone}_v{vocabulary}"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_identity():
    # Include uncommitted implementation bytes for honest local smoke identities.
    files = sorted(path for directory in ("rotquant", "scripts")
                   for path in (ROOT / directory).rglob("*.py") if "__pycache__" not in path.parts)
    return {"revision": _code_revision(), "source_fingerprint": digest({
        str(path.relative_to(ROOT)): file_digest(path) for path in files})}


def resolved_config(path):
    config = experiment.load_config(str(path))
    for suites in ("logit_fidelity_suites", "trajectory_suites"):
        for suite in config.get("eval", {}).get(suites, {}).values():
            if suite.get("prompt_file"):
                candidate = Path(suite["prompt_file"])
                candidate = candidate if candidate.is_absolute() else ROOT / candidate
                suite["prompt_file"] = str(candidate)
    if config.get("patch", {}).get("dynamic") or config.get("patch", {}).get("train_rotation"):
        raise ValueError("factorial screen cannot include dynamic allocation or recovery")
    if config.get("patch", {}).get("activation_bits") is not None:
        raise ValueError("factorial screen does not include activation quantization")
    return config


def trial_identity(config, seed, backbone, vocabulary, source):
    prompts = {suite["prompt_file"]: file_digest(Path(suite["prompt_file"]))
               for key in ("logit_fidelity_suites", "trajectory_suites")
               for suite in config.get("eval", {}).get(key, {}).values()
               if suite.get("prompt_file")}
    return {"protocol": PROTOCOL, "config": config, "seed": seed,
            "backbone_bits": backbone, "vocabulary_bits": vocabulary,
            "source": source, "prompt_files": prompts,
            "runtime": runtime_identity()}


def runtime_identity():
    versions = {}
    for package in ("torch", "transformers", "datasets", "safetensors", "fast-hadamard-transform"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"versions": versions, "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "disable_fast_hadamard": os.environ.get("ROTQUANT_DISABLE_FAST_HADAMARD", ""),
            "tf32": False}


def load_completed(path, identity):
    checksum = path.with_suffix(".sha256")
    if not path.exists():
        return None
    if not checksum.exists() or json.loads(checksum.read_text()).get("sha256") != file_digest(path):
        raise ValueError(f"result checksum missing or mismatched: {path}")
    payload = json.loads(path.read_text())
    if payload.get("identity") != identity or payload.get("trial_fingerprint") != digest(identity):
        raise ValueError(f"result identity mismatch; use a new output directory: {path}")
    if payload.get("status") != "complete":
        return None
    return payload


def component_ledger(model, metrics, vocabulary, config):
    embedding, _ = tied_modules(model)
    shared = embedding.weight
    seen = set()
    vision = small = 0
    for name, parameter in model.named_parameters():
        if parameter is shared or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        size = parameter.numel() * parameter.element_size()
        if "visual" in name or "vision" in name:
            vision += size
        else:
            small += size
    dense_bytes = shared.numel() * shared.element_size()
    retained_parameters = dense_bytes + vision + small
    registered_bytes = int(metrics["registered_model_bytes"])
    buffer_bytes = registered_bytes - retained_parameters
    if buffer_bytes < 0:
        raise ValueError("component ledger exceeds registered model storage")
    existing = int(metrics["complete_persistent_model_bytes"])
    projected_vocabulary = vocabulary.get("packed_vocabulary_payload_bytes", dense_bytes)
    projected = existing - dense_bytes + projected_vocabulary
    overhead = int(config["artifact_overhead_estimate_bytes"])
    return {
        "vocabulary_parameters": shared.numel(), "vocabulary_actual_dense_bytes": dense_bytes,
        "vocabulary_projected_packed_bytes": projected_vocabulary,
        "backbone_packed_bytes": metrics.get("packed_weight_bytes", 0),
        "backbone_quantized_parameters": int(metrics.get("fp16_weight_bytes", 0)) // 2,
        "retained_vision_parameter_bytes": vision,
        "other_retained_parameter_bytes": small,
        "persistent_registered_buffer_bytes": buffer_bytes,
        "registered_tensor_bytes": registered_bytes,
        "registered_components_reconcile": retained_parameters + buffer_bytes == registered_bytes,
        "backbone_effective_bpw": metrics.get("bits_per_weight_mean", 16.0),
        "backbone_codebook_bytes": metrics.get("codebook_bytes", 0),
        "actual_persistent_model_bytes": existing,
        "quality_runtime_model_bytes": metrics["quality_runtime_model_bytes"],
        "projected_persistent_bytes": projected,
        "container_overhead_estimate_bytes": overhead,
        "projected_artifact_bytes": projected + overhead,
        "target_artifact_bytes": config["target_artifact_bytes"],
        "measured_artifact_bytes": None, "artifact_gate_passed": False,
        "boundary": "Dense vocabulary prototype; projected bytes are not a measured export.",
    }


def resource_ledger(model, config, pcfg, root):
    """Shape-based lower bounds; filesystem free space is not a Drive quota API."""
    from rotquant.calibrate import _iter_linears

    embedding, _ = tied_modules(model)
    targets = list(_iter_linears(model, pcfg.include, pcfg.exclude))
    hessian_sizes = [module.in_features ** 2 * 4 for _, module in targets]
    layers_per_pass = int(config.get("hessian_layers_per_pass", 1))
    group_bytes = [sum(hessian_sizes[i:i + layers_per_pass])
                   for i in range(0, len(hessian_sizes), layers_per_pass)]
    suites = [config.get("eval", {}).get("logit_fidelity", {}),
              *config.get("eval", {}).get("logit_fidelity_suites", {}).values()]
    reference_upper_bound = sum(
        int(suite.get("batches", 0)) * max(int(suite.get("prompt_len", 0)) - 1, 0)
        * embedding.num_embeddings * 2 for suite in suites
    )
    return {
        "source_registered_bytes": experiment.footprint_metrics(model, {})["registered_model_bytes"],
        "source_vocabulary_dense_bytes": embedding.weight.numel() * embedding.weight.element_size(),
        "source_hessian_tensor_bytes": sum(hessian_sizes),
        "largest_hessian_group_tensor_bytes": max(group_bytes, default=0),
        "target_layers": len(targets),
        "host_fp16_logit_reference_upper_bound_bytes": reference_upper_bound,
        "filesystem_reported_free_bytes": shutil.disk_usage(root).free,
        "gpu_free_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
        "boundary": (
            "Shape estimates exclude serialization, activations, solver workspaces and "
            "other process memory; filesystem free space may not reflect Drive quota. "
            "No total-runtime or fits-in-memory guarantee. Hessians are shared across W4/W5."
        ),
    }


def run_screen(config_path=DEFAULT_CONFIG, output_dir="results/vocabulary-budget", *,
               seeds=(0,), force=False, heartbeat_seconds=60):
    config = resolved_config(config_path)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # One writer per output root; prevents interleaved chunk/result publication.
    with (root / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another vocabulary runner owns {root}") from exc
        return _run_locked(config, root, seeds, force, heartbeat_seconds)


def _run_locked(config, root, seeds, force, heartbeat_seconds):
    from scripts.assess_qwen35_vocabulary_budget import assess

    source = source_identity()
    environment = environment_record()
    write_result(str(root / "environment.json"), {"source": source, **environment})
    result_dir = root / "model_trials"
    result_dir.mkdir(exist_ok=True)
    os.environ.setdefault("ROTQUANT_TOKEN_CACHE_DIR", str(root / "token_cache"))
    device, dtype = experiment.resolve_device_dtype(config)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # A separately invoked screen must never inherit a teacher captured by a
    # different caller/dtype. Reuse references only inside this frozen run.
    experiment._LOGIT_REFERENCE_CACHE.clear()
    experiment._TRAJECTORY_REFERENCE_CACHE.clear()
    results = []

    def update_progress(phase, seed, *, group=None, arm=None):
        write_result(str(root / "progress.json"), {
            "protocol": PROTOCOL, "status": "running", "phase": phase,
            "group": group, "arm": arm, "seed": seed, "pid": os.getpid(),
            "updated_at": _timestamp(), "completed": len(results),
            "expected": len(ARMS) * len(seeds),
        })

    for seed in seeds:
        set_seed(seed)
        owners = {}
        source_vocab_digest = None
        for backbone in (16, 4, 5):
            pending = []
            for vocabulary in (16, 8, 6):
                arm = arm_name(backbone, vocabulary)
                identity = trial_identity(config, seed, backbone, vocabulary, source)
                path = result_dir / f"{arm}_s{seed}.json"
                saved = None if force else load_completed(path, identity)
                if saved is not None:
                    print(f"resume {arm}/seed-{seed} (verified complete result)", flush=True)
                    results.append(saved)
                else:
                    pending.append((vocabulary, arm, identity, path))
            if not pending:
                continue
            label = f"backbone-W{backbone}/seed-{seed}"
            current = copy.deepcopy(config)
            current.update(seed=seed, stage_code_revision=source["revision"],
                           stage_source_fingerprint=source["source_fingerprint"],
                           stage_runtime_fingerprint=digest(runtime_identity()),
                           hessian_cache_dir=str(root / "hessians"))
            current["quant"]["bits"] = backbone if backbone != 16 else 4
            current["patch"]["enabled"] = backbone != 16
            metrics = {"data_manifest": {}}
            started = time.monotonic()
            try:
                with _Heartbeat(label, seconds=heartbeat_seconds):
                    update_progress("source_load", seed, group=label)
                    print(f"[{_timestamp()}] load source for {label}", flush=True)
                    model, tokenizer, selected_loader = experiment.load_hf_model(
                        config["model"], dtype, device, config["model_loader"],
                        config.get("model_revision"))
                    model.eval()
                    metrics["model_loader"] = selected_loader
                    embedding, head = tied_modules(model)
                    vocab_digest = tensor_digest(embedding.weight)
                    if source_vocab_digest is not None and vocab_digest != source_vocab_digest:
                        raise RuntimeError("reloaded source vocabulary differs between backbones")
                    source_vocab_digest = vocab_digest
                    qcfg = QuantConfig(**current["quant"], seed=seed)
                    pcfg = PatchConfig(quant=qcfg, **current["patch"], seed=seed)
                    resources = resource_ledger(model, current, pcfg, root)
                    write_result(str(root / f"resources_b{backbone}_s{seed}.json"), resources)
                    print("resource preflight: " + json.dumps(resources), flush=True)
                    update_progress("source_calibration", seed, group=label)
                    print(f"[{_timestamp()}] source calibration for {label}", flush=True)
                    calibration = experiment._prepare_calibration(
                        current, model, tokenizer, device, qcfg, pcfg, metrics)
                    update_progress("teacher_capture", seed, group=label)
                    print(f"[{_timestamp()}] capture unchanged FP16 teacher for {label}", flush=True)
                    references = experiment._capture_references(
                        current, current["eval"], model, tokenizer, device, config["model"], metrics)
                    experiment.validate_internal_data_disjointness(metrics["data_manifest"])
                    # Quantize vocabulary from source; do not mutate teacher/model yet.
                    for vocabulary, _, _, _ in pending:
                        if vocabulary == 16 or vocabulary in owners:
                            continue
                        update_progress(f"vocabulary_w{vocabulary}", seed, group=label)
                        print(f"[{_timestamp()}] prepare W{vocabulary} vocabulary", flush=True)
                        owners[vocabulary] = quantize_vocabulary(
                            embedding.weight, VocabularyConfig(bits=vocabulary, seed=seed,
                                                               **config["vocabulary"]),
                            cache_dir=(root / "vocabulary_cache" / source["source_fingerprint"]
                                       / digest(runtime_identity())),
                            progress=lambda event: print(json.dumps(event), flush=True))
                    if tensor_digest(embedding.weight) != source_vocab_digest:
                        raise RuntimeError("vocabulary quantizer mutated the source teacher")
                    update_progress("backbone_quantization", seed, group=label)
                    print(f"[{_timestamp()}] patch {label} (once for all vocabulary arms)", flush=True)
                    experiment._apply_quantization(current, model, pcfg, calibration, metrics)
                    for vocabulary, arm, identity, path in pending:
                        print(f"[{_timestamp()}] run {arm}/seed-{seed} {_gpu_snapshot()}", flush=True)
                        update_progress("evaluation", seed, group=label, arm=arm)
                        arm_metrics = copy.deepcopy(metrics)
                        context = (vocabulary_prototype(model, owners[vocabulary])
                                   if vocabulary != 16 else nullcontext({
                                       "mode": "source_vocabulary", "source_digest": vocab_digest,
                                       "tied_parameter": True}))
                        arm_start = time.monotonic()
                        with context as vocabulary_record:
                            arm_metrics["vocabulary"] = dict(vocabulary_record)
                            arm_metrics["teacher_source_vocab_digest"] = source_vocab_digest
                            arm_metrics["teacher_captured_before_mutation"] = True
                            arm_metrics["byte_ledger"] = component_ledger(
                                model, arm_metrics, vocabulary_record, config)
                            experiment._run_evaluations(
                                current, current["eval"], model, tokenizer, device, seed,
                                references, arm_metrics)
                            experiment.validate_internal_data_disjointness(arm_metrics["data_manifest"])
                        payload = {"protocol": PROTOCOL, "status": "complete", "arm": arm,
                                   "seed": seed, "identity": identity,
                                   "trial_fingerprint": digest(identity), "metrics": arm_metrics,
                                   "seconds": time.monotonic() - arm_start,
                                   "backbone_group_elapsed_seconds": time.monotonic() - started}
                        write_result(str(path), payload)
                        write_result(str(path.with_suffix(".sha256")), {"sha256": file_digest(path)})
                        results.append(payload)
                        write_result(str(root / "screen_summary.json"), assess(results, seeds=seeds))
                        print(f"complete {arm}/seed-{seed}: " + json.dumps({
                            "kl": arm_metrics.get("logit_fidelity", {}).get("mean_teacher_kl"),
                            "top1": arm_metrics.get("logit_fidelity", {}).get("top1_agreement"),
                            "projected_bytes": arm_metrics["byte_ledger"]["projected_artifact_bytes"],
                        }), flush=True)
                del embedding, head, model, tokenizer, calibration, references
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except BaseException as exc:
                write_result(str(root / "progress.json"), {"status": "failed", "group": label,
                             "error_type": type(exc).__name__, "error": str(exc),
                             "updated_at": _timestamp(), "completed": len(results)})
                raise
    summary = assess(results, seeds=seeds)
    write_result(str(root / "screen_summary.json"), summary)
    write_result(str(root / "progress.json"), {"status": "complete", "completed": len(results),
                 "updated_at": _timestamp(), "finalists": summary["finalists"]})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=60)
    args = parser.parse_args()
    seeds = tuple(args.seed or [0])
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("seeds must be unique nonnegative integers")
    if args.dry_run:
        config = resolved_config(args.config)
        print(json.dumps({"protocol": PROTOCOL, "source": source_identity(),
              "arms": [arm_name(*arm) for arm in ARMS], "seeds": seeds,
              "arm_count": len(ARMS) * len(seeds), "backbone_quantizations_per_seed": 2,
              "vocabulary": config["vocabulary"], "output_dir": str(args.output_dir),
              "provider_promotion_enabled": False}, indent=2))
        return
    enable_default_logging()
    run_screen(args.config, args.output_dir, seeds=seeds, force=args.force,
               heartbeat_seconds=args.heartbeat_seconds)


if __name__ == "__main__":
    main()
