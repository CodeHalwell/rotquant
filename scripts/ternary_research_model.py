"""Prepare trained ternary checkpoints and recover an exact saved parent."""
from __future__ import annotations

import copy
import gc
import time
from pathlib import Path

import torch

from rotquant.checkpoint import load_packed_model, verify_checkpoint
from rotquant.ternary_numerics import stable_qwen_fallbacks
from scripts.native_gpu_workflow import fingerprint, write_json
from scripts.ternary_research_plan import PROTOCOL, arm_record, precision_scope
from scripts.w4_research_data import batches
from scripts.w4_research_model import (
    apply_recovery,
    capture_recovery,
    export_packed,
    load_source,
    quantize_backbone,
    quantize_vocabulary_in_place,
)


def train_ternary(model, arm, config, data, root, identity):
    from rotquant.ternary import TernaryConfig
    from rotquant.ternary_train import TernaryTrainConfig, train_and_patch_ternary

    options = copy.deepcopy(config["ternary"])
    options["quant"] = TernaryConfig(**{**options["quant"], "seed": config["seed"]})
    options.update(source_id=identity, data_id=data["fingerprint"],
                   checkpoint_dir=str(Path(root) / "ternary_state" / arm.name))
    if arm.method == "ternary_ptq":
        options["steps"] = 0
    stats = {}
    with stable_qwen_fallbacks(model) as numerics:
        train_and_patch_ternary(
            model, batches(data["documents"]["recovery_train"]),
            batches(data["documents"]["recovery_validation"]), TernaryTrainConfig(**options),
            device=config["device"], stats_out=stats)
        stats["numerics"] = numerics
    return stats


def bind_recovery(root, arm, identity, data, parent):
    manifest = verify_checkpoint(parent)
    binding = {"identity": identity, "data": data["fingerprint"],
               "parent_manifest": fingerprint(manifest)}
    directory = Path(root) / "recovery_state" / arm.name
    path = directory / "binding.json"
    if path.exists():
        import json
        if json.loads(path.read_text()) != binding:
            raise ValueError("recovery parent/data changed; choose a fresh result root")
    write_json(path, binding)
    return directory


def prepare_mlp_remainder(model, tokenizer, config, data, root, identity):
    """Use the same source-calibrated W4 remainder in both MLP study arms."""
    remainder = copy.deepcopy(config)
    remainder["quant"].update(bits=4, scale_refit="hessian")
    remainder["patch"]["exclude"] = [*remainder["patch"]["exclude"], ".mlp."]
    remainder["stage_source_fingerprint"] = fingerprint({
        "identity": identity, "stage": "source-w4-non-mlp-v1"})
    return quantize_backbone(model, tokenizer, remainder, data, Path(root) / "mlp_remainder")


def recover(model, config, calls, directory):
    from rotquant.block_train import _distill_step_calls

    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    stats = apply_recovery(model, config, calls, directory)
    settings = config["recovery"]
    train = calls[:settings["distill_train_batches"]]
    stats.update(wall_seconds_this_attempt=time.monotonic() - started,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                 optimiser_seen_tokens=sum(call.inputs["input_ids"].numel()
                     for step in range(1, stats["steps_run"] + 1)
                     for call in _distill_step_calls(train, step, settings["distill_batches_per_step"])))
    return stats


def prepare_arm(arm, config, data, root, destination, identity, evaluate, *, parent=None):
    cfg = arm.config(config)
    model, tokenizer, processor, loader = load_source(cfg)
    metrics = {}
    if arm.parent:
        if parent is None:
            raise ValueError("recovery needs the exact saved parent checkpoint")
        recovery_dir = bind_recovery(root, arm, identity, data, parent)
        calls = capture_recovery(model, cfg, data)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_packed_model(parent, device=cfg["device"], dtype=torch.float16, fallback=True)
        metrics["recovery"] = recover(model, cfg, calls, recovery_dir)
        metrics["parent_checkpoint"] = str(parent)
    else:
        if arm.method.startswith("ternary"):
            if cfg["ternary"]["scope"] == "mlp":
                metrics["remaining_w4"] = prepare_mlp_remainder(model, tokenizer, cfg, data, root, identity)
            metrics["block_qat"] = train_ternary(model, arm, cfg, data, root, identity)
        else:
            metrics["quantization"] = quantize_backbone(model, tokenizer, cfg, data, root)
        quantize_vocabulary_in_place(model, cfg, root)
    destination = Path(destination)
    selection = evaluate(model, tokenizer, cfg["device"], destination)
    exported = export_packed(model, tokenizer, processor, loader, cfg, destination, identity, protocol=PROTOCOL)
    result = {"arm": arm_record(arm), "selection": selection,
              "artifact_bytes": exported["artifact_bytes"], "export": exported,
              "metrics": metrics, "identity": identity, "precision_scope": precision_scope(arm, cfg)}
    write_json(destination / "prepared.json", result)
    return result
