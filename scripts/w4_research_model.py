"""Source preparation and exact checkpoint export for W4 research arms."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from rotquant.block_train import (
    BlockRotationTrainConfig,
    collect_teacher_calls,
    distill_packed_model,
)
from rotquant.checkpoint import save_packed_checkpoint
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig
from rotquant.quantize import QuantConfig
from rotquant.vocabulary import (
    VocabularyConfig,
    install_packed_vocabulary,
    quantize_vocabulary,
    tied_modules,
)
from scripts import run_experiment as experiment
from scripts.native_gpu_workflow import write_json
from scripts.w4_research_data import batches
from scripts.w4_research_plan import arm_record


def load_source(config):
    model, tokenizer, loader = experiment.load_hf_model(
        config["model"], torch.float16, config["device"],
        config["model_loader"], config["model_revision"])
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config["model"], revision=config["model_revision"])
    return model.eval(), tokenizer, processor, loader


def capture_recovery(model, config, data):
    settings = config["recovery"]
    documents = [item for part in ("train", "validation", "selection")
                 for item in data["documents"][f"recovery_{part}"]]
    return collect_teacher_calls(model, batches(documents), config["device"],
                                 max_batches=len(documents),
                                 topk=settings["distill_teacher_topk"],
                                 temperature=settings["distill_temperature"])


def quantize_backbone(model, tokenizer, config, data, root):
    cfg = copy.deepcopy(config)
    cfg["hessian_cache_dir"] = str(Path(root) / "hessians")
    metrics = {"data_manifest": {}}
    qcfg = QuantConfig(**cfg["quant"], seed=cfg["seed"])
    pcfg = PatchConfig(quant=qcfg, **cfg["patch"], seed=cfg["seed"])
    calibration = experiment._prepare_calibration(
        cfg, model, tokenizer, cfg["device"], qcfg, pcfg, metrics)
    expected = [v["source_row"] for v in data["documents"]["calibration"]]
    actual = metrics["data_manifest"]["hessian_calibration"]
    frozen = experiment.token_batch_manifest(
        batches(data["documents"]["calibration"]), dataset="allenai/c4", split="train",
        revision=cfg["calib_dataset_revision"], skip=0, seq_len=cfg["calib_seq_len"])
    if actual["source_rows"] != expected or actual["token_hashes"] != frozen["token_hashes"]:
        raise ValueError("actual Hessian calibration differs from frozen inputs")
    experiment._apply_quantization(cfg, model, pcfg, calibration, metrics)
    return metrics


def quantize_vocabulary_in_place(model, config, root):
    embedding, _ = tied_modules(model)
    owner = quantize_vocabulary(
        embedding.weight, VocabularyConfig(**config["vocabulary"], seed=config["seed"]),
        cache_dir=Path(root) / "vocabulary_cache")
    owner.projection_mode = "dense_equivalent"
    owner.to(device=config["device"], dtype=torch.float16)
    install_packed_vocabulary(model, owner)


def apply_recovery(model, config, calls, destination):
    cfg = dict(config["recovery"])
    cfg["distill_checkpoint_path"] = str(Path(destination) / "recovery.pt")
    return distill_packed_model(model, calls, config["device"], torch.float16,
                                BlockRotationTrainConfig(**cfg))


def export_packed(model, tokenizer, processor, loader, config, destination, identity,
                  *, protocol="qwen35-w4-research-v1"):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.fallback = False
            module._fp_cache = None
    return save_packed_checkpoint(
        model, Path(destination) / "checkpoint", base_model=config["model"],
        base_model_revision=config["model_revision"], model_loader=loader,
        tokenizer=tokenizer, processor=processor,
        deployment_metadata={"protocol": protocol, "identity": identity})


def prepare_arm(arm, config, data, root, destination, identity, evaluate):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cfg = arm.config(config)
    model, tokenizer, processor, loader = load_source(cfg)
    calls = capture_recovery(model, cfg, data) if arm.recovery else None
    metrics = quantize_backbone(model, tokenizer, cfg, data, root)
    quantize_vocabulary_in_place(model, cfg, root)
    if calls is not None:
        metrics["recovery"] = apply_recovery(model, cfg, calls, Path(root) / "recovery_state" / arm.name)
    selection = evaluate(model, tokenizer, cfg["device"], destination)
    exported = export_packed(model, tokenizer, processor, loader, cfg, destination, identity)
    result = {"arm": arm_record(arm), "selection": selection,
              "artifact_bytes": exported["artifact_bytes"], "export": exported,
              "metrics": metrics, "identity": identity}
    write_json(destination / "prepared.json", result)
    return result
