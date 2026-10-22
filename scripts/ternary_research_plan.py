"""Pinned ternary block-QAT experiment and comparable fresh controls."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.native_gpu_workflow import fingerprint, write_json
from scripts.w4_research_plan import ROOT, source_files, validate_bounds

PROTOCOL = "qwen35-ternary-research-v1"
DEFAULT_CONFIG = ROOT / "configs/qwen35_4b_ternary_research_cuda.yaml"
LABELS = ("fp16", "w6_v6", "w5_v6", "w4_v6", "w4_v6_recovered",
          "ternary_ptq", "ternary_blockqat", "ternary_recovered", "bonsai2_4b")


@dataclass(frozen=True)
class Arm:
    name: str
    bits: int = 4
    method: str = "scalar"
    parent: str | None = None

    def config(self, base):
        config = copy.deepcopy(base)
        config["quant"].update(bits=self.bits, group_size=128,
                               scale_refit="hessian" if self.bits == 4 else "none",
                               scale_refit_passes=config["research"]["scale_refit_passes"],
                               code_refit_passes=0)
        return config


def arms():
    return [Arm("w5_v6", 5), Arm("w6_v6", 6), Arm("w4_v6"),
            Arm("w4_v6_recovered", parent="w4_v6"),
            Arm("ternary_ptq", 2, "ternary_ptq"),
            Arm("ternary_blockqat", 2, "ternary_qat"),
            Arm("ternary_recovered", 2, "ternary_qat", "ternary_blockqat")]


def arm_record(arm):
    return asdict(arm)


def precision_scope(arm, config):
    if arm.method.startswith("ternary"):
        target = config["ternary"]["scope"]
        remainder = "; W4 remaining backbone" if target == "mlp" else ""
        return f"ternary {target}{remainder}; W6 tied vocabulary; protected source tensors; 2-bit slots/FP16 scales"
    return f"W{arm.bits} backbone / W6 tied vocabulary; protected source tensors"


def validate_config(config):
    if config.get("dtype") != "float16" or config.get("device") != "cuda":
        raise ValueError("the comparison requires FP16 CUDA execution")
    for key in ("model_revision", "calib_dataset_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", config.get(key, "")):
            raise ValueError(f"pin {key} to a full commit")
    if config["vocabulary"]["bits"] != 6 or config["quant"]["scale_bits"] != 8:
        raise ValueError("scalar controls require W6 vocabulary and scale8")
    validate_bounds(config)
    for key, value in config["research"].items():
        if key not in {"groups", "speed_contexts", "task_margin"} and (type(value) is not int or value < 1):
            raise ValueError(f"positive integer required: research.{key}")
    if not 0 < config["research"]["task_margin"] < 1:
        raise ValueError("task_margin must be between zero and one")
    validate_recovery(config)
    validate_ternary(config)
    if config["bonsai"]["packing"] not in {"PTQ1_0", "PQ2_0"}:
        raise ValueError("unsupported pinned Bonsai packing")
    return config


def validate_recovery(config):
    from rotquant.block_train import BlockRotationTrainConfig

    recovery = config["recovery"]
    BlockRotationTrainConfig(**{**recovery, "distill_checkpoint_path": "bound-at-run.pt"})
    if (recovery["distill_train_rotations"] or recovery["distill_lora_rank"]
            or recovery["distill_code_refresh_interval"] or not recovery["distill_train_scales"]
            or recovery["distill_steps"] < 1 or recovery["distill_kl_weight"] <= 0):
        raise ValueError("recovery requires fixed-code scale optimisation and teacher KL")
    for part in ("train", "validation", "selection"):
        if recovery[f"distill_{part}_batches"] != config["research"][f"recovery_{part}_documents"]:
            raise ValueError(f"recovery {part} count mismatch")


def validate_ternary(config):
    from rotquant.ternary import TernaryConfig
    from rotquant.ternary_train import TernaryTrainConfig

    ternary = config["ternary"]
    quant = TernaryConfig(**ternary["quant"])
    TernaryTrainConfig(**{**ternary, "quant": quant, "checkpoint_dir": "bound-at-run"})
    if quant.rotation_block != 128 or quant.group_size != 128 or ternary["max_blocks"]:
        raise ValueError("native full-model study requires FWHT128/group128 and all transformer blocks")
    if (ternary["train_batches"] != config["research"]["recovery_train_documents"]
            or ternary["validation_batches"] != config["research"]["recovery_validation_documents"]):
        raise ValueError("QAT counts must match the frozen fitting partitions")
    if ternary["steps"] < 1:
        raise ValueError("the block-QAT arm must train; PTQ separately uses zero steps")


def bind_run(root, config, runtime):
    identity = {"protocol": PROTOCOL, "config": config, "source": source_files(), "runtime": runtime}
    path = Path(root) / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("code, configuration or runtime changed; choose a new result root")
    write_json(path, identity)
    return fingerprint(identity)
