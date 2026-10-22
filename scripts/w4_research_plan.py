"""Frozen controls and exact evidence identities for the W4 research notebook."""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.native_gpu_workflow import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "qwen35-w4-research-v1"
DEFAULT_CONFIG = ROOT / "configs/qwen35_4b_w4_research_cuda.yaml"


@dataclass(frozen=True)
class Arm:
    name: str
    bits: int = 4
    group_size: int = 128
    scale_refit: str = "hessian"
    code_refit_passes: int = 0
    recovery: bool = False

    def config(self, base):
        cfg = copy.deepcopy(base)
        cfg["quant"].update(bits=self.bits, group_size=self.group_size,
                            scale_refit=self.scale_refit,
                            scale_refit_passes=cfg["research"]["scale_refit_passes"],
                            code_refit_passes=self.code_refit_passes)
        return cfg


def initial_arms(config):
    return [Arm("w5_v6", 5, 128, "none"), Arm("w4_g128_mse", scale_refit="none"),
            *[Arm(f"w4_g{g}_h", group_size=g) for g in config["research"]["groups"]]]


def validate_config(config):
    if config.get("dtype") != "float16" or not re.fullmatch(r"[0-9a-f]{40}", config.get("model_revision", "")):
        raise ValueError("a pinned FP16 source is required")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("calib_dataset_revision", "")):
        raise ValueError("a pinned calibration dataset is required")
    if config["research"]["groups"] != [128, 64, 32]:
        raise ValueError("this protocol compares group sizes 128, 64 and 32")
    if config["vocabulary"]["bits"] != 6 or config["quant"]["scale_bits"] != 8:
        raise ValueError("hold vocabulary W6 and scale8 fixed")
    validate_recovery(config)
    for key, value in config["research"].items():
        if key not in {"groups", "speed_contexts", "task_margin"} and (
                type(value) is not int or value < 1):
            raise ValueError(f"positive integer required: research.{key}")
    if not 0 < config["research"]["task_margin"] < 1:
        raise ValueError("task_margin must be between zero and one")
    validate_bounds(config)
    return config


def validate_bounds(config):
    study = config["research"]
    contexts = study["speed_contexts"]
    if not contexts or len(set(contexts)) != len(contexts) or any(
            type(n) is not int or not 2 <= n <= study["long_context"] for n in contexts):
        raise ValueError("unique speed contexts must fit the long document")
    if (study["long_context"] + study["speed_decode_steps"] > 8192
            or study["speed_decode_steps"] > config["calib_seq_len"]
            or study["public_samples_per_benchmark"] > 541):
        raise ValueError("data or runtime context capacity exceeded")
    if config["n_calib"] < 1 or config["calib_seq_len"] < 17:
        raise ValueError("positive calibration count and at least 17 tokens required")


def validate_recovery(config):
    recovery, study = config["recovery"], config["research"]
    if (recovery["distill_train_rotations"] or recovery["distill_lora_rank"]
            or recovery["distill_code_refresh_interval"] or recovery["distill_ce_weight"]):
        raise ValueError("W4 recovery is fixed-rotation, fixed-code, scale-only teacher KL")
    if (not recovery["distill_train_scales"] or recovery["distill_steps"] < 1
            or recovery["distill_kl_weight"] <= 0 or recovery["distill_temperature"] <= 0):
        raise ValueError("recovery requires active scale training and positive teacher KL")
    for part in ("train", "validation", "selection"):
        if recovery[f"distill_{part}_batches"] != study[f"recovery_{part}_documents"]:
            raise ValueError(f"recovery {part} batch/data count mismatch")


def source_files():
    paths = []
    for directory in ("rotquant", "scripts", "configs", "native", "integrations", "requirements", "research/eval_suites"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix in {".py", ".yaml", ".json", ".patch", ".txt", ".cpp", ".h", ".hpp", ".cu"})
    from scripts.native_hashing import digest
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def bind_run(root, config, runtime):
    identity = {"protocol": PROTOCOL, "config": config, "source": source_files(), "runtime": runtime}
    path = Path(root) / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("code, configuration or runtime changed; choose a new result root")
    write_json(path, identity)
    return fingerprint(identity)


def choose_candidate(rows):
    eligible = [r for r in rows if r["arm"]["bits"] == 4]
    if not eligible or any(not math.isfinite(r["selection"]["mean_teacher_kl"]) for r in eligible):
        raise ValueError("complete finite W4 selection results are required")
    return min(eligible, key=lambda r: (r["selection"]["mean_teacher_kl"], r["artifact_bytes"], r["arm"]["name"]))


def arm_record(arm):
    return asdict(arm)
