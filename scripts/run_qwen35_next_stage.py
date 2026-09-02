#!/usr/bin/env python3
"""Run the focused, resumable Qwen3.5-4B optimization ladder.

The default stage compares the two validated W4 controls with streamed GPTQ.
The more expensive W4A8/E8, recovery, and 8k-context KV stages are explicit
opt-ins. Results are reused only when the code revision and fully resolved
trial configuration match.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.utils import enable_default_logging, write_result
from scripts import run_experiment

PROTOCOL_VERSION = "qwen35-next-stage-v2"
MIN_RELIABLE_BOOTSTRAP_SAMPLES = 20
DEFAULT_STAGES = ("w4",)
REGISTERED_STAGES = ("w4", "w4a8", "ablation", "recovery", "long-kv")
PAIRED_ARMS = {
    "w4": (
        ("gptq_gaussian_w4", "gaussian_w4_control"),
        ("gptq_calibrated_w4", "calibrated_w4_control"),
    ),
    "w4a8": (
        ("optimized_w4", "promoted_w4"),
        ("w4a8", "promoted_w4"),
        ("w4a8_e8", "w4a8"),
    ),
    "ablation": (
        ("scale8_w4", "promoted_w4"),
        ("mean_bias_w4", "promoted_w4"),
        ("shared_fwht_w4", "promoted_w4"),
        ("butterfly_control_w4", "promoted_w4"),
        ("butterfly_hessian_w4", "butterfly_control_w4"),
        ("butterfly_hessian_w4", "promoted_w4"),
        ("butterfly_hessian_signs_w4", "butterfly_hessian_w4"),
        ("butterfly_hessian_signs_w4", "promoted_w4"),
        ("shared_butterfly_hessian_signs_w4", "butterfly_hessian_signs_w4"),
        ("shared_butterfly_hessian_signs_w4", "promoted_w4"),
    ),
    "recovery": (("recovered_w4", "unrecovered_w4"),),
    "long-kv": (
        ("promoted_w4_e8", "source_fp16_e8"),
        ("w4a8_e8", "promoted_w4_e8"),
    ),
}


@dataclass(frozen=True)
class Trial:
    stage: str
    arm: str
    config: str
    overrides: tuple[tuple[str, Any], ...]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gpu_snapshot() -> str:
    """Return a bounded one-line NVIDIA status without touching torch state."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        utilization, used, total, power = (value.strip() for value in output.split(","))
        return (
            f"gpu={utilization}% vram={used}/{total}MiB power={power}W"
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            IndexError, ValueError):
        return "gpu=status-unavailable"


class _Heartbeat:
    """Print liveness while a trial is inside a long silent kernel section."""

    def __init__(self, label: str, *, seconds: float):
        self.label = label
        self.seconds = float(seconds)
        self.started = time.monotonic()
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        if self.seconds > 0:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stopped.wait(self.seconds):
            elapsed = (time.monotonic() - self.started) / 60
            print(
                f"[{_timestamp()}] heartbeat {self.label}: "
                f"elapsed={elapsed:.1f}m {_gpu_snapshot()}",
                flush=True,
            )

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=min(max(self.seconds, 0.1), 2.0))


def _trial(
    stage: str,
    arm: str,
    config: str,
    **overrides: Any,
) -> Trial:
    return Trial(stage, arm, config, tuple(sorted(overrides.items())))


def stage_trials(stage: str) -> tuple[Trial, ...]:
    """Return a matched ladder whose individual effects remain identifiable."""

    if stage == "w4":
        config = "configs/qwen35_4b_gptq_cuda.yaml"
        return (
            _trial(
                stage, "source_fp16", config,
                **{"patch.enabled": False, "quant.error_comp": "none"},
            ),
            _trial(
                stage, "gaussian_w4_control", config,
                **{"quant.codebook": "gaussian", "quant.error_comp": "none"},
            ),
            _trial(
                stage, "calibrated_w4_control", config,
                **{"quant.codebook": "calibrated", "quant.error_comp": "none"},
            ),
            _trial(
                stage, "gptq_gaussian_w4", config,
                **{"quant.codebook": "gaussian", "quant.error_comp": "gptq"},
            ),
            _trial(
                stage, "gptq_calibrated_w4", config,
                **{"quant.codebook": "calibrated", "quant.error_comp": "gptq"},
            ),
        )
    if stage == "w4a8":
        config = "configs/qwen35_4b_w4a8_e8_trials_cuda.yaml"
        return (
            _trial(
                stage, "source_fp16", config,
                **{
                    "patch.enabled": False,
                    "patch.activation_bits": None,
                    "eval.kv_cache": False,
                },
            ),
            _trial(
                stage, "promoted_w4", config,
                **{
                    "quant.scale_bits": 16,
                    "quant.bias_correction": "none",
                    "patch.rotation": "fwht",
                    "patch.share_rotations": False,
                    "patch.train_rotation": None,
                    "patch.activation_bits": None,
                    "eval.kv_cache": False,
                },
            ),
            _trial(
                stage, "optimized_w4", config,
                **{"patch.activation_bits": None, "eval.kv_cache": False},
            ),
            _trial(
                stage, "w4a8", config,
                **{
                    "quant.scale_bits": 16,
                    "quant.bias_correction": "none",
                    "patch.rotation": "fwht",
                    "patch.share_rotations": False,
                    "patch.train_rotation": None,
                    "patch.activation_bits": 8,
                    "eval.kv_cache": False,
                },
            ),
            _trial(
                stage, "w4a8_e8", config,
                **{
                    "quant.scale_bits": 16,
                    "quant.bias_correction": "none",
                    "patch.rotation": "fwht",
                    "patch.share_rotations": False,
                    "patch.train_rotation": None,
                },
            ),
        )
    if stage == "ablation":
        config = "configs/qwen35_4b_w4_factor_ablation_cuda.yaml"
        hessian_training = {
            "steps": 200,
            "lr": 0.001,
            "objective": "hessian",
            "assignment_scale": "rms",
            "learn_signs": False,
        }
        hessian_training_with_signs = {
            **hessian_training,
            "learn_signs": True,
            "sign_temperature": 1.0,
        }
        return (
            _trial(stage, "promoted_w4", config),
            _trial(stage, "scale8_w4", config, **{"quant.scale_bits": 8}),
            _trial(
                stage, "mean_bias_w4", config,
                **{"quant.bias_correction": "mean"},
            ),
            _trial(
                stage, "shared_fwht_w4", config,
                **{"patch.share_rotations": True},
            ),
            _trial(
                stage, "butterfly_control_w4", config,
                **{"patch.rotation": "butterfly"},
            ),
            _trial(
                stage, "butterfly_hessian_w4", config,
                **{
                    "patch.rotation": "butterfly",
                    "patch.train_rotation": hessian_training,
                },
            ),
            _trial(
                stage, "butterfly_hessian_signs_w4", config,
                **{
                    "patch.rotation": "butterfly",
                    "patch.train_rotation": hessian_training_with_signs,
                },
            ),
            _trial(
                stage, "shared_butterfly_hessian_signs_w4", config,
                **{
                    "patch.rotation": "butterfly",
                    "patch.share_rotations": True,
                    "patch.train_rotation": hessian_training_with_signs,
                },
            ),
        )
    if stage == "recovery":
        config = "configs/qwen35_4b_recovery_cuda.yaml"
        return (
            _trial(
                stage, "source_fp16", config,
                **{"patch.enabled": False, "patch.train_rotation": None},
            ),
            _trial(
                stage, "unrecovered_w4", config,
                **{"patch.train_rotation": None},
            ),
            _trial(stage, "recovered_w4", config),
        )
    if stage == "long-kv":
        config = "configs/qwen35_4b_long_context_kv_cuda.yaml"
        return (
            _trial(
                stage, "source_fp16_e8", config,
                **{"patch.enabled": False, "patch.activation_bits": None},
            ),
            _trial(
                stage, "promoted_w4_e8", config,
                **{"patch.activation_bits": None},
            ),
            _trial(
                stage, "w4a8_e8", config,
                **{"patch.activation_bits": 8},
            ),
        )
    raise ValueError(f"unknown stage {stage!r}; choose from {REGISTERED_STAGES}")


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolved_trial_config(trial: Trial, seed: int) -> dict[str, Any]:
    config = run_experiment.load_config(str(ROOT / trial.config))
    config = copy.deepcopy(config)
    run_experiment.apply_set_overrides(config, list(trial.overrides))
    config["seed"] = int(seed)
    return config


def trial_fingerprint(trial: Trial, seed: int, code_revision: str) -> str:
    payload = {
        "protocol": PROTOCOL_VERSION,
        "code_revision": code_revision,
        "config": _resolved_trial_config(trial, seed),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_matching_result(
    output_dir: Path,
    *,
    stage: str,
    arm: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    matches: list[tuple[int, dict[str, Any]]] = []
    for path in output_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("config") or {}
        if (
            config.get("stage_protocol") == PROTOCOL_VERSION
            and config.get("stage_name") == stage
            and config.get("stage_arm") == arm
            and int(config.get("seed", -1)) == seed
            and config.get("stage_trial_fingerprint") == fingerprint
            and isinstance(payload.get("metrics"), dict)
        ):
            matches.append((path.stat().st_mtime_ns, payload))
    return max(matches, key=lambda entry: entry[0])[1] if matches else None


def _metric(metrics: dict[str, Any], *path: str) -> Any:
    value: Any = metrics
    for name in path:
        if not isinstance(value, dict):
            return None
        value = value.get(name)
    return value


def summarize_results(
    results: list[tuple[Trial, int, dict[str, Any], float, bool]],
) -> list[dict[str, Any]]:
    source_ppl: dict[tuple[str, int, str], float] = {}
    for trial, seed, payload, _seconds, _resumed in results:
        if trial.arm != "source_fp16":
            continue
        metrics = payload["metrics"]
        for dataset in ("wikitext2", "c4"):
            value = metrics.get(f"ppl_{dataset}")
            if value is not None:
                source_ppl[(trial.stage, seed, dataset)] = float(value)

    rows: list[dict[str, Any]] = []
    for trial, seed, payload, seconds, resumed in results:
        metrics = payload["metrics"]
        row: dict[str, Any] = {
            "stage": trial.stage,
            "arm": trial.arm,
            "seed": seed,
            "run_id": payload.get("run_id"),
            "resumed": resumed,
            "runner_wall_seconds": seconds,
            "model_load_seconds": metrics.get("model_load_seconds"),
            "calib_seconds": metrics.get("calib_seconds"),
            "patch_seconds": metrics.get("patch_seconds"),
            "complete_persistent_model_bytes": metrics.get(
                "complete_persistent_model_bytes"
            ),
            "quality_runtime_model_bytes": metrics.get("quality_runtime_model_bytes"),
            "packed_weight_bytes": metrics.get("packed_weight_bytes"),
            "peak_vram_bytes_patch": metrics.get("peak_vram_bytes_patch"),
            "peak_vram_bytes_eval": metrics.get("peak_vram_bytes_eval"),
            "mean_teacher_kl": _metric(metrics, "logit_fidelity", "mean_teacher_kl"),
            "p95_teacher_kl": _metric(metrics, "logit_fidelity", "p95_teacher_kl"),
            "top1_agreement": _metric(metrics, "logit_fidelity", "top1_agreement"),
            "trajectory_token_agreement": _metric(
                metrics, "trajectory", "token_agreement"
            ),
            "exact_trajectory_rate": _metric(
                metrics, "trajectory", "exact_trajectory_rate"
            ),
            "mean_matching_prefix": _metric(
                metrics, "trajectory", "mean_matching_prefix"
            ),
            "kv_mean_teacher_kl": _metric(
                metrics, "kv_cache", "mean_teacher_kl"
            ),
            "kv_top1_agreement": _metric(metrics, "kv_cache", "top1_agreement"),
            "kv_endpoint_passed": _metric(
                metrics, "kv_cache", "endpoint_check", "passed"
            ),
            "kv_endpoint_mean_teacher_kl": _metric(
                metrics, "kv_cache", "endpoint_check", "mean_teacher_kl"
            ),
            "kv_effective_bpv": _metric(
                metrics, "kv_cache", "effective_kv_bpv"
            ),
            "kv_total_cache_compression_ratio": _metric(
                metrics, "kv_cache", "total_cache_compression_ratio"
            ),
            "evaluation_halted": metrics.get("evaluation_halted", False),
            "evaluation_halt_reasons": metrics.get("evaluation_halt_reasons"),
        }
        for dataset in ("wikitext2", "c4"):
            value = metrics.get(f"ppl_{dataset}")
            row[f"ppl_{dataset}"] = value
            source = source_ppl.get((trial.stage, seed, dataset))
            row[f"ppl_{dataset}_relative_to_source"] = (
                float(value) / source - 1.0
                if value is not None and source not in (None, 0.0)
                else None
            )
        rows.append(row)
    return rows


def _bootstrap_delta(
    candidate: list[float],
    baseline: list[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired metric arrays must be non-empty and shape-matched")
    delta = left - right
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(delta.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "paired_samples": len(delta),
        # A percentile bootstrap of a handful of prompts cannot reach nominal
        # coverage; the interval is reported for completeness but must not be
        # quoted as a 95% interval below this sample count.
        "interval_reliable": len(delta) >= MIN_RELIABLE_BOOTSTRAP_SAMPLES,
    }


def _row_metric_delta(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    identity: str,
    metric: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    candidate = {row[identity]: float(row[metric]) for row in candidate_rows}
    baseline = {row[identity]: float(row[metric]) for row in baseline_rows}
    if len(candidate) != len(candidate_rows) or len(baseline) != len(baseline_rows):
        raise ValueError(f"duplicate {identity} in paired {metric} rows")
    if set(candidate) != set(baseline):
        raise ValueError(f"paired {metric} identities differ")
    identities = sorted(candidate)
    return _bootstrap_delta(
        [candidate[value] for value in identities],
        [baseline[value] for value in identities],
        draws=draws,
        seed=seed,
    )


def paired_comparisons(
    results: list[tuple[Trial, int, dict[str, Any], float, bool]],
    *,
    draws: int = 4000,
) -> list[dict[str, Any]]:
    """Return candidate-minus-control paired intervals without declaring a win."""

    payloads = {
        (trial.stage, trial.arm, seed): payload
        for trial, seed, payload, _seconds, _resumed in results
    }
    comparisons: list[dict[str, Any]] = []
    for stage, pairs in PAIRED_ARMS.items():
        seeds = sorted({key[2] for key in payloads if key[0] == stage})
        for seed in seeds:
            for pair_index, (candidate_arm, baseline_arm) in enumerate(pairs):
                candidate = payloads.get((stage, candidate_arm, seed))
                baseline = payloads.get((stage, baseline_arm, seed))
                if candidate is None or baseline is None:
                    continue
                candidate_metrics = candidate["metrics"]
                baseline_metrics = baseline["metrics"]
                report: dict[str, Any] = {
                    "stage": stage,
                    "seed": seed,
                    "candidate_arm": candidate_arm,
                    "baseline_arm": baseline_arm,
                    "interpretation": "candidate minus baseline; no winner inferred",
                    "aggregate_deltas": {},
                    "metrics": {},
                }
                metric_seed = 1701 + seed * 100 + pair_index * 10
                aggregate_paths = {
                    "ppl_wikitext2": ("ppl_wikitext2",),
                    "ppl_c4": ("ppl_c4",),
                    "complete_persistent_model_bytes": (
                        "complete_persistent_model_bytes",
                    ),
                    "quality_runtime_model_bytes": ("quality_runtime_model_bytes",),
                    "peak_vram_bytes_patch": ("peak_vram_bytes_patch",),
                    "peak_vram_bytes_eval": ("peak_vram_bytes_eval",),
                    "mean_teacher_kl": ("logit_fidelity", "mean_teacher_kl"),
                    "top1_agreement": ("logit_fidelity", "top1_agreement"),
                    "trajectory_token_agreement": (
                        "trajectory", "token_agreement"
                    ),
                    "exact_trajectory_rate": (
                        "trajectory", "exact_trajectory_rate"
                    ),
                    "mean_matching_prefix": (
                        "trajectory", "mean_matching_prefix"
                    ),
                    "kv_mean_teacher_kl": (
                        "kv_cache", "mean_teacher_kl"
                    ),
                    "kv_top1_agreement": ("kv_cache", "top1_agreement"),
                    "kv_nll_delta": ("kv_cache", "nll_delta"),
                }
                for name, path in aggregate_paths.items():
                    left_value = _metric(candidate_metrics, *path)
                    right_value = _metric(baseline_metrics, *path)
                    if (
                        isinstance(left_value, (int, float))
                        and not isinstance(left_value, bool)
                        and isinstance(right_value, (int, float))
                        and not isinstance(right_value, bool)
                    ):
                        report["aggregate_deltas"][name] = (
                            float(left_value) - float(right_value)
                        )
                for dataset_index, dataset in enumerate(("wikitext2", "c4")):
                    left = candidate_metrics.get(f"ppl_{dataset}_details") or {}
                    right = baseline_metrics.get(f"ppl_{dataset}_details") or {}
                    if left.get("window_hashes") and right.get("window_hashes"):
                        if left["window_hashes"] != right["window_hashes"]:
                            raise ValueError(f"paired {dataset} windows differ")
                        report["metrics"][f"{dataset}_mean_nll"] = _bootstrap_delta(
                            left["window_mean_nll"],
                            right["window_mean_nll"],
                            draws=draws,
                            seed=metric_seed + dataset_index,
                        )
                for block_index, (block, metrics) in enumerate((
                    ("logit_fidelity", (
                        "mean_teacher_kl", "top1_agreement", "nll_delta"
                    )),
                    ("trajectory", (
                        "token_agreement", "exact_trajectory_rate",
                        "mean_matching_prefix",
                    )),
                )):
                    left_rows = (candidate_metrics.get(block) or {}).get(
                        "prompt_metrics", []
                    )
                    right_rows = (baseline_metrics.get(block) or {}).get(
                        "prompt_metrics", []
                    )
                    if not left_rows or not right_rows:
                        continue
                    for metric_index, metric in enumerate(metrics):
                        report["metrics"][f"{block}.{metric}"] = _row_metric_delta(
                            left_rows,
                            right_rows,
                            identity="input_hash",
                            metric=metric,
                            draws=draws,
                            seed=(
                                metric_seed + 2 + block_index * len(metrics)
                                + metric_index
                            ),
                        )
                comparisons.append(report)
    return comparisons


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run_trial(
    trial: Trial,
    *,
    seed: int,
    output_dir: Path,
    force: bool,
    dry_run: bool,
    code_revision: str,
    position: int,
    total: int,
    heartbeat_seconds: float,
) -> tuple[dict[str, Any] | None, float, bool]:
    label = f"{trial.stage}/{trial.arm}/seed-{seed}"
    fingerprint = trial_fingerprint(trial, seed, code_revision)
    if not force:
        existing = _load_matching_result(
            output_dir,
            stage=trial.stage,
            arm=trial.arm,
            seed=seed,
            fingerprint=fingerprint,
        )
        if existing is not None:
            print(
                f"[{_timestamp()}] RESUME [{position}/{total}] {label}",
                flush=True,
            )
            return existing, 0.0, True

    print(f"[{_timestamp()}] RUN [{position}/{total}] {label}", flush=True)
    if dry_run:
        return None, 0.0, False
    marker_overrides = (
        ("stage_protocol", PROTOCOL_VERSION),
        ("stage_name", trial.stage),
        ("stage_arm", trial.arm),
        ("stage_code_revision", code_revision),
        ("stage_trial_fingerprint", fingerprint),
    )
    start = time.perf_counter()
    with _Heartbeat(label, seconds=heartbeat_seconds):
        payload = run_experiment.run(
            str(ROOT / trial.config),
            str(output_dir),
            overrides={"seed": seed},
            sets=[*trial.overrides, *marker_overrides],
        )
    elapsed = time.perf_counter() - start
    metrics = payload.get("metrics") or {}
    logit = metrics.get("logit_fidelity") or {}
    digest = {
        "ppl_wikitext2": metrics.get("ppl_wikitext2"),
        "ppl_c4": metrics.get("ppl_c4"),
        "mean_teacher_kl": logit.get("mean_teacher_kl"),
        "top1_agreement": logit.get("top1_agreement"),
        "trajectory_token_agreement": _metric(
            metrics, "trajectory", "token_agreement"),
        "kv_mean_teacher_kl": _metric(metrics, "kv_cache", "mean_teacher_kl"),
        "evaluation_halted": metrics.get("evaluation_halted", False),
    }
    print(
        f"[{_timestamp()}] COMPLETE [{position}/{total}] {label} "
        f"elapsed={elapsed / 60:.1f}m metrics={json.dumps(digest)}",
        flush=True,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload, elapsed, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        choices=REGISTERED_STAGES,
        help="stage to run; repeat for multiple stages (default: w4)",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="run only this arm within the selected stage(s); repeat as needed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=60.0,
        help="print liveness/GPU status this often during a trial; 0 disables",
    )
    return parser.parse_args()


def main() -> None:
    enable_default_logging()
    args = parse_args()
    stages = tuple(dict.fromkeys(args.stage or DEFAULT_STAGES))
    seeds = tuple(dict.fromkeys(args.seeds or [0]))
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if args.heartbeat_seconds < 0:
        raise ValueError("heartbeat-seconds must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_revision = _code_revision()
    results: list[tuple[Trial, int, dict[str, Any], float, bool]] = []
    requested_arms = set(args.arms or [])
    planned_trials: list[tuple[Trial, int]] = []
    for stage in stages:
        for seed in seeds:
            for trial in stage_trials(stage):
                if requested_arms and trial.arm not in requested_arms:
                    continue
                planned_trials.append((trial, seed))

    matched_arms = {trial.arm for trial, _seed in planned_trials}
    unknown_arms = requested_arms - matched_arms
    if unknown_arms:
        raise ValueError(
            "requested arms are not present in the selected stages: "
            + ", ".join(sorted(unknown_arms))
        )
    if not planned_trials:
        raise ValueError("the selected stages/arms produced an empty trial plan")

    planned = [
        {"stage": trial.stage, "arm": trial.arm, "seed": seed}
        for trial, seed in planned_trials
    ]
    print(json.dumps({
        "event": "trial_plan",
        "protocol": PROTOCOL_VERSION,
        "code_revision": code_revision,
        "total_trials": len(planned),
        "planned": planned,
    }, indent=2), flush=True)

    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL_VERSION, "planned": planned}, indent=2))
        return

    run_started = time.monotonic()
    progress: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "code_revision": code_revision,
        "state": "running",
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "total_trials": len(planned),
        "completed_trials": 0,
        "planned": planned,
        "completed": [],
        "current": None,
    }
    progress_path = args.output_dir / "next_stage_progress.json"
    write_result(str(progress_path), progress)

    for position, (trial, seed) in enumerate(planned_trials, start=1):
        current = {"stage": trial.stage, "arm": trial.arm, "seed": seed}
        progress.update({"current": current, "updated_at": _timestamp()})
        write_result(str(progress_path), progress)
        try:
            payload, seconds, resumed = _run_trial(
                trial,
                seed=seed,
                output_dir=args.output_dir,
                force=args.force,
                dry_run=False,
                code_revision=code_revision,
                position=position,
                total=len(planned_trials),
                heartbeat_seconds=args.heartbeat_seconds,
            )
        except Exception as exc:
            progress.update({
                "state": "failed",
                "updated_at": _timestamp(),
                "elapsed_seconds": time.monotonic() - run_started,
                "last_error": {
                    **current,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            write_result(str(progress_path), progress)
            raise
        assert payload is not None
        results.append((trial, seed, payload, seconds, resumed))
        progress["completed"].append({
            **current,
            "resumed": resumed,
            "wall_seconds": seconds,
        })
        progress.update({
            "completed_trials": len(results),
            "updated_at": _timestamp(),
            "elapsed_seconds": time.monotonic() - run_started,
        })
        write_result(str(progress_path), progress)

        partial_rows = summarize_results(results)
        partial = {
            "protocol": PROTOCOL_VERSION,
            "code_revision": code_revision,
            "stages": list(stages),
            "seeds": list(seeds),
            "rows": partial_rows,
            "paired_comparisons": paired_comparisons(results),
            "complete": False,
        }
        write_result(
            str(args.output_dir / "next_stage_partial_summary.json"), partial)
        _write_csv(args.output_dir / "next_stage_partial_summary.csv", partial_rows)

    rows = summarize_results(results)
    comparisons = paired_comparisons(results)
    summary = {
        "protocol": PROTOCOL_VERSION,
        "code_revision": code_revision,
        "stages": list(stages),
        "seeds": list(seeds),
        "rows": rows,
        "paired_comparisons": comparisons,
        "complete": True,
    }
    write_result(str(args.output_dir / "next_stage_summary.json"), summary)
    _write_csv(args.output_dir / "next_stage_summary.csv", rows)
    progress.update({
        "state": "complete",
        "current": None,
        "updated_at": _timestamp(),
        "elapsed_seconds": time.monotonic() - run_started,
    })
    write_result(str(progress_path), progress)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
