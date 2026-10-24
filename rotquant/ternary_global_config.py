"""Explicit objective, storage and independent behaviour gates for global QAT."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TernaryGlobalConfig:
    steps: int = 200
    lr: float = 1e-5
    scale_lr: float = 1e-4
    kl_weight: float = 1.0
    ce_weight: float = 0.1
    temperature: float = 1.0
    topk: int = 128
    required_ids: tuple[int, ...] = ()
    validation_interval: int = 10
    checkpoint_interval: int = 10
    early_stopping_patience: int = 5
    max_grad_norm: float = 1.0
    min_improvement: float = 0.0
    min_completion_rate: float = 0.9
    max_repetition_rate: float = 0.05
    min_format_rate: float = 0.9
    gradient_checkpointing: bool = True
    learn_assignments: bool = True
    checkpoint_dir: str | None = None
    resume: bool = False
    source_id: str = ""
    data_id: str = ""
    behaviour_data_id: str = ""
    loss_chunk_tokens: int = 128
    dense_audit_interval: int = 50

    def __post_init__(self):
        for name, minimum in (("steps", 0), ("topk", 1), ("validation_interval", 1),
                              ("checkpoint_interval", 1), ("early_stopping_patience", 0),
                              ("loss_chunk_tokens", 1), ("dense_audit_interval", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.validate_reals()
        if not all((self.source_id, self.data_id, self.behaviour_data_id)):
            raise ValueError("global QAT requires explicit source, data and behaviour identities")
        if self.resume and not self.checkpoint_dir:
            raise ValueError("global QAT resume requires checkpoint_dir")
        if not self.required_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                        for x in self.required_ids):
            raise ValueError("global QAT requires explicit stop/format IDs")

    def validate_reals(self):
        names = ("lr", "scale_lr", "kl_weight", "ce_weight", "temperature", "max_grad_norm",
                 "min_improvement", "min_completion_rate", "max_repetition_rate", "min_format_rate")
        for name in names:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.lr, self.scale_lr, self.temperature) <= 0 or self.kl_weight + self.ce_weight <= 0:
            raise ValueError("global QAT learning rates, temperature and objective must be positive")
        for name in ("min_completion_rate", "max_repetition_rate", "min_format_rate"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")


def validate_behaviour(metrics, config):
    if metrics.get("data_id") != config.behaviour_data_id or metrics.get("examples", 0) < 1:
        raise ValueError("behaviour holdout identity or example count is invalid")
    for name in ("completion_rate", "repetition_rate", "format_rate"):
        if not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1:
            raise ValueError(f"invalid behavioural metric: {name}")
    return dict(metrics)


def behaviour_acceptance(candidate, baseline, config):
    reasons = []
    thresholds = (("completion_rate", max(config.min_completion_rate, baseline["completion_rate"]), 1),
                  ("format_rate", max(config.min_format_rate, baseline["format_rate"]), 1),
                  ("repetition_rate", min(config.max_repetition_rate, baseline["repetition_rate"]), -1))
    for name, bound, direction in thresholds:
        if direction * candidate[name] < direction * bound:
            reasons.append(name)
    return not reasons, reasons
