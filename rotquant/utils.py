"""Seeding, logging, memory probes, and bits/weight accounting helpers."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency at runtime
    torch = None  # type: ignore


_LOGGER_NAME = "rotquant"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("torch", "numpy", "scipy", "transformers", "datasets", "lm_eval"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not-installed"
    return versions


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": False}
    if torch is not None and torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            info["driver"] = driver.decode().strip().splitlines()[0]
        except Exception:
            info["driver"] = "unknown"
    return info


def peak_vram_bytes() -> int:
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def reset_peak_vram() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def environment_record() -> Dict[str, Any]:
    """Full provenance block to embed in every result file."""
    return {
        "git_sha": git_sha(),
        "library_versions": library_versions(),
        "gpu": gpu_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# --------------------------------------------------------------------------- #
# bits/weight accounting
# --------------------------------------------------------------------------- #
@dataclass
class BitBudget:
    """Accounts for the *true* bits/weight of a quantised group.

    The effective bits/weight for a group of ``group_size`` weights is::

        (group_size * log2(levels) + scale_bits + sign_bits) / group_size

    where ``scale_bits``/``sign_bits`` are per-group metadata costs. This is the
    accounting the spec requires so that a "3-bit" config is never secretly 3.5
    or 8 bits.
    """

    levels: int
    group_size: int
    scale_bits: float = 16.0
    sign_bits: float = 0.0
    extra_metadata_bits: float = 0.0

    @property
    def code_bits(self) -> float:
        return math.log2(self.levels)

    @property
    def bits_per_weight(self) -> float:
        meta = self.scale_bits + self.sign_bits + self.extra_metadata_bits
        return (self.group_size * self.code_bits + meta) / self.group_size

    def assert_matches(self, claimed_bpw: float, tol: float = 1e-6) -> None:
        actual = self.bits_per_weight
        if abs(actual - claimed_bpw) > tol:
            raise AssertionError(
                f"bits/weight mismatch: claimed {claimed_bpw}, actual {actual:.6f} "
                f"(levels={self.levels}, group={self.group_size}, "
                f"scale_bits={self.scale_bits}, sign_bits={self.sign_bits})"
            )


def write_result(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class Timer:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0
