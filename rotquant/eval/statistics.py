"""Deterministic, bounded-memory confidence intervals for paired eval deltas."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def bootstrap_mean_interval(
    values: torch.Tensor | Sequence[float],
    *,
    draws: int = 2_000,
    seed: int = 17,
    confidence: float = 0.95,
    max_resample_entries: int = 1_000_000,
) -> tuple[float, float]:
    """Return a percentile interval for a paired sample's mean.

    Resamples are generated in chunks so long-context token arrays do not
    allocate ``draws * tokens`` indices at once.
    """

    sample = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if sample.numel() < 1:
        raise ValueError("bootstrap requires at least one value")
    if draws < 1:
        raise ValueError("bootstrap draws must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    if sample.numel() == 1:
        value = float(sample.item())
        return value, value
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunk_draws = max(1, max_resample_entries // sample.numel())
    means = []
    remaining = draws
    while remaining:
        count = min(remaining, chunk_draws)
        indices = torch.randint(
            sample.numel(),
            (count, sample.numel()),
            generator=generator,
        )
        means.append(sample[indices].mean(dim=1))
        remaining -= count
    distribution = torch.cat(means)
    tail = (1.0 - confidence) / 2.0
    quantiles = torch.quantile(
        distribution, torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    )
    return float(quantiles[0].item()), float(quantiles[1].item())


def bootstrap_report(
    metrics: dict[str, torch.Tensor | Sequence[float]],
    *,
    draws: int = 2_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Build deterministic intervals for several paired metric arrays."""

    intervals = {
        name: list(bootstrap_mean_interval(
            values,
            draws=draws,
            seed=seed + index,
            confidence=confidence,
        ))
        for index, (name, values) in enumerate(metrics.items())
    }
    return {
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "intervals": intervals,
    }


__all__ = ["bootstrap_mean_interval", "bootstrap_report"]
