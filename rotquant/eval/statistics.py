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


def bootstrap_cluster_mean_interval(
    cluster_sums: torch.Tensor | Sequence[float],
    cluster_counts: torch.Tensor | Sequence[float],
    *,
    draws: int = 2_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval for a pooled mean when observations come in clusters.

    Tokens of one prompt share a context, so their errors are correlated and
    resampling tokens independently understates the uncertainty of a per-token
    mean. Here whole clusters (prompts) are resampled with replacement and the
    statistic is the ratio ``sum(cluster sums) / sum(cluster counts)``, which
    is the pooled per-token mean of each resample.

    With fewer than two clusters the between-cluster variation cannot be
    estimated at all; the interval is returned as ``(nan, nan)`` rather than
    as a zero-width interval that would read as certainty.
    """

    sums = torch.as_tensor(cluster_sums, dtype=torch.float64, device="cpu").reshape(-1)
    counts = torch.as_tensor(cluster_counts, dtype=torch.float64, device="cpu").reshape(-1)
    if sums.numel() != counts.numel() or sums.numel() < 1:
        raise ValueError("cluster bootstrap needs one sum and one count per cluster")
    if (counts <= 0).any():
        raise ValueError("cluster bootstrap counts must be positive")
    if draws < 1:
        raise ValueError("bootstrap draws must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    clusters = sums.numel()
    if clusters < 2:
        return float("nan"), float("nan")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(clusters, (draws, clusters), generator=generator)
    distribution = sums[indices].sum(dim=1) / counts[indices].sum(dim=1)
    tail = (1.0 - confidence) / 2.0
    quantiles = torch.quantile(
        distribution, torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    )
    return float(quantiles[0].item()), float(quantiles[1].item())


def bootstrap_cluster_report(
    cluster_sums: dict[str, torch.Tensor | Sequence[float]],
    cluster_counts: torch.Tensor | Sequence[float],
    *,
    draws: int = 2_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Cluster-resampled intervals for several pooled means sharing counts."""

    intervals = {
        name: list(bootstrap_cluster_mean_interval(
            sums, cluster_counts, draws=draws, seed=seed + index,
            confidence=confidence,
        ))
        for index, (name, sums) in enumerate(cluster_sums.items())
    }
    return {
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "resampling_unit": "prompt",
        "clusters": len(torch.as_tensor(cluster_counts).reshape(-1)),
        "intervals": intervals,
    }


__all__ = [
    "bootstrap_cluster_mean_interval",
    "bootstrap_cluster_report",
    "bootstrap_mean_interval",
    "bootstrap_report",
]
