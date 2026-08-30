"""Matched-budget diagnostics for scalar quantizer variants.

Reconstruction MSE alone favours conditional-mean centroids and can hide the
inner-product shrinkage that matters to retrieval and attention.  These helpers
report both signals, plus the exact packed rate claimed by each candidate.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from rotquant.quantize import QuantConfig, QuantizedWeight, Quantizer


@torch.no_grad()
def quantized_weight_metrics(
    reference: torch.Tensor,
    quantized: QuantizedWeight,
    *,
    probes: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure reconstruction and inner-product fidelity for one packed weight."""
    if reference.ndim != 2:
        raise ValueError("reference weight must have shape [out, in]")
    if tuple(reference.shape) != (
        quantized.out_features, quantized.in_features
    ):
        raise ValueError("reference and quantized weight shapes differ")
    source = reference.detach().float()
    candidate = quantized.dequantize().to(source.device).float()
    source_energy = source.square().sum().clamp_min(1e-12)
    error = candidate - source
    row_energy = source.square().sum(dim=1)
    row_alignment = (source * candidate).sum(dim=1)
    valid_rows = row_energy > 0
    row_ratios = row_alignment[valid_rows] / row_energy[valid_rows]
    budget = quantized.bit_budget()
    metrics = {
        "effective_bpw": float(budget.bits_per_weight),
        "weight_nmse": float(error.square().sum().div(source_energy).item()),
        "global_self_dot_ratio": float(
            (source * candidate).sum().div(source_energy).item()),
        "mean_row_self_dot_ratio": float(
            row_ratios.mean().item() if row_ratios.numel() else 1.0),
        "max_row_self_dot_error": float(
            (row_ratios - 1).abs().max().item() if row_ratios.numel() else 0.0),
    }
    if probes is not None:
        if probes.shape[-1] != source.shape[1]:
            raise ValueError("probe width must match weight input features")
        flat_probes = probes.detach().float().reshape(-1, source.shape[1])
        source_output = flat_probes @ source.T
        candidate_output = flat_probes @ candidate.T
        output_energy = source_output.square().sum().clamp_min(1e-12)
        output_error = candidate_output - source_output
        metrics.update({
            "probe_output_nmse": float(
                output_error.square().sum().div(output_energy).item()),
            "probe_output_bias": float(output_error.mean().item()),
            "probe_output_mae": float(output_error.abs().mean().item()),
        })
    return metrics


@torch.no_grad()
def compare_quantizers(
    weight: torch.Tensor,
    candidates: Mapping[str, QuantConfig],
    *,
    probes: torch.Tensor | None = None,
    require_matched_budget: bool = True,
    budget_tolerance: float = 1e-9,
) -> dict[str, dict[str, float]]:
    """Quantize one matrix with named candidates and compare at equal rate.

    ``weight`` must already be in the basis consumed by the quantizer.  This
    keeps the helper useful for identity, randomized-Hadamard, and learned
    rotations without silently applying a different transform to each arm.
    """
    if not candidates:
        raise ValueError("at least one quantizer candidate is required")
    results = {
        name: quantized_weight_metrics(
            weight,
            Quantizer(config).quantize_weight(weight),
            probes=probes,
        )
        for name, config in candidates.items()
    }
    if require_matched_budget:
        rates = [metrics["effective_bpw"] for metrics in results.values()]
        if max(rates) - min(rates) > budget_tolerance:
            detail = ", ".join(
                f"{name}={metrics['effective_bpw']:.6f}"
                for name, metrics in results.items()
            )
            raise ValueError(f"quantizer candidates are not budget matched: {detail}")
    return results
