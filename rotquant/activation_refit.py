"""Bounded reconstruction refinement under a calibrated input second moment.

For each output row, minimize ``(w - q) H (w - q).T`` in the weight's
already-rotated basis. Scale proposals use the full error projection ``E H``
and each group's exact diagonal curvature, followed by a full-quadratic step.
Code proposals consider the adjacent scalar centroids. Every candidate is
rounded to its deployed representation and accepted against the full row loss.
The scale8 affine grids stay fixed, so accepted rows cannot change other rows'
stored scales. This is bounded local refinement, not a global optimum solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StoredScaleGrid:
    values: torch.Tensor
    stored: torch.Tensor
    bits: float
    block_size: int
    offsets: torch.Tensor | None = None
    steps: torch.Tensor | None = None

    def snap(self, proposed: torch.Tensor, row_start: int):
        if self.bits != 8:
            dtype = torch.float16 if self.bits == 16 else torch.float32
            stored = proposed.clamp_min(2.0 ** -24).to(dtype)
            return stored.float(), stored
        positions = torch.arange(
            proposed.numel(), device=proposed.device
        ).reshape_as(proposed) + row_start * proposed.shape[1]
        blocks = positions // self.block_size
        offsets = self.offsets.float()[blocks]
        steps = self.steps.float()[blocks]
        divisor = torch.where(steps > 0, steps, torch.ones_like(steps))
        codes = torch.round((proposed - offsets) / divisor).clamp(0, 255)
        codes = torch.where(steps > 0, codes, torch.zeros_like(codes))
        return offsets + codes * steps, codes.to(torch.uint8)


@dataclass
class ActivationRefitter:
    hessian: torch.Tensor
    group_size: int
    scale_passes: int = 1
    code_passes: int = 0
    row_batch: int = 128

    def validate(self, width: int) -> None:
        if self.hessian.shape != (width, width):
            raise ValueError("activation refitting requires an input-sized Hessian")
        if not torch.isfinite(self.hessian).all():
            raise ValueError("activation refitting requires a finite Hessian")
        if not torch.allclose(self.hessian, self.hessian.T, rtol=1e-4, atol=1e-6):
            raise ValueError("activation refitting requires a symmetric Hessian")
        if (self.hessian.diagonal() < 0).any():
            raise ValueError("activation refitting requires nonnegative Hessian diagonal")

    def grouped(self, values: torch.Tensor) -> torch.Tensor:
        return F.pad(values, (0, (-values.shape[1]) % self.group_size)).reshape(
            values.shape[0], -1, self.group_size)

    def expanded(self, scales: torch.Tensor, width: int) -> torch.Tensor:
        return scales.repeat_interleave(self.group_size, dim=1)[:, :width]

    def objective(self, error: torch.Tensor) -> torch.Tensor:
        return (error * (error @ self.hessian)).sum(dim=1)

    @staticmethod
    def improves(candidate: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(candidate) & (candidate < baseline - baseline.abs() * 1e-6)

    def group_hessians(self) -> torch.Tensor:
        blocks = []
        for start in range(0, self.hessian.shape[0], self.group_size):
            block = self.hessian[start:start + self.group_size, start:start + self.group_size]
            padding = self.group_size - block.shape[0]
            blocks.append(F.pad(block, (0, padding, 0, padding)))
        return torch.stack(blocks)

    def scale_direction(self, error, coefficients, group_hessians):
        grouped = self.grouped(coefficients)
        curvature = (torch.einsum("rgi,gij->rgj", grouped, group_hessians) * grouped).sum(-1)
        projected = error @ self.hessian
        gradient = self.grouped(projected * coefficients).sum(-1)
        denominator = torch.where(curvature > 0, curvature, torch.ones_like(curvature))
        direction = torch.where(curvature > 0, gradient / denominator, torch.zeros_like(gradient))
        delta = coefficients * self.expanded(direction, error.shape[1])
        numerator = (projected * delta).sum(1)
        denominator = self.objective(delta)
        usable = (denominator > 0) & torch.isfinite(denominator)
        safe = torch.where(usable, denominator, torch.ones_like(denominator))
        rate = torch.where(usable, (numerator / safe).clamp(0, 1), torch.zeros_like(safe))
        return direction * rate.unsqueeze(1)

    def refine_scales(self, weight, reconstruction, coefficients, values, stored, grid, row_start, blocks):
        direction = self.scale_direction(weight - reconstruction, coefficients, blocks)
        original = values
        best_loss = self.objective(weight - reconstruction)
        for fraction in (1.0, 0.5, 0.25):
            trial_values, trial_stored = grid.snap(original + fraction * direction, row_start)
            trial = coefficients * self.expanded(trial_values, weight.shape[1])
            trial_loss = self.objective(weight - trial)
            keep = self.improves(trial_loss, best_loss)
            reconstruction = torch.where(keep[:, None], trial, reconstruction)
            values = torch.where(keep[:, None], trial_values, values)
            stored = torch.where(keep[:, None], trial_stored, stored)
            best_loss = torch.where(keep, trial_loss, best_loss)
        return reconstruction, values, stored

    def code_proposal(self, weight, reconstruction, indices, expanded_scales, centroids):
        gradient = (weight - reconstruction) @ self.hessian
        curvature = self.hessian.diagonal().unsqueeze(0)
        proposal = indices
        gain = torch.zeros_like(reconstruction)
        for shift in (-1, 1):
            codes = (indices + shift).clamp(0, centroids.numel() - 1)
            delta = centroids[codes] * expanded_scales - reconstruction
            candidate_gain = 2 * gradient * delta - curvature * delta.square()
            keep = torch.isfinite(candidate_gain) & (candidate_gain > gain)
            proposal = torch.where(keep, codes, proposal)
            gain = torch.where(keep, candidate_gain, gain)
        return proposal, gain

    def refine_codes(self, weight, reconstruction, indices, scales, centroids):
        width = weight.shape[1]
        expanded = self.expanded(scales, width)
        proposal, gain = self.code_proposal(weight, reconstruction, indices, expanded, centroids)
        order = torch.argsort(gain, dim=1, descending=True, stable=True)
        original = indices
        best_loss = self.objective(weight - reconstruction)
        for count in dict.fromkeys((width, max(1, width // 4), max(1, width // 16), 1)):
            selected = torch.zeros_like(indices, dtype=torch.bool)
            selected.scatter_(1, order[:, :count], True)
            trial_indices = torch.where(selected & (gain > 0), proposal, original)
            trial = centroids[trial_indices] * expanded
            trial_loss = self.objective(weight - trial)
            keep = self.improves(trial_loss, best_loss)
            reconstruction = torch.where(keep[:, None], trial, reconstruction)
            indices = torch.where(keep[:, None], trial_indices, indices)
            best_loss = torch.where(keep, trial_loss, best_loss)
        return reconstruction, indices

    def refit_batch(self, source, trial, codes, scales, storage, centroids, grid, start, blocks):
        for sweep in range(max(self.scale_passes, self.code_passes)):
            if sweep < self.scale_passes:
                trial, scales, storage = self.refine_scales(
                    source, trial, centroids[codes], scales, storage, grid, start, blocks)
            if sweep < self.code_passes:
                trial, codes = self.refine_codes(source, trial, codes, scales, centroids)
        return trial, codes, scales, storage

    def refit(self, weight, reconstruction, indices, centroids, grid):
        self.validate(weight.shape[1])
        blocks = self.group_hessians() if self.scale_passes else None
        values, stored = grid.values.clone(), grid.stored.clone()
        before, after, accepted_rows = 0.0, 0.0, 0
        for start in range(0, weight.shape[0], self.row_batch):
            stop = start + self.row_batch
            source, base = weight[start:stop], reconstruction[start:stop]
            codes = indices[start:stop]
            trial, codes, scales, storage = self.refit_batch(
                source, base, codes, values[start:stop], stored[start:stop], centroids, grid, start, blocks)
            base_loss, trial_loss = self.objective(source - base), self.objective(source - trial)
            keep = self.improves(trial_loss, base_loss)
            reconstruction[start:stop] = torch.where(keep[:, None], trial, base)
            indices[start:stop] = torch.where(keep[:, None], codes, indices[start:stop])
            values[start:stop] = torch.where(keep[:, None], scales, values[start:stop])
            stored[start:stop] = torch.where(keep[:, None], storage, stored[start:stop])
            before += float(base_loss.sum())
            after += float(torch.where(keep, trial_loss, base_loss).sum())
            accepted_rows += int(keep.sum())
        return values, stored, {
            "objective": "full_input_second_moment", "scale_grid": "frozen_stored",
            "scale_passes": self.scale_passes, "code_passes": self.code_passes,
            "objective_before": before, "objective_after": after,
            "accepted_rows": accepted_rows,
        }
