"""Held-out greedy trajectory fidelity for quantized causal language models.

Perplexity and teacher-forced KL only measure one-step predictions on a fixed
prefix.  A small perturbation can instead change one greedy token and send the
student down an entirely different continuation.  This module captures source
model continuations before patching, then measures the deployed model on the
same prompts after quantization and recovery training.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrajectoryConfig:
    batches: int = 4
    prompt_len: int = 64
    new_tokens: int = 32
    # ``build_calib_loader`` starts a fresh C4 stream on each call.  Skipping a
    # substantial prefix prevents these prompts overlapping calibration calls.
    skip: int = 1024
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.batches < 1:
            raise ValueError("trajectory batches must be >= 1")
        if self.prompt_len < 1 or self.new_tokens < 1:
            raise ValueError("trajectory lengths must be >= 1")
        if self.skip < 0:
            raise ValueError("trajectory skip must be >= 0")


@dataclass
class TrajectoryReference:
    inputs: dict[str, torch.Tensor]
    continuation: torch.Tensor


def _device_inputs(inputs: Mapping[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _generation_kwargs(tokenizer, config: TrajectoryConfig) -> dict[str, Any]:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.new_tokens,
        "min_new_tokens": config.new_tokens,
        "do_sample": False,
        "use_cache": config.use_cache,
    }
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    return kwargs


@torch.no_grad()
def capture_trajectories(model, tokenizer, batches: Sequence[Mapping[str, Any]],
                         device, config: TrajectoryConfig,
                         ) -> list[TrajectoryReference]:
    """Generate and retain source-model continuations for held-out prompts."""

    model.eval()
    references = []
    kwargs = _generation_kwargs(tokenizer, config)
    for batch in batches[:config.batches]:
        prompt = {
            key: value[..., :config.prompt_len]
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        device_prompt = _device_inputs(prompt, device)
        output = model.generate(**device_prompt, **kwargs)
        prompt_tokens = device_prompt["input_ids"].shape[-1]
        continuation = output[..., prompt_tokens:].detach().cpu()
        references.append(TrajectoryReference(
            inputs={key: value.detach().cpu() for key, value in prompt.items()},
            continuation=continuation,
        ))
    if not references:
        raise ValueError("trajectory evaluation requires at least one prompt")
    return references


@torch.no_grad()
def evaluate_trajectories(model, tokenizer,
                          references: Sequence[TrajectoryReference], device,
                          config: TrajectoryConfig) -> dict[str, float | int]:
    """Compare free-running deployed continuations with source references."""

    model.eval()
    kwargs = _generation_kwargs(tokenizer, config)
    matching_tokens = 0
    total_tokens = 0
    exact = 0
    prefix_tokens = 0
    examples = 0
    for reference in references:
        inputs = _device_inputs(reference.inputs, device)
        output = model.generate(**inputs, **kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        candidate = output[..., prompt_tokens:].detach().cpu()
        target = reference.continuation
        width = min(candidate.shape[-1], target.shape[-1])
        if width == 0:
            continue
        matches = candidate[..., :width].eq(target[..., :width])
        matching_tokens += int(matches.sum().item())
        total_tokens += matches.numel()
        rows = matches.reshape(-1, width)
        exact += int(rows.all(dim=-1).sum().item())
        prefix_tokens += int(rows.cumprod(dim=-1).sum().item())
        examples += rows.shape[0]
    if not examples or not total_tokens:
        raise ValueError("trajectory references contain no continuation tokens")
    return {
        "prompts": examples,
        "new_tokens": config.new_tokens,
        "token_agreement": matching_tokens / total_tokens,
        "exact_trajectory_rate": exact / examples,
        "mean_matching_prefix": prefix_tokens / examples,
        "mean_first_divergence": prefix_tokens / examples,
    }
