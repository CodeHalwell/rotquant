"""Held-out source/deployed next-token distribution fidelity."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .statistics import bootstrap_report


@dataclass(frozen=True)
class LogitFidelityConfig:
    batches: int = 2
    prompt_len: int = 64
    skip: int = 8192
    temperature: float = 1.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 17

    def __post_init__(self) -> None:
        if self.batches < 1 or self.prompt_len < 2:
            raise ValueError("logit fidelity requires batches >= 1 and prompt_len >= 2")
        if self.skip < 0:
            raise ValueError("logit fidelity skip must be >= 0")
        if self.temperature <= 0:
            raise ValueError("logit fidelity temperature must be > 0")
        if self.bootstrap_draws < 1 or self.bootstrap_seed < 0:
            raise ValueError("bootstrap draws must be positive and seed nonnegative")


@dataclass
class LogitReference:
    inputs: dict[str, torch.Tensor]
    logits: torch.Tensor
    input_hash: str


def _device_inputs(inputs: Mapping[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _input_hash(input_ids: torch.Tensor) -> str:
    return hashlib.sha256(
        input_ids.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


@torch.no_grad()
def capture_logit_references(
    model,
    batches: Sequence[Mapping[str, Any]],
    device,
    config: LogitFidelityConfig,
) -> list[LogitReference]:
    """Capture source logits and exact input identities before quantization."""

    model.eval()
    references = []
    for batch in batches[:config.batches]:
        prompt = {
            key: value[..., :config.prompt_len]
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        device_prompt = _device_inputs(prompt, device)
        output = model(**device_prompt, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        references.append(LogitReference(
            inputs={key: value.detach().cpu() for key, value in prompt.items()},
            # The final position has no in-prompt next-token target.
            logits=logits[..., :-1, :].detach().to(
                device="cpu", dtype=torch.float16
            ),
            input_hash=_input_hash(prompt["input_ids"]),
        ))
    if not references:
        raise ValueError("logit fidelity evaluation requires at least one prompt")
    return references


@torch.no_grad()
def evaluate_logit_fidelity(
    model,
    references: Sequence[LogitReference],
    device,
    config: LogitFidelityConfig,
) -> dict[str, Any]:
    """Compare teacher distributions and next-token losses on fixed prompts."""

    model.eval()
    kl_sums = cosine_sums = top1_matches = 0.0
    source_nll_sum = candidate_nll_sum = 0.0
    tokens = 0
    token_kl_values: list[torch.Tensor] = []
    token_cosine_values: list[torch.Tensor] = []
    token_match_values: list[torch.Tensor] = []
    token_nll_delta_values: list[torch.Tensor] = []
    prompt_metrics: list[dict[str, Any]] = []
    scale = float(config.temperature)
    for reference in references:
        inputs = _device_inputs(reference.inputs, device)
        output = model(**inputs, use_cache=False)
        candidate = (
            output.logits if hasattr(output, "logits") else output[0]
        )[..., :-1, :].float()
        teacher = reference.logits.to(device=device, dtype=torch.float32)
        width = min(candidate.shape[-2], teacher.shape[-2])
        candidate = candidate[..., :width, :]
        teacher = teacher[..., :width, :]
        targets = inputs["input_ids"][..., 1:1 + width]

        teacher_prob = F.softmax(teacher / scale, dim=-1)
        candidate_log_prob = F.log_softmax(candidate / scale, dim=-1)
        token_kl = F.kl_div(
            candidate_log_prob, teacher_prob, reduction="none"
        ).sum(dim=-1) * scale**2
        token_cosine = F.cosine_similarity(teacher, candidate, dim=-1)
        token_matches = teacher.argmax(dim=-1).eq(candidate.argmax(dim=-1))
        source_token_nll = F.cross_entropy(
            teacher.reshape(-1, teacher.shape[-1]), targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        candidate_token_nll = F.cross_entropy(
            candidate.reshape(-1, candidate.shape[-1]), targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        source_nll = source_token_nll.sum()
        candidate_nll = candidate_token_nll.sum()
        prompt_tokens = targets.numel()
        prompt_kl_sum = float(token_kl.sum().item())
        prompt_cosine_sum = float(token_cosine.sum().item())
        prompt_top1 = float(token_matches.sum().item())
        prompt_source_nll = float(source_nll.item())
        prompt_candidate_nll = float(candidate_nll.item())
        kl_sums += prompt_kl_sum
        cosine_sums += prompt_cosine_sum
        top1_matches += prompt_top1
        source_nll_sum += prompt_source_nll
        candidate_nll_sum += prompt_candidate_nll
        tokens += prompt_tokens
        token_kl_values.append(token_kl.detach().cpu().reshape(-1))
        token_cosine_values.append(token_cosine.detach().cpu().reshape(-1))
        token_match_values.append(token_matches.detach().cpu().float().reshape(-1))
        token_nll_delta_values.append(
            (candidate_token_nll - source_token_nll).detach().cpu().reshape(-1)
        )
        prompt_metrics.append({
            "input_hash": reference.input_hash,
            "tokens": prompt_tokens,
            "mean_teacher_kl": prompt_kl_sum / prompt_tokens,
            "mean_logit_cosine": prompt_cosine_sum / prompt_tokens,
            "top1_agreement": prompt_top1 / prompt_tokens,
            "source_nll": prompt_source_nll / prompt_tokens,
            "candidate_nll": prompt_candidate_nll / prompt_tokens,
            "nll_delta": (
                prompt_candidate_nll - prompt_source_nll
            ) / prompt_tokens,
        })
    if tokens <= 0:
        raise ValueError("logit fidelity references contain no scored tokens")
    token_kl = torch.cat(token_kl_values).float()
    bootstrap = bootstrap_report(
        {
            "mean_teacher_kl": token_kl,
            "mean_logit_cosine": torch.cat(token_cosine_values),
            "top1_agreement": torch.cat(token_match_values),
            "nll_delta": torch.cat(token_nll_delta_values),
        },
        draws=config.bootstrap_draws,
        seed=config.bootstrap_seed,
    )
    return {
        "prompts": len(references),
        "tokens": tokens,
        "mean_teacher_kl": kl_sums / tokens,
        "median_teacher_kl": float(torch.quantile(token_kl, 0.5).item()),
        "p95_teacher_kl": float(torch.quantile(token_kl, 0.95).item()),
        "max_teacher_kl": float(token_kl.max().item()),
        "mean_logit_cosine": cosine_sums / tokens,
        "top1_agreement": top1_matches / tokens,
        "source_nll": source_nll_sum / tokens,
        "candidate_nll": candidate_nll_sum / tokens,
        "nll_delta": (candidate_nll_sum - source_nll_sum) / tokens,
        "input_hashes": [reference.input_hash for reference in references],
        "prompt_metrics": prompt_metrics,
        "paired_token_bootstrap": bootstrap,
    }


__all__ = [
    "LogitFidelityConfig",
    "LogitReference",
    "capture_logit_references",
    "evaluate_logit_fidelity",
]
