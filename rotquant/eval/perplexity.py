"""Fixed-protocol perplexity on WikiText-2 and C4.

Sliding-window evaluation with an identical tokenizer and stride for every run.
Report both datasets; C4 catches overfitting to WikiText. The protocol here is
intentionally rigid -- fix it once, never change it mid-study.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import torch

from rotquant.utils import get_logger

logger = get_logger()


@dataclass
class PPLConfig:
    seq_len: int = 2048           # or the model's native context
    stride: int | None = None  # defaults to seq_len (non-overlapping)
    max_samples: int | None = None
    wikitext_revision: str | None = None
    c4_revision: str | None = None
    # Screening can stop after a small paired prefix when degradation is
    # already far outside the admissible region. Reference values are sums,
    # not means, so unequal final windows remain correctly weighted.
    early_stop_after: int | None = None
    early_stop_relative_ppl: float | None = None
    reference_window_nll_sums: list[float] | None = None
    reference_window_tokens: list[int] | None = None

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError("perplexity seq_len must be >= 2")
        if self.stride is not None and self.stride < 1:
            raise ValueError("perplexity stride must be >= 1")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("perplexity max_samples must be >= 1")
        if self.early_stop_after is not None and self.early_stop_after < 1:
            raise ValueError("early_stop_after must be >= 1")
        if (self.early_stop_relative_ppl is not None
                and self.early_stop_relative_ppl < 0):
            raise ValueError("early_stop_relative_ppl must be >= 0")
        supplied = (
            self.reference_window_nll_sums is not None,
            self.reference_window_tokens is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("paired reference NLL sums and token counts are required")
        if all(supplied) and (
            len(self.reference_window_nll_sums or [])
            != len(self.reference_window_tokens or [])
        ):
            raise ValueError("paired reference arrays must have equal length")


@lru_cache(maxsize=8)
def _load_text(
    dataset: str,
    *,
    wikitext_revision: str | None = None,
    c4_revision: str | None = None,
) -> str:
    from datasets import load_dataset
    if dataset == "wikitext2":
        # Bare "wikitext" ids are rejected by current huggingface_hub.
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="test",
            revision=wikitext_revision,
        )
        return "\n\n".join(ds["text"])
    if dataset == "c4":
        ds = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
            revision=c4_revision,
        )
        chunks = []
        for n, row in enumerate(ds, start=1):
            chunks.append(row["text"])
            if n >= 2000:
                break
        return "\n\n".join(chunks)
    raise ValueError(f"unknown dataset: {dataset}")


_TOKEN_CACHE: dict[str, torch.Tensor] = {}


def _tokenizer_fingerprint(tokenizer) -> str:
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    identity = {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens": getattr(tokenizer, "special_tokens_map", None),
        "revision": init_kwargs.get("revision"),
        "commit_hash": init_kwargs.get("_commit_hash"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _tokenized_corpus(tokenizer, dataset: str, config: PPLConfig) -> torch.Tensor:
    key_payload = {
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "dataset": dataset,
        "wikitext_revision": config.wikitext_revision,
        "c4_revision": config.c4_revision,
    }
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode()
    ).hexdigest()
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    text = _load_text(
        dataset,
        wikitext_revision=config.wikitext_revision,
        c4_revision=config.c4_revision,
    )
    # Cache on CPU and transfer one bounded window at a time. This avoids
    # repeatedly tokenising the same pinned corpus across notebook trials and
    # avoids retaining the entire corpus in VRAM.
    encoded = tokenizer(text, return_tensors="pt", verbose=False).input_ids.cpu()
    _TOKEN_CACHE[key] = encoded
    return encoded


def _paired_reference_ppl(config: PPLConfig, windows: int) -> float | None:
    sums = config.reference_window_nll_sums
    counts = config.reference_window_tokens
    if sums is None or counts is None or len(sums) < windows:
        return None
    tokens = sum(int(value) for value in counts[:windows])
    if tokens <= 0:
        return None
    return float(torch.exp(torch.tensor(
        sum(float(value) for value in sums[:windows]) / tokens,
        dtype=torch.float64,
    )).item())


@torch.no_grad()
def perplexity_details(model, tokenizer, dataset: str = "wikitext2",
                       config: PPLConfig | None = None, device=None,
                       ) -> dict[str, object]:
    """Return aggregate PPL plus auditable, paired per-window statistics."""

    config = config or PPLConfig()
    device = device or next(model.parameters()).device
    model.eval()
    seq_len = config.seq_len
    stride = config.stride or seq_len
    encoded = _tokenized_corpus(tokenizer, dataset, config)
    n_tokens = encoded.shape[1]

    nlls: list[torch.Tensor] = []
    nll_sums: list[float] = []
    token_counts: list[int] = []
    window_hashes: list[str] = []
    total = 0
    prev_end = 0
    stopped_early = False
    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + seq_len, n_tokens)
        trg_len = end - prev_end
        input_ids = encoded[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        out = model(input_ids, labels=target_ids)
        n_scored = int((target_ids[:, 1:] != -100).sum().item())
        if n_scored == 0:
            prev_end = end
            continue
        nll_sum = out.loss.float() * n_scored
        nlls.append(nll_sum)
        nll_sums.append(float(nll_sum.item()))
        token_counts.append(n_scored)
        total += n_scored
        window_hashes.append(hashlib.sha256(
            input_ids.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest())
        prev_end = end

        windows = len(nlls)
        if (config.early_stop_after is not None
                and windows == config.early_stop_after
                and config.early_stop_relative_ppl is not None):
            source_ppl = _paired_reference_ppl(config, windows)
            if source_ppl is not None:
                candidate_ppl = float(torch.exp(
                    torch.stack(nlls).sum() / total
                ).item())
                stopped_early = (
                    candidate_ppl / source_ppl - 1
                    > config.early_stop_relative_ppl
                )
        if stopped_early:
            break
        if config.max_samples and windows >= config.max_samples:
            break
        if end == n_tokens:
            break
    if not nlls or total <= 0:
        raise ValueError("perplexity evaluation produced no scored tokens")
    ppl = float(torch.exp(torch.stack(nlls).sum() / total).item())
    digest = hashlib.sha256(
        "".join(window_hashes).encode()
    ).hexdigest()
    logger.info(
        "%s perplexity (seq=%d stride=%d windows=%d stopped=%s): %.4f",
        dataset, seq_len, stride, len(nlls), stopped_early, ppl,
    )
    return {
        "ppl": ppl,
        "window_nll_sums": nll_sums,
        "window_tokens": token_counts,
        "window_mean_nll": [
            value / count for value, count in zip(nll_sums, token_counts)
        ],
        "window_hashes": window_hashes,
        "input_digest": digest,
        "windows": len(nlls),
        "tokens": total,
        "stopped_early": stopped_early,
    }


@torch.no_grad()
def perplexity(model, tokenizer, dataset: str = "wikitext2",
               config: PPLConfig | None = None, device=None) -> float:
    """Sliding-window perplexity. Stride defaults to ``seq_len`` (no overlap)."""
    return float(perplexity_details(
        model, tokenizer, dataset, config, device
    )["ppl"])
