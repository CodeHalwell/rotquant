"""Fixed-protocol perplexity on WikiText-2 and C4.

Sliding-window evaluation with an identical tokenizer and stride for every run.
Report both datasets; C4 catches overfitting to WikiText. The protocol here is
intentionally rigid -- fix it once, never change it mid-study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rotquant.utils import get_logger

logger = get_logger()


@dataclass
class PPLConfig:
    seq_len: int = 2048           # or the model's native context
    stride: Optional[int] = None  # defaults to seq_len (non-overlapping)
    max_samples: Optional[int] = None


def _load_text(dataset: str) -> str:
    from datasets import load_dataset
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])
    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation",
                          streaming=True)
        chunks, n = [], 0
        for row in ds:
            chunks.append(row["text"])
            n += 1
            if n >= 2000:
                break
        return "\n\n".join(chunks)
    raise ValueError(f"unknown dataset: {dataset}")


@torch.no_grad()
def perplexity(model, tokenizer, dataset: str = "wikitext2",
               config: Optional[PPLConfig] = None, device=None) -> float:
    """Sliding-window perplexity. Stride defaults to ``seq_len`` (no overlap)."""
    config = config or PPLConfig()
    device = device or next(model.parameters()).device
    seq_len = config.seq_len
    stride = config.stride or seq_len

    text = _load_text(dataset)
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n_tokens = enc.shape[1]

    nlls, total = [], 0
    model.eval()
    prev_end = 0
    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + seq_len, n_tokens)
        trg_len = end - prev_end
        input_ids = enc[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # only score the new tokens
        out = model(input_ids, labels=target_ids)
        # out.loss is mean over scored tokens; rescale to a token sum.
        n_scored = (target_ids != -100).sum().item()
        nlls.append(out.loss.float() * n_scored)
        total += n_scored
        prev_end = end
        if config.max_samples and len(nlls) >= config.max_samples:
            break
        if end == n_tokens:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / total).item()
    logger.info("%s perplexity (seq=%d stride=%d): %.4f", dataset, seq_len, stride, ppl)
    return ppl
