"""Decode throughput + peak VRAM for E8 (footprint & speed).

Greedy generation over random in-vocab prompts, timed after a warmup pass. Run
it once on the packed config and once with ``patch: {fallback: true}`` (e.g.
``--set patch.fallback=true``): storage numbers only ever come from the packed
path, speed is reported for both.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rotquant.utils import Timer, get_logger, peak_vram_bytes, reset_peak_vram

logger = get_logger()


@dataclass
class ThroughputConfig:
    prompt_len: int = 64
    new_tokens: int = 128
    batch_size: int = 1
    warmup: int = 1


@torch.no_grad()
def measure_throughput(model, tokenizer, device,
                       config: ThroughputConfig = None) -> dict:
    cfg = config or ThroughputConfig()
    model.eval()
    vocab = int(getattr(model.config, "vocab_size", tokenizer.vocab_size))
    gen = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(0, vocab, (cfg.batch_size, cfg.prompt_len),
                              generator=gen).to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    kwargs = dict(max_new_tokens=cfg.new_tokens, min_new_tokens=cfg.new_tokens,
                  do_sample=False, use_cache=True, pad_token_id=pad_id)

    for _ in range(cfg.warmup):
        model.generate(input_ids, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reset_peak_vram()
    with Timer() as t:
        out = model.generate(input_ids, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    n_new = (out.shape[1] - cfg.prompt_len) * cfg.batch_size
    result = {
        "tokens_per_s": n_new / max(t.elapsed, 1e-9),
        "seconds": t.elapsed,
        "new_tokens": n_new,
        "prompt_len": cfg.prompt_len,
        "batch_size": cfg.batch_size,
        "peak_vram_bytes": peak_vram_bytes(),
    }
    logger.info("throughput: %.2f tok/s (%d new tokens, batch=%d)",
                result["tokens_per_s"], n_new, cfg.batch_size)
    return result
