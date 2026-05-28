"""Zero-shot evaluation via EleutherAI's lm-evaluation-harness.

Perplexity alone hides reasoning collapse -- which is exactly where low-bit quant
fails -- so we always report the zero-shot bundle mean alongside it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rotquant.utils import get_logger

logger = get_logger()

# The fixed bundle; never change mid-study.
DEFAULT_TASKS: List[str] = [
    "arc_challenge", "arc_easy", "boolq", "piqa", "winogrande", "hellaswag",
]


def zeroshot(model, tokenizer, tasks: Optional[List[str]] = None,
             batch_size: int = 8, device: str = "cuda",
             limit: Optional[int] = None) -> Dict[str, float]:
    """Run the lm-eval harness on an already-loaded HF model.

    Returns a dict of per-task accuracy plus ``bundle_mean``.
    """
    tasks = tasks or DEFAULT_TASKS
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except Exception as exc:  # pragma: no cover - requires lm-eval install
        raise ImportError(
            "lm-eval is required for zero-shot eval: pip install lm-eval"
        ) from exc

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
              device=device)
    res = simple_evaluate(model=lm, tasks=tasks, limit=limit)

    scores: Dict[str, float] = {}
    for task, metrics in res["results"].items():
        acc = metrics.get("acc_norm,none", metrics.get("acc,none"))
        if acc is not None:
            scores[task] = float(acc)
    if scores:
        scores["bundle_mean"] = sum(scores.values()) / len(scores)
    logger.info("zero-shot bundle mean: %.4f", scores.get("bundle_mean", float("nan")))
    return scores
