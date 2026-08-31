"""Held-out next-token distribution fidelity diagnostics."""
from types import SimpleNamespace

import torch
from torch import nn

from rotquant.eval.logit_fidelity import (
    LogitFidelityConfig,
    capture_logit_references,
    evaluate_logit_fidelity,
)


class TinyLogitModel(nn.Module):
    def __init__(self, vocab: int = 11):
        super().__init__()
        self.vocab = vocab
        self.offset = 0

    def eval(self):
        return self

    def forward(self, input_ids, use_cache=False):
        del use_cache
        target = (input_ids + 1 + self.offset) % self.vocab
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits.scatter_(-1, target.unsqueeze(-1), 8.0)
        return SimpleNamespace(logits=logits)


def test_logit_fidelity_detects_distribution_drift():
    model = TinyLogitModel()
    batches = [
        {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])},
        {"input_ids": torch.tensor([[5, 4, 3, 2, 1]])},
    ]
    config = LogitFidelityConfig(batches=2, prompt_len=5, skip=0)
    references = capture_logit_references(model, batches, "cpu", config)
    perfect = evaluate_logit_fidelity(model, references, "cpu", config)
    assert perfect["top1_agreement"] == 1.0
    assert perfect["mean_teacher_kl"] < 1e-6
    assert perfect["median_teacher_kl"] < 1e-6
    assert perfect["p95_teacher_kl"] < 1e-6
    assert perfect["nll_delta"] == 0.0
    assert len(perfect["input_hashes"]) == 2
    assert len(perfect["prompt_metrics"]) == 2

    model.offset = 1
    divergent = evaluate_logit_fidelity(model, references, "cpu", config)
    assert divergent["top1_agreement"] == 0.0
    assert divergent["mean_teacher_kl"] > 1.0
    assert divergent["p95_teacher_kl"] >= divergent["median_teacher_kl"]
    assert divergent["nll_delta"] > 1.0
