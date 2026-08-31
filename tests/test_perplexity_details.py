"""Paired perplexity details and safe screening stops."""
from types import SimpleNamespace

import torch
from torch import nn

import rotquant.eval.perplexity as ppl


class ConstantLossModel(nn.Module):
    def __init__(self, loss: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.loss_value = loss

    def forward(self, input_ids, labels):
        del input_ids, labels
        return SimpleNamespace(loss=self.anchor * 0 + self.loss_value)


def test_perplexity_details_are_paired_and_can_stop_early(monkeypatch):
    encoded = torch.arange(1, 31).reshape(1, -1)
    monkeypatch.setattr(ppl, "_tokenized_corpus", lambda *_args: encoded)
    source = ppl.perplexity_details(
        ConstantLossModel(1.0), object(), config=ppl.PPLConfig(
            seq_len=5, max_samples=4
        ), device="cpu",
    )
    assert source["windows"] == 4
    assert len(source["window_hashes"]) == 4
    assert source["ppl"] == torch.exp(torch.tensor(1.0)).item()

    candidate = ppl.perplexity_details(
        ConstantLossModel(2.0), object(), config=ppl.PPLConfig(
            seq_len=5,
            max_samples=4,
            early_stop_after=2,
            early_stop_relative_ppl=0.25,
            reference_window_nll_sums=source["window_nll_sums"],
            reference_window_tokens=source["window_tokens"],
        ), device="cpu",
    )
    assert candidate["stopped_early"] is True
    assert candidate["windows"] == 2
    assert candidate["tokens"] == sum(candidate["window_tokens"])
