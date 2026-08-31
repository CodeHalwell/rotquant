"""Dataset revisions must reach every Hugging Face loading boundary."""

from __future__ import annotations

import sys
import types

import torch

from rotquant.eval.perplexity import _load_text
from scripts.run_experiment import build_calib_loader


def test_perplexity_dataset_revisions_are_forwarded(monkeypatch) -> None:
    calls = []

    def load_dataset(name, subset, **kwargs):
        calls.append((name, subset, kwargs))
        if name == "Salesforce/wikitext":
            return {"text": ["first", "second"]}
        return [{"text": "c4 row"}]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    assert _load_text("wikitext2", wikitext_revision="wiki-sha")
    assert _load_text("c4", c4_revision="c4-sha")

    assert calls[0][2]["revision"] == "wiki-sha"
    assert calls[1][2]["revision"] == "c4-sha"


def test_calibration_dataset_revision_is_forwarded(monkeypatch) -> None:
    captured = {}

    def load_dataset(name, subset, **kwargs):
        captured.update(kwargs)
        return [{"text": "enough tokens"}]

    class Tokenizer:
        def __call__(self, _text, return_tensors):
            assert return_tensors == "pt"
            return types.SimpleNamespace(input_ids=torch.arange(8).reshape(1, 8))

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    batches = build_calib_loader(
        Tokenizer(), 1, 4, "cpu", revision="c4-calibration-sha"
    )

    assert captured["revision"] == "c4-calibration-sha"
    assert batches[0]["input_ids"].shape == (1, 4)
