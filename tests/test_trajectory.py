"""Free-running source/deployed trajectory fidelity."""
from types import SimpleNamespace

import torch

from rotquant.eval.trajectory import (
    TrajectoryConfig,
    capture_trajectories,
    evaluate_trajectories,
)


class TinyGenerator:
    def __init__(self):
        self.config = SimpleNamespace()
        self.offset = 0

    def eval(self):
        return self

    def generate(self, input_ids, max_new_tokens, **kwargs):
        del kwargs
        token = (input_ids[:, -1:] + self.offset + 1) % 31
        continuation = token.repeat(1, max_new_tokens)
        return torch.cat((input_ids, continuation), dim=-1)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def test_trajectory_fidelity_detects_free_running_divergence():
    model = TinyGenerator()
    config = TrajectoryConfig(batches=2, prompt_len=4, new_tokens=8, skip=0)
    batches = [
        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
        {"input_ids": torch.tensor([[5, 6, 7, 8]])},
    ]
    references = capture_trajectories(
        model, TinyTokenizer(), batches, "cpu", config)
    perfect = evaluate_trajectories(
        model, TinyTokenizer(), references, "cpu", config)
    assert perfect["token_agreement"] == 1.0
    assert perfect["exact_trajectory_rate"] == 1.0
    assert perfect["mean_matching_prefix"] == 8.0

    model.offset = 1
    divergent = evaluate_trajectories(
        model, TinyTokenizer(), references, "cpu", config)
    assert divergent["token_agreement"] == 0.0
    assert divergent["exact_trajectory_rate"] == 0.0
    assert divergent["mean_first_divergence"] == 0.0
