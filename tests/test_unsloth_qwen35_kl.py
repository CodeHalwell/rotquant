"""Pinned artifact and full-distribution scoring checks for the GGUF control."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run_unsloth_qwen35_4b_kl.py"
    spec = importlib.util.spec_from_file_location("run_unsloth_qwen35_4b_kl", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_released_unsloth_artifacts_are_immutable_and_complete():
    module = _load_script()
    assert module.GGUF_REVISION == "e87f176479d0855a907a41277aca2f8ee7a09523"
    assert module.UD_Q4.bytes == 2_912_109_728
    assert module.UD_Q4.sha256 == (
        "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7"
    )
    assert module.UD_Q4.bytes + module.MM_PROJ_F16.bytes == 3_584_533_344
    assert len(module.BF16.sha256) == 64
    assert len(module.LLAMA_CPP_ENGINE_REVISION) == 40
    assert len(module.ROTQUANT_W4A8_INPUT_HASHES) == 4
    assert all(len(value) == 64 for value in module.ROTQUANT_W4A8_INPUT_HASHES)


def test_distribution_metrics_matches_direct_torch_calculation():
    module = _load_script()
    teacher = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=np.float16)
    candidate = np.array([[1.5, 0.5, -1.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    targets = np.array([0, 2], dtype=np.int64)
    result = module.distribution_metrics(
        teacher, candidate, targets, device="cpu", chunk_tokens=1
    )

    left = torch.tensor(teacher, dtype=torch.float32)
    right = torch.tensor(candidate, dtype=torch.float32)
    left_log = torch.log_softmax(left, -1)
    right_log = torch.log_softmax(right, -1)
    expected_kl = torch.sum(left_log.exp() * (left_log - right_log), -1)
    assert result["teacher_kl"] == pytest.approx(expected_kl.tolist())
    assert result["top1_matches"] == [True, False]
    assert result["source_nll"] == pytest.approx(
        [-left_log[0, 0].item(), -left_log[1, 2].item()]
    )
    assert result["candidate_nll"] == pytest.approx(
        [-right_log[0, 0].item(), -right_log[1, 2].item()]
    )


def test_distribution_metrics_rejects_misaligned_vocab_or_targets():
    module = _load_script()
    with pytest.raises(ValueError, match="shapes differ"):
        module.distribution_metrics(
            np.zeros((2, 3)), np.zeros((2, 4)), np.zeros(2), device="cpu"
        )
    with pytest.raises(ValueError, match="one ID"):
        module.distribution_metrics(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(1), device="cpu"
        )
