from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from rotquant import RotQuantConfig, optimize_model
from rotquant.checkpoint import load_packed_model, save_packed_checkpoint
from rotquant.dynamic import CandidateScore, DynamicQuantConfig, _candidate_specs, _pareto_selection
from rotquant.eval.kv_cache import KVCacheEvalConfig, evaluate_kv_cache
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig, Quantizer
from scripts.score_cache_identity import score_cache_identity
from tests.test_checkpoint import _tiny_packed_llama
from tests.test_kv_cache_eval import _CacheModel

ROOT = Path(__file__).resolve().parents[1]


def test_gptq_requires_calibration_before_any_model_mutation():
    config = QuantConfig(bits=4, group_size=8, error_comp="gptq")
    with pytest.raises(ValueError, match="calibrated Hessian"):
        Quantizer(config).quantize_weight(torch.randn(8, 8))
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    original = list(model.children())
    with pytest.raises(ValueError, match="Hessian for 1"):
        patch_model(model, PatchConfig(quant=config, block=8), hessians={"0": torch.eye(8)})
    assert list(model.children()) == original
    layer = QuantLinear.from_linear(original[0], config, H=torch.eye(8))
    layer.retain_recovery_source(original[0].weight)
    before = layer.qweight
    with pytest.raises(ValueError, match="calibrated Hessian"):
        layer.refresh_quantization()
    assert layer.qweight is before
    layer.refresh_quantization(torch.eye(8))
    assert torch.isfinite(layer.qweight.dequantize()).all()


def test_shared_projections_keep_one_replacement_and_reject_filter_conflicts():
    model = nn.ModuleDict({"a": nn.Linear(8, 8)})
    model["b"] = model["a"]
    with pytest.raises(ValueError, match="alias conflicts"):
        optimize_model(model, RotQuantConfig(group_size=8, rotation_block=8, include=("a",)))
    assert model["a"] is model["b"]
    optimize_model(model, RotQuantConfig(group_size=8, rotation_block=8))
    assert isinstance(model["a"], QuantLinear) and model["a"] is model["b"]


def test_shared_projection_cannot_replace_an_unsafe_attention_alias():
    model = nn.ModuleDict({"safe": nn.Linear(8, 8), "unsafe": nn.MultiheadAttention(8, 2)})
    model["unsafe"].out_proj = model["safe"]
    with pytest.raises(ValueError, match="alias conflicts"):
        optimize_model(model, RotQuantConfig(group_size=8, rotation_block=8))
    assert model["safe"] is model["unsafe"].out_proj


def test_shared_projection_checkpoint_restores_all_aliases(tmp_path):
    model = _tiny_packed_llama("fwht")
    attention = model.model.layers[0].self_attn
    attention.k_proj = attention.q_proj
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(inputs, use_cache=False).logits
    save_packed_checkpoint(model, tmp_path / "model", model_loader="causal_lm")
    restored = load_packed_model(tmp_path / "model", dtype=torch.float32, fallback=True)
    attention = restored.model.layers[0].self_attn
    assert attention.q_proj is attention.k_proj
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs, use_cache=False).logits, expected, rtol=0, atol=0)


def test_root_projection_is_rejected_without_attaching_an_unused_child():
    model = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="root projection"):
        optimize_model(model, RotQuantConfig(group_size=8, rotation_block=8))
    assert not list(model.children())


def test_replacing_codebook_resolves_automatic_bounds_and_keeps_explicit_bounds():
    gaussian = QuantConfig(scale="mse_search")
    assert replace(gaussian, codebook="uniform").mse_search_bounds == (1.5, 4.0)
    explicit = replace(gaussian, mse_search_lo=.7, mse_search_hi=1.1)
    assert replace(explicit, codebook="uniform").mse_search_bounds == (.7, 1.1)
    config = DynamicQuantConfig(candidate_formats=({"name": "uniform", "bits": 4, "codebook": "uniform"},))
    assert _candidate_specs("layer", config, gaussian)[0][1].mse_search_bounds == (1.5, 4.0)


def score(size, distortion):
    return CandidateScore(4, QuantConfig(bits=4), size, 0, 0, size,
                          distortion, 1, distortion, 0, distortion)


def solve_budget(candidates, budget):
    selected, _ = _pareto_selection(candidates, size_attr="packed_bytes", fixed_bytes=0,
                                   target_bytes=budget, tolerance_bytes=0, granularity_bytes=1,
                                   minimum_bytes=0)
    return sum(row.score for row in selected.values())


def feasible_score(choice, budget):
    return sum(row.score for row in choice) if sum(row.packed_bytes for row in choice) <= budget else float("inf")


def test_bit_budget_solver_minimizes_distortion_over_all_smaller_recipes():
    assert solve_budget({"a": [score(2, 1), score(10, 0)], "b": [score(4, 100), score(10, 0)]}, 14) == 1
    rng = random.Random(19)
    for _ in range(20):
        candidates = {str(i): [score(rng.randrange(2, 12), rng.randrange(100)) for _ in range(3)]
                      for i in range(3)}
        budget = sum(min(row.packed_bytes for row in rows) for rows in candidates.values()) + 5
        optimum = min(feasible_score(choice, budget) for choice in itertools.product(*candidates.values()))
        assert solve_budget(candidates, budget) == optimum


@pytest.mark.parametrize("scale_bits", [8, 16])
def test_scale_overflow_fails_before_returning_a_packed_weight(scale_bits):
    quantizer = Quantizer(QuantConfig(bits=4, group_size=8, scale_bits=scale_bits))
    with pytest.raises(ValueError, match="overflow"):
        quantizer.quantize_weight(torch.full((2, 8), 1e6))
    with pytest.raises(ValueError, match="finite"):
        quantizer.quantize_weight(torch.full((2, 8), float("nan")))
    assert torch.isfinite(quantizer.quantize_weight(torch.full((2, 8), 1e-12)).dequantize()).all()


def test_score_cache_identity_tracks_weights_dtype_code_and_runtime():
    model = nn.Linear(8, 8)
    before = score_cache_identity(model, {})
    assert score_cache_identity(model, {"allocation": "random"}) == before
    assert score_cache_identity(model, {"stage_code_revision": "changed"}) != before
    with torch.no_grad():
        model.weight.add_(1)
    assert score_cache_identity(model, {})["model_sha256"] != before["model_sha256"]
    assert score_cache_identity(model.half(), {})["dtypes"] != before["dtypes"]
    assert "rotquant/quantize.py" in before["sources"]
    assert set(before["libraries"]) == {"torch", "numpy", "scipy", "transformers"}


class BypassedCacheModel(_CacheModel):
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        if past_key_values is not None and hasattr(past_key_values, "rotquant_write_coverage"):
            past_key_values.update = lambda keys, values, layer: past_key_values.layers[layer].update(keys, values)
        return super().forward(input_ids, attention_mask, past_key_values, use_cache)


def test_kv_write_coverage_rejects_bypass_and_records_every_batch():
    config = KVCacheEvalConfig(group_size=4, rotation_block=8, prompt_len=8,
                               continuation_len=4, batches=2, bootstrap_draws=8)
    batches = [{"input_ids": torch.arange(1, 16).reshape(1, -1)}] * 2
    with pytest.raises(RuntimeError, match="bypassed quantization"):
        evaluate_kv_cache(BypassedCacheModel(), batches, config, "cpu")
    metrics = evaluate_kv_cache(_CacheModel(), batches, config, "cpu")
    assert metrics["decode_write_coverage"] == [{"0": {"tokens": 4, "calls": 4}}] * 2


@pytest.mark.parametrize("historical", [False, True])
def test_publication_gate_carries_withdrawal_status(tmp_path, historical):
    report, tex = tmp_path / "report.json", tmp_path / "results.tex"
    command = [sys.executable, "-m", "scripts.audit_publication", "--report", str(report), "--tex-output", str(tex)]
    result = subprocess.run([*command, *(["--historical"] if historical else [])], cwd=ROOT, capture_output=True, check=False)
    assert result.returncode == (0 if historical else 1)
    receipt = json.loads(report.read_text())
    assert not receipt["passed"] and not receipt["publication_eligible"]
    assert receipt["arithmetic_passed"]
    assert "withdrawn" in receipt["evidence_status"]["cache_transfer"]
    assert "short_cache_kl_reduction_pct" not in receipt["derived"]
    assert ("historical_derived" in receipt) == historical
    if not historical:
        assert "\\newcommand{\\CacheShortReductionPct}{withdrawn}" in tex.read_text()


def test_ci_builds_current_cpu_graph_and_compiles_current_cuda_integration():
    jobs = yaml.safe_load((ROOT / ".github/workflows/llama-cpp-patch.yml").read_text())["jobs"]
    cpu = "\n".join(step.get("run", "") for step in jobs["v2-cpu"]["steps"])
    cuda = "\n".join(step.get("run", "") for step in jobs["v2-cuda-compile"]["steps"])
    assert "scripts.build_rq3_runtime" in cpu and "scripts.check_rq3_cpu_model" in cpu
    assert "for bits in 6 8" in cpu and "scripts.check_rq3_gpu" in cpu
    assert "prepare_source" in cuda and "-DCMAKE_CUDA_ARCHITECTURES=80" in cuda
    assert "--target rotquant_ggml_test" in cuda
