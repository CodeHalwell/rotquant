"""Regressions for the allocator and evaluation defects of the 2026-11-17 review.

Each test fails on the code before the fix (section 4 of
docs/mathematical_review_2026-11-17.md, "Found, confirmed, not fixed").
"""
from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from rotquant import dynamic
from rotquant.block_train import TeacherCall
from rotquant.calibrate import HessianAccumulator, collect_hessians, collect_hessians_streamed
from rotquant.dynamic import (
    CandidateScore,
    _candidate_context,
    _candidate_linear,
    _local_error,
    _pareto_selection,
    select_dynamic_quantization,
    teacher_logit_kl,
)
from rotquant.eval.logit_fidelity import (
    LogitFidelityConfig,
    capture_logit_references,
    evaluate_logit_fidelity,
)
from rotquant.patch import PatchConfig
from rotquant.quantize import QuantConfig, Quantizer
from rotquant.rotate import ButterflyRotation
from rotquant.utils import BitBudget


class TinyLM(nn.Module):
    """Residual toy LM; layer ``i`` sees activations of scale ``1 + i``."""

    def __init__(self, layers: int = 8, width: int = 32):
        super().__init__()
        self.embed = nn.Embedding(32, width)
        self.layers = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(layers)])
        self.lm_head = nn.Linear(width, 32, bias=False)

    def forward(self, input_ids, use_cache=False, attention_mask=None):
        del use_cache, attention_mask
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = hidden + 0.1 * torch.tanh(layer(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


def _patch_cfg(seed: int = 0, **dynamic_config) -> PatchConfig:
    config = PatchConfig(
        quant=QuantConfig(bits=4, scale="rms", group_size=16),
        rotation="fwht", block=32,
        include=("layers.",), exclude=(),
        dynamic=dynamic_config,
    )
    config.seed = seed
    return config


def _activations(layers: int = 8, width: int = 32) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        f"layers.{index}": torch.randn(64, width, generator=generator) * (1 + index)
        for index in range(layers)
    }


def _select(model, seed=0, **dynamic_config):
    dynamic._CANDIDATE_SCORE_CACHE.clear()
    return select_dynamic_quantization(
        model, _patch_cfg(seed, **dynamic_config), activations=_activations())


# --------------------------------------------------------------------------- #
# 1. random_pareto must spend a bits-per-weight budget like the guided arm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target_bpw", [5.0, 6.5])
def test_random_pareto_spends_the_bits_per_weight_budget(target_bpw):
    torch.manual_seed(0)
    model = TinyLM().eval()
    _recipe, guided = _select(
        model, candidate_bits=[2, 3, 4, 5, 6], target_bpw=target_bpw)
    # The guided arm keeps its one-sided bpw target.
    assert guided["achieved_bpw"] <= target_bpw
    assert "budget_matching" not in guided["solver"]
    for seed in range(4):
        _recipe, control = _select(
            model, seed=seed, candidate_bits=[2, 3, 4, 5, 6],
            target_bpw=target_bpw, allocation="random_pareto")
        # Before the fix the control took the lowest random score anywhere at
        # or below the target: 4.5-4.75 bpw for both targets.
        assert control["achieved_bpw"] == pytest.approx(guided["achieved_bpw"])
        assert control["target_reached"] is True
        assert control["solver"]["budget_matching"] == "closest_to_target"


def _scored(size: int, score: float, bits: int = 4) -> CandidateScore:
    return CandidateScore(
        bits=bits, config=QuantConfig(bits=bits), packed_bytes=size,
        registered_bytes=0, codebook_bytes=0, complete_bytes=size,
        local_error=score, reference_energy=1.0, local_relative_error=score,
        global_kl=0.0, score=score, format_id=f"w{bits}",
    )


def test_pareto_objectives_under_a_one_sided_target():
    candidates = {
        "a": [_scored(10, 0.1, 2), _scored(30, 0.9, 4)],
        "b": [_scored(10, 0.2, 2), _scored(30, 0.8, 4)],
    }
    kwargs = {
        "size_attr": "packed_bytes", "fixed_bytes": 0, "target_bytes": 60,
        "tolerance_bytes": 0, "granularity_bytes": 1, "minimum_bytes": 0,
    }
    by_score, _ = _pareto_selection(candidates, **kwargs)
    closest, stats = _pareto_selection(candidates, objective="closest", **kwargs)
    assert sum(item.packed_bytes for item in by_score.values()) == 20
    assert sum(item.packed_bytes for item in closest.values()) == 60
    assert stats["search_within_tolerance"] is True
    with pytest.raises(ValueError, match="objective"):
        _pareto_selection(candidates, objective="largest", **kwargs)


# --------------------------------------------------------------------------- #
# 2-3. local score: deployed forward (QJL sketch) and a bias-free denominator
# --------------------------------------------------------------------------- #
def _linear_and_inputs(bias: bool):
    torch.manual_seed(3)
    source = nn.Linear(64, 32, bias=bias)
    if bias:
        with torch.no_grad():
            source.bias.copy_(torch.randn(32) * 2.0)
    mixing = torch.randn(64, 64) / 8
    inputs = torch.randn(256, 64) @ mixing + 0.3 * torch.randn(64)
    return source, inputs


def _candidate(source, quant, inputs=None):
    patch_cfg = PatchConfig(quant=quant, rotation="fwht", block=64)
    context = _candidate_context(source, patch_cfg, 0, None)
    mean = None if inputs is None else inputs.mean(dim=0)
    return _candidate_linear(source, quant, context, mean)


def test_local_error_scores_the_qjl_sketch_term():
    source, inputs = _linear_and_inputs(bias=True)
    quant = QuantConfig(bits=3, scale="turboquant", group_size=64,
                        error_comp="turboquant", sketch_k=16)
    candidate = _candidate(source, quant)
    assert candidate.qweight.sketch is not None
    with torch.no_grad():
        deployed = candidate(inputs) - source(inputs)
        weight_only = F.linear(
            candidate.act_rotation.rotate_activation(inputs),
            candidate.qweight.dequantize().float(), candidate.bias,
        ) - source(inputs)
    expected = float(deployed.pow(2).sum(dim=-1).mean())
    dequantized_only = float(weight_only.pow(2).sum(dim=-1).mean())
    # The sketch term matters here, so the old dequantize-only score differs.
    assert abs(expected - dequantized_only) > 0.1 * expected

    error, _energy, _relative = _local_error(source, candidate, inputs, 256)
    assert error == pytest.approx(expected, rel=1e-5)

    # Without activations the identity covariance gives the squared Frobenius
    # norm of the deployed effective weight's error, sketch included.
    with torch.no_grad():
        identity = torch.eye(64)
        effective = (candidate(identity) - candidate.bias).T
    weight = source.weight.detach()
    frobenius = float((effective - weight).pow(2).sum())
    error, energy, _relative = _local_error(source, candidate, None, 256)
    assert error == pytest.approx(frobenius, rel=1e-5)
    assert energy == pytest.approx(float(weight.pow(2).sum()), rel=1e-5)


def test_relative_error_denominator_excludes_the_bias():
    source, inputs = _linear_and_inputs(bias=True)
    with torch.no_grad():
        bias_free_energy = float(
            F.linear(inputs, source.weight).pow(2).sum(dim=-1).mean())
        with_bias_energy = float(source(inputs).pow(2).sum(dim=-1).mean())
    assert with_bias_energy > 1.5 * bias_free_energy

    for quant, mean_inputs in (
        (QuantConfig(bits=3, scale="rms", group_size=64), None),
        # A bias-corrected candidate changes the bias; that difference must
        # still count as error.
        (QuantConfig(bits=3, scale="rms", group_size=64,
                     bias_correction="mean"), inputs),
    ):
        candidate = _candidate(source, quant, mean_inputs)
        error, energy, relative = _local_error(source, candidate, inputs, 256)
        with torch.no_grad():
            expected_error = float(
                (candidate(inputs) - source(inputs)).pow(2).sum(dim=-1).mean())
        assert energy == pytest.approx(bias_free_energy, rel=1e-5)
        assert error == pytest.approx(expected_error, rel=1e-5)
        assert relative == pytest.approx(error / bias_free_energy, rel=1e-9)


# --------------------------------------------------------------------------- #
# 4. greedy / random adjacent downgrades must pass equal-size formats
# --------------------------------------------------------------------------- #
_EQUAL_SIZE_FORMATS = [
    {"name": "g2", "bits": 2, "codebook": "gaussian"},
    {"name": "g3", "bits": 3, "codebook": "gaussian"},
    {"name": "u3", "bits": 3, "codebook": "uniform"},
    {"name": "g4", "bits": 4, "codebook": "gaussian"},
]


@pytest.mark.parametrize("allocation", ["greedy", "random"])
def test_adjacent_downgrades_pass_equal_size_formats(allocation):
    torch.manual_seed(0)
    model = TinyLM().eval()
    # 3.5 bpw needs 2-bit layers: before the fix both allocators stopped at
    # u3 everywhere (4.0 bpw, target missed) behind the zero-byte u3 -> g3 step.
    _recipe, stats = _select(
        model, candidate_formats=_EQUAL_SIZE_FORMATS, target_bpw=3.5,
        allocation=allocation)
    assert stats["target_reached"] is True
    assert stats["solver"]["search_within_tolerance"] is True
    assert stats["achieved_bpw"] <= 3.5
    if allocation == "greedy":
        # One candidate per size, and the better-scored one at 3 bits.
        rows = stats["candidate_table"]
        for detail in stats["details"]:
            if detail["bits"] != 3:
                continue
            same_size = [
                row for row in rows
                if row["name"] == detail["name"] and row["bits"] == 3
            ]
            assert detail["score"] == min(row["score"] for row in same_size)


# --------------------------------------------------------------------------- #
# 5. bucketed DP with a window narrower than a bucket
# --------------------------------------------------------------------------- #
def _bucketed_instance(rng: random.Random, layers: int, choices: int):
    candidates = {}
    for layer in range(layers):
        base = rng.randint(20_000, 160_000)
        sensitivity = rng.choice([1, 3, 10, 30, 100])
        items = []
        for index in range(choices):
            bits = index + 2
            size = base * bits // 2 + rng.randint(0, 2_000)
            score = sensitivity * 4.0 ** (-bits) * (1 + 0.2 * rng.random())
            items.append(_scored(size, score, bits))
        candidates[f"L{layer}"] = sorted(items, key=lambda item: item.packed_bytes)
    return candidates


def _brute_force(candidates, lower, upper):
    best = None
    for recipe in itertools.product(*candidates.values()):
        total = sum(item.packed_bytes for item in recipe)
        if lower <= total <= upper:
            score = sum(item.score for item in recipe)
            best = score if best is None else min(best, score)
    return best


def test_narrow_byte_window_uses_the_exact_milp_and_keeps_the_better_recipe():
    rng = random.Random(7)
    solved = []
    for _ in range(40):
        candidates = _bucketed_instance(rng, rng.randint(5, 6), 3)
        low = sum(min(i.packed_bytes for i in v) for v in candidates.values())
        high = sum(max(i.packed_bytes for i in v) for v in candidates.values())
        granularity = high // 80          # bucketed: high > 64 buckets
        target = rng.randint(low + (high - low) // 5, high - (high - low) // 5)
        tolerance = max(1, round(target * 0.0005))   # window << one bucket
        optimum = _brute_force(candidates, target - tolerance, target + tolerance)
        if optimum is None:
            continue
        selected, stats = _pareto_selection(
            candidates, size_attr="packed_bytes", fixed_bytes=0,
            target_bytes=target, tolerance_bytes=tolerance,
            granularity_bytes=granularity,
        )
        total = sum(item.packed_bytes for item in selected.values())
        assert target - tolerance <= total <= target + tolerance
        # Before the fix some bucketed recipes here were far from optimal.
        assert sum(item.score for item in selected.values()) == pytest.approx(
            optimum, rel=1e-9)
        solved.append(stats)
    assert len(solved) >= 20
    assert all(stats["bucketed_exact"] is False for stats in solved)
    assert all(stats["repair"]["attempted"] is True for stats in solved)
    assert any(
        stats["solver"] == "bucketed_multiple_choice_with_exact_milp"
        for stats in solved
    )


def test_wide_byte_window_keeps_the_bucketed_solution_without_milp():
    rng = random.Random(11)
    candidates = _bucketed_instance(rng, 6, 3)
    high = sum(max(i.packed_bytes for i in v) for v in candidates.values())
    low = sum(min(i.packed_bytes for i in v) for v in candidates.values())
    target = (low + high) // 2
    _selected, stats = _pareto_selection(
        candidates, size_attr="packed_bytes", fixed_bytes=0,
        target_bytes=target, tolerance_bytes=round(target * 0.05),
        granularity_bytes=high // 80,
    )
    assert stats["solver"] == "bucketed_multiple_choice_pareto"
    assert stats["repair"] == {"attempted": False}


# --------------------------------------------------------------------------- #
# 6. random controls must not inherit the sensitivity-chosen protected set
# --------------------------------------------------------------------------- #
def test_random_controls_draw_their_protected_layers_at_random():
    torch.manual_seed(0)
    model = TinyLM().eval()
    protection = {
        "candidate_bits": [3, 4], "target_bpw": 4.5,
        "protect_top_fraction": 0.25, "protect_min_bits": 4,
        "protect_metric": "local_error",
    }
    _recipe, guided = _select(model, **protection)
    guided_set = {row["name"] for row in guided["protected_layers"]}
    rows = guided["candidate_table"]
    sensitivity = {
        name: next(r["local_error"] for r in rows if r["name"] == name and r["bits"] == 3)
        - next(r["local_error"] for r in rows if r["name"] == name and r["bits"] == 4)
        for name in {row["name"] for row in rows}
    }
    assert guided_set == set(sorted(sensitivity, key=sensitivity.get)[-2:])

    for allocation in ("random", "random_pareto"):
        controls = [
            _select(model, seed=seed, allocation=allocation, **protection)[1]
            for seed in range(6)
        ]
        drawn = [
            {row["name"] for row in control["protected_layers"]}
            for control in controls
        ]
        # Before the fix every seed protected exactly the guided set.
        assert any(names != guided_set for names in drawn)
        assert all(len(names) == len(guided_set) for names in drawn)
        for control in controls:
            assert control["protection_policy"] == "seeded_random"
            assert all(row["policy"] == "seeded_random"
                       for row in control["protected_layers"])
        _recipe, repeat = _select(
            model, seed=0, allocation=allocation, **protection)
        assert {row["name"] for row in repeat["protected_layers"]} == drawn[0]
    assert guided["protection_policy"] == "measured_sensitivity"


# --------------------------------------------------------------------------- #
# 7. logit fidelity ignores padding; teacher_logit_kl is token-weighted
# --------------------------------------------------------------------------- #
class PaddingNoiseLM(nn.Module):
    """Real logits on unmasked positions, seeded noise on padding."""

    def __init__(self, noise_seed: int):
        super().__init__()
        torch.manual_seed(5)
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.noise_seed = noise_seed

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del use_cache
        logits = self.head(torch.tanh(self.embed(input_ids)))
        if attention_mask is not None:
            generator = torch.Generator().manual_seed(self.noise_seed)
            noise = torch.randn(logits.shape, generator=generator) * 10
            logits = torch.where(attention_mask[..., None].bool(), logits, noise)
        return SimpleNamespace(logits=logits)


def test_logit_fidelity_excludes_padded_inputs_and_targets():
    source = PaddingNoiseLM(noise_seed=1).eval()
    candidate = copy.deepcopy(source)
    candidate.noise_seed = 2          # identical except on padding
    torch.manual_seed(6)
    ids = torch.randint(1, 16, (2, 8))
    ids[1, 5:] = 0
    mask = torch.ones_like(ids)
    mask[1, 5:] = 0
    lone = torch.randint(1, 16, (1, 8))
    lone[0, 1:] = 0
    lone_mask = torch.zeros_like(lone)
    lone_mask[0, 0] = 1
    config = LogitFidelityConfig(batches=2, prompt_len=8, bootstrap_draws=50)
    references = capture_logit_references(
        source,
        [{"input_ids": ids, "attention_mask": mask},
         {"input_ids": lone, "attention_mask": lone_mask}],
        "cpu", config,
    )
    result = evaluate_logit_fidelity(candidate, references, "cpu", config)
    # Row 0 scores 7 positions; row 1 has 5 real tokens, so 4 scored
    # positions; the one-token prompt has none. Before the fix: 21 tokens.
    assert result["tokens"] == 11
    assert result["unscored_input_hashes"] == [references[1].input_hash]
    assert [row["tokens"] for row in result["prompt_metrics"]] == [11]
    assert result["top1_agreement"] == 1.0
    assert result["mean_teacher_kl"] < 1e-5
    assert result["max_teacher_kl"] < 1e-5

    with torch.no_grad():
        logits = source(ids, attention_mask=mask).logits[:, :-1].float()
    valid = (mask[:, :-1] & mask[:, 1:]).bool()
    expected_nll = F.cross_entropy(logits[valid], ids[:, 1:][valid])
    assert result["candidate_nll"] == pytest.approx(float(expected_nll), rel=1e-5)


def test_teacher_logit_kl_is_a_token_weighted_mean():
    torch.manual_seed(1)
    model = TinyLM(layers=2, width=16).eval()
    short = {"input_ids": torch.randint(0, 32, (1, 1))}
    long = {"input_ids": torch.randint(0, 32, (1, 7))}
    with torch.no_grad():
        short_logits = model(**short).logits.clone()
        long_logits = model(**long).logits.clone()
    perturbed = short_logits + torch.randn(short_logits.shape) * 3
    calls = [
        TeacherCall(inputs=short, logits=perturbed),
        TeacherCall(inputs=long, logits=long_logits),
    ]
    short_kl = float(F.kl_div(
        F.log_softmax(short_logits, dim=-1), F.softmax(perturbed, dim=-1),
        reduction="none").sum())
    long_kl = float(F.kl_div(
        F.log_softmax(long_logits, dim=-1), F.softmax(long_logits, dim=-1),
        reduction="none").sum())
    observed = teacher_logit_kl(model, calls, "cpu", torch.float32)
    # Before the fix: the mean of per-call means, (short_kl + long_kl/7) / 2.
    assert short_kl > 0.1
    assert observed == pytest.approx((short_kl + long_kl) / 8, rel=1e-5)


# --------------------------------------------------------------------------- #
# 8. nominal BitBudget of vector codebooks
# --------------------------------------------------------------------------- #
def test_bit_budget_nominal_rate_counts_vector_codes_per_weight():
    vector = BitBudget(levels=2**16, group_size=128, scale_bits=16.0, vector_dim=8)
    assert vector.code_bits == 16
    assert vector.code_bits_per_weight == 2
    assert vector.bits_per_weight == pytest.approx(2.125)
    vector.assert_matches(2.125)
    scalar = BitBudget(levels=8, group_size=128)
    assert scalar.code_bits_per_weight == scalar.code_bits == 3
    assert scalar.bits_per_weight == pytest.approx(3.125)
    stored = BitBudget(levels=2**16, group_size=128, vector_dim=8,
                       stored_bits=1000.0, stored_weights=400)
    assert stored.bits_per_weight == 2.5
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError, match="vector_dim"):
            BitBudget(levels=4, group_size=32, vector_dim=invalid)

    # With the codebook dimension supplied, the nominal formula reproduces
    # the stored rate of a real vector-quantized tensor.
    weight = torch.randn(16, 128, generator=torch.Generator().manual_seed(2))
    qweight = Quantizer(QuantConfig(
        bits=2, codebook="vector", vector_dim=2, scale="rms", group_size=32,
        vector_samples=1024, vector_iters=5,
    )).quantize_weight(weight)
    budget = qweight.bit_budget()
    nominal = replace(budget, stored_bits=None, stored_weights=None,
                      vector_dim=qweight.codebook.dim)
    assert nominal.bits_per_weight == pytest.approx(budget.bits_per_weight)


# --------------------------------------------------------------------------- #
# 9. Hessians are undamped by default (GPTQ damps them itself)
# --------------------------------------------------------------------------- #
def test_collected_hessians_are_undamped_by_default(tmp_path):
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(8, 4))
    batches = [torch.randn(5, 8) for _ in range(3)]
    stacked = torch.cat(batches)
    second_moment = stacked.T @ stacked / stacked.shape[0]

    joint = collect_hessians(model, batches, "cpu")
    streamed = collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path)
    assert joint.damp_frac == streamed.damp_frac == 0.0
    assert torch.allclose(joint.hessians["0"], second_moment, atol=1e-6)
    assert torch.allclose(streamed.hessians["0"], second_moment, atol=1e-6)

    accumulator = HessianAccumulator(8)
    accumulator.update(stacked)
    assert torch.equal(accumulator.finalize(), accumulator.H)
    # An explicit ridge is still available to consumers that need one.
    damped = collect_hessians(model, batches, "cpu", damp_frac=0.01)
    ridge = 0.01 * torch.diagonal(second_moment).mean()
    assert torch.allclose(
        damped.hessians["0"], second_moment + ridge * torch.eye(8), atol=1e-6)


# --------------------------------------------------------------------------- #
# 10. ButterflyRotation float64 reference path
# --------------------------------------------------------------------------- #
def test_butterfly_float64_path_is_orthogonal_to_float64_precision():
    rotation = ButterflyRotation(64, block=64, seed=0)
    with torch.no_grad():
        rotation.theta.add_(torch.randn(rotation.theta.shape,
                                        generator=torch.Generator().manual_seed(1)))
    rotation.eval()
    with torch.no_grad():
        matrix = rotation.as_matrix(dtype=torch.float64)
    deviation = (matrix @ matrix.T - torch.eye(64, dtype=torch.float64)).abs().max()
    # FP32 trigonometry left this at about 2e-7.
    assert float(deviation) < 1e-13

    # The FP32 evaluation cache is untouched by float64 calls.
    x = torch.randn(3, 64)
    with torch.no_grad():
        first = rotation.rotate_activation(x)
        cached_cos = rotation._cached_cos
        assert cached_cos is not None and cached_cos.dtype == torch.float32
        rotated64 = rotation.rotate_activation(x.double())
        assert rotation._cached_cos is cached_cos
        assert torch.equal(rotation.rotate_activation(x), first)
    assert math.isclose(
        float(rotated64.norm()), float(x.double().norm()), rel_tol=1e-12)
