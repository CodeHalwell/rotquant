"""Regression tests for the 2026-11-17 mathematical review.

Each test pins a statement the review either corrected or established, so a
later change cannot quietly reintroduce the defect.
"""
import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from rotquant.block_train import (
    BlockRotationTrainConfig,
    _distill_eval_loss,
    _project_log_multiplier,
    _scale_override,
    collect_block_calls,
    collect_teacher_calls,
    distill_packed_model,
    find_transformer_blocks,
    train_and_patch_blocks,
)
from rotquant.codebooks import (
    ScalarCodebook,
    _gaussian_cells,
    _lloyd_max_grid,
    build_finite_e8_codebook,
    build_scalar_codebook,
    lloyd_max_gaussian,
    lloyd_max_gaussian_grid,
    lloyd_max_samples,
    lloyd_max_spherical,
    quantizer_mse,
    quantizer_mse_spherical,
    turboquant_mse_bound,
)
from rotquant.eval.statistics import (
    bootstrap_cluster_mean_interval,
    bootstrap_mean_interval,
)
from rotquant.patch import PatchConfig
from rotquant.quantize import QuantConfig
from rotquant.rotate import ButterflyRotation
from tests.test_block_train import ToyCausalLM

# --------------------------------------------------------------------------- #
# Scalar codebooks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bits", range(1, 9))
def test_gaussian_grid_satisfies_the_lloyd_max_conditions(bits):
    centroids = lloyd_max_gaussian(2**bits)
    assert centroids.shape == (2**bits,)
    assert np.all(np.diff(centroids) > 0)
    assert np.array_equal(centroids, -centroids[::-1])
    _, _, mass, first_moment = _gaussian_cells(centroids)
    # Every centroid is the conditional mean of its own cell.
    assert np.max(np.abs(centroids - first_moment / mass)) < 1e-10
    assert abs(mass.sum() - 1.0) < 1e-12
    # A stationary Lloyd-Max quantiser's distortion is the variance it loses.
    assert quantizer_mse(centroids) == pytest.approx(
        1.0 - float((centroids**2 * mass).sum()), abs=1e-12)


def test_gaussian_distortion_matches_the_published_table_and_the_bounds():
    published = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497}  # Max (1960)
    previous_ratio = 0.0
    for bits in range(1, 9):
        distortion = quantizer_mse(lloyd_max_gaussian(2**bits))
        if bits in published:
            assert distortion == pytest.approx(published[bits], rel=2e-3)
        assert 4.0**-bits < distortion <= turboquant_mse_bound(bits)
        # The gap to the Shannon bound grows towards sqrt(3) * pi / 2 = 2.72.
        ratio = distortion * 4.0**bits
        assert previous_ratio < ratio < math.sqrt(3) * math.pi / 2
        previous_ratio = ratio


def test_earlier_grid_solver_is_kept_and_is_not_converged_above_four_bits():
    for levels, excess in ((16, 1e-5), (32, 7e-4), (64, 6e-3), (256, 0.26)):
        legacy = quantizer_mse(lloyd_max_gaussian_grid(levels))
        exact = quantizer_mse(lloyd_max_gaussian(levels))
        assert legacy >= exact
        assert legacy / exact - 1.0 == pytest.approx(excess, rel=0.2, abs=1e-5)
    # The legacy name reproduces the old centroids, e.g. its narrow 8-bit tail.
    legacy = build_scalar_codebook("gaussian_legacy", 256).centroids
    assert float(legacy[-1]) == pytest.approx(4.2053, abs=1e-3)
    assert float(build_scalar_codebook("gaussian", 256).centroids[-1]) == pytest.approx(
        4.6035, abs=1e-3)
    assert QuantConfig(bits=5, codebook="gaussian_legacy").codebook == "gaussian_legacy"


def test_prefix_sum_lloyd_iteration_equals_the_direct_cell_sums():
    generator = np.random.default_rng(0)
    xs = np.sort(generator.normal(size=5000))
    weights = generator.uniform(0.5, 1.5, size=5000)
    initial = np.quantile(xs, (np.arange(16) + 0.5) / 16)

    centroids = initial.copy()
    for _ in range(7):
        bounds = (centroids[:-1] + centroids[1:]) / 2.0
        indices = np.searchsorted(bounds, xs, side="right")
        centroids = np.array([
            (xs[indices == k] * weights[indices == k]).sum() / weights[indices == k].sum()
            for k in range(16)])
    np.testing.assert_allclose(
        _lloyd_max_grid(xs, weights, 16, initial=initial, iters=7), centroids,
        rtol=0, atol=1e-11)


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_calibrated_fit_reaches_the_source_optimum_on_gaussian_samples(bits):
    generator = torch.Generator().manual_seed(0)
    samples = torch.randn(65536, generator=generator)
    held_out = torch.randn(1_000_000, generator=generator)

    def held_out_mse(centroids):
        reconstruction, _ = ScalarCodebook(centroids).quantize(held_out)
        return float((held_out - reconstruction).square().mean())

    fitted = held_out_mse(lloyd_max_samples(samples, 2**bits))
    optimum = held_out_mse(lloyd_max_gaussian(2**bits))
    # Before the fix the fit was 0.4%, 12% and 64% above the optimum here.
    assert fitted <= optimum * 1.02


def test_spherical_grid_is_converged_and_beats_the_gaussian_grid_on_its_own_source():
    for bits in (4, 6):
        spherical = lloyd_max_spherical(2**bits, 128)
        longer = lloyd_max_spherical(2**bits, 128, iters=200_000)
        np.testing.assert_allclose(spherical, longer, atol=1e-8)
        assert quantizer_mse_spherical(spherical, 128) < quantizer_mse_spherical(
            lloyd_max_gaussian(2**bits), 128)


# --------------------------------------------------------------------------- #
# Finite E8 codebook
# --------------------------------------------------------------------------- #
E8_SHELL_SIZES = {2.0: 240, 4.0: 2160, 6.0: 6720, 8.0: 17520, 10.0: 30240, 12.0: 60480}


@pytest.mark.parametrize("bits", [1, 2])
def test_finite_e8_is_a_symmetric_lowest_energy_subset_of_the_lattice(bits):
    codebook = build_finite_e8_codebook(bits)
    raw = codebook.raw_points
    assert raw.shape == (2 ** (8 * bits), 8)
    doubled = torch.round(raw * 2).to(torch.int64)
    # Every point is in D8 or D8 + 1/2: all-even or all-odd doubled
    # coordinates, with an even coordinate sum.
    parity = doubled.remainder(2)
    assert bool(((parity == 0).all(dim=1) | (parity == 1).all(dim=1)).all())
    assert bool((raw.sum(dim=1).remainder(2) == 0).all())
    keys = {tuple(row) for row in doubled.tolist()}
    assert len(keys) == raw.shape[0]
    assert all(tuple(-value for value in key) in keys for key in keys)
    assert float(raw.mean(dim=0).abs().max()) == 0.0
    assert float(raw.square().sum(dim=1).min()) > 0.0

    energy = raw.square().sum(dim=1)
    shells = dict(zip(*[values.tolist() for values in torch.unique(energy, return_counts=True)]))
    last = max(shells)
    for shell_energy, count in shells.items():
        if shell_energy < last:
            assert count == E8_SHELL_SIZES[shell_energy]   # complete shells below
        else:
            assert count <= E8_SHELL_SIZES[shell_energy]


def test_finite_e8_scale_minimises_gaussian_distortion():
    samples = torch.randn(20_000, 8, generator=torch.Generator().manual_seed(7))

    def distortion(codebook, factor=1.0):
        from rotquant.codebooks import VectorCodebook
        scaled = VectorCodebook(codebook.centroids * factor)
        reconstruction, _ = scaled.quantize(samples)
        return float((samples - reconstruction).square().mean())

    one_bit = build_finite_e8_codebook(1)
    two_bit = build_finite_e8_codebook(2)
    assert distortion(one_bit) < 0.323           # 0.338 at unit RMS before the fix
    assert distortion(two_bit) < 0.0910          # 0.0926 for the asymmetric box
    assert distortion(two_bit) < quantizer_mse(lloyd_max_gaussian(4))
    for codebook in (one_bit, two_bit):
        centre = distortion(codebook)
        assert distortion(codebook, 0.93) > centre
        assert distortion(codebook, 1.07) > centre
    # The lattice fast path and the exhaustive search agree on the encoding.
    fast, _ = two_bit.quantize(samples[:2000])
    from rotquant.codebooks import VectorCodebook
    slow, _ = VectorCodebook(two_bit.centroids).quantize(samples[:2000])
    assert float((samples[:2000] - fast).square().sum()) == pytest.approx(
        float((samples[:2000] - slow).square().sum()), rel=1e-6)


# --------------------------------------------------------------------------- #
# Block training and recovery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_learned_scale_override_rotates_in_single_precision(dtype):
    torch.manual_seed(0)
    linear = nn.Linear(256, 32, bias=False).to(dtype)
    rotation = ButterflyRotation(256, block=128, seed=3)
    with torch.no_grad():
        rotation.theta.add_(0.02 * torch.randn_like(rotation.theta))
    config = QuantConfig(bits=4, group_size=128, scale="mse_search")
    multiplier = (1.0 + 0.1 * torch.randn(32, 2)).clamp(0.5, 1.5)
    override = _scale_override(linear, rotation, config, {"scale_multiplier": multiplier})
    rotated = rotation.rotate_weight(linear.weight.detach().float())
    expected = rotated.reshape(32, 2, 128).square().mean(dim=-1).sqrt() * multiplier
    assert torch.equal(override, expected)
    # Rotating the low-precision tensor rounds before the RMS and differs.
    rounded = rotation.rotate_weight(linear.weight.detach()).float()
    assert not torch.equal(
        rounded.reshape(32, 2, 128).square().mean(dim=-1).sqrt() * multiplier, expected)


def test_scale_multiplier_is_projected_back_inside_its_bounds():
    parameter = nn.Parameter(torch.tensor([-2.0, 0.0, 0.1, 3.0]))
    _project_log_multiplier(parameter, 0.8, 1.25)
    expected = torch.tensor([math.log(0.8), 0.0, 0.1, math.log(1.25)])
    assert torch.allclose(parameter.detach(), expected)
    # Inside the interval exp(p).clamp(...) has a non-zero gradient again.
    value = parameter.exp().clamp(0.8, 1.25).sum()
    value.backward()
    assert bool((parameter.grad[1:3] > 0).all())
    _project_log_multiplier(None, 0.8, 1.25)


@pytest.mark.parametrize("min_improvement,init", [(0.9, "residual_svd"), (0.0, "residual_svd")])
def test_reported_final_recovery_losses_describe_the_deployed_model(min_improvement, init):
    torch.manual_seed(5)
    model = ToyCausalLM(vocab=32, d=32, n=2).eval()
    batches = [{"input_ids": torch.randint(0, 32, (2, 16)),
                "attention_mask": torch.ones(2, 16, dtype=torch.long)} for _ in range(12)]
    calls = collect_block_calls(
        model, batches[:3], "cpu", blocks=find_transformer_blocks(model),
        max_batches=3, storage_dtype=torch.float32)
    teacher = collect_teacher_calls(
        model, batches[3:], "cpu", max_batches=9, storage_dtype=torch.float32)
    base = {
        "objective": "block", "steps": 2, "lr": 1e-2, "train_batches": 1,
        "validation_batches": 1, "selection_batches": 1, "assignment_scale": "rms",
        "selection_min_improvement": 0.0, "learn_scales": True, "distill_steps": 6,
        "distill_train_batches": 3, "distill_validation_batches": 3,
        "distill_selection_batches": 3, "distill_early_stopping_patience": 0,
        "distill_train_rotations": False, "distill_train_scales": True,
        "distill_lora_rank": 4, "distill_lora_alpha": 8.0, "distill_lora_lr": 1e-2,
        "distill_lora_init": init, "distill_selection_min_improvement": min_improvement,
    }
    patch = PatchConfig(
        quant=QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=16),
        rotation="butterfly", block=16, fallback=True, train_rotation=dict(base))
    block_only = copy.copy(patch)
    block_only.train_rotation = {**base, "distill_steps": 0}
    train_and_patch_blocks(model, block_only, calls)
    config = BlockRotationTrainConfig(**base)
    stats = distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert stats["accepted"] is (min_improvement == 0.0)
    deployed_train = _distill_eval_loss(model, teacher[:3], "cpu", torch.float32, config)
    deployed_validation = _distill_eval_loss(model, teacher[3:6], "cpu", torch.float32, config)
    assert stats["final_train_loss"] == pytest.approx(deployed_train, abs=1e-9)
    assert stats["final_validation_loss"] == pytest.approx(deployed_validation, abs=1e-9)


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #


def test_prompt_resampling_is_wider_than_token_resampling_for_correlated_tokens():
    generator = torch.Generator().manual_seed(0)
    prompts, tokens = 24, 64
    # A shared per-prompt offset makes the tokens of one prompt correlated.
    values = torch.randn(prompts, 1, generator=generator) + 0.5 * torch.randn(
        prompts, tokens, generator=generator)
    token_low, token_high = bootstrap_mean_interval(values.reshape(-1))
    prompt_low, prompt_high = bootstrap_cluster_mean_interval(
        values.sum(dim=1), torch.full((prompts,), float(tokens)))
    assert prompt_high - prompt_low > 3 * (token_high - token_low)
    mean = float(values.mean())
    assert prompt_low < mean < prompt_high

    # With independent tokens the two agree.
    independent = torch.randn(prompts, tokens, generator=generator)
    token = bootstrap_mean_interval(independent.reshape(-1))
    prompt = bootstrap_cluster_mean_interval(
        independent.sum(dim=1), torch.full((prompts,), float(tokens)))
    assert (prompt[1] - prompt[0]) == pytest.approx(token[1] - token[0], rel=0.5)


def test_cluster_bootstrap_refuses_to_report_certainty_from_one_prompt():
    low, high = bootstrap_cluster_mean_interval([3.0], [10.0])
    assert math.isnan(low) and math.isnan(high)
    with pytest.raises(ValueError):
        bootstrap_cluster_mean_interval([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        bootstrap_cluster_mean_interval([1.0, 2.0], [1.0, 0.0])
