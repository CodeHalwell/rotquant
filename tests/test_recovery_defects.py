"""Regressions for the recovery defects in docs/mathematical_review_2026-11-17.md, section 4.

Each test failed on the code before the fix:

* code refresh broke "restore best" and "reject -> restore" and re-selected
  learned scales;
* Adam stepped FP16/BF16 parameters directly;
* residual-SVD adapters were attached at patch time, inside block training;
* the block proxy differed from the packed block (length correction with
  learned scales, uniform-grid scale bounds, unrounded scales, FP32 angles).
"""
import itertools

import pytest
import torch
from torch import nn

from rotquant import block_train
from rotquant.block_train import (
    BlockRotationTrainConfig,
    FakeQuantButterflyLinear,
    _distill_eval_loss,
    _fake_quant_block,
    _packed_cpu_block,
    _rotation_states,
    collect_block_calls,
    collect_teacher_calls,
    distill_packed_model,
    find_transformer_blocks,
    train_and_patch_blocks,
    train_and_patch_blocks_streamed,
)
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig
from rotquant.quantize import QuantConfig
from tests.test_block_train import ToyCausalLM, ToyTransformer

_GAUSSIAN_RMS = QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=16)
_GAUSSIAN_MSE = QuantConfig(bits=3, codebook="gaussian", scale="mse_search", group_size=16)


def _block_config(**extra):
    return {
        "objective": "block", "steps": 2, "lr": 1e-2, "train_batches": 1,
        "validation_batches": 1, "selection_batches": 1,
        "assignment_scale": "rms", "selection_min_improvement": 0.0, **extra,
    }


def _toy(*, quant=_GAUSSIAN_RMS, storage="float32", seed=5, **block_extra):
    """Block-trained two-block toy LM and nine teacher calls.

    ``block_extra`` may carry ``distill_*`` keys: with ``distill_steps`` left
    at zero they only decide what patching retains for a later recovery run.
    """
    torch.manual_seed(seed)
    model = ToyCausalLM(vocab=32, d=32, n=2).eval()
    batches = [{"input_ids": torch.randint(0, 32, (2, 16)),
                "attention_mask": torch.ones(2, 16, dtype=torch.long)}
               for _ in range(12)]
    calls = collect_block_calls(
        model, batches[:3], "cpu", blocks=find_transformer_blocks(model),
        max_batches=3, storage_dtype=torch.float32)
    teacher = collect_teacher_calls(
        model, batches[3:], "cpu", max_batches=9, storage_dtype=torch.float32)
    block = _block_config(**block_extra)
    patch = PatchConfig(quant=quant, rotation="butterfly", block=16, fallback=True,
                        train_rotation=block, rotation_storage_dtype=storage)
    stats = {}
    train_and_patch_blocks(model, patch, calls, stats_out=stats)
    return model, teacher, block, stats


def _distill(block, **overrides):
    values = {
        **block, "distill_steps": 6, "distill_train_batches": 3,
        "distill_validation_batches": 3, "distill_selection_batches": 3,
        "distill_early_stopping_patience": 0, "distill_train_rotations": False,
        "distill_train_scales": True, "distill_scale_lr": 0.02,
        "distill_batches_per_step": 1,
    }
    values.update(overrides)
    return BlockRotationTrainConfig(**values)


def _quant_linears(model):
    return [module for module in model.modules() if isinstance(module, QuantLinear)]


def _packed(model):
    return [(module.qweight.packed.data.clone(), module.qweight.scales.clone(),
             module.qweight.dequantize().clone()) for module in _quant_linears(model)]


def _assert_same_packed(actual, expected):
    assert len(actual) == len(expected)
    for layer, (got, want) in enumerate(zip(actual, expected)):
        for got_tensor, want_tensor in zip(got, want):
            assert torch.equal(got_tensor, want_tensor), f"layer {layer} differs"


def _select_last_step(monkeypatch):
    """Make every later evaluation better, so the last step is selected and accepted."""
    counter = itertools.count()
    monkeypatch.setattr(block_train, "_distill_eval_loss",
                        lambda *args, **kwargs: 1.0 - 1e-3 * next(counter))


# --------------------------------------------------------------------------- #
# 1. code refresh, best-step restore and rejection
# --------------------------------------------------------------------------- #

def test_rejected_refresh_run_restores_the_exact_pre_distillation_state():
    model, teacher, block, _ = _toy(learn_scales=True, distill_code_refresh_interval=1)
    config = _distill(block, distill_steps=12, distill_code_refresh_interval=1,
                      distill_selection_min_improvement=0.9)
    before = _packed(model)
    stats = distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert not stats["accepted"] and stats["steps_run"] == 12
    # Before the fix every layer moved by about 20% and every scale changed.
    _assert_same_packed(_packed(model), before)
    deployed = _distill_eval_loss(model, teacher[6:9], "cpu", torch.float32, config)
    assert deployed == stats["selection_reference_loss"]


def test_restoring_the_best_refreshed_step_restores_the_validated_model():
    model, teacher, block, _ = _toy(learn_scales=True, distill_code_refresh_interval=3)
    config = _distill(block, distill_steps=12, distill_code_refresh_interval=3)
    stats = distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert stats["best_step"] > 0
    # Before the fix the restored candidate was refreshed once more: 0.363721
    # against a best validation of 0.364167.
    assert stats["candidate_validation_loss"] == stats["best_validation_loss"]


def test_refresh_keeps_learned_scales_and_is_idempotent(monkeypatch):
    model, teacher, block, stats = _toy(
        quant=_GAUSSIAN_MSE, learn_scales=True, scale_lr=3e-2, steps=6,
        distill_code_refresh_interval=1)
    assert stats["rotation_train"]["selection_acceptance_rate"] == 1.0
    before = _packed(model)
    for module in _quant_linears(model):
        module.refresh_quantization()
    # A refresh with nothing changed re-selected mse_search scales: 8% weight change.
    _assert_same_packed(_packed(model), before)

    _select_last_step(monkeypatch)
    config = _distill(block, distill_steps=4, distill_code_refresh_interval=1,
                      distill_train_scales=False, distill_train_rotations=True,
                      distill_lr=5e-2)
    result = distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert result["accepted"] and result["best_step"] == 4
    after = _packed(model)
    # Codes follow the trained rotations; the gated block-learned scales stay.
    assert any(not torch.equal(new[0], old[0]) for new, old in zip(after, before))
    for new, old in zip(after, before):
        assert torch.equal(new[1], old[1])


def test_refresh_checkpoint_resume_continues_from_the_checkpointed_packed_state(tmp_path):
    def run(steps, path, resume):
        model, teacher, block, _ = _toy(learn_scales=True, distill_code_refresh_interval=2)
        config = _distill(block, distill_steps=steps, distill_code_refresh_interval=2,
                          distill_checkpoint_path=str(path), distill_checkpoint_interval=1,
                          distill_resume=resume)
        return model, distill_packed_model(model, teacher, "cpu", torch.float32, config)

    uninterrupted, expected = run(5, tmp_path / "full.pt", False)
    run(3, tmp_path / "split.pt", False)  # checkpoint after step 3, between refreshes
    resumed, stats = run(5, tmp_path / "split.pt", True)
    assert stats["resumed"]
    assert stats["best_step"] == expected["best_step"]
    assert stats["best_validation_loss"] == expected["best_validation_loss"]
    _assert_same_packed(_packed(resumed), _packed(uninterrupted))

    state = torch.load(tmp_path / "split.pt", weights_only=True)
    del state["packed_state"]
    torch.save(state, tmp_path / "legacy.pt")
    model, teacher, block, _ = _toy(learn_scales=True, distill_code_refresh_interval=2)
    before = _packed(model)
    with pytest.raises(ValueError, match="lacks the packed state"):
        distill_packed_model(model, teacher, "cpu", torch.float32, _distill(
            block, distill_code_refresh_interval=2, distill_resume=True,
            distill_checkpoint_path=str(tmp_path / "legacy.pt")))
    assert all(module._log_scale_multiplier is None for module in _quant_linears(model))
    _assert_same_packed(_packed(model), before)


# --------------------------------------------------------------------------- #
# 2. FP32 optimiser masters for FP16/BF16 parameters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_reduced_precision_rotations_train_through_fp32_masters(storage, monkeypatch):
    model, teacher, block, _ = _toy(storage=storage)
    thetas = [module.act_rotation.theta for module in _quant_linears(model)]
    before = [theta.detach().clone() for theta in thetas]
    _select_last_step(monkeypatch)
    # 5e-4 per step is under half a BF16 ulp at pi/4 (2e-3) but ten steps are not.
    config = _distill(block, distill_steps=10, distill_train_rotations=True,
                      distill_train_scales=False, distill_lr=5e-4)
    stats = distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert stats["accepted"] and stats["best_step"] == 10
    dtype = getattr(torch, storage)
    for theta, initial in zip(thetas, before):
        assert theta.dtype == dtype and torch.isfinite(theta).all()
        # FP16 angles became non-finite after one step; BF16 angles never moved.
        assert (theta != initial).float().mean() > 0.25


def test_fp16_existing_parameters_train_through_fp32_masters(monkeypatch):
    model, teacher, block, _ = _toy()
    model.half()
    norms = {name: parameter for name, parameter in model.named_parameters()
             if "norm" in name}
    before = {name: parameter.detach().clone() for name, parameter in norms.items()}
    _select_last_step(monkeypatch)
    config = _distill(block, distill_steps=10, distill_train_scales=False,
                      distill_existing_patterns=("norm",), distill_existing_lr=1e-4)
    stats = distill_packed_model(model, teacher[:9], "cpu", torch.float16, config)
    assert stats["accepted"]
    for name, parameter in norms.items():
        assert parameter.dtype == torch.float16 and torch.isfinite(parameter).all()
        assert not torch.equal(parameter, before[name])


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)

    def forward(self, hidden_states):
        return self.fc1(hidden_states)


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_block_proxy_rotation_uses_the_deployed_angle_precision(storage):
    torch.manual_seed(0)
    source = _Projection()
    quant = QuantConfig(bits=3, group_size=128, codebook="gaussian", scale="mse_search")
    train = {"objective": "block", "learn_scales": True}
    patch = PatchConfig(quant=quant, rotation="butterfly", block=128, fallback=True,
                        train_rotation=train, rotation_storage_dtype=storage)
    seeds = {"layers.0.fc1": 3}
    proxy, fake = _fake_quant_block(source, "layers.0", quant, patch, seeds, "cpu",
                                    BlockRotationTrainConfig(**train))
    with torch.no_grad():
        fake["fc1"].rotation.theta.add_(0.03 * torch.randn_like(fake["fc1"].rotation.theta))
    candidate = _packed_cpu_block(source, "layers.0", quant, patch, seeds,
                                  _rotation_states(fake))
    x = torch.randn(64, 256)
    with torch.no_grad():
        # FP32 proxy angles against committed FP16/BF16 angles: 1.5% / 3.5% apart.
        assert torch.equal(fake["fc1"]._assigned_weight(),
                           candidate.fc1.qweight.dequantize())
        assert torch.equal(proxy(x), candidate(x))


# --------------------------------------------------------------------------- #
# 3. residual-SVD LoRA starts with distillation, not inside block training
# --------------------------------------------------------------------------- #

_RESIDUAL_LORA = {"distill_lora_rank": 2, "distill_lora_alpha": 4.0,
                  "distill_lora_init": "residual_svd", "distill_lora_svd_oversample": 2,
                  "distill_lora_svd_niter": 2}


@pytest.mark.parametrize("streamed", [False, True])
def test_residual_svd_adapters_never_reach_block_training(streamed):
    def run(**extra):
        torch.manual_seed(7)
        model = ToyTransformer(n=2).eval()
        batches = [torch.randn(1, 8, 16) for _ in range(3)]
        patch = PatchConfig(quant=_GAUSSIAN_RMS, rotation="butterfly", block=16,
                            fallback=True, train_rotation=_block_config(
                                steps=2, propagate_quantized_inputs=True,
                                stream_blocks=streamed, **extra))
        stats = {}
        if streamed:
            train_and_patch_blocks_streamed(model, patch, batches, stats_out=stats)
        else:
            calls = collect_block_calls(model, batches, "cpu", max_batches=3,
                                        storage_dtype=torch.float32)
            train_and_patch_blocks(model, patch, calls, stats_out=stats)
        return model, stats["rotation_train"]

    plain, plain_stats = run()
    adapted, adapted_stats = run(**_RESIDUAL_LORA)
    layers = _quant_linears(adapted)
    assert all(module.lora_A is None for module in layers)
    assert all(module._recovery_source_weight is not None for module in layers)
    # Later blocks saw the same (adapter-free) prefix activations.
    assert adapted_stats["mean_input_drift_mse"] == plain_stats["mean_input_drift_mse"]
    _assert_same_packed(_packed(adapted), _packed(plain))
    for new, old in zip(layers, _quant_linears(plain)):
        assert type(new.act_rotation) is type(old.act_rotation)
        assert all(torch.equal(a, b) for a, b in zip(
            new.act_rotation.parameters(), old.act_rotation.parameters()))


def test_residual_svd_lora_initialises_from_the_retained_source(monkeypatch):
    model, teacher, block, _ = _toy(**_RESIDUAL_LORA)
    layers = _quant_linears(model)
    assert all(module.lora_A is None for module in layers)
    expected = {
        id(module): module.act_rotation.rotate_weight(
            module._recovery_source_weight.float()) - module.qweight.dequantize().float()
        for module in layers}
    recorded = []
    enable_lora = QuantLinear.enable_lora

    def spy(self, rank, alpha, **kwargs):
        matrix_a, matrix_b = enable_lora(self, rank, alpha, **kwargs)
        recorded.append((self, kwargs["residual"],
                         (matrix_b @ matrix_a).detach() * (alpha / rank)))
        return matrix_a, matrix_b

    monkeypatch.setattr(QuantLinear, "enable_lora", spy)
    config = _distill(block, distill_steps=1, distill_train_scales=False, **_RESIDUAL_LORA)
    distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert len(recorded) == len(layers)
    for module, residual, product in recorded:
        assert torch.equal(residual, expected[id(module)])
        assert (residual - product).pow(2).mean() < residual.pow(2).mean()
    assert all(module._recovery_source_weight is None for module in layers)


def test_residual_svd_lora_without_a_retained_source_fails_before_mutation():
    model, teacher, block, _ = _toy()
    before = _packed(model)
    config = _distill(block, **_RESIDUAL_LORA)
    with pytest.raises(ValueError, match="retained source weight"):
        distill_packed_model(model, teacher, "cpu", torch.float32, config)
    assert all(module._log_scale_multiplier is None and module.lora_A is None
               for module in _quant_linears(model))
    _assert_same_packed(_packed(model), before)


# --------------------------------------------------------------------------- #
# 4. block proxy versus packed block
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("correction", ["length", "length_mean"])
def test_learned_scales_with_length_correction_fail_before_mutation(correction):
    torch.manual_seed(3)
    model = ToyTransformer(n=1).eval()
    calls = collect_block_calls(model, [torch.randn(1, 8, 16) for _ in range(3)], "cpu",
                                max_batches=3, storage_dtype=torch.float32)
    quant = QuantConfig(bits=3, codebook="gaussian", scale="mse_search", group_size=16,
                        bias_correction=correction)
    patch = PatchConfig(quant=quant, rotation="butterfly", block=16, fallback=True,
                        train_rotation=_block_config(learn_scales=True))
    with pytest.raises(ValueError, match="length correction"):
        train_and_patch_blocks(model, patch, calls)
    assert type(model.layers[0].fc1) is nn.Linear and type(model.layers[0].fc2) is nn.Linear


@pytest.mark.parametrize("scale", ["mse_search", "absmax"])
def test_uniform_learned_scales_start_at_the_selected_scales(scale):
    torch.manual_seed(5)
    linear = nn.Linear(64, 32)
    quant = QuantConfig(bits=3, codebook="uniform", scale=scale, group_size=16)
    fake = FakeQuantButterflyLinear(linear, quant, block=16, seed=3, learn_scales=True)
    packed = QuantLinear.from_linear(linear, quant, weight_rotation=fake.rotation,
                                     act_rotation=fake.rotation, fallback=True,
                                     fallback_dtype=torch.float32)
    # The fixed [0.5, 1.5] bounds clamped every uniform group (searched at 1.5-4 x RMS).
    with torch.no_grad():
        assert torch.equal(fake._assigned_weight(), packed.qweight.dequantize())
    low, high = quant.mse_search_bounds
    assert fake.scale_multiplier_min <= low and fake.scale_multiplier_max >= high


def test_gaussian_learned_scale_bounds_are_unchanged():
    linear = nn.Linear(64, 32)
    for scale in ("rms", "mse_search"):
        quant = QuantConfig(bits=3, codebook="gaussian", scale=scale, group_size=16)
        fake = FakeQuantButterflyLinear(linear, quant, block=16, seed=3, learn_scales=True)
        assert (fake.scale_multiplier_min, fake.scale_multiplier_max) == (0.5, 1.5)
    explicit = FakeQuantButterflyLinear(linear, quant, block=16, seed=3, learn_scales=True,
                                        scale_multiplier_min=0.7, scale_multiplier_max=1.2)
    assert (explicit.scale_multiplier_min, explicit.scale_multiplier_max) == (0.7, 1.2)


@pytest.mark.parametrize("scale_bits", [16.0, 8.0])
@pytest.mark.parametrize("correction", ["none", "length"])
def test_unlearned_proxy_assigns_codes_exactly_like_the_packer(scale_bits, correction):
    torch.manual_seed(0)
    linear = nn.Linear(160, 48)  # partial last group
    quant = QuantConfig(bits=3, codebook="gaussian", scale="mse_search", group_size=64,
                        scale_bits=scale_bits, bias_correction=correction,
                        scale_quant_group_size=32)
    fake = FakeQuantButterflyLinear(linear, quant, block=32, seed=3)
    packed = QuantLinear.from_linear(linear, quant, weight_rotation=fake.rotation,
                                     act_rotation=fake.rotation, fallback=True,
                                     fallback_dtype=torch.float32)
    # Codes were assigned against unrounded scales: 3e-3 (16-bit) and 1e-2 (8-bit) apart.
    assert torch.equal(fake._assigned_weight(), packed.qweight.dequantize())
