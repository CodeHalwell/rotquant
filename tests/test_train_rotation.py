"""The learned-rotation (E1) training arm: data-free alternating minimisation on
the Cayley parameters must actually reduce rotated-domain quantisation MSE,
stay exactly orthogonal, and preserve the linear map end to end.
"""
import pytest
import torch
from torch import nn

from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.rotate import ButterflyRotation, LearnedRotation, RandomizedHadamard
from rotquant.train_rotation import (
    RotationTrainConfig,
    select_butterfly_checkpoint,
    train_layer_rotation,
)


def _outlier_weight(out=16, d=32, seed=0) -> torch.Tensor:
    """A weight with a few high-magnitude input channels: the worst case for a
    Gaussian codebook under identity rotation, the case rotation exists to fix."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, d, generator=g)
    w[:, :4] *= 8.0
    return w


def _quant_cfg(d=32) -> QuantConfig:
    # One scale per row (group covers the full input dim) so the outlier columns
    # and the small columns must share a scale -- rotation is the only fix.
    return QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=d)


def test_training_reduces_quant_mse():
    torch.manual_seed(0)
    d = 32
    w = _outlier_weight(d=d)
    rot = LearnedRotation(d, seed=0)
    stats = train_layer_rotation(rot, w, _quant_cfg(d),
                                 RotationTrainConfig(steps=120, lr=1e-2))
    assert stats["final_mse"] < stats["initial_mse"] * 0.9, \
        f"training did not reduce quant MSE: {stats}"
    assert rot.training is False and rot._cached_R is not None  # eval cache warm


def test_trained_rotation_stays_orthogonal_and_preserves_map():
    torch.manual_seed(0)
    d = 32
    w = _outlier_weight(d=d)
    rot = LearnedRotation(d, seed=1)
    train_layer_rotation(rot, w, _quant_cfg(d),
                         RotationTrainConfig(steps=60, lr=1e-2))
    R = rot.matrix()
    eye = torch.eye(d, dtype=R.dtype)
    assert torch.allclose(R @ R.T, eye, atol=1e-4), "Cayley map left the manifold?!"
    # (x R^T)(W R^T)^T == x W^T -- the invariance every rotation must satisfy.
    x = torch.randn(5, d)
    lhs = rot.rotate_activation(x) @ rot.rotate_weight(w).T
    assert torch.allclose(lhs, x @ w.T, atol=1e-4)


def test_patch_model_trains_learned_rotation_and_reports_stats():
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.fc1 = nn.Linear(d, d)
            self.fc2 = nn.Linear(d, d)
            with torch.no_grad():
                self.fc1.weight.copy_(_outlier_weight(d, d, seed=1))
                self.fc2.weight.copy_(_outlier_weight(d, d, seed=2))

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    m = Toy()
    stats: dict = {}
    patch_model(m, PatchConfig(quant=_quant_cfg(32), rotation="learned",
                               train_rotation={"steps": 40, "lr": 1e-2}, seed=0),
                stats_out=stats)
    agg = stats["rotation_train"]
    assert agg["layers"] == 2
    assert agg["mean_final_mse"] < agg["mean_initial_mse"]
    # The patched model must still run, with the trained rotation shared between
    # weight and activation (consistent mode).
    y = m(torch.randn(3, 32))
    assert y.shape == (3, 32) and torch.isfinite(y).all()


def test_untrained_learned_rotation_warns(caplog):
    m = nn.Sequential(nn.Linear(32, 32))
    with caplog.at_level("WARNING", logger="rotquant"):
        patch_model(m, PatchConfig(quant=_quant_cfg(32), rotation="learned", seed=0))
    assert any("no-rotation control" in r.message for r in caplog.records)


def test_learned_rotation_cache_invalidates_on_state_load():
    torch.manual_seed(0)
    rotation = LearnedRotation(4, seed=0).eval()
    x = torch.randn(2, 4)
    before = rotation.rotate_activation(x)  # warm cache
    state = rotation.state_dict()
    state["theta"] = torch.full_like(state["theta"], 0.25)
    rotation.load_state_dict(state)
    after = rotation.rotate_activation(x)
    assert not torch.allclose(before, after)


def test_butterfly_initialises_exactly_from_seeded_fwht():
    torch.manual_seed(0)
    x = torch.randn(7, 256)
    fwht = RandomizedHadamard(256, block=128, seed=17)
    butterfly = ButterflyRotation(256, block=128, seed=17)
    assert torch.allclose(butterfly.rotate_activation(x),
                          fwht.rotate_activation(x), atol=2e-6)
    assert butterfly.theta.numel() == 256 * 7 // 2


def test_activation_aware_butterfly_training_reduces_output_error():
    torch.manual_seed(3)
    d = 32
    w = _outlier_weight(out=24, d=d, seed=4)
    # Anisotropic calibration inputs make this genuinely activation-aware: the
    # first channels matter much more than an isotropic weight-MSE objective says.
    x = torch.randn(96, d)
    x[:, :4] *= 5.0
    rot = ButterflyRotation(d, block=32, seed=2)
    stats = train_layer_rotation(
        rot, w, _quant_cfg(d),
        RotationTrainConfig(steps=100, lr=2e-2, objective="activation",
                            max_tokens=64, assignment_scale="rms"),
        activations=x,
    )
    assert stats["objective"] == "activation"
    assert stats["tokens"] == 64
    assert stats["final_mse"] < stats["initial_mse"] * 0.9, stats
    assert not rot.theta.requires_grad
    restored = rot.inverse_activation(rot.rotate_activation(x[:8]))
    assert torch.allclose(restored, x[:8], atol=2e-5)


def test_short_butterfly_training_never_discards_better_fwht_start():
    torch.manual_seed(9)
    d = 32
    w = torch.randn(20, d)
    x = torch.randn(12, d)
    rot = ButterflyRotation(d, block=32, seed=0)
    before = rot.theta.detach().clone()
    stats = train_layer_rotation(
        rot, w, _quant_cfg(d),
        RotationTrainConfig(steps=1, lr=1.0, objective="activation",
                            max_tokens=12, assignment_scale="rms",
                            restore_best=True),
        activations=x,
    )
    assert stats["final_mse"] <= stats["initial_mse"] + 1e-8
    if stats["best_step"] == 0:
        assert torch.equal(rot.theta, before)


def test_final_quantizer_gate_restores_fwht_when_margin_is_not_met():
    torch.manual_seed(12)
    d = 32
    w = torch.randn(18, d)
    x = torch.randn(16, d)
    reference = ButterflyRotation(d, block=32, seed=5).eval()
    candidate = ButterflyRotation(d, block=32, seed=5).eval()
    with torch.no_grad():
        candidate.theta.add_(0.03 * torch.randn_like(candidate.theta))
    stats = select_butterfly_checkpoint(
        candidate, reference, w, _quant_cfg(d), x,
        max_tokens=16, min_improvement=0.99)
    assert stats["selection_accepted"] is False
    assert torch.equal(candidate.theta, reference.theta)


def test_rotation_train_config_validates_selection_split():
    try:
        RotationTrainConfig(selection_tokens=-1)
    except ValueError as exc:
        assert "selection_tokens" in str(exc)
    else:
        raise AssertionError("negative selection token count was accepted")


def test_butterfly_learned_signs_commit_to_exact_orthogonal_buffer():
    torch.manual_seed(17)
    d = 16
    weight = torch.randn(12, d)
    hessian = torch.randn(d, d)
    hessian = hessian.T @ hessian / d
    rotation = ButterflyRotation(d, block=16, seed=4)
    stats = train_layer_rotation(
        rotation,
        weight,
        _quant_cfg(d),
        RotationTrainConfig(
            steps=3,
            lr=2e-2,
            objective="hessian",
            assignment_scale="rms",
            learn_signs=True,
        ),
        hessian=hessian,
    )
    assert rotation.sign_logits is None
    assert set(rotation.signs.tolist()) <= {-1.0, 1.0}
    assert 0.0 <= stats["sign_flip_rate"] <= 1.0
    x = torch.randn(5, d)
    assert torch.allclose(
        rotation.inverse_activation(rotation.rotate_activation(x)),
        x,
        atol=2e-5,
    )


def test_sign_training_starts_at_configured_magnitude_and_commits_flips():
    from rotquant.rotate import ButterflyRotation

    rotation = ButterflyRotation(64, block=32, seed=3)
    x = torch.randn(4, 64)
    before = rotation.rotate_activation(x)
    original_signs = rotation.signs.clone()
    logits = rotation.enable_sign_training(1.0, init_magnitude=0.05)
    assert torch.allclose(logits.abs(), torch.full_like(logits, 0.05))
    # Hard signs in the forward pass: enabling training changes nothing.
    assert torch.allclose(rotation.rotate_activation(x), before)
    with torch.no_grad():
        logits[0] = -logits[0]
    rotation.commit_signs()
    assert rotation.signs[0] == -original_signs[0]
    assert torch.equal(rotation.signs[1:], original_signs[1:])
    with pytest.raises(ValueError, match="init_magnitude"):
        ButterflyRotation(64, block=32, seed=3).enable_sign_training(1.0, 0.0)


def test_hessian_gate_restores_fwht_for_a_worse_candidate_and_keeps_an_equal_one():
    from rotquant.rotate import ButterflyRotation
    from rotquant.train_rotation import (
        hessian_reconstruction_error,
        select_butterfly_checkpoint_hessian,
    )

    torch.manual_seed(5)
    d, out = 64, 32
    weight = torch.randn(out, d)
    weight[:, :4] *= 20.0  # outlier input channels that only mixing tames
    x = torch.randn(512, d) * torch.linspace(0.5, 2.0, d)
    hessian = x.T @ x / 512
    quant_cfg = QuantConfig(bits=3, group_size=32)

    reference = ButterflyRotation(d, block=32, seed=7)
    candidate = ButterflyRotation(d, block=32, seed=7)
    with torch.no_grad():
        candidate.theta.zero_()  # theta=0 stages do not mix coordinates
    stats = select_butterfly_checkpoint_hessian(
        candidate, reference, weight, quant_cfg, hessian)
    assert stats["selection_objective"] == "hessian"
    assert not stats["selection_accepted"]
    assert stats["selection_candidate_mse"] > stats["selection_reference_mse"]
    assert torch.equal(candidate.theta, reference.theta)

    same = ButterflyRotation(d, block=32, seed=7)
    stats = select_butterfly_checkpoint_hessian(
        same, reference, weight, quant_cfg, hessian)
    assert stats["selection_accepted"]

    plain = hessian_reconstruction_error(reference, weight, quant_cfg, hessian)
    gptq = hessian_reconstruction_error(
        reference, weight, QuantConfig(bits=3, group_size=32, error_comp="gptq"),
        hessian)
    assert 0 < gptq <= plain * 1.001


def test_patch_model_gates_hessian_objective_against_fwht():
    from rotquant.patch import PatchConfig, patch_model

    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.Linear(16, 8))
    x = torch.randn(256, 64) * torch.linspace(0.5, 2.0, 64)
    hessians = {"0": x.T @ x / 256}
    config = PatchConfig(
        quant=QuantConfig(bits=3, group_size=32),
        rotation="butterfly",
        block=32,
        include=["0"],
        train_rotation={"steps": 2, "lr": 1e-3, "objective": "hessian",
                        "assignment_scale": "rms"},
    )
    stats = {}
    patch_model(model, config, hessians=hessians, stats_out=stats)
    train = stats["rotation_train"]
    assert train["objective"] == "hessian"
    assert "selection_acceptance_rate" in train
    assert train["mean_selection_deployed_mse"] <= train["mean_selection_reference_mse"]


def test_hessian_error_with_a_mean_matches_the_deployed_bias_corrected_layer():
    """The gate's score must be the error that survives the deployed bias."""
    from rotquant.quantize import Quantizer
    from rotquant.train_rotation import hessian_reconstruction_error

    torch.manual_seed(0)
    out, d, n = 16, 32, 4096
    # Post-attention inputs carry a substantial per-channel mean; that is the
    # component apply_mean_bias_correction cancels.
    x = torch.randn(n, d) * 0.6 + torch.randn(d) * 1.5
    weight = torch.randn(out, d) * 0.1
    hessian = x.T @ x / n
    mean = x.mean(dim=0)
    quant_cfg = QuantConfig(bits=4, group_size=8, bias_correction="mean")
    rotation = RandomizedHadamard(d, block=d, seed=0)

    # Reproduce exactly what patch_model deploys for this layer.
    rotated = rotation.rotate_weight(weight)
    packed = Quantizer(quant_cfg).quantize_weight(rotated).dequantize()
    correction = (rotation.rotate_activation(mean.reshape(1, -1))
                  @ (rotated - packed).T).reshape(-1)
    deployed = rotation.rotate_activation(x) @ packed.T + correction
    empirical = (deployed - x @ weight.T).pow(2).sum(dim=1).mean()

    centered_signal = ((weight @ hessian * weight).sum()
                       - (mean.reshape(1, -1) @ weight.T).square().sum())
    scored = hessian_reconstruction_error(
        rotation, weight, quant_cfg, hessian, mean) * centered_signal
    assert scored == pytest.approx(empirical.item(), rel=1e-4)

    # Without the mean the gate scores a component the deployment removes.
    uncentered = hessian_reconstruction_error(
        rotation, weight, quant_cfg, hessian) * (
            weight @ hessian * weight).sum()
    assert uncentered > empirical * 2


def test_hessian_gate_decision_follows_the_mean_corrected_error():
    from rotquant.train_rotation import (
        hessian_reconstruction_error,
        select_butterfly_checkpoint_hessian,
    )

    torch.manual_seed(0)
    out, d, n = 16, 32, 4096
    x = torch.randn(n, d) * 0.5 + torch.randn(d) * 2.0
    weight = torch.randn(out, d) * 0.1
    hessian, mean = x.T @ x / n, x.mean(dim=0)
    quant_cfg = QuantConfig(bits=3, group_size=8, bias_correction="mean")

    reference = ButterflyRotation(d, block=d, seed=0)
    candidate = ButterflyRotation(d, block=d, seed=500)
    with torch.no_grad():
        candidate.theta.add_(0.3)
    trained_theta = candidate.theta.detach().clone()

    # This pair ranks one way on H and the other way on H - mean^T mean.
    assert (hessian_reconstruction_error(candidate, weight, quant_cfg, hessian)
            > hessian_reconstruction_error(
                reference, weight, quant_cfg, hessian))
    assert (hessian_reconstruction_error(
        candidate, weight, quant_cfg, hessian, mean)
        < hessian_reconstruction_error(
            reference, weight, quant_cfg, hessian, mean))

    stats = select_butterfly_checkpoint_hessian(
        candidate, reference, weight, quant_cfg, hessian,
        activation_mean=mean)
    assert stats["selection_accepted"]
    assert torch.equal(candidate.theta, trained_theta)

    # The same pair without the mean discards the better deployed rotation.
    reverted = ButterflyRotation(d, block=d, seed=500)
    with torch.no_grad():
        reverted.theta.add_(0.3)
    assert not select_butterfly_checkpoint_hessian(
        reverted, reference, weight, quant_cfg, hessian)["selection_accepted"]


def test_patch_model_passes_the_activation_mean_to_the_hessian_gate():
    import rotquant.train_rotation as train_module

    seen = {}
    original = train_module.select_butterfly_checkpoint_hessian

    def spy(*args, **kwargs):
        seen[args[2].shape] = kwargs.get("activation_mean")
        return original(*args, **kwargs)

    torch.manual_seed(9)
    x = torch.randn(256, 64) * torch.linspace(0.5, 2.0, 64)
    hessians = {"0": x.T @ x / 256}
    means = {"0": x.mean(dim=0)}

    def run(bias_correction):
        seen.clear()
        model = torch.nn.Sequential(nn.Linear(64, 16), nn.Linear(16, 8))
        config = PatchConfig(
            quant=QuantConfig(bits=3, group_size=32,
                              bias_correction=bias_correction),
            rotation="butterfly", block=32, include=["0"],
            train_rotation={"steps": 2, "lr": 1e-3, "objective": "hessian",
                            "assignment_scale": "rms"},
        )
        patch_model(model, config, hessians=hessians, activation_means=means)
        return next(iter(seen.values()))

    monkey = pytest.MonkeyPatch()
    monkey.setattr(train_module, "select_butterfly_checkpoint_hessian", spy)
    try:
        for correction in ("mean", "length_mean"):
            passed = run(correction)
            assert passed is not None, correction
            assert torch.equal(passed, means["0"])
        for correction in ("none", "length"):
            assert run(correction) is None, correction
    finally:
        monkey.undo()
