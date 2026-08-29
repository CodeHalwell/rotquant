"""The learned-rotation (E1) training arm: data-free alternating minimisation on
the Cayley parameters must actually reduce rotated-domain quantisation MSE,
stay exactly orthogonal, and preserve the linear map end to end.
"""
import torch
import torch.nn as nn

from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.rotate import ButterflyRotation, LearnedRotation, RandomizedHadamard
from rotquant.train_rotation import RotationTrainConfig, train_layer_rotation
from rotquant.train_rotation import select_butterfly_checkpoint


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
