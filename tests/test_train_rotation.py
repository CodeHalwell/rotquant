"""The learned-rotation (E1) training arm: data-free alternating minimisation on
the Cayley parameters must actually reduce rotated-domain quantisation MSE,
stay exactly orthogonal, and preserve the linear map end to end.
"""
import torch
import torch.nn as nn

from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.rotate import LearnedRotation
from rotquant.train_rotation import RotationTrainConfig, train_layer_rotation


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
