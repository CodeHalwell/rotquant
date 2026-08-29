"""Dynamic mixed-precision selection and teacher-logit fidelity metrics."""
from types import SimpleNamespace

import torch
from torch import nn

from rotquant.block_train import TeacherCall
from rotquant.dynamic import (
    DynamicQuantConfig,
    select_dynamic_quantization,
    teacher_logit_kl,
)
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([
            nn.Linear(16, 16, bias=False),
            nn.Linear(16, 16, bias=False),
        ])
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, input_ids, use_cache=False, attention_mask=None):
        del use_cache, attention_mask
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


def _patch_cfg(**dynamic):
    return PatchConfig(
        quant=QuantConfig(bits=4, scale="rms", group_size=16),
        rotation="fwht", block=16,
        include=("layers.",), exclude=(),
        dynamic=dynamic,
    )


def test_dynamic_config_validation_and_rule_selection():
    config = DynamicQuantConfig(
        candidate_bits=(4, 3, 4), target_bpw=4.5,
        rules=({"match": "layers.0", "bits": 4},),
    )
    assert config.candidate_bits == (3, 4)

    torch.manual_seed(0)
    model = TinyLM().eval()
    patch_cfg = _patch_cfg(
        candidate_bits=[3, 4], target_bpw=4.5,
        global_kl_weight=0.0,
        rules=[{"match": "layers.0", "bits": 4}],
    )
    activations = {
        "layers.0": torch.randn(16, 16),
        "layers.1": torch.randn(16, 16),
    }
    recipe, stats = select_dynamic_quantization(
        model, patch_cfg, activations=activations)
    assert recipe["layers.0"].bits == 4
    assert recipe["layers.1"].bits == 3
    assert stats["counts_by_bits"] == {"4": 1, "3": 1}
    assert stats["target_reached"]

    patch_cfg.layer_quant = recipe
    patch_model(model, patch_cfg)
    assert model.layers[0].qweight.packed.bits == 4
    assert model.layers[1].qweight.packed.bits == 3


def test_teacher_logit_kl_is_zero_for_source_and_positive_after_perturbation():
    torch.manual_seed(1)
    model = TinyLM().eval()
    inputs = {"input_ids": torch.randint(0, 32, (1, 8))}
    with torch.no_grad():
        logits = model(**inputs).logits.detach().clone()
    calls = [TeacherCall(inputs=inputs, logits=logits)]
    assert teacher_logit_kl(
        model, calls, "cpu", torch.float32) < 1e-7

    with torch.no_grad():
        model.layers[0].weight.add_(0.5)
    assert teacher_logit_kl(
        model, calls, "cpu", torch.float32) > 1e-6
