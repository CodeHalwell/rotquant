"""Dynamic mixed-precision selection and teacher-logit fidelity metrics."""
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from rotquant.block_train import TeacherCall
from rotquant.dynamic import (
    DynamicQuantConfig,
    select_dynamic_quantization,
    teacher_logit_kl,
)
from rotquant.linear import QuantLinear
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


def test_dynamic_allocator_can_score_and_patch_vector_candidates():
    torch.manual_seed(4)
    model = TinyLM().eval()
    patch_cfg = PatchConfig(
        quant=QuantConfig(
            bits=3,
            codebook="vector",
            scale="rms",
            group_size=16,
            vector_dim=2,
            vector_samples=1024,
            vector_iters=5,
        ),
        rotation="fwht",
        block=16,
        include=("layers.",),
        exclude=(),
        dynamic={
            "candidate_bits": [2, 3],
            "target_bpw": 3.0,
            "global_kl_weight": 0.0,
        },
    )
    activations = {
        "layers.0": torch.randn(16, 16),
        "layers.1": torch.randn(16, 16),
    }

    recipe, stats = select_dynamic_quantization(
        model, patch_cfg, activations=activations
    )
    assert stats["target_reached"]
    assert all(config.codebook == "vector" for config in recipe.values())

    patch_cfg.layer_quant = recipe
    patch_model(model, patch_cfg)
    for layer in model.layers:
        assert layer.qweight.packed.bits == layer._quant_config.bits * 2


def test_random_allocator_is_a_seeded_matched_format_control():
    torch.manual_seed(5)
    activations = {
        "layers.0": torch.randn(16, 16),
        "layers.1": torch.randn(16, 16),
    }
    recipes = []
    for _ in range(2):
        model = TinyLM().eval()
        patch_cfg = _patch_cfg(
            candidate_bits=[2, 3, 4], target_bpw=3.0,
            global_kl_weight=0.0, allocation="random",
        )
        recipe, stats = select_dynamic_quantization(
            model, patch_cfg, activations=activations
        )
        assert stats["config"]["allocation"] == "random"
        assert stats["target_reached"]
        recipes.append({name: quant.bits for name, quant in recipe.items()})
    assert recipes[0] == recipes[1]


def test_dynamic_allocator_targets_complete_persistent_bytes():
    torch.manual_seed(8)
    activations = {
        "layers.0": torch.randn(16, 16),
        "layers.1": torch.randn(16, 16),
    }
    probe = TinyLM().eval()
    probe_cfg = _patch_cfg(
        candidate_bits=[2, 4], target_bpw=2.0,
        global_kl_weight=0.0,
    )
    _recipe, probe_stats = select_dynamic_quantization(
        probe, probe_cfg, activations=activations
    )
    candidates = probe_stats["candidate_table"]
    high = {
        row["name"]: row for row in candidates if row["bits"] == 4
    }
    low = {
        row["name"]: row for row in candidates if row["bits"] == 2
    }
    first_name = min(high)
    target = (
        probe_stats["fixed_complete_bytes"]
        + sum(row["complete_bytes"] for row in high.values())
        - (high[first_name]["complete_bytes"] - low[first_name]["complete_bytes"])
    )

    model = TinyLM().eval()
    patch_cfg = _patch_cfg(
        candidate_bits=[2, 4],
        target_bpw=4.0,
        target_complete_bytes=target,
        target_tolerance_fraction=0.0,
        require_target_match=True,
        global_kl_weight=0.0,
    )
    recipe, stats = select_dynamic_quantization(
        model, patch_cfg, activations=activations
    )
    assert stats["estimated_complete_bytes"] == target
    assert stats["within_target_tolerance"] is True
    assert sorted(config.bits for config in recipe.values()) == [2, 4]
    assert len(stats["candidate_table"]) == 4

    patch_cfg.layer_quant = recipe
    patch_model(model, patch_cfg)
    seen: set[int] = set()
    registered_bytes = 0
    for module in model.modules():
        for tensor in [*module._parameters.values(), *module._buffers.values()]:
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                registered_bytes += tensor.numel() * tensor.element_size()
    packed_bytes = 0
    codebook_bytes = 0
    for module in model.modules():
        if not isinstance(module, QuantLinear):
            continue
        packed_bytes += module.packed_state_bytes()
        centroids = module.qweight.codebook.centroids
        codebook_bytes += centroids.numel() * centroids.element_size()
    assert registered_bytes + packed_bytes + codebook_bytes == target


def test_candidate_score_cache_reuses_screen_but_not_allocation(monkeypatch):
    import rotquant.dynamic as dynamic_module

    torch.manual_seed(9)
    model = TinyLM().eval()
    patch_cfg = _patch_cfg(
        candidate_bits=[2, 4], target_bpw=3.0, global_kl_weight=0.0
    )
    calls = {"count": 0}
    original = dynamic_module._score_candidates

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dynamic_module, "_score_candidates", counted)
    dynamic_module._CANDIDATE_SCORE_CACHE.clear()
    _, first = select_dynamic_quantization(
        model, patch_cfg, score_cache_key="same-source-and-scoring"
    )
    random_patch = replace(
        patch_cfg,
        dynamic={**patch_cfg.dynamic, "allocation": "random"},
    )
    _, second = select_dynamic_quantization(
        model, random_patch, score_cache_key="same-source-and-scoring"
    )
    assert calls["count"] == 1
    assert first["candidate_score_cache_hit"] is False
    assert second["candidate_score_cache_hit"] is True
