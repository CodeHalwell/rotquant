"""Runner plumbing that must hold before any model run is trusted: base-config
merging, run-id derivation (seed sweeps must not overwrite results), device
fallback, the lm_head exclusion default, partial-group scale handling, and the
aggregator seeing nested (zero-shot) metrics.
"""
import importlib.util
import os

import pytest
import torch
import torch.nn as nn
import yaml

from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig, Quantizer, _group_scales_rms
from rotquant.linear import QuantLinear
from rotquant.rotate import Identity

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    path = os.path.join(_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_experiment = _load_script("run_experiment")
aggregate_mod = _load_script("aggregate")


# --------------------------------------------------------------------------- #
# config loading
# --------------------------------------------------------------------------- #
def test_base_config_deep_merge(tmp_path):
    (tmp_path / "_base.yaml").write_text(yaml.safe_dump({
        "model": "base/model", "device": "cuda", "seed": 0,
        "eval": {"perplexity": True, "ppl": {"seq_len": 2048},
                 "ppl_datasets": ["wikitext2", "c4"]},
    }))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({
        "seed": 7,
        "eval": {"ppl": {"max_samples": 4}, "ppl_datasets": ["wikitext2"]},
    }))
    cfg = run_experiment.load_config(str(tmp_path / "exp.yaml"))
    assert cfg["model"] == "base/model"          # inherited
    assert cfg["seed"] == 7                      # overridden
    assert cfg["eval"]["perplexity"] is True     # nested inherit
    assert cfg["eval"]["ppl"] == {"seq_len": 2048, "max_samples": 4}  # nested merge
    assert cfg["eval"]["ppl_datasets"] == ["wikitext2"]  # lists replaced


def test_shipped_configs_all_resolve_a_model():
    cfg_dir = os.path.join(_ROOT, "configs")
    for fname in sorted(os.listdir(cfg_dir)):
        if not fname.endswith(".yaml") or fname == run_experiment.BASE_CONFIG_NAME:
            continue
        cfg = run_experiment.load_config(os.path.join(cfg_dir, fname))
        assert cfg.get("model"), f"{fname} resolves no model even after _base merge"
        # quant/patch blocks must construct without unknown-key TypeErrors
        quant_kwargs = dict(cfg.get("quant") or {})
        patch_kwargs = dict(cfg.get("patch") or {})
        qcfg = QuantConfig(seed=quant_kwargs.pop("seed", 0), **quant_kwargs)
        PatchConfig(quant=qcfg, seed=patch_kwargs.pop("seed", 0), **patch_kwargs)


def test_run_id_seed_suffix_and_explicit():
    assert run_experiment.derive_run_id({"label": "e1_fwht", "seed": 2},
                                        "configs/e1.yaml") == "e1_fwht_s2"
    assert run_experiment.derive_run_id({}, "configs/e2_codebook.yaml") == \
        "e2_codebook_s0"
    # explicit run_id is used verbatim (no suffix)
    assert run_experiment.derive_run_id({"run_id": "fixed", "seed": 3},
                                        "x.yaml") == "fixed"


def test_run_id_reflects_cli_overrides():
    # --seed on a config with an explicit run_id must not overwrite the default run
    assert run_experiment.derive_run_id(
        {"run_id": "e3b", "seed": 1}, "x.yaml", seed_overridden=True) == "e3b_s1"
    # --model / --set slugs keep sweep results apart
    slug = run_experiment.override_slug("meta-llama/Llama-2-7b-hf",
                                        [("patch.rotation", "dense")])
    assert slug == "Llama-2-7b-hf_patch.rotation=dense"
    rid = run_experiment.derive_run_id({"label": "e1", "seed": 0}, "x.yaml",
                                       slug=slug)
    assert rid == "e1_Llama-2-7b-hf_patch.rotation=dense_s0"


def test_apply_set_overrides_types_and_nesting():
    cfg = {"quant": {"bits": 3}, "eval": {"perplexity": True}}
    run_experiment.apply_set_overrides(cfg, [
        ("quant.bits", 4),            # already parsed by the CLI via yaml
        ("patch.rotation", "dense"),  # creates the missing patch block
        ("eval.zeroshot", True),
    ])
    assert cfg["quant"]["bits"] == 4
    assert cfg["patch"] == {"rotation": "dense"}
    assert cfg["eval"] == {"perplexity": True, "zeroshot": True}


def test_device_fallback_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU-only fallback behaviour")
    device, dtype = run_experiment.resolve_device_dtype(
        {"device": "cuda", "dtype": "float16"})
    assert device == "cpu"
    assert dtype == torch.float32


# --------------------------------------------------------------------------- #
# patch targeting
# --------------------------------------------------------------------------- #
class _ToyLM(nn.Module):
    def __init__(self, d=64, vocab=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.lm_head(self.mlp(self.q_proj(x)))


def _qcfg():
    return QuantConfig(bits=3, codebook="gaussian", scale="rms", group_size=32)


def test_lm_head_excluded_by_default():
    m = _ToyLM()
    patch_model(m, PatchConfig(quant=_qcfg(), rotation="fwht", block=64, seed=0))
    assert isinstance(m.q_proj, QuantLinear)
    assert isinstance(m.mlp, QuantLinear)
    assert isinstance(m.lm_head, nn.Linear), "lm_head must stay fp by default"
    # opt-in: empty exclude quantises the head too
    m2 = _ToyLM()
    patch_model(m2, PatchConfig(quant=_qcfg(), rotation="fwht", block=64,
                                seed=0, exclude=()))
    assert isinstance(m2.lm_head, QuantLinear)


def test_zero_targets_warns(caplog):
    class NoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 8)

    with caplog.at_level("WARNING", logger="rotquant"):
        patch_model(NoLinear(), PatchConfig(quant=_qcfg()))
    assert any("NO nn.Linear" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# partial-group scales
# --------------------------------------------------------------------------- #
def test_partial_group_rms_ignores_padding():
    torch.manual_seed(0)
    w = torch.randn(4, 100)
    scales = _group_scales_rms(w, 64)  # groups: 64 + 36
    ref_last = w[:, 64:].pow(2).mean(dim=1).sqrt()
    assert torch.allclose(scales[:, 1], ref_last, atol=1e-6), \
        "last-group RMS must average over the 36 real weights, not 64 slots"


def test_partial_group_quantize_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(8, 100)
    for scale in ("rms", "mse_search"):
        qw = Quantizer(QuantConfig(bits=4, scale=scale, group_size=64)).quantize_weight(w)
        deq = qw.dequantize()
        assert deq.shape == w.shape
        rel = (w - deq).pow(2).sum() / w.pow(2).sum()
        assert rel < 0.02, f"scale={scale}: partial-group path degraded ({rel:.4f})"


# --------------------------------------------------------------------------- #
# storage accounting matches what is actually stored
# --------------------------------------------------------------------------- #
def test_scales_stored_at_claimed_precision():
    """The default budget charges 16 bits per scale, so the stored tensor must
    actually be fp16 (and quantisation must agree with the rounded values);
    a non-16-bit budget keeps fp32 and is charged its true element size."""
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    qw16 = Quantizer(QuantConfig(bits=3, scale="rms", group_size=32,
                                 error_comp="residual")).quantize_weight(w)
    assert qw16.scales.dtype == torch.float16
    assert qw16.residual_scales.dtype == torch.float16
    rel = (w - qw16.dequantize()).pow(2).sum() / w.pow(2).sum()
    assert rel < 0.02

    qw16p = Quantizer(QuantConfig(bits=3, scale="rms",
                                  group_size=32)).quantize_weight(w)
    qw32 = Quantizer(QuantConfig(bits=3, scale="rms", group_size=32,
                                 scale_bits=32.0)).quantize_weight(w)
    assert qw32.scales.dtype == torch.float32
    lin16 = QuantLinear(qw16p, act_rotation=Identity(64))
    lin32 = QuantLinear(qw32, act_rotation=Identity(64))
    # fp16 scales charged 2 bytes each; fp32 scales charged their true 4.
    assert (lin32.packed_state_bytes() - lin16.packed_state_bytes()
            == 2 * qw32.scales.numel())


def test_zero_group_scales_survive_fp16_floor():
    # An all-zero input group must not underflow the fp16 scale to 0 -> NaN.
    w = torch.zeros(4, 64)
    w[:, :32] = torch.randn(4, 32)
    qw = Quantizer(QuantConfig(bits=3, scale="rms", group_size=32)).quantize_weight(w)
    deq = qw.dequantize()
    assert torch.isfinite(deq).all()
    assert torch.allclose(deq[:, 32:], torch.zeros(4, 32), atol=1e-3)


def test_footprint_bpw_is_size_weighted():
    """bits_per_weight_mean must be total bits / total weights, not a per-layer
    average -- TurboQuant per-row overhead depends on in_features, so small
    layers would otherwise skew the model-wide rate."""
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 32)   # bpw = 3 + 16/64  = 3.25
            self.fc2 = nn.Linear(128, 32)  # bpw = 3 + 16/128 = 3.125

    m = Toy()
    qcfg = QuantConfig(bits=3, codebook="gaussian", scale="turboquant",
                       group_size=64)
    patch_model(m, PatchConfig(quant=qcfg, rotation="fwht", block=64, seed=0))
    metrics = run_experiment.footprint_metrics(m, {})
    weighted = (3.25 * 64 * 32 + 3.125 * 128 * 32) / (64 * 32 + 128 * 32)
    assert abs(metrics["bits_per_weight_mean"] - weighted) < 1e-9
    assert abs(metrics["bits_per_weight_mean"] - (3.25 + 3.125) / 2) > 1e-3
    assert metrics["bits_per_weight_min"] == 3.125
    assert metrics["bits_per_weight_max"] == 3.25


# --------------------------------------------------------------------------- #
# module moves / caching
# --------------------------------------------------------------------------- #
def test_quantlinear_survives_dtype_moves():
    """model.half()/float() after patching must keep forward working: _apply has
    to carry the packed dataclass tensors (int32 codes stay int32) and caches."""
    torch.manual_seed(0)
    m = _ToyLM()
    patch_model(m, PatchConfig(quant=QuantConfig(bits=3, scale="rms", group_size=32,
                                                 error_comp="residual"),
                               rotation="fwht", block=64, seed=0))
    x = torch.randn(2, 64)
    y32 = m(x)
    m.half()
    assert m.q_proj.qweight.packed.data.dtype == torch.int32
    assert m.q_proj.qweight.scales.dtype == torch.float16
    y16 = m(x.half())
    assert y16.shape == y32.shape and torch.isfinite(y16).all()
    m.float()
    y32b = m(x)
    assert torch.allclose(y32, y32b, atol=2e-2), "float->half->float round trip drifted"


def test_fallback_cache_follows_dtype_moves():
    m = _ToyLM()
    patch_model(m, PatchConfig(quant=_qcfg(), rotation="fwht", block=64,
                               seed=0, fallback=True))
    m.half()
    assert m.q_proj._fp_cache.dtype == torch.float16
    y = m(torch.randn(2, 64).half())
    assert torch.isfinite(y).all()


def test_codebook_cache_shared():
    from rotquant.codebooks import build_scalar_codebook
    a = build_scalar_codebook("gaussian", 8)
    b = build_scalar_codebook("gaussian", 8)
    assert a is b, "codebooks must be cached (Lloyd-Max is expensive per layer)"
    assert build_scalar_codebook("gaussian", 16) is not a


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_flattens_zeroshot():
    runs = [
        {"run_id": f"e1_s{s}",
         "config": {"experiment": "E1", "model": "m", "label": "e1"},
         "metrics": {"ppl_wikitext2": 6.0 + s,
                     "zeroshot": {"boolq": 0.6 + 0.01 * s, "bundle_mean": 0.5}}}
        for s in range(2)
    ]
    table = aggregate_mod.aggregate(runs)
    (key, agg), = table.items()
    assert agg["n_seeds"] == 2
    assert abs(agg["ppl_wikitext2"]["mean"] - 6.5) < 1e-9
    assert abs(agg["zeroshot.boolq"]["mean"] - 0.605) < 1e-9
    assert "zeroshot.bundle_mean" in agg


def test_aggregate_keeps_sweep_variants_apart():
    """Seeds of one cell merge; --set variants (baked into run_id) must not be
    averaged together even though they share the config label."""
    def mk(rid, ppl):
        return {"run_id": rid,
                "config": {"experiment": "E1", "model": "m", "label": "e1"},
                "metrics": {"ppl_wikitext2": ppl}}
    runs = [mk("e1_s0", 6.0), mk("e1_s1", 7.0),
            mk("e1_quant.bits=4_s0", 5.0), mk("e1_quant.bits=4_s1", 5.5)]
    table = aggregate_mod.aggregate(runs)
    assert len(table) == 2
    base = table["E1|m|e1"]
    swept = table["E1|m|e1_quant.bits=4"]
    assert base["n_seeds"] == 2 and abs(base["ppl_wikitext2"]["mean"] - 6.5) < 1e-9
    assert swept["n_seeds"] == 2 and abs(swept["ppl_wikitext2"]["mean"] - 5.25) < 1e-9
