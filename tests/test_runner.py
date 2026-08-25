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
