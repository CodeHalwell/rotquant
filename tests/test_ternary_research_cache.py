"""MLP-only arms share a frozen source-calibrated W4 remainder."""
import copy

import yaml

from scripts import ternary_research_model as work
from scripts.ternary_research_plan import DEFAULT_CONFIG, Arm


def test_mlp_remainder_uses_same_source_and_cache_for_ptq_and_qat(tmp_path, monkeypatch):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    calls = []
    monkeypatch.setattr(work, "quantize_backbone", lambda model, tokenizer, cfg, data, root:
                        calls.append((cfg, root)))
    for method in ("ternary_ptq", "ternary_qat"):
        original = copy.deepcopy(config)
        work.prepare_mlp_remainder(None, None, Arm(method, 2, method).config(config),
                                   {}, tmp_path, "run-identity")
        assert config == original
    assert calls[0] == calls[1]
    cfg, cache = calls[0]
    assert cfg["quant"]["bits"] == 4 and cfg["quant"]["scale_refit"] == "hessian"
    assert ".mlp." in cfg["patch"]["exclude"]
    assert "linear_attn.in_proj_a" in cfg["patch"]["exclude"]
    assert cache == tmp_path / "mlp_remainder"
    assert cfg["stage_source_fingerprint"]
