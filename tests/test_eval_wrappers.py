"""Coverage for previously untested wrappers: throughput, zeroshot scoring,
and the GGUF verification script's name mapping.

The zero-shot test stubs lm-eval so it exercises rotquant's own scoring
aggregation (acc_norm preference, bundle mean, non-accuracy tasks skipped)
without downloading tasks; the throughput test runs real greedy generation on
an in-process tiny Llama.
"""
import importlib.util
import os
import sys
from types import ModuleType

import pytest
import torch

transformers = pytest.importorskip("transformers")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StubTokenizer:
    vocab_size = 64
    pad_token_id = 0
    eos_token_id = 0


def _tiny_causal_lm():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).eval()


def test_measure_throughput_accounts_exact_new_tokens():
    from rotquant.eval.throughput import ThroughputConfig, measure_throughput

    result = measure_throughput(
        _tiny_causal_lm(), _StubTokenizer(), "cpu",
        ThroughputConfig(prompt_len=8, new_tokens=4, batch_size=2, warmup=1),
    )
    # min_new_tokens == max_new_tokens, so the count is exact, not a bound.
    assert result["new_tokens"] == 4 * 2
    assert result["prompt_len"] == 8
    assert result["batch_size"] == 2
    assert result["seconds"] > 0
    assert result["tokens_per_s"] > 0
    assert result["peak_vram_bytes"] == 0  # CPU run must not report VRAM


def test_zeroshot_prefers_acc_norm_and_reports_bundle_mean(monkeypatch):
    calls = {}

    lm_eval = ModuleType("lm_eval")

    def simple_evaluate(model=None, tasks=None, limit=None):
        calls["tasks"] = tasks
        calls["limit"] = limit
        return {"results": {
            "arc_easy": {"acc_norm,none": 0.5, "acc,none": 0.25},
            "boolq": {"acc,none": 0.75},
            "wikitext": {"word_perplexity,none": 12.0},
        }}

    lm_eval.simple_evaluate = simple_evaluate
    models_mod = ModuleType("lm_eval.models")
    hf_mod = ModuleType("lm_eval.models.huggingface")

    class HFLM:
        def __init__(self, pretrained=None, tokenizer=None, batch_size=None,
                     device=None):
            calls["device"] = device

    hf_mod.HFLM = HFLM
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_mod)

    from rotquant.eval.zeroshot import DEFAULT_TASKS, zeroshot

    scores = zeroshot(object(), object(), device="cpu")
    assert scores["arc_easy"] == 0.5  # acc_norm wins over plain acc
    assert scores["boolq"] == 0.75
    assert "wikitext" not in scores  # no accuracy metric -> excluded
    assert scores["bundle_mean"] == pytest.approx((0.5 + 0.75) / 2)
    assert calls["tasks"] == DEFAULT_TASKS
    assert calls["device"] == "cpu"


def _load_verify_script():
    pytest.importorskip("safetensors")
    path = os.path.join(_ROOT, "scripts", "verify_rotquant_gguf.py")
    spec = importlib.util.spec_from_file_location("verify_rotquant_gguf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_rotquant_gguf"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_verify_gguf_stem_mapping_is_exact():
    verify = _load_verify_script()
    assert verify._gguf_stem(
        "model.language_model.layers.3.self_attn.q_proj") == "blk.3.attn_q"
    assert verify._gguf_stem(
        "model.language_model.layers.0.mlp.down_proj") == "blk.0.ffn_down"
    assert verify._gguf_stem(
        "model.language_model.layers.12.linear_attn.in_proj_qkv"
    ) == "blk.12.attn_qkv"


def test_verify_gguf_stem_rejects_unknown_names():
    verify = _load_verify_script()
    # A wrong prefix must fail loudly, never map to a plausible-looking stem.
    with pytest.raises(ValueError):
        verify._gguf_stem("model.layers.0.self_attn.q_proj")
    with pytest.raises(ValueError):
        verify._gguf_stem("model.language_model.layers.0.self_attn.qkv_proj")
