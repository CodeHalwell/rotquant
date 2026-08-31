"""Runner plumbing that must hold before any model run is trusted: base-config
merging, run-id derivation (seed sweeps must not overwrite results), device
fallback, the lm_head exclusion default, partial-group scale handling, and the
aggregator seeing nested (zero-shot) metrics.
"""
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from rotquant.patch import PatchConfig, patch_model, _cpu_staging_linear
from rotquant.calibrate import collect_activations
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
_baseline_spec = importlib.util.spec_from_file_location(
    "run_baseline", os.path.join(_ROOT, "baselines", "run_baseline.py"))
baseline_mod = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(baseline_mod)


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


def test_calibration_batches_are_cached_and_content_addressed(monkeypatch):
    calls = {"loads": 0, "tokens": 0}

    def load_dataset(*args, **kwargs):
        del args, kwargs
        calls["loads"] += 1
        return [{"text": "abcdefghijk"}, {"text": "mnopqrstuvw"}]

    datasets = ModuleType("datasets")
    datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    class Tokenizer:
        name_or_path = "tiny/tokenizer"
        vocab_size = 32
        special_tokens_map = {}

        def __call__(self, text, return_tensors="pt"):
            del return_tensors
            calls["tokens"] += 1
            return SimpleNamespace(
                input_ids=torch.tensor([[ord(char) % 32 for char in text]])
            )

    run_experiment._CALIB_CACHE.clear()
    first = run_experiment.build_calib_loader(
        Tokenizer(), 1, 8, "cpu", revision="pinned"
    )
    second = run_experiment.build_calib_loader(
        Tokenizer(), 1, 8, "cpu", revision="pinned"
    )
    assert calls == {"loads": 1, "tokens": 1}
    assert torch.equal(first[0]["input_ids"], second[0]["input_ids"])
    manifest = run_experiment.token_batch_manifest(
        first, dataset="allenai/c4", split="train", revision="pinned",
        skip=0, seq_len=8,
    )
    assert manifest["batches"] == 1
    assert manifest["source_rows"] == [0]
    assert len(manifest["digest"]) == 64


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

    # Device changes affect dtype fallback, numerical results, and throughput.
    cpu_slug = run_experiment.override_slug(None, [], device="cpu")
    cuda_slug = run_experiment.override_slug(None, [], device="cuda:1")
    assert cpu_slug == "device=cpu"
    assert cuda_slug == "device=cuda-1"
    assert cpu_slug != cuda_slug


def test_long_run_id_is_bounded_hashed_and_keeps_seed_suffix():
    long_slug = "override=" + "x" * 400
    rid0 = run_experiment.derive_run_id(
        {"label": "experiment", "seed": 0}, "x.yaml", slug=long_slug)
    rid1 = run_experiment.derive_run_id(
        {"label": "experiment", "seed": 1}, "x.yaml", slug=long_slug)
    assert len(rid0.encode()) <= run_experiment.MAX_RUN_ID_LENGTH
    assert rid0.endswith("_s0") and rid1.endswith("_s1")
    assert rid0[:-3] == rid1[:-3]
    assert aggregate_mod._key({
        "run_id": rid0, "config": {"experiment": "E", "model": "m"}
    }).endswith(rid0[:-3])


def test_baseline_run_id_includes_quantization_options():
    protocol = {
        "revision": "abc123",
        "calib_n": 256,
        "calib_min_chars": 2048,
        "eval": {
            "datasets": ["wikitext2", "c4"],
            "ppl_seq_len": 2048,
            "ppl_stride": 1024,
            "ppl_max_samples": 32,
            "zeroshot": True,
            "tasks": ["boolq", "piqa"],
            "zeroshot_limit": 100,
            "zeroshot_batch_size": 8,
        },
    }
    base = baseline_mod.baseline_run_id(
        "gptq", "org/model", 4, 128, False, "cuda:0", protocol)
    assert base.startswith("baseline_gptq_model_4bit_g128_quantized_cuda-0_")
    assert len(base.rsplit("_", 1)[-1]) == 12
    assert baseline_mod.baseline_run_id(
        "gptq", "org/model", 4, 64, False, "cuda:0", protocol) != base
    assert baseline_mod.baseline_run_id(
        "gptq", "org/model", 4, 128, True, "cuda:0", protocol) != base
    assert baseline_mod.baseline_run_id(
        "gptq", "org/model", 4, 128, False, "cpu", protocol) != base
    assert baseline_mod.baseline_run_id(
        "gptq", "other-org/model", 4, 128, False, "cuda:0", protocol) != base
    assert baseline_mod.baseline_run_id(
        "gptq", "org/model", 4, 128, False, "cuda:0",
        {"eval": protocol["eval"], "calib_min_chars": 2048,
         "calib_n": 256, "revision": "abc123"},
    ) == base
    changed_protocols = [
        {**protocol, "revision": "def456"},
        {**protocol, "calib_n": 128},
        {**protocol, "calib_min_chars": 1024},
        {**protocol, "eval": {**protocol["eval"], "datasets": ["c4"]}},
        {**protocol, "eval": {**protocol["eval"], "ppl_seq_len": 1024}},
        {**protocol, "eval": {**protocol["eval"], "ppl_stride": 512}},
        {**protocol, "eval": {**protocol["eval"], "ppl_max_samples": 16}},
        {**protocol, "eval": {**protocol["eval"], "zeroshot": False}},
        {**protocol, "eval": {**protocol["eval"], "tasks": ["boolq"]}},
        {**protocol, "eval": {**protocol["eval"], "zeroshot_limit": 50}},
        {**protocol, "eval": {
            **protocol["eval"], "zeroshot_batch_size": 4}},
    ]
    assert all(
        baseline_mod.baseline_run_id(
            "gptq", "org/model", 4, 128, False, "cuda:0", changed
        ) != base
        for changed in changed_protocols
    )
    assert baseline_mod.IMPLEMENTED_BACKENDS == ("gptq", "awq", "aqlm")


def test_awq_receives_the_recorded_calibration_protocol(monkeypatch):
    calls = {}
    tokenizer = object()

    class FakeAWQModel:
        def __init__(self):
            self.model = nn.Identity()

        def quantize(self, received_tokenizer, **kwargs):
            calls["tokenizer"] = received_tokenizer
            calls.update(kwargs)

    fake_model = FakeAWQModel()

    class FakeAutoAWQ:
        @staticmethod
        def from_pretrained(model_name):
            calls["model_name"] = model_name
            return fake_model

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer_model"] = model_name
            calls["tokenizer_kwargs"] = kwargs
            return tokenizer

    awq_module = ModuleType("awq")
    awq_module.AutoAWQForCausalLM = FakeAutoAWQ
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "awq", awq_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    def fake_calib_texts(n, min_chars):
        calls["calibration_request"] = (n, min_chars)
        return ["sample-a", "sample-b"]

    monkeypatch.setattr(baseline_mod, "_calib_texts", fake_calib_texts)
    model, received_tokenizer = baseline_mod.load_baseline(
        "awq",
        "org/model",
        4,
        "cpu",
        group_size=64,
        calib_n=17,
        calib_min_chars=333,
    )

    assert model is fake_model.model
    assert received_tokenizer is tokenizer
    assert calls["calibration_request"] == (17, 333)
    assert calls["calib_data"] == ["sample-a", "sample-b"]
    assert calls["max_calib_samples"] == 17
    assert calls["quant_config"] == {
        "w_bit": 4,
        "q_group_size": 64,
        "zero_point": True,
        "version": "GEMM",
    }


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


def test_mps_quality_runs_enable_cached_fallback():
    cfg = {"patch": {"rotation": "fwht", "fallback": False}}
    changed = run_experiment.apply_device_defaults(cfg, torch.device("mps"))
    assert changed is True
    assert cfg["patch"]["fallback"] is True
    assert run_experiment.apply_device_defaults(cfg, torch.device("mps")) is False

    cpu_cfg = {"patch": {"rotation": "fwht", "fallback": False}}
    assert run_experiment.apply_device_defaults(cpu_cfg, torch.device("cpu")) is False
    assert cpu_cfg["patch"]["fallback"] is False

    baseline_cfg = {"patch": {"enabled": False}}
    assert run_experiment.apply_device_defaults(
        baseline_cfg, torch.device("mps")) is False
    assert "fallback" not in baseline_cfg["patch"]


def test_model_loader_auto_detects_unified_multimodal_configs():
    text = SimpleNamespace(vision_config=None)
    vision_text = SimpleNamespace(vision_config=SimpleNamespace())
    assert run_experiment.resolve_model_loader(text) == "causal_lm"
    assert run_experiment.resolve_model_loader(vision_text) == "multimodal_lm"
    assert run_experiment.resolve_model_loader(
        vision_text, "causal_lm") == "causal_lm"
    with pytest.raises(ValueError, match="unknown model_loader"):
        run_experiment.resolve_model_loader(vision_text, "fast_vision")


def test_cpu_staging_linear_preserves_half_source_values_in_fp32():
    source = nn.Linear(32, 16).half()
    staged = _cpu_staging_linear(source)
    assert staged.weight.device.type == "cpu"
    assert staged.weight.dtype == torch.float32
    assert torch.equal(staged.weight, source.weight.float())
    assert torch.equal(staged.bias, source.bias.float())


def test_activation_collection_is_bounded_and_removes_hooks():
    model = _ToyLM(d=16, vocab=8)
    batches = [torch.randn(2, 5, 16), torch.randn(2, 5, 16)]
    result = collect_activations(model, batches, "cpu", max_tokens=7,
                                 storage_dtype=torch.float32)
    assert set(result.activations) == {"q_proj", "mlp", "lm_head"}
    assert all(x.shape[0] == 7 for x in result.activations.values())
    assert all(x.dtype == torch.float32 for x in result.activations.values())
    assert all(not module._forward_hooks for module in model.modules())


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


def test_disabled_patch_is_a_true_source_model_baseline():
    model = _ToyLM()
    original_layers = {name: layer for name, layer in model.named_modules()
                       if isinstance(layer, nn.Linear)}
    returned = patch_model(
        model, PatchConfig(quant=_qcfg(), enabled=False, rotation="fwht"))
    assert returned is model
    assert all(dict(model.named_modules())[name] is layer
               for name, layer in original_layers.items())
    assert not any(isinstance(layer, QuantLinear) for layer in model.modules())


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


def test_partial_group_bit_budget_matches_stored_bytes():
    torch.manual_seed(0)
    w = torch.randn(3, 130)  # two scale groups, with a 2-weight final group
    qw = Quantizer(QuantConfig(bits=3, scale="rms", group_size=128)).quantize_weight(w)
    qlin = QuantLinear(qw, act_rotation=Identity(130))
    assert qw.bit_budget().bits_per_weight == \
        qlin.packed_state_bytes() * 8 / w.numel()


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


def test_unimplemented_scale_precision_is_rejected():
    with pytest.raises(ValueError, match="scale_bits must be 16 or 32"):
        QuantConfig(scale_bits=8.0)


@pytest.mark.parametrize("value", ["gptqq", "Residual", ""])
def test_unknown_error_comp_is_rejected(value):
    with pytest.raises(ValueError, match="unknown error compensation"):
        QuantConfig(error_comp=value)


def test_zero_group_scales_survive_fp16_floor():
    # An all-zero input group must not underflow the fp16 scale to 0 -> NaN.
    w = torch.zeros(4, 64)
    w[:, :32] = torch.randn(4, 32)
    qw = Quantizer(QuantConfig(bits=3, scale="rms", group_size=32)).quantize_weight(w)
    deq = qw.dequantize()
    assert torch.isfinite(deq).all()
    assert torch.allclose(deq[:, 32:], torch.zeros(4, 32), atol=1e-3)


def test_explicit_scale_override_is_validated_and_stored_exactly():
    cfg = QuantConfig(bits=3, group_size=4, scale="rms")
    weight = torch.randn(3, 7)
    scales = torch.full((3, 2), 0.375, dtype=torch.float32)
    qw = Quantizer(cfg).quantize_weight(weight, scales_override=scales)
    assert torch.equal(qw.scales, scales.to(torch.float16))

    with pytest.raises(ValueError, match="shape"):
        Quantizer(cfg).quantize_weight(
            weight, scales_override=torch.ones(3, 1))
    with pytest.raises(ValueError, match="finite and positive"):
        Quantizer(cfg).quantize_weight(
            weight, scales_override=torch.zeros(3, 2))


def test_quantlinear_scale_finetuning_commits_without_storage_growth():
    torch.manual_seed(9)
    linear = nn.Linear(8, 4)
    qlinear = QuantLinear.from_linear(
        linear, QuantConfig(bits=3, group_size=4), fallback=True,
        fallback_dtype=torch.float32)
    packed_before = qlinear.qweight.packed.data.clone()
    bytes_before = qlinear.packed_state_bytes()
    scales_before = qlinear.qweight.scales.clone()

    parameter = qlinear.enable_scale_finetuning(0.5, 1.5)
    output = qlinear(torch.randn(2, 8)).pow(2).mean()
    output.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        parameter.add_(0.1)
    qlinear.commit_scale_finetuning()

    assert qlinear._log_scale_multiplier is None
    assert torch.equal(qlinear.qweight.packed.data, packed_before)
    assert not torch.equal(qlinear.qweight.scales, scales_before)
    assert qlinear.qweight.scales.dtype == torch.float16
    assert qlinear.packed_state_bytes() == bytes_before
    assert torch.isfinite(qlinear(torch.randn(2, 8))).all()


def test_quantlinear_lora_is_zero_initialized_and_fully_accounted():
    torch.manual_seed(10)
    linear = nn.Linear(8, 4)
    qlinear = QuantLinear.from_linear(
        linear, QuantConfig(bits=3, group_size=4), fallback=True,
        fallback_dtype=torch.float32)
    inputs = torch.randn(3, 8)
    baseline = qlinear(inputs).detach()
    packed_bytes = qlinear.packed_state_bytes()
    lora_a, lora_b = qlinear.enable_lora(rank=2, alpha=4.0)
    assert torch.equal(qlinear(inputs), baseline)

    qlinear(inputs).pow(2).mean().backward()
    assert lora_b.grad is not None and torch.isfinite(lora_b.grad).all()
    with torch.no_grad():
        lora_b.add_(0.01)
    assert not torch.equal(qlinear(inputs), baseline)
    qlinear.commit_lora()

    expected_adapter_bytes = 2 * (2 * 8 + 4 * 2)
    assert lora_a.dtype == torch.float16 and lora_b.dtype == torch.float16
    assert qlinear.adapter_state_bytes() == expected_adapter_bytes
    assert qlinear.packed_state_bytes() == packed_bytes

    wrapper = nn.Sequential(qlinear)
    metrics = run_experiment.footprint_metrics(wrapper, {})
    assert metrics["adapter_parameter_bytes"] == expected_adapter_bytes
    assert (metrics["packed_plus_auxiliary_bytes"]
            == metrics["packed_weight_bytes"]
            + metrics["rotation_parameter_bytes"]
            + expected_adapter_bytes)
    assert metrics["complete_persistent_model_bytes"] > packed_bytes
    assert metrics["quality_runtime_model_bytes"] == (
        metrics["complete_persistent_model_bytes"]
        + metrics["fallback_cache_bytes"]
    )


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


def test_fallback_cache_starts_in_source_weight_dtype():
    linear = nn.Linear(64, 32, bias=False).half()
    qlin = QuantLinear.from_linear(linear, _qcfg(), fallback=True)
    assert qlin._fp_cache.dtype == torch.float16
    assert qlin._fp_cache.element_size() == 2


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
