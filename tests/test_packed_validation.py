"""Artifact gates, finite-precision head semantics, and bounded execution checks."""
import copy
import json
import sys
from types import SimpleNamespace

import nbformat
import pytest
import torch

from rotquant.checkpoint import load_packed_model, save_packed_checkpoint
from rotquant.format import FormatValidationError, validate_checkpoint_manifest
from rotquant.linear import QuantLinear
from rotquant.patch import PatchConfig, patch_model
from rotquant.quantize import QuantConfig
from rotquant.validation import (
    audit_artifact,
    capture_probes,
    compare_probes,
    packed_residency,
    probe_inputs,
    save_probes,
)
from rotquant.vocabulary import VocabularyConfig, quantize_vocabulary
from scripts import run_qwen35_packed_validation as runner
from scripts.build_qwen35_packed_validation_notebook import (
    OUTPUT,
    REVALIDATION_OUTPUT,
    build_notebook,
)


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(34)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=33, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
        eos_token_id=32, pad_token_id=0)).eval()


@pytest.mark.parametrize("bits", [6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_equivalent_projection_matches_stored_reconstruction(bits, dtype):
    model = tiny_llama().to(dtype=dtype)
    owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                VocabularyConfig(bits=bits, group_size=32, block=32, chunk_rows=8))
    owner.projection_mode = "dense_equivalent"
    hidden = torch.randn(2, 4, 64).to(dtype)
    dense = torch.cat([chunk for _, chunk in owner.reconstructed_chunks()])
    # The weights are identical; GEMM reduction order can differ by a few FP32
    # ulps when tiling the output dimension, especially the last partial tile.
    torch.testing.assert_close(owner.project(hidden), torch.nn.functional.linear(hidden, dense),
                               rtol=0, atol=2e-7 if dtype == torch.float32 else 0)
    assert not any(tuple(t.shape) == (33, 64) for t in owner.buffers())


def test_probe_checks_reject_bad_inputs_nonfinite_and_changed_generation(tmp_path):
    model = tiny_llama()
    prompts = [{"input_ids": torch.tensor([[1, 2, 3, 4]])}]
    original = capture_probes(model, prompts, "cpu", prompt_tokens=4, new_tokens=2)
    save_probes(tmp_path / "probes.safetensors", original)
    from safetensors.torch import load_file
    reloaded = load_file(str(tmp_path / "probes.safetensors"))
    assert compare_probes(reloaded, original, reload=True)["passed"]
    assert torch.equal(probe_inputs(reloaded)[0]["input_ids"], prompts[0]["input_ids"])
    wrong = copy.deepcopy(original)
    wrong["p0.input.input_ids"][0, 0] += 1
    with pytest.raises(ValueError, match="inputs differ"):
        compare_probes(wrong, original)
    wrong = copy.deepcopy(original)
    wrong["p0.logits"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare_probes(wrong, original)
    wrong = copy.deepcopy(original)
    wrong["p0.logits"] += 1
    assert not compare_probes(wrong, original)["passed"]
    wrong = copy.deepcopy(original)
    wrong["p0.generated"][0, -1] += 1
    assert not compare_probes(wrong, original, reload=True)["passed"]
    with pytest.raises(ValueError, match="empty"):
        compare_probes({}, {})


def test_checkpoint_preserves_mode_metadata_and_rejects_extra_or_corrupt_files(tmp_path):
    model = tiny_llama()
    owner = quantize_vocabulary(model.get_input_embeddings().weight,
                                VocabularyConfig(bits=6, group_size=32, block=32, chunk_rows=8))
    owner.projection_mode = "dense_equivalent"
    patch_model(model, PatchConfig(quant=QuantConfig(bits=5, group_size=32, scale_bits=8),
                                   block=32, fallback=True))
    original_parameter = model.get_input_embeddings().weight
    with runner.packed_context(model, owner, "cpu", torch.float32):
        assert packed_residency(model)["backbone_fallback_cache_bytes"] == 0
        report = save_packed_checkpoint(model, tmp_path / "packed", model_loader="causal_lm")
    assert model.get_input_embeddings().weight is original_parameter
    assert model.get_output_embeddings().weight is original_parameter
    checkpoint = tmp_path / "packed"
    ledger = audit_artifact(checkpoint, expected_manifest_sha256=report["manifest_sha256"])
    assert ledger["measured_artifact_bytes"] == report["artifact_bytes"]
    assert ledger["vocabulary_codes_scales_codebook_bytes"] + ledger["vocabulary_rotation_bytes"] == owner.packed_bytes()
    restored = load_packed_model(checkpoint, dtype="bfloat16", fallback=False)
    assert restored.get_input_embeddings().owner.projection_mode == "dense_equivalent"
    assert restored.get_input_embeddings().owner.execution_dtype == torch.bfloat16
    assert packed_residency(restored)["passed"]
    cached = next(m for m in restored.modules() if isinstance(m, QuantLinear))
    cached._fp_cache = torch.ones(1)
    with pytest.raises(ValueError, match="fallback"):
        packed_residency(restored)
    manifest = json.loads((checkpoint / "rotquant_config.json").read_text())
    invalid = copy.deepcopy(manifest)
    invalid["tied_vocabulary"]["projection_mode"] = "unknown"
    with pytest.raises(FormatValidationError, match="projection_mode"):
        validate_checkpoint_manifest(invalid)
    # Old v3 artifacts had no projection_mode and keep their rotated semantics.
    del manifest["tied_vocabulary"]["projection_mode"]
    validate_checkpoint_manifest(manifest)
    (checkpoint / "unrecorded.txt").write_text("extra")
    with pytest.raises(ValueError, match="unrecorded"):
        audit_artifact(checkpoint)
    (checkpoint / "rotquant_packed.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        audit_artifact(checkpoint)


def test_tiny_multimodal_preflight_runs_fresh_processes():
    from scripts.preflight_packed_validation import preflight

    report = preflight("cpu")
    assert report["passed"]
    assert len(report["checks"]) == 2
    for row in report["checks"]:
        assert row["fresh_process_reload"]["parity"]["max_abs_error"] == 0
        assert row["fresh_process_reload"]["residency_after"]["passed"]
        assert set(row["fresh_process_reload"]["rotary_buffers"].values()) == {"torch.float32"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_qwen_preserves_fp32_rotary_values_and_exact_reload(tmp_path, dtype):
    from scripts.preflight_packed_validation import rotary_buffer_check, tiny_model

    torch.manual_seed(20260907)
    model = tiny_model(dtype=dtype)
    rotary_buffer_check(model)
    original_buffers = {name: b.clone() for name, b in model.named_buffers() if "inv_freq" in name}
    assert any(not torch.equal(b, b.to(dtype).float()) for b in original_buffers.values())
    owner = quantize_vocabulary(model.get_input_embeddings().weight,
                               VocabularyConfig(bits=6, chunk_rows=64))
    owner.projection_mode = "dense_equivalent"
    patch_model(model, PatchConfig(quant=QuantConfig(bits=5, group_size=128, scale_bits=8),
                                  block=128, fallback=True,
                                  include=["model.language_model.layers."],
                                  exclude=["linear_attn.in_proj_a", "linear_attn.in_proj_b"]))
    prompts = [{"input_ids": torch.arange(1, 65).unsqueeze(0)}]
    with runner.packed_context(model, owner, "cpu", dtype):
        expected = capture_probes(model, prompts, "cpu", new_tokens=8)
        save_packed_checkpoint(model, tmp_path / "packed", model_loader="multimodal_lm")
    loaded = load_packed_model(tmp_path / "packed", dtype=dtype)
    rotary_buffer_check(loaded)
    for name, before in original_buffers.items():
        after = loaded.get_buffer(name)
        assert after.dtype == before.dtype
        assert torch.equal(after, before)  # FP16 then .float() is not sufficient.
    actual = capture_probes(loaded, prompts, "cpu", new_tokens=8)
    assert compare_probes(actual, expected, reload=True)["max_abs_error"] == 0
    assert all(torch.equal(actual[name], value) for name, value in expected.items())


def test_prepare_validate_resume_and_identity_gates(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    revalidation_root = tmp_path / "revalidation"
    tmp_path = tmp_path / "original"
    tmp_path.mkdir()
    config = runner.load_config(runner.DEFAULT_CONFIG)
    config.update(model="offline-tiny", model_loader="causal_lm", device="cpu", dtype="float32")
    config["quant"].update(group_size=32, error_comp="none")
    config["patch"].update(block=32, include=["model.layers."])
    config["vocabulary"].update(block=32, group_size=32, chunk_rows=8)
    config["eval"] = {}
    config["packed_validation"].update(probe_prompts=1, probe_prompt_tokens=4, probe_new_tokens=2)
    tok = Tokenizer(WordLevel({f"t{i}": i for i in range(33)}, unk_token="t0"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t0", pad_token="t0", eos_token="t32")
    calls = []
    ids = torch.tensor([[1, 2, 3, 4]])

    def load(*args):
        calls.append("load")
        return tiny_llama(), tokenizer, "causal_lm"

    def capture(*args):
        return SimpleNamespace(logit_references=[SimpleNamespace(inputs={"input_ids": ids})])

    def evaluate(cfg, eval_cfg, model, tokenizer, device, seed, refs, metrics):
        with torch.no_grad():
            assert torch.isfinite(model(ids).logits).all()
        record = {"input_hash": "fixture", "tokens": 3, "mean_teacher_kl": 0.01}
        common = {"input_hashes": ["fixture"], "prompt_metrics": [record],
                  "mean_teacher_kl": 0.01, "top1_agreement": 0.99}
        metrics.update(logit_fidelity=common, logit_fidelity_suites={"diverse": common},
                       trajectory_suites={"diverse": {**common, "token_agreement": 0.9}},
                       ppl_wikitext2=10., ppl_c4=14., evaluation_halted=False)
        metrics["data_manifest"] = {name: {"window_hashes": [name + "-fixture"]}
                                    for name in ("ppl_wikitext2", "ppl_c4")}

    monkeypatch.setattr(runner.experiment, "load_hf_model", load)
    monkeypatch.setattr(runner.experiment, "_prepare_calibration",
                        lambda *args: runner.experiment._CalibrationArtifacts())
    monkeypatch.setattr(runner.experiment, "_capture_references", capture)
    monkeypatch.setattr(runner.experiment, "_run_evaluations", evaluate)
    runner.prepare_seed(config, tmp_path, 0, 0)
    assert calls == ["load"]  # one backbone for both vocabularies
    # Simulate interruption after checkpoint publication but before completion.
    (tmp_path / "b5_v6_s0/prepared.json").unlink()
    runner.prepare_seed(config, tmp_path, 0, 0)
    assert calls == ["load"]
    for arm in runner.ARMS:
        result = runner.validate_worker(config, tmp_path, arm, 0, 0)
        assert result["passed"]
        assert result["artifact"]["measured_artifact_bytes"] > 0
        assert not result["provider_competitive"]
    assert calls == ["load"] * 3
    again = runner.validate_worker(config, tmp_path, runner.ARMS[0], 0, 0)
    assert again["passed"] and len(calls) == 3
    summary = runner.summarize(tmp_path, config, (0,))
    assert summary["complete"] and summary["artifact_validated_arms"] == list(runner.ARMS)
    assert not summary["independent_confirmation"]
    # A historical failure must survive a new-code validation unchanged.
    old_failure_path = tmp_path / "b5_v6_s0/validation.json"
    old_failure = runner.read_record(old_failure_path)
    old_failure["passed"] = False
    runner.save_record(old_failure_path, old_failure)
    files_before = {str(p.relative_to(tmp_path)): p.read_bytes()
                    for p in tmp_path.rglob("*") if p.is_file()}
    with monkeypatch.context() as newer:
        newer.setattr(runner, "source_identity", lambda: {"revision": "new-code", "source_fingerprint": "new-bytes"})
        newer.setattr(runner, "load_config", lambda path: config)
        newer.setattr(runner, "prepare_seed", lambda *args: pytest.fail("revalidation must not quantize"))

        def child(command, **kwargs):
            assert "--revalidate-from" in command
            arm = command[command.index("--worker-arm") + 1]
            result = runner.validate_worker(config, revalidation_root, arm, 0, 0, revalidate_from=tmp_path)
            assert result["passed"]
            assert result["identity"]["source"]["revision"] == "new-code"
            assert result["preparation_identity"] == old_failure["identity"] | {"arm": arm}
            return SimpleNamespace(returncode=0)

        newer.setattr(runner.subprocess, "run", child)
        newer.setenv("ROTQUANT_TOKEN_CACHE_DIR", str(tmp_path / "token_cache"))
        # The report binds both code identities and the old artifact hashes;
        # source evidence and failed validation are never relabelled/rewritten.
        result = runner.run(runner.DEFAULT_CONFIG, revalidation_root, heartbeat_seconds=0,
                            revalidate_from=tmp_path)
        assert result["complete"] and len(calls) == 5
        runner.run(runner.DEFAULT_CONFIG, revalidation_root, heartbeat_seconds=0, revalidate_from=tmp_path)
        assert len(calls) == 5
        import os
        assert os.environ["ROTQUANT_TOKEN_CACHE_DIR"] == str(revalidation_root / "token_cache")
        changed = copy.deepcopy(config)
        changed["target_artifact_bytes"] += 1
        with pytest.raises(ValueError, match="unchanged"):
            runner.validation_input(changed, revalidation_root, "b5_v6", 0, tmp_path)
        with pytest.raises(ValueError, match="overlap"):
            runner.run(runner.DEFAULT_CONFIG, tmp_path / "nested", revalidate_from=tmp_path)
        with pytest.raises(ValueError, match="missing"):
            runner.validation_input(config, revalidation_root, "b5_v6", 1, tmp_path)
    assert {str(p.relative_to(tmp_path)): p.read_bytes()
            for p in tmp_path.rglob("*") if p.is_file()} == files_before
    root = tmp_path / "b5_v6_s0"
    (root / "packed_probes.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="probe checksum"):
        runner.validate_worker(config, tmp_path, "b5_v6", 0, 0)
    with pytest.raises(ValueError, match="probe checksum"):
        runner.validation_input(config, revalidation_root, "b5_v6", 0, tmp_path)
    changed = copy.deepcopy(config)
    changed["vocabulary"]["seed"] = 99
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.prepare_seed(changed, tmp_path, 0, 0)


def test_revalidation_identity_allows_only_code_and_path_relocation():
    config = runner.load_config(runner.DEFAULT_CONFIG)
    original = runner.identity_for(config, 0, "b5_v6")
    moved = copy.deepcopy(original)
    moved["source"] = {"revision": "new", "source_fingerprint": "new"}
    paths = {path: "/new-checkout/" + str(i) for i, path in enumerate(moved["prompt_files"])}
    moved["prompt_files"] = {paths[path]: value for path, value in moved["prompt_files"].items()}
    for key in ("logit_fidelity_suites", "trajectory_suites"):
        for suite in moved["config"]["eval"].get(key, {}).values():
            if suite.get("prompt_file"):
                suite["prompt_file"] = paths[suite["prompt_file"]]
    assert runner.comparable_identity(moved) == runner.comparable_identity(original)
    for key in ("runtime", "reload_parity", "prototype_parity", "quality_guards"):
        changed = copy.deepcopy(moved)
        changed[key]["changed"] = True
        assert runner.comparable_identity(changed) != runner.comparable_identity(original)
    changed = copy.deepcopy(moved)
    changed["prompt_files"][next(iter(changed["prompt_files"]))] = "different-content"
    assert runner.comparable_identity(changed) != runner.comparable_identity(original)


def test_paired_quality_missing_inputs_or_windows_cannot_pass():
    with pytest.raises(ValueError, match="pairing mismatch"):
        runner.paired_quality({}, {}, "b5_v6", 0)
    metrics = {"logit_fidelity": {"input_hashes": ["a"]}}
    with pytest.raises(ValueError, match="per-prompt records"):
        runner.paired_quality(metrics, metrics, "b5_v6", 0)


def test_archived_screen_is_byte_preserved_and_reproducible(tmp_path):
    from pathlib import Path

    from rotquant.vocabulary import file_digest
    from scripts.archive_vocabulary_screen import archive

    source = Path("research/results/raw/qwen35_vocabulary_budget_ce6c8ec861a2")
    original_index = json.loads((source / "evidence_index.json").read_text())
    for name, record in original_index["files"].items():
        assert file_digest(source / name) == record["sha256"]
    copied = archive(source, tmp_path / "archive")
    assert copied["trial_records"] == 9
    assert copied["files"] == original_index["files"]
    damaged = tmp_path / "archive" / next(iter(original_index["files"]))
    damaged.write_text("corrupted")
    with pytest.raises(ValueError, match="collision"):
        archive(source, tmp_path / "archive")


def test_record_checksums_and_lock(tmp_path):
    path = tmp_path / "record.json"
    identity = {"source": "a"}
    runner.save_record(path, {"identity": identity})
    assert runner.read_record(path, identity)["identity"] == identity
    with pytest.raises(ValueError, match="identity"):
        runner.read_record(path, {"source": "b"})
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_record(path)
    with (runner.writer_lock(tmp_path / ".lock"),
          pytest.raises(RuntimeError, match="another runner"),
          runner.writer_lock(tmp_path / ".lock")):
        pass


def test_notebook_and_live_logs(tmp_path):
    from scripts.colab_runtime import run_live

    notebook = build_notebook()
    nbformat.validate(notebook)
    assert notebook == nbformat.read(OUTPUT, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "SEEDS = (0,)" in text and "PREPARE_ONLY = False" in text
    assert "--assess-only" in text and "preflight_packed_validation.py" in text
    assert "glob(\"*.safetensors\")" not in text
    revalidation = build_notebook(revalidate=True)
    nbformat.validate(revalidation)
    assert revalidation == nbformat.read(REVALIDATION_OUTPUT, as_version=4)
    for cell in revalidation.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    recovery_text = "\n".join(cell.source for cell in revalidation.cells)
    assert 'REVALIDATE_FROM = Path("/content/drive/MyDrive/rotquant/qwen35_packed_validation/8f10ee60fc7f")' in recovery_text
    assert '--revalidate-from' in recovery_text and 'RELOAD_PARITY' not in recovery_text
    log = run_live([sys.executable, "-c", "print('durable output', flush=True)"], "smoke",
                   repo_dir=tmp_path, log_root=tmp_path)
    assert "durable output" in log.read_text()
