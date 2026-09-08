"""Frozen-ID input contracts and narrowly reviewed, read-only result reuse."""

import ast
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_qwen35_fresh_eval as runner


def test_segmentation_difference_is_audited_without_changing_scoring_inputs():
    raw = "का".encode()
    ids = [0, 1]
    tokenizer = SimpleNamespace(decode=lambda *a, **k: "का")

    class Tokenizer:
        def __call__(self, *a, **k):
            return SimpleNamespace(input_ids=ids)

        decode = staticmethod(tokenizer.decode)

    model = SimpleNamespace(tokenize=lambda *a, **k: [2], detokenize=lambda *a, **k: raw)
    item = {"id": "hi", "kind": "task", "rendered": "का", "input_ids": ids,
            "input_hash": runner._input_hash(np.array(ids))}
    manifest = {"fingerprint": "m", "items": [item]}
    before = copy.deepcopy(manifest)
    strict = runner.audit_gguf_inputs(model, Tokenizer(), manifest, "strict")
    frozen = runner.audit_gguf_inputs(model, Tokenizer(), manifest, "frozen-hf")
    assert not strict["passed"] and frozen["passed"]
    assert frozen["native_mismatches"] == ["hi"]
    assert frozen["records"][0]["frozen_ids"] == [0, 1]
    assert frozen["records"][0]["native_ids"] == [2]
    assert manifest == before
    model.detokenize = lambda *a, **k: b"different text"
    assert not runner.audit_gguf_inputs(model, Tokenizer(), manifest, "frozen-hf")["passed"]
    model.detokenize = lambda *a, **k: raw
    ids.append(3)  # Frozen hash/tokens disagree; the full manifest gate detects this.
    Tokenizer.decode = staticmethod(lambda *a, **k: "different HF text")
    assert not runner.audit_gguf_inputs(model, Tokenizer(), manifest, "frozen-hf")["passed"]


def test_llama_generation_consumes_frozen_ids_without_native_retokenization():
    class Model:
        n_tokens = 0
        scores = np.array([[0., 0., 1.]])
        _model = SimpleNamespace(token_get_text=lambda i: str(i))

        def n_vocab(self):
            return 3

        def tokenize(self, *a, **k):
            pytest.fail("scoring/generation must not invoke native retokenization")

        def reset(self):
            self.seen = []

        def eval(self, ids):
            self.seen.append(list(ids))
            self.n_tokens = 1

    model = Model()
    tokenizer = SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(3)})
    backend = runner.LlamaBackend(model, tokenizer, [2], output_size=3)
    assert backend.generate([0, 1]) == [2]
    assert model.seen == [[0, 1]]


@pytest.fixture
def reusable(tmp_path, monkeypatch):
    previous, output, evidence = [tmp_path / name for name in ("previous", "output", "evidence")]
    current_source = {"revision": "new", "source_fingerprint": "new"}
    expected = {"source": current_source, "runtime": {"version": "fixed"}, "settings": runner.SETTINGS}
    monkeypatch.setattr(runner, "source_identity", lambda: current_source)
    monkeypatch.setattr(runner, "freeze_context", lambda *a: ([], [], expected))
    monkeypatch.setattr(runner, "stop_ids_for_source", lambda *a: [4])
    items = []
    for i in range(120):
        ids = [i % 5, i // 5 % 5, i // 25 % 5]
        items.append({"id": f"item-{i}", "kind": "text" if i < 24 else "task",
                      "domain": "c4" if i < 24 else "task", "family": str(i),
                      "input_ids": ids, "input_hash": runner._input_hash(np.array(ids))})
    recipes = {}
    for arm in ("b5_v6", "b5_v8"):
        prepared = {"identity": {"config": {"model": "fixture"}}, "export": {"manifest_sha256": "artifact"}}
        runner.save_record(evidence / f"{arm}_s0/prepared.json", prepared)
        recipes[arm] = runner.recipe_signature(prepared["identity"])
    manifest = {"identity": {**expected, "source": runner.REUSABLE_SOURCE},
                "items": items, "recipes": recipes, "tokenizer": {"size": 5}}
    manifest["fingerprint"] = runner.fingerprint(manifest)
    runner.save_record(previous / "manifest.json", manifest)
    for label in runner.REUSABLE_LABELS:
        directory = previous / "runs" / label
        identity = {"manifest": manifest["fingerprint"], "source": runner.REUSABLE_SOURCE,
                    "runtime": expected["runtime"], "stop_ids": [4], "device": "cpu", "label": label}
        if label.startswith("b5_"):
            identity.update(prepared_sha256=runner.file_digest(evidence / label / "prepared.json"),
                            manifest_sha256="artifact")
            runner.save_record(directory / "reload_probe_report.json",
                               {"identity": identity, "parity": {"passed": True}})
        runner.save_record(directory / "identity.json", identity)
        runner.save_record(directory / "complete.json",
                           {"complete": True, "identity": identity, "manifest": manifest["fingerprint"]})
        for item in items:
            path = directory / f"{item['id']}.json"
            value = {**item, "manifest": manifest["fingerprint"], "collection": runner.fingerprint(identity),
                     "tokens": 2, "mean_teacher_kl": 0., "top1_agreement": 1.,
                     "task_success": True, "source_task_success": True, "json_valid": True,
                     "truncated": False, "trajectory_token_agreement": 1., "exact_trajectory": True}
            if label == "source_fp16":
                runner._atomic_npz(path.with_suffix(".npz"), logits=np.zeros((2, 5), dtype=np.float32))
                value["reference_sha256"] = runner.file_digest(path.with_suffix(".npz"))
            runner.save_record(path, value)
    return previous, output, evidence, manifest


def test_reuse_preserves_original_results_and_reads_external_references(reusable):
    previous, output, evidence, manifest = reusable
    before = {str(p): runner.file_digest(p) for p in previous.rglob("*") if p.is_file()}
    runner.reuse_completed(output, evidence, previous, runner.REUSABLE_LABELS, "cpu")
    assert runner.freeze(output, evidence) == manifest
    assert not list(output.rglob("*.npz"))
    assert not (output / "runs").exists()
    path = runner.paths_for(output, "source_fp16", manifest["items"][0])
    assert path.is_relative_to(previous)
    _, logits = runner.load_reference(output, "source_fp16", manifest["items"][0], manifest)
    assert logits.shape == (2, 5)
    summary = runner.summarize(output, manifest, [*runner.REUSABLE_LABELS, "gguf_bf16_bridge"])
    assert summary["missing"] == ["gguf_bf16_bridge"]
    assert summary["reuse"]["producer_source"] == runner.REUSABLE_SOURCE
    assert all(row["producer_source"] == runner.REUSABLE_SOURCE for row in summary["rows"])
    assert not summary["provider_competitive"]
    runner.reuse_completed(output, evidence, previous, runner.REUSABLE_LABELS, "cpu")
    assert before == {str(p): runner.file_digest(p) for p in previous.rglob("*") if p.is_file()}


def test_reused_source_cell_verifies_instead_of_loading_model(reusable, monkeypatch):
    previous, output, evidence, manifest = reusable
    runner.reuse_completed(output, evidence, previous, ["source_fp16"], "cpu")
    monkeypatch.setattr(runner, "load_tokenizer", lambda: object())
    monkeypatch.setattr(runner, "tokenizer_identity", lambda *a: manifest["tokenizer"])
    monkeypatch.setattr(runner, "fresh_runtime", lambda: manifest["identity"]["runtime"])

    def forbidden(*a, **k):
        pytest.fail("completed source must not load or rerun a model")

    monkeypatch.setattr(runner.experiment, "load_hf_model", forbidden)
    runner.run_collection(output, manifest, "source_fp16", device="cpu")
    assert not (output / "runs").exists()
    with pytest.raises(ValueError, match="parameters changed"):
        runner.run_collection(output, manifest, "source_fp16", device="cuda")


@pytest.mark.parametrize("change", ["source", "runtime", "incomplete", "record", "tensor", "parity"])
def test_reuse_fails_closed_on_unreviewed_changed_or_incomplete_evidence(reusable, change):
    previous, output, evidence, manifest = reusable
    if change in ("source", "runtime"):
        changed = copy.deepcopy(manifest)
        changed["identity"][change] = {"changed": True}
        changed["fingerprint"] = runner.fingerprint({k: v for k, v in changed.items() if k != "fingerprint"})
        runner.save_record(previous / "manifest.json", changed)
    elif change == "incomplete":
        (previous / "runs/b5_v6_s0/complete.json").unlink()
    elif change == "record":
        path = previous / "runs/source_fp16/item-0.json"
        value = runner.read_record(path)
        runner.save_record(path, {**value, "collection": "different"})
    elif change == "tensor":
        (previous / "runs/source_fp16/item-0.npz").write_bytes(b"corrupted")
    else:
        path = previous / "runs/b5_v6_s0/reload_probe_report.json"
        value = runner.read_record(path)
        runner.save_record(path, {**value, "parity": {"passed": False}})
    with pytest.raises(ValueError):
        runner.reuse_completed(output, evidence, previous, runner.REUSABLE_LABELS, "cpu")
    assert not (output / "reuse.json").exists()


def test_reuse_rejects_changes_after_adoption_and_does_not_mix_consumers(reusable, monkeypatch):
    previous, output, evidence, manifest = reusable
    runner.reuse_completed(output, evidence, previous, ["source_fp16"], "cpu")
    path = previous / "runs/source_fp16/item-0.json"
    value = runner.read_record(path)
    runner.save_record(path, {**value, "mean_teacher_kl": 9.})
    with pytest.raises(ValueError, match="changed or missing"):
        runner.paths_for(output, "source_fp16", manifest["items"][0])
    monkeypatch.setattr(runner, "source_identity", lambda: {"revision": "different consumer"})
    with pytest.raises(ValueError, match="consumer code changed"):
        runner.freeze(output, evidence)


def test_reuse_cannot_import_provider_results_or_overlap_original(reusable):
    previous, output, evidence, _ = reusable
    with pytest.raises(ValueError, match="reviewed"):
        runner.reuse_completed(output, evidence, previous, ["source_fp16", "gguf_bf16_bridge"], "cpu")
    with pytest.raises(ValueError, match="overlap"):
        runner.reuse_completed(previous, evidence, previous, ["source_fp16"], "cpu")


def test_reuse_compatibility_keeps_reviewed_hf_scoring_and_core_code_unchanged():
    revision = runner.REUSABLE_SOURCE["revision"]
    old = subprocess.check_output(["git", "show", f"{revision}:scripts/run_qwen35_fresh_eval.py"],
                                  cwd=runner.ROOT, text=True)
    new = Path(runner.__file__).read_text()
    def functions(text):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for name in ("HFBackend", "LlamaBackend", "stop_ids_for_source", "score_one", "metric_record"):
        assert functions(old)[name] == functions(new)[name], f"re-review reuse compatibility: {name}"
    code = subprocess.check_output(["git", "diff", revision, "--", "rotquant",
                                    "scripts/run_experiment.py", "scripts/run_unsloth_qwen35_4b_kl.py",
                                    "scripts/run_qwen35_packed_validation.py"], cwd=runner.ROOT, text=True)
    assert not code, "re-review compatibility when model execution/scoring code changes"
