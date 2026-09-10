"""CPU-only validation of the native-GPU handoff's bounded, fail-closed controls."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

from scripts import build_rq3_runtime as build
from scripts import run_rq3_retained_gpu as retained
from scripts.build_qwen35_native_gpu_notebook import build_notebook
from scripts.check_rq3_model import numerical_metrics
from scripts.rq3_test_runtime import NativeModel


def test_notebook_valid_and_bounded():
    notebook = build_notebook()
    nbformat.validate(notebook)
    codes = [c.source for c in notebook.cells if c.cell_type == "code"]
    for source in codes:
        ast.parse(source)
    joined = "\n".join(codes)
    assert "SESSION_BUDGET_MINUTES" in joined and "timeout_seconds=min(remaining, phase_limit)" in joined
    assert 'RUN_TIMING = False' in joined
    assert 'operators-CUDA0.json' in joined and 'model-w8.json' in joined
    assert 'assert gate["passed"]' in joined and 'gate["runtime_files"] == current_runtime' in joined
    assert "--force" not in joined and '"reset"' not in joined
    assert "run_qwen35_public_tasks" not in joined
    path = Path(__file__).resolve().parents[1] / "notebooks/qwen35_4b_native_gpu_validation_colab.ipynb"
    saved = nbformat.read(path, as_version=4)
    assert [(c.cell_type, c.source) for c in saved.cells] == [(c.cell_type, c.source) for c in notebook.cells]


def test_cuda_build_fails_before_source_mutations(tmp_path, monkeypatch):
    monkeypatch.setattr(build.shutil, "which", lambda _: None)
    monkeypatch.setattr(build, "prepare_source", lambda _: pytest.fail("must not prepare a CPU fallback"))
    with pytest.raises(ValueError, match="nvcc missing"):
        build.build(tmp_path / "source", tmp_path / "build", "CUDA", 2)
    assert not list(tmp_path.iterdir())


def test_metrics_are_teacher_to_candidate_kl():
    candidate = np.array([[2., -1., 0.], [1., 3., -2.]])
    teacher = np.array([[1., -1., 0.], [1., 3., -2.]])
    result = numerical_metrics(candidate, teacher)
    q, p = torch.tensor(candidate).log_softmax(-1), torch.tensor(teacher).log_softmax(-1)
    assert result["mean_kl"] == pytest.approx((p.exp() * (p-q)).sum(-1).mean().item())
    assert numerical_metrics(teacher, teacher)["mean_kl"] == 0
    with pytest.raises(ValueError):
        numerical_metrics([[float("nan")]], [[0.]])


class FakeModel:
    vocab = 16
    def evaluate(self, ids, reset=False, selected=1):
        result = np.zeros((selected, self.vocab), dtype=np.float32)
        result[:, 7] = 1.
        return result[0] if selected == 1 else result

    def is_eog(self, token):
        return False


def probes():
    return {"p0.input.input_ids": torch.tensor([[3, 4, 5, 6]]),
            "p0.input.attention_mask": torch.ones((1, 4), dtype=torch.int64),
            "p0.logits": torch.nn.functional.one_hot(torch.tensor([[7, 7]]), 16).float(),
            "p0.generated": torch.tensor([[3, 4, 5, 6, 7, 7, 7]])}


def test_native_capture_preserves_probes_and_stops_on_eog():
    expected = probes()
    actual = retained.capture_native(FakeModel(), expected)
    assert all(torch.equal(actual[k], expected[k]) for k in actual)
    model = FakeModel()
    model.is_eog = lambda _: True
    actual = retained.capture_native(model, expected)
    assert actual["p0.generated"].tolist() == [[3, 4, 5, 6, 7]]
    assert not torch.equal(actual["p0.generated"], expected["p0.generated"])


@pytest.mark.parametrize("mutation", ["mask", "multimodal", "batch", "long_trace", "prefix", "unknown"])
def test_native_capture_rejects_unrepresented_inputs(mutation):
    expected = probes()
    if mutation == "mask":
        expected["p0.input.attention_mask"][0, 0] = 0
    elif mutation == "multimodal":
        expected["p0.input.pixel_values"] = torch.ones(1)
    elif mutation == "batch":
        expected["p0.input.input_ids"] = expected["p0.input.input_ids"].repeat(2, 1)
    elif mutation == "long_trace":
        expected["p0.generated"] = torch.tensor([[3, 4, 5, 6] + [7] * 9])
    elif mutation == "prefix":
        expected["p0.generated"][0, 0] = 9
    else:
        expected["p0.unrecognised"] = torch.ones(1)
    with pytest.raises(ValueError):
        retained.capture_native(FakeModel(), expected)


@pytest.mark.parametrize("ids", [[2**32 + 3], [-1], [1.5], [], [[3]]])
def test_native_binding_rejects_invalid_ids_before_ctypes(ids):
    model = NativeModel.__new__(NativeModel)
    model.handle, model.vocab = 1, 16
    with pytest.raises(ValueError):
        model.evaluate(ids)


def evidence_fixture(tmp_path, monkeypatch):
    arm, exported = tmp_path / "arm", tmp_path / "export"
    arm.mkdir(); exported.mkdir()
    checkpoint = arm / "checkpoint"; checkpoint.mkdir()
    (checkpoint / retained.MANIFEST_NAME).write_text("{}")
    (arm / "packed_probes.safetensors").write_bytes(b"opaque fixture")
    core = {"prototype_parity": {"passed": True}, "probe_files": {
        "packed_probes.safetensors": build.digest(arm / "packed_probes.safetensors")}}
    (arm / "preparation.json").write_text(json.dumps({**core, "status": "preparing"}))
    sha = build.digest(checkpoint / retained.MANIFEST_NAME)
    (arm / "prepared.json").write_text(json.dumps({**core, "status": "prepared", "export": {"manifest_sha256": sha}}))
    manifest = {"deployment": {"preparation_sha256": build.digest(arm / "preparation.json")}}
    def verify(path, expected_manifest_sha256):
        assert build.digest(path / retained.MANIFEST_NAME) == expected_manifest_sha256
        return manifest
    monkeypatch.setattr(retained, "verify_checkpoint", verify)
    (exported / "model.gguf").write_bytes(b"gguf fixture")
    audit = {"protocol": "rotquant-gguf-v2-export-v1", "checkpoint_manifest_sha256": sha,
             "artifact_files": {"model.gguf": {"bytes": 12, "sha256": build.digest(exported / "model.gguf")}},
             "complete_model_payload_bytes": 12}
    (exported / "export.json").write_text(json.dumps(audit))
    return arm, exported, audit


def test_preflight_verifies_saved_artifacts(tmp_path, monkeypatch):
    arm, exported, _ = evidence_fixture(tmp_path, monkeypatch)
    assert retained.verified_evidence(arm, exported)[0].name == "packed_probes.safetensors"
    (exported / "model.gguf").write_bytes(b"changed file")
    with pytest.raises(ValueError, match="artifact changed"):
        retained.verified_evidence(arm, exported)


@pytest.mark.parametrize("mutation", ["probe", "source", "bytes", "path"])
def test_preflight_rejects_mismatched_provenance(tmp_path, monkeypatch, mutation):
    arm, exported, audit = evidence_fixture(tmp_path, monkeypatch)
    if mutation == "probe":
        (arm / "packed_probes.safetensors").write_bytes(b"changed")
    elif mutation == "source":
        audit["checkpoint_manifest_sha256"] = "wrong"
    elif mutation == "bytes":
        audit["complete_model_payload_bytes"] += 1
    else:
        audit["artifact_files"]["../escape"] = audit["artifact_files"]["model.gguf"]
    (exported / "export.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        retained.verified_evidence(arm, exported)


def test_patch_manifest_is_self_consistent():
    contract = json.loads((build.INTEGRATION / "rotquant-native-v2-files.json").read_text())
    assert contract["base_revision"] == build.REVISION
    patch = build.INTEGRATION / "rotquant-native-v2.patch"
    assert contract["patch_sha256"] == build.digest(patch)
    changed = {line.split(" b/", 1)[1] for line in patch.read_text().splitlines() if line.startswith("diff --git ")}
    assert changed == set(contract["files_sha256"])
    assert "ggml/src/ggml-cuda/rq3.cu" in changed and "src/llama-rq3.cpp" in changed
