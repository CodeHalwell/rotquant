"""Offline end-to-end runner, resume, decision and notebook conformance."""
import copy
import json

import nbformat
import pytest
import torch
import yaml
from torch import nn

from rotquant.calibrate import collect_hessians_streamed
from rotquant.vocabulary import tensor_digest
from scripts import run_qwen35_vocabulary_budget as runner
from scripts.assess_qwen35_vocabulary_budget import assess, paired_contrast
from scripts.build_qwen35_vocabulary_budget_notebook import OUTPUT, build_notebook


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 64)
        self.backbone = nn.Linear(64, 64, bias=False)
        self.head = nn.Linear(64, 17, bias=False)
        self.head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, ids):
        return self.head(self.backbone(self.embedding(ids)))


def test_grouped_screen_reuses_backbones_and_resumes(tmp_path, monkeypatch):
    loads, patches, references = [], [], []
    ids = torch.tensor([[1, 2, 3]])
    config = runner.resolved_config(runner.DEFAULT_CONFIG)
    config.update(model="tiny", device="cpu", dtype="float32")
    config["quant"].update(error_comp="none", group_size=32)
    config["patch"].update(block=32, include=["backbone"])
    config["vocabulary"].update(block=32, group_size=32, chunk_rows=4)
    config["eval"] = {}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def load(*args):
        torch.manual_seed(30)
        model = TinyModel().eval()
        loads.append(1)
        return model, None, "tiny"

    def capture(cfg, eval_cfg, model, tokenizer, device, model_name, metrics):
        references.append(tensor_digest(model.embedding.weight))
        return model(ids).detach().clone()

    original_patch = runner.experiment._apply_quantization

    def patch(cfg, model, pcfg, art, metrics):
        if pcfg.enabled:
            patches.append(pcfg.quant.bits)
        return original_patch(cfg, model, pcfg, art, metrics)

    def evaluate(cfg, eval_cfg, model, tokenizer, device, seed, refs, metrics):
        error = float((model(ids) - refs).square().mean().detach())
        records = [{"input_hash": str(i), "tokens": 3, "mean_teacher_kl": error}
                   for i in range(24)]
        common = {"input_hashes": [str(i) for i in range(24)], "mean_teacher_kl": error,
                  "top1_agreement": 0.99, "prompt_metrics": records}
        metrics.update(logit_fidelity=common, logit_fidelity_suites={"diverse": common},
                       trajectory_suites={"diverse": {**common, "token_agreement": 0.8}},
                       ppl_wikitext2=10., ppl_c4=14., evaluation_halted=False)

    monkeypatch.setattr(runner.experiment, "load_hf_model", load)
    monkeypatch.setattr(runner.experiment, "_prepare_calibration",
                        lambda *a: runner.experiment._CalibrationArtifacts())
    monkeypatch.setattr(runner.experiment, "_capture_references", capture)
    monkeypatch.setattr(runner.experiment, "_apply_quantization", patch)
    monkeypatch.setattr(runner.experiment, "_run_evaluations", evaluate)
    summary = runner.run_screen(config_path, tmp_path / "results", heartbeat_seconds=0)
    assert summary["complete"] and len(summary["rows"]) == 9
    assert len(loads) == 3 and patches == [4, 5]
    assert len(set(references)) == 1
    assert not summary["artifact_promoted"] and not summary["provider_competitive"]
    assert all(row["measured_artifact_bytes"] is None for row in summary["rows"])
    again = runner.run_screen(config_path, tmp_path / "results", heartbeat_seconds=0)
    assert again == summary and len(loads) == 3
    trials = [json.loads(path.read_text()) for path in (tmp_path / "results/model_trials").glob("*.json")]
    damaged = copy.deepcopy(trials)
    damaged[0]["metrics"]["teacher_source_vocab_digest"] = "wrong"
    rejected = assess(damaged)
    assert not rejected["complete"] and not rejected["finalists"]
    path = next((tmp_path / "results/model_trials").glob("*.json"))
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        runner.run_screen(config_path, tmp_path / "results", heartbeat_seconds=0)


def test_resumable_hessians_skip_completed_layers_and_validate_hashes(tmp_path):
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).eval()
    calls = []
    batches = [torch.randn(4, 8) for _ in range(2)]
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    first = collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                                     resume_key="source-model-and-token-digest")
    count = len(calls)
    second = collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                                      resume_key="source-model-and-token-digest")
    assert len(calls) == count
    for name in first.hessians:
        assert torch.equal(first.hessians[name], second.hessians[name])
        assert torch.equal(first.means[name], second.means[name])
    with pytest.raises(ValueError, match="identity"):
        collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                                 resume_key="other-source")
    next(tmp_path.glob("*.pt")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid resumed Hessian"):
        collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                                 resume_key="source-model-and-token-digest")
    hook.remove()


def test_notebook_generated_cells_are_valid_and_compile():
    notebook = build_notebook()
    nbformat.validate(notebook)
    saved = nbformat.read(OUTPUT, as_version=4)
    assert saved == notebook
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{index}", "exec")
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "SEEDS = (0,)" in text and "stdout=sink" in text
    assert "--dry-run" in text and "--device\", \"cuda" in text
    assert "projected" in text and "--no-build-isolation" in text


def test_pairing_rejects_mismatched_tokens():
    def trial(token):
        return {"metrics": {"logit_fidelity": {"prompt_metrics": [
            {"input_hash": token, "tokens": 3, "mean_teacher_kl": 0.1}]}}}
    with pytest.raises(ValueError, match="identical prompt"):
        paired_contrast({"a": trial("a"), "b": trial("b")}, {"a": 1, "b": -1})


def test_partial_hessian_resume_and_invalid_mean(tmp_path):
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).eval()
    batches = [torch.randn(4, 8)]
    collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                             layers_per_pass=1, resume_key="source")
    path = tmp_path / "calibration.json"
    record = json.loads(path.read_text())
    del record["layers"]["1"]
    path.write_text(json.dumps(record))
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    resumed = collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                                       layers_per_pass=1, resume_key="source")
    hook.remove()
    assert len(calls) == 1 and len(resumed.hessians) == 2
    record = json.loads(path.read_text())
    record["layers"]["0"]["mean"] = [0.0]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="invalid resumed Hessian"):
        collect_hessians_streamed(model, batches, "cpu", offload_dir=tmp_path,
                                 resume_key="source")
