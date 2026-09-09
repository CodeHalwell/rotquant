"""CPU-only contracts; no public-dataset download or model inference in CI."""

from typing import ClassVar

import nbformat
import numpy as np
import pytest
import torch

from scripts import public_task_suite as suite
from scripts import run_qwen35_public_tasks as runner
from scripts.build_qwen35_public_tasks_notebook import OUTPUT, build_notebook


@pytest.mark.parametrize("text", ["#### 42", "Reason: 6 * 7 = 42.\n#### 42.0", "#### +42"])
def test_gsm_numeric_correctness(text):
    item = {"benchmark": "gsm8k", "answer": "42"}
    assert suite.score(item, text)["task_success"]
    assert not suite.score(item, text, truncated=True)["task_success"]


@pytest.mark.parametrize("text", ["42", "#### 43", "#### NaN", "#### Infinity", "#### 42\nextra",
                                  "I used 42 items and got 43.", "#### 4,2", "text #### 42"])
def test_gsm_does_not_accept_incidental_digits(text):
    assert not suite.score({"benchmark": "gsm8k", "answer": "42"}, text)["task_success"]


def test_numeric_commas_and_signs():
    assert suite.numeric_answer("#### -1,234.5") == suite.numeric_answer("#### -1234.50")


@pytest.mark.parametrize("text", ["__import__('os').getcwd()", "[0] * 10000000000", "{'x':1,'x':2}",
                                  "{True:1,1:2}", "[v for v in []]", "lambda: 4"])
def test_literal_is_inert_and_bounded(text):
    with pytest.raises((ValueError, TypeError, SyntaxError)):
        suite.literal(text)
    with pytest.raises(ValueError, match="long"):
        suite.literal(" " * 65537)


def test_crux_strict_types_markers_and_truncation():
    item = {"benchmark": "cruxeval_o", "answer": "{'x': [1, 2]}"}
    result = suite.score(item, "[ANSWER]{'x':[1,2]}[/ANSWER]")
    assert result["task_success"]
    assert not suite.score(item, "[ANSWER]{'x':[True,2]}[/ANSWER]")["task_success"]
    assert not suite.score(item, "[ANSWER]{'x':(1,2)}[/ANSWER]")["task_success"]
    assert not suite.score(item, "[ANSWER]{'x':[1,2]}[/ANSWER]", True)["task_success"]
    assert not suite.score(item, "[ANSWER]1[/ANSWER][ANSWER]{'x':[1,2]}[/ANSWER]")["task_success"]
    assert not suite.same_literal({1: "value"}, {True: "value"})
    assert not suite.same_literal({1}, {True})
    assert not suite.same_literal({(1,): 3}, {(True,): 3})


class Scorer:
    identity: ClassVar[dict] = {"fixture": "not-real-ifeval"}

    def score(self, row, text):
        valid = text == "yes"
        return {"prompt_strict": valid, "prompt_loose": valid,
                "instructions_strict": [valid, valid], "instructions_loose": [valid, valid]}


def test_ifeval_records_upstream_score_before_guard():
    item = {"benchmark": "ifeval", "oracle": {}}
    result = suite.score(item, "yes", True, Scorer())
    assert result["prompt_strict"] and result["checker_success_before_truncation_guard"]
    assert not result["task_success"]
    with pytest.raises(ValueError, match="required"):
        suite.score(item, "yes")


def test_dataset_pins_and_selection_are_stable(monkeypatch):
    import datasets

    sources = {
        "gsm8k": [{"question": f"Question {i}", "answer": "#### 42"} for i in range(10)],
        "cruxeval_o": [{"code": "def f(x): return x", "input": "42", "output": "42", "id": str(i)} for i in range(10)],
        "ifeval": [{"key": i, "prompt": f"Prompt {i}", "instruction_id_list": ["x"], "kwargs": [{}]} for i in range(10)],
    }
    specs = {k: {**v, "rows": 10} for k, v in suite.DATASETS.items()}
    monkeypatch.setattr(suite, "DATASETS", specs)
    calls = []

    def load(repo, config, *, split, revision):
        name = next(k for k, v in specs.items() if v["repo"] == repo)
        assert split == specs[name]["split"] and revision == specs[name]["revision"]
        calls.append(repo)
        return sources[name]

    monkeypatch.setattr(datasets, "load_dataset", load)
    a, receipts = suite.load_items(4)
    b, _ = suite.load_items(6)
    assert len(a) == 12 and len(b) == 18
    assert {i["id"] for i in a} < {i["id"] for i in b}
    assert len(calls) == 6 and all(len(v["revision"]) == 40 for v in receipts.values())
    assert len(suite.load_items(0)[0]) == 30
    with pytest.raises(ValueError, match="samples"):
        suite.load_items(-1)
    sources["gsm8k"].pop()
    with pytest.raises(ValueError, match="row count"):
        suite.load_items(4)


def fixture_manifest():
    items = [suite.make_item("gsm8k", {"question": "6*7?", "answer": "#### 42"}, 0),
             suite.make_item("cruxeval_o", {"code": "def f(x): return x", "input": "42", "output": "42", "id": "x"}, 0),
             suite.make_item("ifeval", {"key": 1, "prompt": "Say yes", "instruction_id_list": ["x"], "kwargs": [{}]}, 0)]
    for i, item in enumerate(items):
        item.update(input_ids=[i + 1], input_hash=runner.fresh._input_hash(np.array([i + 1])), rendered=item["prompt"])
    manifest = {"identity": {"labels": ["source_fp16"], "source": {}, "runtime": {}, "scorer": Scorer.identity},
                "items": items, "datasets": {name: {"selected": 1} for name in suite.BENCHMARKS},
                "stop_ids": [4]}
    manifest["fingerprint"] = suite.fingerprint(manifest)
    return manifest


class Tokenizer:
    def get_vocab(self):
        return {str(i): i for i in range(5)}

    def decode(self, tokens, **kwargs):
        return {1: "#### 42", 2: "[ANSWER]42[/ANSWER]", 3: "yes"}[tokens[0]]


class Generator:
    tokenizer = Tokenizer()
    stop_ids: ClassVar[list] = [4]

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def generate_task(self, ids, cap):
        self.calls.append(ids)
        if len(self.calls) == self.fail:
            raise RuntimeError("simulated disconnect")
        return [ids[0], 4]


def identity_for(manifest):
    return {"manifest": manifest["fingerprint"], "source": {}, "runtime": {},
            "scorer": Scorer.identity, "label": "source_fp16", "stop_ids": [4]}


def test_manifest_refuses_corrupt_or_duplicate_inputs():
    manifest = fixture_manifest()
    runner.validate_manifest(manifest)
    manifest["items"][0]["input_ids"] = [2]
    with pytest.raises(ValueError, match="corrupt"):
        runner.validate_manifest(manifest)
    manifest["fingerprint"] = suite.fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="hash"):
        runner.validate_manifest(manifest)


def test_resume_after_interruption_and_summary_reconciles(tmp_path):
    manifest = fixture_manifest()
    identity = identity_for(manifest)
    directory = tmp_path / "runs/source_fp16"
    runner.save_record(directory / "identity.json", identity)
    backend = Generator(fail=2)
    with pytest.raises(RuntimeError, match="disconnect"):
        runner.collect_prompts(tmp_path, manifest, "source_fp16", identity, backend, Scorer())
    first = directory / (manifest["items"][0]["id"] + ".json")
    first_bytes = first.read_bytes()
    report = runner.summarize(tmp_path, manifest, Scorer())
    assert not report["complete"] and report["missing"][0]["completed_prompts"] == 1
    backend = Generator()
    runner.collect_prompts(tmp_path, manifest, "source_fp16", identity, backend, Scorer())
    assert len(backend.calls) == 2 and first.read_bytes() == first_bytes
    runner.save_record(directory / "complete.json", {"complete": True, "identity": identity, "ledger": None})
    report = runner.summarize(tmp_path, manifest, Scorer())
    assert report["complete"] and len(report["rows"]) == 3
    assert all(row["accuracy"] == 1.0 for row in report["rows"])
    assert not report["promoted"]
    again = Generator()
    runner.collect_prompts(tmp_path, manifest, "source_fp16", identity, again, Scorer())
    assert not again.calls
    saved = runner.read_record(first)
    saved["task_success"] = False
    runner.save_record(first, saved)
    with pytest.raises(ValueError, match="oracle"):
        runner.summarize(tmp_path, manifest, Scorer())


def test_worker_rejects_bad_generation_and_changed_identity(tmp_path):
    manifest = fixture_manifest()
    identity = identity_for(manifest)
    runner.collect_prompts(tmp_path, manifest, "source_fp16", identity, Generator(), Scorer())
    with pytest.raises(ValueError, match="identity"):
        runner.collect_prompts(tmp_path, manifest, "source_fp16", {**identity, "runtime": {"changed": True}}, Generator(), Scorer())
    first = manifest["items"][0]
    value = runner.read_record(tmp_path / "runs/source_fp16" / (first["id"] + ".json"))
    value["continuation"] = [1, 4, 2]
    with pytest.raises(ValueError, match="past stop"):
        runner.validate_record(value, first, identity)


def test_packed_summary_requires_reload_gate(tmp_path):
    with pytest.raises(ValueError, match="gate"):
        runner.verify_gates(tmp_path, "b5_v6_s0", {"manifest": "x"})


def test_summary_refuses_different_provider_bridge_binaries(tmp_path, monkeypatch):
    manifest = fixture_manifest()
    labels = ["gguf_bf16_bridge", "unsloth_ud_q4"]
    manifest["identity"]["labels"] = labels
    manifest["fingerprint"] = suite.fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    monkeypatch.setattr(runner, "verify_gates", lambda *a: None)
    for label in labels:
        identity = {**identity_for(manifest), "label": label, "llama_build": label}
        directory = tmp_path / "runs" / label
        runner.save_record(directory / "identity.json", identity)
        runner.collect_prompts(tmp_path, manifest, label, identity, Generator(), Scorer())
        runner.save_record(directory / "complete.json", {"complete": True, "identity": identity, "ledger": None})
    with pytest.raises(ValueError, match="different engine"):
        runner.summarize(tmp_path, manifest, Scorer())


def test_hf_generator_uses_per_task_cap_and_neutral_config():
    class Model:
        def eval(self):
            return self

        def generate(self, **kwargs):
            config = kwargs["generation_config"]
            assert config.max_new_tokens == 71 and not config.do_sample
            assert config.repetition_penalty == 1.0 and config.eos_token_id == [4]
            guard = kwargs["logits_processor"][0]
            with pytest.raises(ValueError, match="non-finite"):
                guard(None, torch.tensor([[float("nan")]]))
            return torch.tensor([[1, 2, 3, 4]])

    backend = runner.HFGenerator(Model(), Tokenizer(), "cpu", [4])
    assert backend.generate_task([1, 2], 71) == [3, 4]


def test_gguf_greedy_rollout_resets_and_stops_without_extra_eval():
    class Model:
        def reset(self):
            self.calls = []
            self.n_tokens = 0
            self.scores = np.zeros((5, 5))

        def eval(self, ids):
            self.calls.append(ids)
            self.n_tokens += len(ids)
            self.scores[self.n_tokens - 1, 3 if len(self.calls) == 1 else 4] = 1

    backend = object.__new__(runner.GGUFGenerator)
    backend.model, backend.stop_ids = Model(), [4]
    assert backend.generate_task([1, 2], 10) == [3, 4]
    assert backend.model.calls == [[1, 2], [3]]
    assert backend.generate_task([1, 2], 1) == [3]
    assert backend.model.calls == [[1, 2]]


def test_freeze_only_all_registered_recipes_not_lucky_seed(monkeypatch):
    monkeypatch.setattr(runner, "preparation_binding", lambda *a: ({"recipe": "same"}, {}))
    with pytest.raises(ValueError, match="register"):
        runner.freeze_identity(128, ["source_fp16", "b5_v8_s1"], ".", ".", Scorer())


def test_wilson_finite_and_bounded():
    assert runner.wilson(0, 0) is None
    for correct in (0, 10, 20):
        lo, hi = runner.wilson(correct, 20)
        assert -1e-15 <= lo <= correct / 20 <= hi <= 1 + 1e-15


def test_runtime_reports_actual_tf32_flags(monkeypatch):
    monkeypatch.setattr(runner.fresh, "fresh_runtime", lambda: {"tf32": False})
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    assert runner.runtime()["tf32"] and runner.runtime()["cudnn_tf32"]
    runner.configure_runtime()
    assert not runner.runtime()["tf32"] and not runner.runtime()["cudnn_tf32"]
    assert runner.runtime()["float32_matmul_precision"] == "highest"


def test_notebook_reproducible_valid_compilable_and_safe():
    notebook = build_notebook()
    nbformat.validate(notebook)
    assert nbformat.read(OUTPUT, as_version=4) == notebook
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "public-task-cell", "exec")
            assert not cell.outputs and cell.execution_count is None
    text = "\n".join(c.source for c in notebook.cells)
    assert "SMOKE_RUN = True" in text and "run_live" in text
    assert "run_qwen35_packed_validation.py" not in text  # Never prepare/quantize.
    assert "manifest[\"identity\"][\"labels\"]" in text
    assert "No model output/program is executed" in text


def test_changed_scorer_cache_fails_before_import(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.importlib.metadata, "version", lambda p: suite.SCORER_PACKAGES[p])
    with pytest.raises((FileNotFoundError, ValueError)):
        suite.scorer_identity(tmp_path)


def test_archive_rejects_overlap_and_symlinks(tmp_path):
    from scripts.archive_fresh_quality import archive

    with pytest.raises(ValueError, match="overlap"):
        archive(tmp_path, tmp_path / "nested")
    source = tmp_path / "original"
    source.mkdir()
    (source / "link.json").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        archive(source, tmp_path / "copy")
