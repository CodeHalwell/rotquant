"""Offline oracles, token alignment, provenance and resumability checks."""

import copy
import json
from collections import Counter
from types import SimpleNamespace
from typing import ClassVar

import nbformat
import numpy as np
import pytest
import torch

from rotquant.eval.fresh_tasks import fingerprint, paired_family_interval, score_task, task_suite
from scripts import run_qwen35_fresh_eval as runner
from scripts.build_qwen35_fresh_eval_notebook import OUTPUT, build_notebook


def test_authored_suite_is_deterministic_unique_and_inert():
    tasks = task_suite()
    assert len(tasks) == 96
    assert Counter(t["domain"] for t in tasks) == dict.fromkeys(
        ["multilingual", "code_trace", "structured", "tool_selection"], 24
    )
    assert fingerprint(tasks) == fingerprint(task_suite())
    assert len({t["language"] for t in tasks if "language" in t}) == 6
    for task in tasks:
        assert score_task(json.dumps(task["expected"]), task["expected"])["task_success"]
    assert len({t["family"] for t in tasks}) == 13


@pytest.mark.parametrize(
    "answer",
    [
        '{"answer":true}',
        '{"answer":1.0}',
        '{"answer":1,"extra":2}',
        '{"answer":1,"answer":1}',
        '```json\n{"answer":1}\n```',
        '{"answer":NaN}',
        'reasoning {"answer":1}',
        "[1]",
        '__import__("os")',
    ],
)
def test_oracle_rejects_types_extra_keys_duplicates_and_code(answer):
    assert not score_task(answer, {"answer": 1})["task_success"]


def test_truncated_but_valid_json_is_not_a_success():
    result = score_task('{"answer":1}', {"answer": 1}, truncated=True)
    assert result == {"json_valid": True, "task_success": False, "truncated": True}


def test_family_interval_pairs_and_does_not_inflate_template_evidence():
    left = [
        {"id": str(i), "input_hash": str(i), "family": str(i // 2), "tokens": 4, "value": 1.0}
        for i in range(6)
    ]
    right = [{**r, "value": 1.5} for r in left]
    result = paired_family_interval(left, right, "value", draws=100)
    assert result["families"] == 3 and result["prompts"] == 6
    assert result["ci95"] == [0.5, 0.5]
    with pytest.raises(ValueError, match="mismatch"):
        paired_family_interval(left, [{**r, "tokens": 3} for r in right], "value")
    with pytest.raises(ValueError, match="duplicate"):
        paired_family_interval(left + left, right + right, "value")
    one_family = [{**r, "family": "same"} for r in left]
    assert paired_family_interval(one_family, one_family, "value")["ci95"] is None


class Tokenizer:
    chat_template = "frozen-template"
    eos_token_id = 4
    special_tokens_map: ClassVar[dict] = {"eos_token": "stop"}

    def __len__(self):
        return 5

    def get_vocab(self):
        return {str(i): i for i in range(5)}

    def decode(self, ids, **kwargs):
        return '{"answer":7}'


class Backend:
    tokenizer = Tokenizer()
    stop_ids: ClassVar[list] = [4]

    def __init__(self, offset=0):
        self.offset = offset
        self.seen = []

    def generate(self, ids):
        return [3, 4]

    def predict(self, ids):
        self.seen.append(ids)
        logits = np.arange((len(ids) - 1) * 5, dtype=np.float32).reshape(-1, 5) / 10
        logits[:, 0] += self.offset
        return logits


def test_missing_generation_file_uses_nested_model_eos_and_chat_eos(monkeypatch):
    import huggingface_hub
    import transformers
    from huggingface_hub.errors import EntryNotFoundError

    calls = []

    def missing(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        raise EntryNotFoundError("generation_config.json is absent at the pinned revision")

    def model_config(repo, **kwargs):
        calls.append((repo, "config.json", kwargs))
        # The actual model has EOS only in its nested text config.
        return transformers.Qwen3_5Config(text_config={"eos_token_id": 3})

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", model_config)
    assert runner.stop_ids_for_source(Tokenizer()) == [3, 4]
    assert calls == [
        (runner.MODEL_ID, name, {"revision": runner.MODEL_REVISION})
        for name in ("generation_config.json", "config.json")
    ]


@pytest.mark.parametrize("eos, expected", [(3, [3, 4]), ([3, 4, 3], [3, 4]), (4, [4])])
def test_generation_file_takes_precedence_and_deduplicates_eos(
    tmp_path, monkeypatch, eos, expected
):
    import huggingface_hub
    import transformers

    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"eos_token_id": eos}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))

    def unexpected(*a, **k):
        pytest.fail("existing generation config must not be replaced with model defaults")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unexpected)
    assert runner.stop_ids_for_source(Tokenizer()) == expected


@pytest.mark.parametrize("eos", [None, [], True, -1, 3.0, "3", [3, False], [99]])
def test_invalid_or_unmapped_source_eos_is_rejected(tmp_path, monkeypatch, eos):
    import huggingface_hub

    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"eos_token_id": eos}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    with pytest.raises(ValueError):
        runner.stop_ids_for_source(Tokenizer())


@pytest.mark.parametrize("failure", ["offline", "permission", "network", "json"])
def test_stop_resolution_does_not_hide_unrelated_failures(tmp_path, monkeypatch, failure):
    import huggingface_hub
    import transformers
    from huggingface_hub.errors import LocalEntryNotFoundError

    path = tmp_path / "generation_config.json"
    path.write_text("{broken json")
    error = {
        "offline": LocalEntryNotFoundError("not cached; cannot verify absence"),
        "permission": PermissionError("denied"),
        "network": ConnectionError("unavailable"),
    }.get(failure)

    def download(*a, **k):
        if error is not None:
            raise error
        return str(path)

    def unexpected(*a, **k):
        pytest.fail("only a confirmed missing Hub entry permits fallback")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unexpected)
    with pytest.raises(json.JSONDecodeError if error is None else type(error)):
        runner.stop_ids_for_source(Tokenizer())


def save_reference(tmp_path, label, item, manifest, record, logits):
    path = runner.paths_for(tmp_path, label, item)
    runner._atomic_npz(path.with_suffix(".npz"), logits=logits)
    runner.save_record(
        path, {**record, "reference_sha256": runner.file_digest(path.with_suffix(".npz"))}
    )


def test_teacher_forcing_slices_task_prefix_and_common_context_for_bridge(tmp_path):
    item = {
        "id": "task",
        "domain": "code",
        "family": "f",
        "kind": "task",
        "input_hash": "h",
        "input_ids": [0, 1, 2],
        "expected": {"answer": 7},
    }
    manifest = {"fingerprint": "x", "tokenizer": {"size": 5}}
    backend = Backend()
    source, logits = runner.score_one(backend, item, manifest, tmp_path, "source_fp16")
    assert backend.seen == [[0, 1, 2, 3, 4]]
    assert logits.shape == (2, 5) and source["tokens"] == 2
    assert source["task_success"] and source["exact_trajectory"]
    assert source["mean_teacher_kl"] == 0
    save_reference(tmp_path, "source_fp16", item, manifest, source, logits)
    bridge, bridge_logits = runner.score_one(
        Backend(0.2), item, manifest, tmp_path, "gguf_bf16_bridge"
    )
    save_reference(tmp_path, "gguf_bf16_bridge", item, manifest, bridge, bridge_logits)
    candidate, _ = runner.score_one(Backend(0.2), item, manifest, tmp_path, "unsloth_ud_q4")
    assert candidate["mean_teacher_kl"] > 0
    assert candidate["same_engine_bf16"]["mean_teacher_kl"] == 0
    path = runner.paths_for(tmp_path, "source_fp16", item).with_suffix(".npz")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        runner.load_reference(tmp_path, "source_fp16", item, manifest)


def test_text_scoring_uses_every_shifted_position(tmp_path):
    item = {
        "id": "text",
        "domain": "c4",
        "family": "row1",
        "kind": "text",
        "input_hash": "h",
        "input_ids": [0, 1, 2, 3],
    }
    manifest = {"fingerprint": "x", "tokenizer": {"size": 5}}
    record, logits = runner.score_one(Backend(), item, manifest, tmp_path, "source_fp16")
    assert record["tokens"] == 3 and logits.shape == (3, 5)
    assert "task_success" not in record
    with pytest.raises(ValueError, match="full vocabulary"):
        runner.score_one(
            Backend(), item, {"fingerprint": "x", "tokenizer": {"size": 6}}, tmp_path, "source_fp16"
        )


def test_llama_axis_mismatch_is_not_silently_renormalized():
    model = SimpleNamespace(
        n_vocab=lambda: 5, _model=SimpleNamespace(token_get_text=lambda i: str(i))
    )
    runner.LlamaBackend(model, Tokenizer(), [4])
    model._model.token_get_text = lambda i: "wrong"
    with pytest.raises(ValueError, match="token-axis"):
        runner.LlamaBackend(model, Tokenizer(), [4])
    model.n_vocab = lambda: 6
    with pytest.raises(ValueError, match="vocabulary sizes"):
        runner.LlamaBackend(model, Tokenizer(), [4])


def test_full_padded_output_axis_is_preserved_and_padding_names_verified():
    model = SimpleNamespace(
        n_vocab=lambda: 7,
        _model=SimpleNamespace(token_get_text=lambda i: str(i) if i < 5 else f"[PAD{i}]"),
    )
    runner.LlamaBackend(model, Tokenizer(), [4], output_size=7)
    identity = runner.tokenizer_identity(Tokenizer(), 7)
    assert identity["size"] == 7 and identity["tokenizer_size"] == 5
    model._model.token_get_text = lambda i: str(i)
    with pytest.raises(ValueError, match="padded output slot"):
        runner.LlamaBackend(model, Tokenizer(), [4], output_size=7)


def test_tiny_hf_backend_prediction_alignment_and_generation():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(20)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=5,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            eos_token_id=4,
        )
    ).eval()
    backend = runner.HFBackend(model, Tokenizer(), "cpu", [4])
    ids = [0, 1, 2]
    actual = backend.predict(ids)
    with torch.no_grad():
        expected = model(torch.tensor([ids]), use_cache=False).logits[0, :-1].numpy()
    np.testing.assert_allclose(actual, expected)
    generated = backend.generate(ids)
    assert 1 <= len(generated) <= 128
    assert generated[-1] == 4 or len(generated) == 128


def test_summary_keeps_missing_arms_visible_and_checks_pairing(tmp_path):
    item = {
        "id": "c4-000",
        "domain": "c4",
        "family": "row1",
        "kind": "text",
        "input_hash": "h",
        "input_ids": [0, 1, 2],
    }
    manifest = {"fingerprint": "x", "tokenizer": {"size": 5}, "items": [item]}
    result = runner.summarize(tmp_path, manifest, ["source_fp16", "b5_v6_s0"])
    assert not result["complete"] and len(result["missing"]) == 2
    identity = {"manifest": "x"}
    runner.save_record(
        tmp_path / "runs/source_fp16/complete.json",
        {"identity": identity, "manifest": "x", "complete": True},
    )
    record, _ = runner.score_one(Backend(), item, manifest, tmp_path, "source_fp16")
    record["collection"] = fingerprint(identity)
    runner.save_record(runner.paths_for(tmp_path, "source_fp16", item), record)
    result = runner.summarize(tmp_path, manifest, ["source_fp16", "b5_v6_s0"])
    assert result["missing"] == ["b5_v6_s0"]
    record["input_hash"] = "wrong"
    runner.save_record(runner.paths_for(tmp_path, "source_fp16", item), record)
    with pytest.raises(ValueError, match="pairing"):
        runner.summarize(tmp_path, manifest, ["source_fp16"])


def test_frozen_manifest_rejects_corruption_and_overlap_paths(tmp_path):
    items = [
        {
            "id": str(i),
            "kind": "text" if i < 24 else "task",
            "input_ids": [i, i + 1],
            "input_hash": runner._input_hash(np.array([i, i + 1])),
        }
        for i in range(120)
    ]
    manifest = {"items": items}
    manifest["fingerprint"] = fingerprint(manifest)
    runner.validate_manifest(manifest)
    changed = copy.deepcopy(manifest)
    changed["items"][0]["input_ids"][0] = 200
    with pytest.raises(ValueError, match="fingerprint"):
        runner.validate_manifest(changed)
    changed["fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="token hash"):
        runner.validate_manifest(changed)
    with pytest.raises(ValueError, match="overlap"):
        runner.separate(tmp_path, tmp_path / "original")


def test_notebook_matches_builder_and_all_python_cells_compile():
    notebook = build_notebook()
    nbformat.validate(notebook)
    assert nbformat.read(OUTPUT, as_version=4) == notebook
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    text = "\n".join(c.source for c in notebook.cells)
    assert "--prepare-only" in text and "RUN_REPLICATION = True" in text
    assert "run_live" in text and 'expected = ["source_fp16", "b5_v6_s0", "b5_v8_s0"]' in text


def test_collection_resumes_prompts_and_recovers_missing_completion(tmp_path, monkeypatch):
    tokenizer = Tokenizer()
    items = [
        {
            "id": f"c4-{i}",
            "domain": "c4",
            "family": f"row{i}",
            "kind": "text",
            "input_hash": str(i),
            "input_ids": [0, 1, 2],
        }
        for i in range(2)
    ]
    manifest = {
        "fingerprint": "x",
        "tokenizer": runner.tokenizer_identity(tokenizer),
        "items": items,
    }
    calls = []
    monkeypatch.setattr(runner, "load_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(runner, "stop_ids_for_source", lambda *a: [4])
    monkeypatch.setattr(runner, "fresh_runtime", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(runner, "source_identity", lambda: {"source": "fixed"})
    monkeypatch.setattr(
        runner.experiment, "load_hf_model", lambda *a: (calls.append("load"), tokenizer, "fixture")
    )
    monkeypatch.setattr(runner, "HFBackend", lambda *a: Backend())
    runner.run_collection(tmp_path, manifest, "source_fp16", device="cpu")
    assert calls == ["load"]
    original = runner.paths_for(tmp_path, "source_fp16", items[0]).read_bytes()
    runner.run_collection(tmp_path, manifest, "source_fp16", device="cpu")
    assert calls == ["load"]
    # Simulate interruption after the last prompt but before the final marker.
    marker = tmp_path / "runs/source_fp16/complete.json"
    marker.unlink()
    runner.run_collection(tmp_path, manifest, "source_fp16", device="cpu")
    assert marker.exists() and calls == ["load", "load"]
    assert runner.paths_for(tmp_path, "source_fp16", items[0]).read_bytes() == original
    monkeypatch.setattr(runner, "fresh_runtime", lambda: {"runtime": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        runner.run_collection(tmp_path, manifest, "source_fp16", device="cpu")


def test_archived_revalidation_checksums_and_preserved_failure(tmp_path):
    import shutil

    from scripts.archive_packed_revalidation import archive

    original = runner.ROOT / "research/results/raw/qwen35_packed_revalidation_89d25f3"
    index = json.loads((original / "evidence_index.json").read_text())
    assert index["checksum_pairs"] == 14 and len(index["files"]) == 37
    for name, metadata in index["files"].items():
        assert runner.file_digest(original / name) == metadata["sha256"]
        assert (original / name).stat().st_size == metadata["bytes"]
    old = runner.read_record(original / "original/b5_v6_s0/validation.json")
    new = runner.read_record(original / "b5_v6_s0/validation.json")
    assert not old["passed"] and new["passed"]
    source = tmp_path / "source"
    for name in index["files"]:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, source / name)
    saved = archive(source, tmp_path / "destination")
    assert saved["checksum_pairs"] == 14
    assert archive(source, tmp_path / "destination") == saved
    (tmp_path / "destination/summary.json").write_text("{}")
    with pytest.raises(ValueError, match="collision"):
        archive(source, tmp_path / "destination")


def test_freeze_excludes_archived_rows_pins_padding_and_resumes_without_reselection(
    tmp_path, monkeypatch
):
    import transformers

    class FreezeTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"] + "[assistant]"

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[int(c, 16) % 5 for c in fingerprint(text)])

    seen = []

    def batches(tokenizer, count, length, device, **kwargs):
        seen.append(kwargs)
        return [
            runner.experiment.TokenBatch(
                torch.tensor([([int(c, 16) % 5 for c in fingerprint(i)] * 8)[:512]]), 30_000 + i
            )
            for i in range(count)
        ]

    monkeypatch.setattr(runner, "load_tokenizer", FreezeTokenizer)
    monkeypatch.setattr(runner.experiment, "build_calib_loader", batches)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(text_config=SimpleNamespace(vocab_size=7)),
    )
    source = runner.ROOT / "research/results/raw/qwen35_packed_revalidation_89d25f3/original"
    manifest = runner.freeze(tmp_path, source)
    assert len(manifest["items"]) == 120
    assert len(seen) == 1 and len(seen[0]["exclude_source_rows"]) >= 128
    assert manifest["tokenizer"]["size"] == 7 and manifest["tokenizer"]["tokenizer_size"] == 5
    assert runner.freeze(tmp_path, source) == manifest and len(seen) == 1
    monkeypatch.setitem(runner.SETTINGS, "c4_skip", 17000)
    with pytest.raises(ValueError, match="protocol changed"):
        runner.freeze(tmp_path, source)
