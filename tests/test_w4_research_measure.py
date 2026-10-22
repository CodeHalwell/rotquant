"""Small, offline regressions for the frozen W4 comparison measurements."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import w4_research_measure as research


class FakeHF(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache):
        self.inputs = input_ids
        assert not torch.is_grad_enabled() and not use_cache
        assert torch.all(attention_mask == 1)
        logits = torch.nn.functional.one_hot(input_ids % 5, 5).float()
        return SimpleNamespace(logits=logits)


class FakeNative:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def evaluate(self, ids, reset=False, selected=1):
        self.calls.append((list(map(int, ids)), reset, selected))
        if len(self.calls) == self.fail_after:
            raise RuntimeError("fake backend interruption")
        logits = np.eye(5, dtype=np.float32)[(np.asarray(ids) + 1) % 5]
        return logits[-1] if selected == 1 else logits[-selected:]


class FakeMemory:
    def report(self):
        return {"sampled_peak_process_vram_mib": None, "samples": 0}


class FakeTokenizer:
    def get_vocab(self):
        return {str(value): value for value in range(5)}

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return "#### 2" if 2 in ids else "no answer"


def task(identifier="gsm0", prompt=(1,), cap=3):
    return {"id": identifier, "benchmark": "gsm8k", "answer": "2",
            "input_ids": list(prompt), "max_new_tokens": cap}


def test_hf_teacher_returns_all_predictor_positions_without_gradients():
    model = FakeHF()
    result = research.HFTeacher(model, "cpu").predict([1, 2, 3, 4])
    assert result.shape == (3, 5) and result.dtype == np.float32
    assert model.inputs.tolist() == [[1, 2, 3]]
    assert not model.training


def test_native_predict_chunks_each_position_and_resets_only_first_chunk():
    model = FakeNative()
    ids = np.arange(34)
    result = research.NativeBackend(model, {0}).predict(ids)
    assert result.shape == (33, 5) and result.dtype == np.float32
    np.testing.assert_array_equal(result.argmax(-1), (ids[:-1] + 1) % 5)
    assert [(len(ids), reset, count) for ids, reset, count in model.calls] == [
        (16, True, 16), (16, False, 16), (1, False, 1)]


def test_native_generation_includes_eos_and_obeys_token_cap():
    model = FakeNative()
    backend = research.NativeBackend(model, {0})
    assert backend.generate([2], 8) == [3, 4, 0]
    assert backend.generate([1], 2) == [2, 3]
    assert model.calls[0] == ([2], True, 1)
    assert model.calls[3] == ([1], True, 1)


def test_quality_loads_full_logits_records_inputs_and_parity(tmp_path):
    teacher_dir = tmp_path / "teacher"
    parity_dir = tmp_path / "packed"
    teacher_dir.mkdir()
    parity_dir.mkdir()
    backend = research.NativeBackend(FakeNative(), {0})
    documents = [{"id": "a", "domain": "text", "source_row": 7, "input_ids": [0, 1, 2]},
                 {"id": "b", "domain": "text", "source_row": 7, "input_ids": [2, 3, 4, 0]}]
    for document in documents:
        values = backend.predict(document["input_ids"])
        np.save(teacher_dir / f"{document['id']}.npy", values.astype(np.float16))
        np.save(parity_dir / f"{document['id']}.npy", values)
    report = research.evaluate_quality(tmp_path / "quality", backend, documents, teacher_dir,
                                       parity_dir=parity_dir, bootstrap_draws=50)
    assert report["summary"]["tokens"] == 5
    assert report["summary"]["documents"] == 1  # Same source row is one bootstrap cluster.
    assert report["summary"]["records"] == 2
    assert report["summary"]["mean_teacher_kl"] == pytest.approx(0, abs=1e-7)
    assert report["summary"]["top1_agreement"] == 1
    assert report["summary"]["ppl_interval"] is None
    assert report["rows"][0]["packed_parity"]["max_absolute_error"] == 0
    assert report["rows"][0]["input_ids"] == [0, 1, 2]
    assert (tmp_path / "quality" / "quality_rows" / "a.json").exists()
    assert json.loads((tmp_path / "quality" / "quality.json").read_text()) == report


def test_quality_bootstrap_is_token_weighted_and_reproducible():
    rows = [{"id": str(i), "source_row": i, "domain": "text", "tokens": count,
             "mean_teacher_kl": value, "candidate_nll": value, "source_nll": 0.5,
             "top1_agreement": 1} for i, count, value in [(0, 1, 1.), (1, 3, 3.)]]
    summary = research.quality_summary(rows, bootstrap_draws=100)
    assert summary["mean_teacher_kl"] == 2.5
    assert summary["ppl"] == pytest.approx(np.exp(2.5))
    assert summary["confidence_intervals"]["candidate_nll"] == [1., 3.]
    assert summary == research.quality_summary(rows, bootstrap_draws=100)


def test_quality_wrong_teacher_positions_fail_closed(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((1, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="positions differ"):
        research.evaluate_quality(tmp_path / "output", research.NativeBackend(FakeNative(), {0}),
                                  [{"id": "a", "domain": "text", "input_ids": [1, 2, 3]}], tmp_path)


def test_public_tasks_capture_eos_truncation_and_real_scorer(tmp_path):
    backend = research.NativeBackend(FakeNative(), {3})
    report = research.evaluate_public_tasks(tmp_path, backend,
                                           [task(), task("gsm1", cap=1)], FakeTokenizer())
    first, second = report["rows"]
    assert first["continuation"] == [2, 3] and first["stopped_at_eos"]
    assert first["task_success"] and first["output_text"] == "#### 2"
    assert second["truncated"] and not second["task_success"]
    assert second["checker_success_before_truncation_guard"]
    assert report["summary"] == {"gsm8k": {"items": 2, "correct": 1, "accuracy": .5, "truncated": 1}}
    assert (tmp_path / "task_rows" / "gsm0.json").exists()


def test_public_tasks_use_supplied_ifeval_oracle(tmp_path):
    oracle = {"key": 123}
    seen = []

    def score(row, text):
        seen.append((row, text))
        return {"prompt_strict": True, "prompt_loose": True}

    item = {**task(), "benchmark": "ifeval", "oracle": oracle}
    report = research.evaluate_public_tasks(tmp_path, research.NativeBackend(FakeNative(), {3}),
                                           [item], FakeTokenizer(), SimpleNamespace(score=score))
    assert seen == [(oracle, "#### 2")]
    assert report["summary"]["ifeval"]["accuracy"] == 1


def test_paired_no_discordance_retains_conservative_uncertainty():
    rows = [{"id": str(i), "benchmark": "gsm8k", "input_hash": "frozen", "task_success": True}
            for i in range(8)]
    report = research.paired_task_deltas(rows, rows, draws=100)["gsm8k"]
    assert report["bootstrap_interval"] == [0., 0.]
    assert report["hoeffding_interval"][0] < -.9
    assert report["hoeffding_interval"][1] > .9
    assert report["mean_accuracy_delta"] == 0
    assert not any("noninferiority" in key for key in report)


def test_paired_tasks_reject_different_frozen_prompts():
    row = {"id": "a", "benchmark": "gsm8k", "input_hash": "a", "task_success": False}
    with pytest.raises(ValueError, match="frozen prompt"):
        research.paired_task_deltas([row], [{**row, "input_hash": "b"}])


def test_paired_tasks_reject_different_generation_caps():
    row = {"id": "a", "benchmark": "gsm8k", "input_hash": "a", "task_success": False,
           "max_new_tokens": 8}
    with pytest.raises(ValueError, match="token cap"):
        research.paired_task_deltas([row], [{**row, "max_new_tokens": 16}])


def test_unknown_generated_token_ids_fail_closed(tmp_path):
    backend = SimpleNamespace(generate=lambda ids, cap: [999, 0], stop_ids={0})
    with pytest.raises(ValueError, match="absent from the frozen tokenizer"):
        research.evaluate_public_tasks(tmp_path, backend, [task()], FakeTokenizer())


def test_native_speed_uses_frozen_prompt_and_decode_excludes_warmup(tmp_path):
    model = FakeNative()
    report = research.native_speeds(model, {"3": {"prompt_ids": [1, 2, 3], "decode_ids": [2, 4]}},
                                    2, tmp_path / "speed.json", memory=FakeMemory())
    assert report["completed"] and len(report["rows"]) == 3
    assert report["summary"]["3"]["measured_repetitions"] == 2
    assert report["summary"]["3"]["decode_tokens_per_second"] > 0
    assert report["frozen_inputs"]["3"]["prompt_sha256"] == research.input_hash([1, 2, 3])
    assert [row["warmup"] for row in report["rows"]] == [True, False, False]
    assert all(row["decode_input_ids"] == [2, 4] for row in report["rows"])
    assert [ids for ids, reset, _ in model.calls if reset] == [[1, 2, 3]] * 3
    assert json.loads((tmp_path / "speed.json").read_text()) == report


def test_native_speed_never_fabricates_a_short_prompt(tmp_path):
    model = FakeNative()
    with pytest.raises(ValueError, match="no repetition or resizing"):
        research.native_speeds(model, {8: {"prompt_ids": [1, 2], "decode_ids": [3]}},
                               1, tmp_path / "speed.json", memory=FakeMemory())
    assert not model.calls


def test_native_speed_preserves_all_contexts(tmp_path):
    report = research.native_speeds(FakeNative(),
                                    {2: {"prompt_ids": [1, 2], "decode_ids": [3]},
                                     3: {"prompt_ids": [1, 2, 3], "decode_ids": [4]}},
                                    1, tmp_path / "speed.json", memory=FakeMemory())
    assert [row["input_tokens"] for row in report["rows"]] == [2, 2, 3, 3]
    assert set(report["summary"]) == {"2", "3"}


def test_native_speed_preserves_partial_rows_on_interruption(tmp_path):
    path = tmp_path / "speed.json"
    with pytest.raises(RuntimeError, match="interruption"):
        research.native_speeds(FakeNative(fail_after=3),
                               {2: {"prompt_ids": [1, 2], "decode_ids": [3, 4]}}, 1,
                               path, memory=FakeMemory())
    report = json.loads(path.read_text())
    assert not report["completed"]
    assert report["rows"][0]["decode_steps"] == 1
    assert report["rows"][0]["decode_input_ids"] == [3]
