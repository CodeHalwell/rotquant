import copy
import json

import pytest
import torch
from torch import nn

from scripts.native_gpu_workflow import fingerprint
from scripts.ternary_chat_data import (
    chat_batches,
    freeze_chat_dataset,
    load_prompt_jsonl,
    partition_prompts,
    prompt_identity,
    validate_chat_manifest,
)


class Tokenizer:
    chat_template = "source-template-v1"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(x) + 3 for x in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        return [1] + self.encode(messages[-1]["content"]) + [2]


class Teacher(nn.Module):
    def __init__(self, complete=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.complete = complete

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, use_cache,
                 eos_token_id, pad_token_id):
        assert not self.training and not do_sample and use_cache and max_new_tokens > 0
        response = torch.tensor([[10, 11, eos_token_id[0] if self.complete else 12]])
        return torch.cat((input_ids, response), -1)


def prompts():
    return [{"id": str(i), "prompt": f"Fresh private prompt number {i}.", "split": split}
            for i, split in enumerate(("train", "validation", "selection") * 3)]


def frozen(**kwargs):
    return freeze_chat_dataset(Tokenizer(), kwargs.pop("model", Teacher()),
                                kwargs.pop("prompts", prompts()), model_revision="a" * 40,
                                tokenizer_revision="b" * 40, stop_ids=(99,), **kwargs)


def test_complete_turns_have_source_template_disjoint_hashes_and_eos_supervision(tmp_path):
    model = Teacher()
    output = tmp_path / "chat.json"
    manifest = frozen(model=model, destination=output)
    assert model.training
    assert json.loads(output.read_text()) == manifest
    assert set(manifest["partition_hashes"]) == {"train", "validation", "selection"}
    for split, rows in manifest["records"].items():
        assert [x["context_length"] for x in rows] == [512, 1024, 2048]
        for row in rows:
            assert row["response_mask"][:row["prompt_length"]] == [0] * row["prompt_length"]
            assert row["response_mask"][-3:] == [1, 1, 1]
            assert row["input_ids"][-1] == 99 and row["split"] == split
    batch = chat_batches(manifest, "train", pad_to_context=True)[0]
    assert batch["input_ids"].shape == (1, 512)
    assert batch["response_mask"].sum() == 3
    assert batch["attention_mask"].sum() < 512
    assert not batch["response_mask"][0, -1]


def test_incomplete_teacher_turn_rejected_and_training_mode_restored():
    model = Teacher(complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        frozen(model=model)
    assert model.training


def test_partition_and_frozen_public_prompt_exclusion_fail_closed():
    source = prompts()
    with pytest.raises(ValueError, match="overlap"):
        partition_prompts(source, (prompt_identity(source[0]),))
    duplicate = copy.deepcopy(source)
    duplicate[2]["prompt"] = duplicate[0]["prompt"]
    with pytest.raises(ValueError, match="overlap"):
        partition_prompts(duplicate)
    ids = Tokenizer().apply_chat_template([{"role": "user", "content": source[0]["prompt"]}],
                                          True, True, False)
    with pytest.raises(ValueError, match="public evaluation"):
        frozen(forbidden_token_hashes=(fingerprint(ids),))


def test_rehashed_manifest_still_rejects_cross_split_duplicates_and_dropped_eos():
    manifest = frozen()
    row = manifest["records"]["selection"][0]
    row["prompt_hash"] = manifest["records"]["train"][0]["prompt_hash"]
    manifest["partition_hashes"]["selection"] = fingerprint(manifest["records"]["selection"])
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="overlap"):
        validate_chat_manifest(manifest)
    manifest = frozen()
    row = manifest["records"]["train"][0]
    row["input_ids"][-1] = 98
    row["tokens_sha256"] = fingerprint(row["input_ids"])
    manifest["partition_hashes"]["train"] = fingerprint(manifest["records"]["train"])
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="stop token"):
        validate_chat_manifest(manifest)


def test_prompt_jsonl_rejects_public_labels(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"id": "1", "prompt": "question", "answer": "label"}))
    with pytest.raises(ValueError, match="evaluation labels"):
        load_prompt_jsonl(path)


def test_unpinned_source_template_rejected_before_generation():
    with pytest.raises(ValueError, match="pinned"):
        freeze_chat_dataset(Tokenizer(), Teacher(), prompts(), model_revision="main",
                            tokenizer_revision="b" * 40, stop_ids=(99,))
