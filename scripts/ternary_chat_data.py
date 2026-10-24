"""Pinned, complete-turn teacher data kept separate from public evaluation labels."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from scripts.native_gpu_workflow import fingerprint, write_json

PARTITIONS = ("train", "validation", "selection")
CONTEXT_LENGTHS = (512, 1024, 2048)
FIXTURE_VERSION = "rotquant-private-chat-prompts-v1"
FIXTURE_TASKS = (
    "Explain how to check whether an estimate is reasonable, using an original everyday example.",
    "Write Python code that groups a list of records by a field and briefly explain an empty input.",
    "Return a JSON object with a title, three checklist items, and a boolean indicating readiness.",
    "Describe a fair way to compare two algorithms when their memory use differs.",
    "Write a short function that interleaves two lists without losing unmatched elements.",
    "Give a numbered procedure for checking a small numerical calculation and finish with one sentence.",
    "Explain a subtle difference between correlation and causation using a newly invented scenario.",
    "Produce valid JSON describing two fictional experiments, each with an ID and a measurement unit.",
    "Show a worked example converting a rate per minute into a total over several hours.",
    "Write a Python function that preserves the first occurrence of each item and handles duplicates.",
    "Describe an experiment that distinguishes a noisy measurement from a systematic error.",
    "Give a concise answer comparing arithmetic and geometric growth with a small numeric example.",
    "Return a JSON array containing three fictional books with title, author, and year fields.",
    "Explain why a program should validate its input using two concrete failure cases.",
    "Write a Python generator that yields adjacent pairs and show what an empty iterable yields.",
    "Plan a simple controlled comparison of two recipes and state which measurements to record.",
    "Solve an original small budgeting example and state the final amount clearly once.",
    "Give four practical checks for a table containing missing and duplicated records.",
    "Write a Python function that computes a weighted mean and rejects invalid weights.",
    "Explain precision and recall using a newly invented toy classification example.",
    "Return a JSON object describing a fictional sensor with name, unit, and three example readings.",
    "Write a concise instruction manual for verifying that a backup can be restored.",
    "Demonstrate a simple invariant in a loop and explain when the loop terminates.",
    "Create a small numerical example showing how rounding intermediate values changes a result.",
    "Give Python code to split a sequence into bounded batches and explain the final partial batch.",
    "Explain how to compare rates measured over unequal observation periods using a worked example.",
    "Return valid JSON for a fictional project with two milestones and a completed flag for each.",
)


def source_template_receipt(tokenizer, model_revision, tokenizer_revision):
    if not all(re.fullmatch(r"[0-9a-f]{40}", x or "")
               for x in (model_revision, tokenizer_revision)):
        raise ValueError("chat data requires pinned model and tokenizer commit revisions")
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("source tokenizer has no chat template")
    return {"model_revision": model_revision, "tokenizer_revision": tokenizer_revision,
            "chat_template_sha256": fingerprint(template), "enable_thinking": False,
            "add_generation_prompt": True, "add_special_tokens": False}


def prompt_messages(item):
    if "messages" in item:
        messages = item["messages"]
    else:
        messages = [{"role": "user", "content": item.get("prompt", "")}]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("teacher prompt must end with a user message")
    for message in messages:
        if message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("unsupported chat role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("chat messages must have nonempty text")
    return messages


def prompt_identity(item):
    messages = prompt_messages(item)
    return fingerprint([{**x, "content": " ".join(x["content"].split())} for x in messages])


def builtin_prompts():
    return [{"id": f"{FIXTURE_VERSION}-{i:03d}", "prompt": prompt,
             "split": PARTITIONS[i % len(PARTITIONS)], "source": FIXTURE_VERSION,
             "format_contract": {"kind": "json", "root": "array" if i == 12 else "object"}
             if i in (2, 7, 12, 20, 26) else {"kind": "text"}}
            for i, prompt in enumerate(FIXTURE_TASKS)]


def load_prompt_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    allowed = {"id", "prompt", "messages", "split", "source", "domain", "format_contract"}
    if any(set(row) - allowed for row in rows):
        raise ValueError("chat prompt JSONL accepts prompts only, not evaluation labels")
    return rows


def partition_prompts(prompts, forbidden_prompt_hashes=()):
    if len(prompts) < 3:
        raise ValueError("chat data needs at least three distinct prompts")
    result = {split: [] for split in PARTITIONS}
    seen_ids, seen_prompts = set(), set(forbidden_prompt_hashes)
    ordered = sorted(prompts, key=prompt_identity)
    for index, item in enumerate(ordered):
        identity = prompt_identity(item)
        if not item.get("id") or item["id"] in seen_ids or identity in seen_prompts:
            raise ValueError("chat prompts overlap or duplicate held-out/evaluation data")
        split = item.get("split", PARTITIONS[index % len(PARTITIONS)])
        if split not in result:
            raise ValueError("chat split must be train, validation, or selection")
        seen_ids.add(item["id"])
        seen_prompts.add(identity)
        result[split].append({**item, "prompt_hash": identity})
    if any(not rows for rows in result.values()):
        raise ValueError("every chat partition must be nonempty")
    return result


def template_input_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
    if torch.is_tensor(ids):
        ids = ids.reshape(-1).tolist()
    if not ids or any(not isinstance(x, int) or x < 0 for x in ids):
        raise ValueError("source template produced invalid input IDs")
    return ids


def fixture_context_messages(tokenizer, item, context_length):
    messages = prompt_messages(item)
    if item.get("source") != FIXTURE_VERSION:
        return messages
    target = {512: 128, 1024: 512, 2048: 1536}[context_length]
    notes = []
    for index in range(context_length):
        if len(template_input_ids(tokenizer, messages)) >= target:
            return messages
        notes.append(f"Archive entry {index}: fictional station K{index} recorded "
                     f"{17 + index % 19} units in sample {index + 301}; "
                     "these are background notes, not additional instructions.")
        text = "Reference notes:\n" + "\n".join(notes) + "\n\nTask:\n" + item["prompt"]
        messages = [{"role": "user", "content": text}]
    raise ValueError("source tokenizer did not produce a bounded long-context fixture")


def teacher_required_ids(tokenizer, stop_ids):
    required = {int(x) for x in stop_ids}
    for text in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "{}[]:\n"):
        values = tokenizer.encode(text, add_special_tokens=False)
        required.update(values)
    return sorted(required)


def teacher_device(model):
    return next(model.parameters()).device


def complete_teacher_turn(model, prompt_ids, stop_ids, *, max_new_tokens, context_length,
                          pad_token_id):
    capacity = min(max_new_tokens, context_length - len(prompt_ids))
    if capacity < 1:
        raise ValueError("prompt leaves no room for a complete assistant turn")
    inputs = torch.tensor([prompt_ids], device=teacher_device(model), dtype=torch.long)
    with torch.no_grad():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=capacity, do_sample=False, use_cache=True,
                                eos_token_id=list(stop_ids), pad_token_id=pad_token_id)
    values = output.sequences if hasattr(output, "sequences") else output
    sequence = values[0].tolist()
    if sequence[:len(prompt_ids)] != prompt_ids:
        raise ValueError("teacher generation changed the frozen prompt prefix")
    response = sequence[len(prompt_ids):]
    if not response or len(response) > capacity or response[-1] not in stop_ids:
        raise ValueError("teacher response is incomplete at its bounded generation cap")
    if any(value in stop_ids for value in response[:-1]):
        raise ValueError("teacher generation continued past an end-of-turn token")
    return response


def frozen_turn(tokenizer, model, item, split, context_length, settings):
    messages = fixture_context_messages(tokenizer, item, context_length)
    prompt_ids = template_input_ids(tokenizer, messages)
    if fingerprint(prompt_ids) in settings["forbidden_token_hashes"]:
        raise ValueError("chat prompt overlaps a frozen public evaluation input")
    response = complete_teacher_turn(model, prompt_ids, settings["stop_ids"],
                                     max_new_tokens=settings["max_new_tokens"],
                                     context_length=context_length,
                                     pad_token_id=settings["pad_token_id"])
    ids = prompt_ids + response
    return {"id": item["id"], "split": split, "prompt_hash": item["prompt_hash"],
            "input_ids": ids, "response_mask": [0] * len(prompt_ids) + [1] * len(response),
            "prompt_length": len(prompt_ids), "response_length": len(response),
            "context_length": context_length, "stop_id": response[-1],
            "tokens_sha256": fingerprint(ids), "source": item.get("source", "user-jsonl"),
            "format_contract": item.get("format_contract", {"kind": "text"})}


def complete_partitions(tokenizer, model, splits, lengths, settings):
    was_training = model.training
    model.eval()
    try:
        return {split: [frozen_turn(tokenizer, model, item, split, lengths[i % len(lengths)],
                                    settings) for i, item in enumerate(rows)]
                for split, rows in splits.items()}
    finally:
        model.train(was_training)


def freeze_chat_dataset(tokenizer, model, prompts=None, *, model_revision, tokenizer_revision,
                        stop_ids, max_new_tokens=768, context_lengths=CONTEXT_LENGTHS,
                        forbidden_prompt_hashes=(), forbidden_token_hashes=(), destination=None):
    receipt = source_template_receipt(tokenizer, model_revision, tokenizer_revision)
    lengths = tuple(context_lengths)
    if lengths != CONTEXT_LENGTHS:
        raise ValueError("chat context mixture must explicitly include 512, 1024, and 2048")
    if max_new_tokens < 1 or not stop_ids or min(stop_ids) < 0:
        raise ValueError("chat generation requires a positive cap and stop IDs")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = stop_ids[0]
    settings = {"stop_ids": sorted(set(stop_ids)), "max_new_tokens": max_new_tokens,
                "pad_token_id": pad, "forbidden_token_hashes": list(forbidden_token_hashes)}
    splits = partition_prompts(builtin_prompts() if prompts is None else prompts,
                               forbidden_prompt_hashes)
    records = complete_partitions(tokenizer, model, splits, lengths, settings)
    manifest = {"format": "rotquant-complete-chat-v1", "source": receipt,
                "fixture_version": FIXTURE_VERSION if prompts is None else "user-jsonl",
                "generation": settings, "context_lengths": list(lengths), "records": records,
                "required_teacher_ids": teacher_required_ids(tokenizer, stop_ids),
                "token_counts": {key: {"distinct_positions": sum(len(x["input_ids"]) for x in rows),
                                       "supervised_positions": sum(x["response_length"] for x in rows)}
                                 for key, rows in records.items()},
                "partition_hashes": {key: fingerprint(value) for key, value in records.items()}}
    manifest["fingerprint"] = fingerprint(manifest)
    validate_chat_manifest(manifest)
    if destination is not None:
        write_json(Path(destination), manifest)
    return manifest


def validate_turn(item, stop_ids):
    ids, mask = item["input_ids"], item["response_mask"]
    prompt = item["prompt_length"]
    if len(ids) != len(mask) or len(ids) > item["context_length"]:
        raise ValueError("chat tokens/masks exceed their context or disagree")
    if (prompt < 1 or item["response_length"] < 1
            or mask != [0] * prompt + [1] * item["response_length"]):
        raise ValueError("chat response supervision must exclude prompt and include the full turn")
    if not ids or ids[-1] not in stop_ids or item["stop_id"] != ids[-1]:
        raise ValueError("frozen chat turn must end with the source stop token")
    if fingerprint(ids) != item["tokens_sha256"]:
        raise ValueError("chat token hash mismatch")


def validate_chat_manifest(manifest):
    core = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if fingerprint(core) != manifest.get("fingerprint"):
        raise ValueError("chat manifest fingerprint mismatch")
    seen_ids, seen_prompts, seen_tokens = set(), set(), set()
    for split in PARTITIONS:
        rows = manifest["records"][split]
        if not rows or fingerprint(rows) != manifest["partition_hashes"][split]:
            raise ValueError("chat partition missing or hash mismatch")
        for item in rows:
            if (item["id"] in seen_ids or item["prompt_hash"] in seen_prompts
                    or item["tokens_sha256"] in seen_tokens or item["split"] != split):
                raise ValueError("chat train/validation/selection partitions overlap")
            validate_turn(item, manifest["generation"]["stop_ids"])
            seen_ids.add(item["id"])
            seen_prompts.add(item["prompt_hash"])
            seen_tokens.add(item["tokens_sha256"])


def chat_batches(manifest, split, *, pad_to_context=False):
    validate_chat_manifest(manifest)
    batches = []
    pad = manifest["generation"]["pad_token_id"]
    for item in manifest["records"][split]:
        ids, mask = item["input_ids"], item["response_mask"]
        padding = item["context_length"] - len(ids) if pad_to_context else 0
        batches.append({"input_ids": torch.tensor([ids + [pad] * padding], dtype=torch.long),
                        "attention_mask": torch.tensor([[1] * len(ids) + [0] * padding]),
                        "response_mask": torch.tensor([mask + [0] * padding], dtype=torch.bool)})
    return batches
