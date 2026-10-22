"""Disjoint, immutable fitting, selection and confirmation inputs."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import public_task_suite as tasks
from scripts import run_experiment as experiment
from scripts.native_gpu_workflow import fingerprint, write_json
from scripts.w4_research_plan import ROOT


def partition_counts(config):
    study = config["research"]
    return {"calibration": config["n_calib"],
            **{f"recovery_{part}": study[f"recovery_{part}_documents"]
               for part in ("train", "validation", "selection")},
            "selection": study["selection_documents"], "test": study["test_documents"]}


def document(batch, split, index):
    return {"id": f"{split}-{index:04d}", "domain": "c4", "source_row": batch.source_row,
            "input_ids": batch["input_ids"][0].tolist()}


def prior_rows():
    path = ROOT / "research/eval_suites/w4_prior_c4_rows_v1.json"
    value = json.loads(path.read_text())
    return set(value["source_rows"]), fingerprint(value)


def collect_documents(config, tokenizer):
    counts, output = partition_counts(config), {}
    excluded, history = prior_rows()
    calibration = experiment.build_calib_loader(
        tokenizer, counts.pop("calibration"), config["calib_seq_len"], "cpu",
        revision=config["calib_dataset_revision"])
    output["calibration"] = [document(batch, "calibration", i) for i, batch in enumerate(calibration)]
    excluded.update(batch.source_row for batch in calibration)
    remaining = experiment.build_calib_loader(
        tokenizer, sum(counts.values()), config["calib_seq_len"], "cpu",
        revision=config["calib_dataset_revision"], exclude_source_rows=excluded)
    offset = 0
    for split, count in counts.items():
        output[split] = [document(batch, split, i) for i, batch in enumerate(remaining[offset:offset + count])]
        offset += count
    excluded.update(batch.source_row for batch in remaining)
    study = config["research"]
    batches = experiment.build_calib_loader(
        tokenizer, study["long_documents"], study["long_context"], "cpu",
        revision=config["calib_dataset_revision"], exclude_source_rows=excluded)
    output["long_test"] = [document(batch, "long-test", i) for i, batch in enumerate(batches)]
    return output, history


def public_inputs(tokenizer, count, seed):
    items, receipts = tasks.load_items(0)
    selected = []
    for name in tasks.BENCHMARKS:
        candidates = sorted((x for x in items if x["benchmark"] == name),
                            key=lambda x: fingerprint([20261019, seed, x["id"]]))[:count]
        for item in candidates:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            item["input_ids"] = tokenizer(prompt, add_special_tokens=False).input_ids
            if not 0 < len(item["input_ids"]) <= tasks.MAX_PROMPT_TOKENS:
                raise ValueError(f"public prompt exceeds frozen bound: {item['id']}")
            item["max_new_tokens"] = tasks.CAPS[name]
        selected.extend(candidates)
        receipts[name]["selected"] = len(candidates)
    return selected, receipts


def freeze(config, tokenizer, root):
    documents, history = collect_documents(config, tokenizer)
    public, datasets = public_inputs(tokenizer, config["research"]["public_samples_per_benchmark"], config["seed"])
    speed = speed_inputs(config, documents)
    manifest = {"documents": documents, "public_items": public, "datasets": datasets,
                "speed_inputs": speed, "prior_c4_rows_fingerprint": history,
                "tokenizer_size": len(tokenizer), "model_revision": config["model_revision"]}
    manifest["fingerprint"] = fingerprint(manifest)
    validate_manifest(manifest)
    write_json(Path(root) / "data.json", manifest)
    return manifest


def speed_inputs(config, documents):
    study = config["research"]
    tokens = documents["long_test"][0]["input_ids"]
    continuation = documents["test"][0]["input_ids"][:study["speed_decode_steps"]]
    if len(continuation) != study["speed_decode_steps"]:
        raise ValueError("not enough fixed decode tokens")
    result = {}
    for length in study["speed_contexts"]:
        if not 1 <= length <= len(tokens):
            raise ValueError("speed context exceeds a real frozen document")
        result[str(length)] = {"prompt_ids": tokens[:length], "decode_ids": continuation}
    return result


def validate_manifest(manifest):
    core = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if manifest.get("fingerprint") != fingerprint(core):
        raise ValueError("frozen data fingerprint mismatch")
    rows, hashes, ids = set(), set(), set()
    for documents in manifest["documents"].values():
        for item in documents:
            token_hash = fingerprint(item["input_ids"])
            if item["source_row"] in rows or token_hash in hashes or item["id"] in ids:
                raise ValueError("fitting/selection/test documents overlap")
            if len(item["input_ids"]) < 2:
                raise ValueError("empty scoring document")
            rows.add(item["source_row"])
            hashes.add(token_hash)
            ids.add(item["id"])
    validate_public_inputs(manifest["public_items"])
    validate_speed_inputs(manifest)


def validate_public_inputs(items):
    public_ids = [item["id"] for item in items]
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("duplicate public task IDs")
    for item in items:
        if not 0 < len(item["input_ids"]) <= tasks.MAX_PROMPT_TOKENS or item["max_new_tokens"] != tasks.CAPS[item["benchmark"]]:
            raise ValueError("public task input/cap differs from the protocol")


def validate_speed_inputs(manifest):
    original = manifest["documents"]["long_test"][0]["input_ids"]
    continuation = manifest["documents"]["test"][0]["input_ids"]
    for length, item in manifest["speed_inputs"].items():
        if len(item["prompt_ids"]) != int(length) or item["prompt_ids"] != original[:int(length)]:
            raise ValueError("speed prompts differ from frozen document prefixes")
        if not item["decode_ids"] or item["decode_ids"] != continuation[:len(item["decode_ids"])]:
            raise ValueError("speed continuation differs from frozen tokens")


def batches(documents):
    return [experiment.TokenBatch(torch.tensor([item["input_ids"]]), item["source_row"])
            for item in documents]
