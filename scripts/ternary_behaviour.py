"""Held-out completion, repetition, and explicit output-format acceptance metrics."""
from __future__ import annotations

import json
import re

import torch

from scripts.native_gpu_workflow import fingerprint
from scripts.ternary_chat_data import validate_chat_manifest


def repeated_span(tokens, *, minimum_span=64, maximum_period=128):
    for period in range(1, min(maximum_period, len(tokens) // 3) + 1):
        run = 0
        for position in range(period, len(tokens)):
            run = run + 1 if tokens[position] == tokens[position - period] else 0
            span = run + period
            if run >= 2 * period and span >= minimum_span:
                return {"start": position + 1 - span, "period": period, "length": span}
    return None


def response_text(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def valid_format(text, contract):
    kind = contract.get("kind", "text")
    if kind == "text":
        return bool(text.strip())
    if kind != "json":
        raise ValueError("behaviour format contract must be text or json")
    root = contract.get("root", "object")
    expected = {"object": dict, "array": list}.get(root)
    if expected is None:
        raise ValueError("JSON format root must be object or array")
    body = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", text.strip())
    try:
        value = json.loads(body)
    except (ValueError, TypeError):
        return False
    if not isinstance(value, expected):
        return False
    return not isinstance(value, dict) or set(contract.get("required_keys", ())) <= value.keys()


def selection_identity(manifest):
    return fingerprint({"chat_data": manifest["fingerprint"], "split": "selection",
                        "partition_hash": manifest["partition_hashes"]["selection"],
                        "protocol": "complete-turn-behaviour-v1"})


def candidate_response(model, item, generation, device):
    prompt = item["input_ids"][:item["prompt_length"]]
    capacity = min(generation["max_new_tokens"], item["context_length"] - len(prompt))
    inputs = torch.tensor([prompt], device=device, dtype=torch.long)
    with torch.no_grad():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=capacity, do_sample=False, use_cache=True,
                                eos_token_id=generation["stop_ids"],
                                pad_token_id=generation["pad_token_id"])
    values = output.sequences if hasattr(output, "sequences") else output
    sequence = values[0].tolist()
    if sequence[:len(prompt)] != prompt:
        raise ValueError("behaviour generation changed the frozen selection prompt")
    response = sequence[len(prompt):]
    if len(response) > capacity:
        raise ValueError("behaviour generation exceeded its frozen cap")
    return response, capacity


def behaviour_row(model, item, manifest, tokenizer, device):
    ids, cap = candidate_response(model, item, manifest["generation"], device)
    stops = set(manifest["generation"]["stop_ids"])
    completed = bool(ids and ids[-1] in stops and not stops.intersection(ids[:-1]))
    text = response_text(tokenizer, ids)
    loop = repeated_span(ids)
    return {"id": item["id"], "completed": completed, "repetition": loop,
            "format_valid": valid_format(text, item.get("format_contract", {"kind": "text"})),
            "generated_tokens": len(ids), "cap": cap, "output_ids": ids,
            "output_sha256": fingerprint(ids), "text": text}


def evaluate_behaviour(model, manifest, tokenizer, device):
    validate_chat_manifest(manifest)
    was_training = model.training
    model.eval()
    try:
        rows = [behaviour_row(model, item, manifest, tokenizer, device)
                for item in manifest["records"]["selection"]]
    finally:
        model.train(was_training)
    count = len(rows)
    return {"data_id": selection_identity(manifest), "examples": count,
            "completion_rate": sum(x["completed"] for x in rows) / count,
            "repetition_rate": sum(x["repetition"] is not None for x in rows) / count,
            "format_rate": sum(x["format_valid"] for x in rows) / count,
            "protocol": "complete-turn-behaviour-v1", "split": "selection", "rows": rows}


def build_behaviour_evaluator(manifest, tokenizer, device):
    validate_chat_manifest(manifest)
    for item in manifest["records"]["selection"]:
        contract = item.get("format_contract", {"kind": "text"})
        valid_format("", contract)
    return lambda model: evaluate_behaviour(model, manifest, tokenizer, device)
