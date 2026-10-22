"""Frozen-input quality, task and native timing measurements for W4 research."""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from scripts import public_task_suite
from scripts import run_qwen35_fresh_eval as fresh
from scripts.native_gpu_workflow import write_json
from scripts.run_rq3_performance_pilot import measure, summarize
from scripts.run_rq3_retained_gpu import ProcessVRAM

QUALITY_METRICS = ("mean_teacher_kl", "candidate_nll", "source_nll", "top1_agreement")


def token_ids(ids, minimum=1):
    values = np.asarray(ids)
    if values.ndim != 1 or values.dtype.kind not in "iu" or len(values) < minimum:
        raise ValueError("expected a nonempty one-dimensional integer token sequence")
    if (values < 0).any():
        raise ValueError("token IDs must be nonnegative")
    return values.astype(np.int64, copy=False)


def record_name(identifier):
    name = str(identifier)
    if not name or name.startswith(".") or Path(name).name != name:
        raise ValueError("record IDs must be simple file names")
    return name


def input_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


class HFTeacher:
    def __init__(self, model, device):
        self.model, self.device = model.eval(), device

    @torch.no_grad()
    def predict(self, ids):
        values = torch.tensor(token_ids(ids, 2)[:-1], device=self.device).unsqueeze(0)
        output = self.model(input_ids=values, attention_mask=torch.ones_like(values), use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        result = logits[0].float().cpu().numpy()
        if result.ndim != 2 or len(result) != values.shape[-1] or not np.isfinite(result).all():
            raise ValueError("teacher returned invalid full-position logits")
        return result


class NativeBackend:
    def __init__(self, native_model, stop_ids):
        self.model, self.stop_ids = native_model, set(map(int, stop_ids))

    def predict(self, ids):
        values = token_ids(ids, 2)[:-1]
        chunks = []
        for start in range(0, len(values), 16):
            chunk = values[start:start + 16]
            logits = self.model.evaluate(chunk, reset=start == 0, selected=len(chunk))
            logits = np.asarray(logits, dtype=np.float32).reshape(len(chunk), -1)
            if not np.isfinite(logits).all():
                raise ValueError("native model returned non-finite logits")
            chunks.append(logits)
        return np.concatenate(chunks)

    def generate(self, ids, max_new_tokens):
        values = token_ids(ids)
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("generation requires a positive integer token cap")
        logits = None
        for start in range(0, len(values), 8192):
            logits = self.model.evaluate(values[start:start + 8192], reset=start == 0)
        generated = []
        for step in range(max_new_tokens):
            if not np.isfinite(logits).all():
                raise ValueError("native generation returned non-finite logits")
            token = int(np.asarray(logits).argmax())
            generated.append(token)
            if token in self.stop_ids or step + 1 == max_new_tokens:
                break
            logits = self.model.evaluate([token])
        return generated


def packed_parity(student, path):
    expected = np.load(path, mmap_mode="r", allow_pickle=False)
    count = min(16, len(student))
    actual, reference = student[-count:], expected[-count:]
    if actual.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError("packed parity positions or vocabulary disagree")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {"reference": str(path), "positions": count,
            "max_absolute_error": float(np.abs(delta).max()),
            "root_mean_squared_error": float(np.sqrt(np.mean(delta ** 2))),
            "top1_agreement": float(np.mean(actual.argmax(-1) == reference.argmax(-1)))}


def quality_row(backend, item, teacher_dir, parity_dir):
    identifier = record_name(item["id"])
    ids = token_ids(item["input_ids"], 2)
    teacher = np.load(Path(teacher_dir) / f"{identifier}.npy", mmap_mode="c", allow_pickle=False)
    student = backend.predict(ids)
    if teacher.shape != student.shape or teacher.ndim != 2 or len(teacher) != len(ids) - 1:
        raise ValueError("teacher and candidate full-vocabulary positions differ")
    result = {"id": identifier, "domain": item["domain"], "source_row": item.get("source_row"),
              "input_ids": ids.tolist(), "input_hash": input_hash(ids), "tokens": len(ids) - 1,
              **fresh.metric_record(teacher, student, ids[1:])}
    if parity_dir is not None:
        result["packed_parity"] = packed_parity(student, Path(parity_dir) / f"{identifier}.npy")
    return result


def quality_clusters(rows):
    clusters = {}
    for row in rows:
        source = row["source_row"] if row.get("source_row") is not None else row["id"]
        key = json.dumps([row["domain"], source], sort_keys=True)
        values = np.array([row[name] * row["tokens"] for name in QUALITY_METRICS] + [row["tokens"]])
        clusters[key] = clusters.get(key, np.zeros(len(values))) + values
    return np.asarray(list(clusters.values()), dtype=np.float64)


def quality_summary(rows, bootstrap_draws=2000, seed=17):
    if not rows or bootstrap_draws < 1:
        raise ValueError("quality summary requires rows and positive bootstrap draws")
    clusters = quality_clusters(rows)
    totals = clusters.sum(0)
    means = totals[:-1] / totals[-1]
    result = dict(zip(QUALITY_METRICS, means.tolist()))
    result.update(tokens=int(totals[-1]), documents=len(clusters), records=len(rows),
                  ppl=math.exp(result["candidate_nll"]), source_ppl=math.exp(result["source_nll"]))
    intervals = {name: None for name in QUALITY_METRICS}
    if len(clusters) > 1:
        distribution = cluster_resamples(clusters, bootstrap_draws, seed)
        bounds = np.quantile(distribution, [0.025, 0.975], axis=0)
        intervals = {name: bounds[:, index].tolist() for index, name in enumerate(QUALITY_METRICS)}
    result["confidence_intervals"] = intervals
    result["ppl_interval"] = ([math.exp(value) for value in intervals["candidate_nll"]]
                              if intervals["candidate_nll"] is not None else None)
    result["uncertainty"] = {"method": "document-cluster percentile bootstrap", "confidence": 0.95,
                             "draws": bootstrap_draws, "seed": seed,
                             "boundary": "Exploratory intervals; not proof of non-inferiority. One document has no interval."}
    return result


def cluster_resamples(clusters, draws, seed):
    generator = np.random.default_rng(seed)
    chunk_size = max(1, 1_000_000 // len(clusters))
    samples = []
    for start in range(0, draws, chunk_size):
        indices = generator.integers(len(clusters), size=(min(chunk_size, draws - start), len(clusters)))
        totals = clusters[indices].sum(axis=1)
        samples.append(totals[:, :-1] / totals[:, -1:])
    return np.concatenate(samples)


def evaluate_quality(rootdir, backend, documents, teacher_dir, *, parity_dir=None,
                     bootstrap_draws=2000):
    rootdir, rows = Path(rootdir), []
    identifiers = [record_name(item["id"]) for item in documents]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate quality document IDs")
    for item in documents:
        row = quality_row(backend, item, teacher_dir, parity_dir)
        write_json(rootdir / "quality_rows" / f"{row['id']}.json", row)
        rows.append(row)
    result = {"rows": rows, "summary": quality_summary(rows, bootstrap_draws)}
    write_json(rootdir / "quality.json", result)
    return result


def public_task_row(backend, item, tokenizer, ifeval):
    ids = token_ids(item["input_ids"])
    continuation = backend.generate(ids, item["max_new_tokens"])
    if len(continuation) > item["max_new_tokens"]:
        raise ValueError("generation exceeded its frozen token cap")
    known = set(tokenizer.get_vocab().values())
    unmapped = sorted(set(continuation) - known)
    if unmapped:
        raise ValueError(f"generation returned IDs absent from the frozen tokenizer: {unmapped[:8]}")
    text = tokenizer.decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos = bool(continuation and continuation[-1] in backend.stop_ids)
    return {"id": record_name(item["id"]), "benchmark": item["benchmark"],
            "input_hash": input_hash(ids), "input_ids": ids.tolist(), "continuation": continuation,
            "generated_tokens": len(continuation), "max_new_tokens": item["max_new_tokens"],
            "stopped_at_eos": eos, "unmapped_ids": unmapped, "output_text": text,
            **public_task_suite.score(item, text, truncated=not eos, ifeval=ifeval)}


def task_summary(rows):
    benchmarks = sorted({row["benchmark"] for row in rows})
    result = {}
    for name in benchmarks:
        selected = [row for row in rows if row["benchmark"] == name]
        correct = sum(row["task_success"] for row in selected)
        result[name] = {"items": len(selected), "correct": correct, "accuracy": correct / len(selected),
                        "truncated": sum(row["truncated"] for row in selected)}
    return result


def evaluate_public_tasks(rootdir, backend, items, tokenizer, ifeval=None):
    rootdir, rows = Path(rootdir), []
    identifiers = [record_name(item["id"]) for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate public task IDs")
    for item in items:
        row = public_task_row(backend, item, tokenizer, ifeval)
        write_json(rootdir / "task_rows" / f"{row['id']}.json", row)
        rows.append(row)
    result = {"rows": rows, "summary": task_summary(rows)}
    write_json(rootdir / "tasks.json", result)
    return result


def paired_rows(candidate_rows, reference_rows):
    candidate = {row["id"]: row for row in candidate_rows}
    reference = {row["id"]: row for row in reference_rows}
    if (len(candidate) != len(candidate_rows) or len(reference) != len(reference_rows)
            or candidate.keys() != reference.keys() or not candidate):
        raise ValueError("paired tasks require the same unique nonempty item IDs")
    result = []
    for key in sorted(candidate):
        left, right = candidate[key], reference[key]
        if (left["benchmark"] != right["benchmark"]
                or left.get("input_hash") != right.get("input_hash")
                or left.get("max_new_tokens") != right.get("max_new_tokens")):
            raise ValueError("paired tasks disagree on benchmark, frozen prompt or token cap")
        result.append((left["benchmark"], float(left["task_success"]) - float(right["task_success"])))
    return result


def paired_delta_report(values, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.column_stack((values, np.ones(len(values))))
    distribution = cluster_resamples(clusters, draws, seed)[:, 0]
    mean = float(values.mean())
    radius = math.sqrt(2 * math.log(40) / len(values))
    return {"items": len(values), "mean_accuracy_delta": mean,
            "bootstrap_interval": np.quantile(distribution, [0.025, 0.975]).tolist(),
            "hoeffding_interval": [max(-1., mean - radius), min(1., mean + radius)],
            "candidate_wins": int((values > 0).sum()), "candidate_losses": int((values < 0).sum()),
            "boundary": "Exploratory paired estimates, never a non-inferiority claim. Hoeffding assumes independent sampled prompts; bootstrap can collapse with no discordance."}


def paired_task_deltas(candidate_rows, reference_rows, *, draws=2000, seed=17):
    if draws < 1:
        raise ValueError("paired bootstrap draws must be positive")
    pairs = paired_rows(candidate_rows, reference_rows)
    return {name: paired_delta_report([value for benchmark, value in pairs if benchmark == name], draws, seed)
            for name in sorted({name for name, _ in pairs})}


def speed_inputs_for(context, spec):
    prompt, decode = token_ids(spec["prompt_ids"]), token_ids(spec["decode_ids"])
    if len(prompt) != int(context):
        raise ValueError("speed prompt must contain exactly the frozen context length; no repetition or resizing")
    return prompt, decode


def native_speeds(model, speed_inputs, repetitions, output, *, memory=None,
                  maximum_vram=float("inf")):
    if repetitions < 1 or not speed_inputs:
        raise ValueError("speed measurement requires inputs and measured repetitions")
    frozen = {str(context): speed_inputs_for(context, spec) for context, spec in speed_inputs.items()}
    identities = {length: {"prompt_sha256": input_hash(prompt), "decode_sha256": input_hash(decode),
                           "prompt_tokens": len(prompt), "decode_tokens": len(decode)}
                  for length, (prompt, decode) in frozen.items()}
    report = {"completed": False, "rows": [], "summary": {}, "repetitions": repetitions,
              "frozen_inputs": identities}
    context = ProcessVRAM() if memory is None else nullcontext(memory)
    with context as tracker:
        for length, (prompt, decode) in frozen.items():
            previous = list(report["rows"])

            def persist(rows, active, previous=previous):
                report["rows"] = previous + rows + ([] if active is None else [active])
                report["memory"] = tracker.report()
                write_json(output, report)

            rows = measure(model, prompt, context=int(length), decode=len(decode),
                           repetitions=repetitions, minimum_tps=0, maximum_vram=maximum_vram,
                           memory=tracker, persist=persist, forced_decode=decode.tolist())
            report["summary"][length] = summarize(rows)
        report.update(completed=True, memory=tracker.report())
    write_json(output, report)
    return report
