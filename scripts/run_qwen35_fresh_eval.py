#!/usr/bin/env python3
"""Fresh packed-artifact diagnostics with a common HF teacher and GGUF bridge.

Every prompt is durable and checksum-bound. This protocol deliberately does not
reuse the fixed 300-prompt competitive protocol or imply a provider promotion.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.checkpoint import load_packed_model
from rotquant.eval.fresh_tasks import fingerprint, paired_family_interval, score_task, task_suite
from rotquant.utils import enable_default_logging, set_seed, write_result
from rotquant.validation import capture_probes, compare_probes, packed_residency, probe_inputs
from rotquant.vocabulary import file_digest
from scripts import run_experiment as experiment
from scripts.run_qwen35_next_stage import _Heartbeat
from scripts.run_qwen35_packed_validation import (
    probe_settings,
    read_record,
    save_record,
    verify_prepared,
    writer_lock,
)
from scripts.run_qwen35_vocabulary_budget import runtime_identity, source_identity
from scripts.run_unsloth_qwen35_4b_kl import (
    BF16,
    C4_REVISION,
    GGUF_REVISION,
    LLAMA_CPP_PYTHON_REVISION,
    MM_PROJ_F16,
    MODEL_ID,
    MODEL_REVISION,
    UD_Q4,
    _atomic_npz,
    _download,
    _input_hash,
    _llama,
    distribution_metrics,
    evaluate_logits,
)

PROTOCOL = "qwen35-fresh-packed-diagnostics-v1"
SETTINGS = {
    "c4_batches": 24,
    "c4_tokens": 512,
    "c4_skip": 32768,
    "max_prompt_tokens": 512,
    "max_new_tokens": 128,
    "fidelity_new_tokens": 32,
    "enable_thinking": False,
    "temperature": 0,
    "seed": 0,
    "stop_policy": "source-config-plus-tokenizer-eos-v1",
    "task_scope": "96 authored unit diagnostics; not a real-world agent/coding benchmark",
}


def progress(root, phase, **details):
    value = {
        "phase": phase,
        "pid": os.getpid(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **details,
    }
    write_result(str(root / "progress.json"), value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def separate(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output and artifact/evidence trees must not overlap")


def history(root):
    """Require original preparation records, including calibration row provenance."""
    manifests, bindings = [], {}
    for arm in ("b5_v6", "b5_v8"):
        path = root / f"{arm}_s0/prepared.json"
        record = read_record(path)
        if record is None or record.get("status") != "prepared":
            raise ValueError(f"missing original preparation: {path}")
        bindings[arm] = file_digest(path)
        manifests.append(record["metrics"]["data_manifest"])
    return manifests, bindings


def recipe_signature(identity):
    config = identity["config"]
    return {
        key: config.get(key)
        for key in (
            "model",
            "model_revision",
            "model_loader",
            "dtype",
            "quant",
            "patch",
            "vocabulary",
            "n_calib",
            "calib_seq_len",
            "calib_dataset_revision",
            "packed_validation",
        )
    }


def fresh_runtime():
    runtime = runtime_identity()
    for name in ("numpy", "scipy", "huggingface-hub", "jinja2", "accelerate"):
        try:
            runtime["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime["versions"][name] = None
    return runtime


def tokenizer_identity(tokenizer, output_size=None):
    vocab = tokenizer.get_vocab()
    # Bind token IDs, not just tokenizer class/name. Full output axis checked at runtime.
    ordered = sorted(vocab.items(), key=lambda pair: pair[1])
    return {
        "vocab_sha256": fingerprint(ordered),
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
        "special_tokens_sha256": fingerprint(tokenizer.special_tokens_map),
        "size": len(vocab) if output_size is None else output_size,
        "tokenizer_size": len(vocab),
    }


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def archived_data_manifests():
    """Read row/hash provenance from all archived runs, not only the last screen."""
    entries, files = [], {}
    for path in sorted((ROOT / "research/results/raw").rglob("*.json")):
        if path.is_symlink():
            raise ValueError("historical evidence cannot contain symlinks")
        files[str(path.relative_to(ROOT))] = file_digest(path)
        stack = [json.loads(path.read_text())]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if value.get("dataset") in ("allenai/c4", "c4") and value.get("source_rows"):
                    entries.append(value)
                stack.extend(v for v in value.values() if isinstance(v, (dict, list)))
            elif isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, (dict, list)))
    return entries, fingerprint(files)


def freeze(root, evidence):
    separate(root, evidence)
    previous, bindings = history(evidence)
    archived, archive_hash = archived_data_manifests()
    identity = {
        "protocol": PROTOCOL,
        "settings": SETTINGS,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "c4_revision": C4_REVISION,
        "history": bindings,
        "historical_archive_sha256": archive_hash,
        "source": source_identity(),
        "runtime": fresh_runtime(),
        "authored_tasks_sha256": fingerprint(task_suite()),
    }
    existing = read_record(root / "manifest.json")
    if existing is not None:
        if existing["identity"] != identity:
            raise ValueError("frozen protocol changed; preserve this run and choose a fresh root")
        validate_manifest(existing)
        print("resume frozen inputs; no retokenization", flush=True)
        return existing
    tokenizer = load_tokenizer()
    old_rows, old_hashes = set(), set()
    for manifest in [*previous, {"archived-" + str(i): entry for i, entry in enumerate(archived)}]:
        for entry in manifest.values():
            if not isinstance(entry, dict):
                continue
            old_hashes.update(entry.get("token_hashes", []))
            old_hashes.update(entry.get("window_hashes", []))
            if entry.get("dataset") in ("allenai/c4", "c4"):
                old_rows.update(r for r in entry.get("source_rows", []) if r is not None)
    os.environ["ROTQUANT_TOKEN_CACHE_DIR"] = str(root / "token_cache")
    progress(root, "freeze_fresh_c4", excluded_source_rows=len(old_rows))
    batches = experiment.build_calib_loader(
        tokenizer,
        SETTINGS["c4_batches"],
        SETTINGS["c4_tokens"],
        "cpu",
        skip=SETTINGS["c4_skip"],
        revision=C4_REVISION,
        exclude_source_rows=old_rows,
    )
    if len(batches) != SETTINGS["c4_batches"]:
        raise ValueError("incomplete fresh C4 selection")
    items = []
    for i, batch in enumerate(batches):
        ids = batch["input_ids"][0].tolist()
        if batch.source_row in old_rows:
            raise ValueError("C4 source row was used in development/calibration")
        items.append(
            {
                "id": f"c4-{i:03d}",
                "family": f"c4-row-{batch.source_row}",
                "domain": "c4",
                "kind": "text",
                "input_ids": ids,
                "source_row": int(batch.source_row),
            }
        )
    old_texts = json.loads(
        (ROOT / "research/eval_suites/qwen35_diverse_development_v1.json").read_text()
    )["prompts"]
    for task in task_suite():
        if any(task["prompt"] == old[field] for old in old_texts for field in ("prompt", "text")):
            raise ValueError("authored prompt repeats development text")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if not 1 <= len(ids) <= SETTINGS["max_prompt_tokens"]:
            raise ValueError(f"task exceeds prompt budget; never silently truncate: {task['id']}")
        items.append({**task, "input_ids": ids, "rendered": rendered})
    for item in items:
        item["input_hash"] = _input_hash(np.array(item["input_ids"]))
        if item["input_hash"] in old_hashes:
            raise ValueError("fresh input repeats an old token sequence")
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    output_size = getattr(model_config, "text_config", model_config).vocab_size
    if max(tokenizer.get_vocab().values()) >= output_size:
        raise ValueError("tokenizer IDs exceed model output vocabulary")
    manifest = {
        "identity": identity,
        "tokenizer": tokenizer_identity(tokenizer, output_size),
        "items": items,
        "recipes": {
            arm: recipe_signature(read_record(evidence / f"{arm}_s0/prepared.json")["identity"])
            for arm in ("b5_v6", "b5_v8")
        },
        "disjointness": {
            "excluded_c4_source_rows": sorted(old_rows),
            "exact_old_token_hashes_checked": len(old_hashes),
            "authored_exact_text_checked": True,
            "historical_c4_manifests_checked": len(archived),
            "boundary": "Exact source-row/hash checks, not semantic near-dedup or pretraining contamination detection.",
        },
    }
    manifest["fingerprint"] = fingerprint(manifest)
    validate_manifest(manifest)
    save_record(root / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest):
    if (
        fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
        != manifest["fingerprint"]
    ):
        raise ValueError("manifest fingerprint mismatch")
    items = manifest["items"]
    if (
        len(items) != 120
        or len({i["id"] for i in items}) != 120
        or len({i["input_hash"] for i in items}) != 120
    ):
        raise ValueError("incomplete or duplicate frozen inputs")
    if sum(i["kind"] == "text" for i in items) != 24:
        raise ValueError("expected 24 C4 and 96 authored tasks")
    for item in items:
        if _input_hash(np.array(item["input_ids"])) != item["input_hash"]:
            raise ValueError("token hash mismatch")


class HFBackend:
    def __init__(self, model, tokenizer, device, stop_ids):
        self.model, self.tokenizer, self.device, self.stop_ids = (
            model.eval(),
            tokenizer,
            device,
            stop_ids,
        )

    @torch.no_grad()
    def predict(self, ids):
        value = torch.tensor([ids], device=self.device)
        result = self.model(
            input_ids=value, attention_mask=torch.ones_like(value), use_cache=False
        ).logits
        return result[0, :-1].detach().float().cpu().numpy()

    @torch.no_grad()
    def generate(self, ids):
        from transformers import GenerationConfig

        value = torch.tensor([ids], device=self.device)
        output = self.model.generate(
            input_ids=value,
            attention_mask=torch.ones_like(value),
            generation_config=GenerationConfig(
                max_new_tokens=SETTINGS["max_new_tokens"],
                do_sample=False,
                use_cache=True,
                eos_token_id=self.stop_ids,
                pad_token_id=self.stop_ids[0],
                repetition_penalty=1.0,
            ),
        )
        return output[0, len(ids) :].tolist()


class LlamaBackend:
    def __init__(self, model, tokenizer, stop_ids, output_size=None):
        self.model, self.tokenizer, self.stop_ids = model, tokenizer, stop_ids
        output_size = len(tokenizer) if output_size is None else output_size
        if model.n_vocab() != output_size:
            raise ValueError("full output vocabulary sizes differ; do not slice/renormalize for KL")
        for text, index in tokenizer.get_vocab().items():
            if model._model.token_get_text(index) != text:
                raise ValueError(
                    f"GGUF/HF token-axis mismatch at {index}; no cross-engine KL is valid"
                )
        used = set(tokenizer.get_vocab().values())
        for index in range(output_size):
            if index not in used and model._model.token_get_text(index) != f"[PAD{index}]":
                raise ValueError(f"unrecognized GGUF padded output slot: {index}")

    def predict(self, ids):
        return evaluate_logits(self.model, np.array(ids, dtype=np.int64))

    def generate(self, ids):
        self.model.reset()
        output = []
        # Direct argmax avoids engine-specific sampler penalties and EOS rules.
        self.model.eval(ids)
        for _ in range(SETTINGS["max_new_tokens"]):
            token = int(np.argmax(self.model.scores[self.model.n_tokens - 1]))
            output.append(token)
            if token in self.stop_ids or len(output) == SETTINGS["max_new_tokens"]:
                break
            self.model.eval([token])
        return output


def stop_ids_for_source(tokenizer=None):
    """Resolve pinned generation defaults and the tokenizer's chat terminator.

    A separate generation_config.json is optional on the Hub. Only a confirmed
    missing entry permits fallback: offline-cache misses, auth failures and bad
    JSON must not silently change the experiment's stopping rules.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    from transformers import AutoConfig, GenerationConfig

    origin = "generation_config.json"
    try:
        path = hf_hub_download(MODEL_ID, "generation_config.json", revision=MODEL_REVISION)
    except LocalEntryNotFoundError:
        raise
    except EntryNotFoundError:
        config = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        )
        origin = "config.json (including nested text_config)"
    else:
        config = GenerationConfig.from_dict(json.loads(Path(path).read_text()))
    ids = config.eos_token_id
    ids = ids if isinstance(ids, list) else [ids]
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("source EOS IDs missing")
    tokenizer = load_tokenizer() if tokenizer is None else tokenizer
    # Qwen's model EOS is endoftext, whereas completed chat replies use im_end.
    # Apply this explicit union to HF, packed and GGUF arms, not engine defaults.
    chat_eos = tokenizer.eos_token_id
    if chat_eos is not None:
        if type(chat_eos) is not int or chat_eos < 0:
            raise ValueError("invalid tokenizer EOS ID")
        ids = [*ids, chat_eos]
    ids = list(dict.fromkeys(ids))
    if not set(ids).issubset(tokenizer.get_vocab().values()):
        raise ValueError("source EOS IDs are not defined tokenizer tokens")
    print(json.dumps({"stop_config": origin, "stop_ids": ids,
                      "stop_policy": SETTINGS["stop_policy"]}), flush=True)
    return ids


def llama_build_identity():
    """Refuse a version-string-only installation; bind VCS metadata and binary."""
    dist = importlib.metadata.distribution("llama-cpp-python")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != LLAMA_CPP_PYTHON_REVISION:
        raise ValueError("install the pinned llama-cpp-python Git revision, not an arbitrary wheel")
    libraries = {
        str(p): file_digest(Path(dist.locate_file(p)))
        for p in dist.files or []
        if str(p).endswith((".so", ".dylib"))
    }
    if not libraries:
        raise ValueError("cannot identify installed llama shared libraries")
    return {
        "commit": LLAMA_CPP_PYTHON_REVISION,
        "direct_url": direct,
        "libraries": libraries,
        "version": dist.version,
    }


def paths_for(root, label, item):
    return root / "runs" / label / (item["id"] + ".json")


def reference_record(root, label, item, manifest):
    path = paths_for(root, label, item)
    record = read_record(path)
    if record is None or record.get("manifest") != manifest["fingerprint"]:
        raise ValueError(f"missing/mismatched reference record: {path}")
    if record["id"] != item["id"] or record["input_hash"] != item["input_hash"]:
        raise ValueError("reference item pairing mismatch")
    reference = path.with_suffix(".npz")
    if not reference.exists() or not record.get("reference_sha256"):
        raise ValueError(f"missing reference tensor or digest: {reference}")
    return record, reference


def load_reference(root, label, item, manifest):
    record, reference = reference_record(root, label, item, manifest)
    if file_digest(reference) != record["reference_sha256"]:
        raise ValueError(f"missing/corrupt reference tensor: {reference}")
    with np.load(reference, allow_pickle=False) as values:
        logits = values["logits"].copy()
    if (
        logits.shape != (record["tokens"], manifest["tokenizer"]["size"])
        or not np.isfinite(logits).all()
    ):
        raise ValueError("invalid full-logit reference shape/values")
    return record, logits


def metric_record(teacher, student, targets):
    if not np.isfinite(teacher).all() or not np.isfinite(student).all():
        raise ValueError("non-finite model logits")
    metrics = distribution_metrics(
        teacher, student, np.asarray(targets), device="cpu", chunk_tokens=8
    )
    result = {
        "mean_teacher_kl": float(np.mean(metrics["teacher_kl"])),
        "top1_agreement": float(np.mean(metrics["top1_matches"])),
        "source_nll": float(np.mean(metrics["source_nll"])),
        "candidate_nll": float(np.mean(metrics["candidate_nll"])),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("non-finite scored metrics")
    return result


def score_one(backend, item, manifest, root, label):
    ids = item["input_ids"]
    continuation = backend.generate(ids) if item["kind"] == "task" else []
    if item["kind"] == "task" and not continuation:
        raise ValueError("generation returned no tokens")
    truncated = bool(continuation and continuation[-1] not in backend.stop_ids)
    unmapped = sorted(set(continuation) - set(backend.tokenizer.get_vocab().values()))
    text = (
        f"Unmapped generated output IDs: {unmapped}"
        if unmapped
        else backend.tokenizer.decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    )
    reference, teacher = (
        (None, None)
        if label == "source_fp16"
        else load_reference(root, "source_fp16", item, manifest)
    )
    if item["kind"] == "text":
        scoring_ids, offset = ids, 0
        targets = ids[1:]
    else:
        suffix = (continuation if reference is None else reference["continuation"])[
            : SETTINGS["fidelity_new_tokens"]
        ]
        scoring_ids, offset, targets = ids + suffix, len(ids) - 1, suffix
    logits = backend.predict(scoring_ids)[offset:]
    if logits.shape != (len(targets), manifest["tokenizer"]["size"]):
        raise ValueError("full vocabulary/position count mismatch")
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")
    result = {key: item[key] for key in ("id", "domain", "family", "input_hash", "kind")}
    result.update(
        manifest=manifest["fingerprint"],
        tokens=len(targets),
        continuation=continuation,
        output_text=text,
        unmapped_generated_ids=unmapped,
        truncated=truncated,
        **metric_record(logits if teacher is None else teacher, logits, targets),
    )
    if item["kind"] == "task":
        result.update(score_task(text, item["expected"], truncated=truncated))
        expected_tokens = continuation if reference is None else reference["continuation"]
        length = max(len(expected_tokens), len(continuation))
        result["trajectory_token_agreement"] = (
            sum(a == b for a, b in zip(expected_tokens, continuation)) / length
        )
        result["exact_trajectory"] = expected_tokens == continuation
        result["source_task_success"] = (
            result["task_success"] if reference is None else reference["task_success"]
        )
    if label == "unsloth_ud_q4":
        bridge, bridge_logits = load_reference(root, "gguf_bf16_bridge", item, manifest)
        if bridge["tokens"] != len(targets):
            raise ValueError("bridge context/target mismatch")
        result["same_engine_bf16"] = metric_record(bridge_logits, logits, targets)
    return result, logits


def run_collection(
    root, manifest, label, *, artifact_root=None, seed=0, gguf_dir=None, device="cuda"
):
    tokenizer = load_tokenizer()
    if tokenizer_identity(tokenizer, manifest["tokenizer"]["size"]) != manifest["tokenizer"]:
        raise ValueError("tokenizer/template changed after freeze")
    stops = stop_ids_for_source(tokenizer)
    base_identity = {
        "manifest": manifest["fingerprint"],
        "source": source_identity(),
        "runtime": fresh_runtime(),
        "stop_ids": stops,
        "device": device,
        "label": label,
    }
    ledger = None
    if label.startswith("b5_"):
        if artifact_root is None:
            raise ValueError("packed phase requires --artifact-root")
        separate(root, artifact_root)
        arm = label.rsplit("_s", 1)[0]
        artifact = artifact_root / f"{arm}_s{seed}"
        prepared = read_record(artifact / "prepared.json")
        if prepared is None:
            raise ValueError("missing prepared artifact")
        if prepared["identity"]["arm"] != arm or prepared["identity"]["seed"] != seed:
            raise ValueError("artifact arm/seed mismatch")
        config = prepared["identity"]["config"]
        if config["model"] != MODEL_ID or config["model_revision"] != MODEL_REVISION:
            raise ValueError("packed source model differs from common teacher")
        if recipe_signature(prepared["identity"]) != manifest["recipes"][arm]:
            raise ValueError("packed recipe differs from frozen seed-0 recipe")
        # New seeds must still be disjoint from the now-frozen C4 evaluation.
        train_rows = experiment.calibration_source_rows(prepared["metrics"]["data_manifest"])
        if train_rows.intersection(
            i["source_row"] for i in manifest["items"] if i["kind"] == "text"
        ):
            raise ValueError("new seed calibration overlaps frozen evaluation rows")
        ledger = verify_prepared(artifact, prepared, prepared["identity"])
        base_identity.update(
            prepared_sha256=file_digest(artifact / "prepared.json"),
            manifest_sha256=prepared["export"]["manifest_sha256"],
        )
    elif label in ("gguf_bf16_bridge", "unsloth_ud_q4"):
        base_identity["llama_build"] = llama_build_identity()
        base_identity["gguf_revision"] = GGUF_REVISION
        descriptor = BF16 if label == "gguf_bf16_bridge" else UD_Q4
        base_identity["gguf"] = descriptor.__dict__
    base_identity = json.loads(json.dumps(base_identity))
    run_dir = root / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    old = read_record(run_dir / "identity.json")
    if old is not None and old != base_identity:
        raise ValueError("collection identity changed; use a new run root")
    save_record(run_dir / "identity.json", base_identity)
    # Check reference metadata before load; hash/read large arrays once when
    # each prompt is consumed, not a second full Drive scan before every model.
    if label != "source_fp16":
        for item in manifest["items"]:
            reference_record(root, "source_fp16", item, manifest)
            if label == "unsloth_ud_q4":
                reference_record(root, "gguf_bf16_bridge", item, manifest)
    pending = []
    for item in manifest["items"]:
        saved = read_record(paths_for(root, label, item))
        if saved is not None:
            if saved.get("collection") != fingerprint(base_identity):
                raise ValueError("prompt result collection mismatch")
            if label in ("source_fp16", "gguf_bf16_bridge"):
                load_reference(root, label, item, manifest)
        else:
            pending.append(item)
    completion = read_record(run_dir / "complete.json")
    if completion is not None and completion["identity"] != base_identity:
        raise ValueError("completion identity mismatch")
    if not pending and completion is not None:
        print(f"resume {label}: all {len(manifest['items'])} prompts verified", flush=True)
        return
    progress(root, "model_load", label=label, remaining=len(pending))
    if label.startswith("b5_"):
        from safetensors.torch import load_file

        model = load_packed_model(
            artifact / "checkpoint", device=device, dtype=torch.float16, fallback=False
        )
        before = packed_residency(model)
        if before["backbone_fallback_cache_bytes"] or not before["one_shared_vocabulary_owner"]:
            raise ValueError("packed residency failed")
        expected = load_file(str(artifact / "packed_probes.safetensors"))
        probes = capture_probes(model, probe_inputs(expected), device, **probe_settings(config))
        parity = compare_probes(probes, expected, reload=True)
        save_record(
            run_dir / "reload_probe_report.json", {"identity": base_identity, "parity": parity}
        )
        if not parity["passed"]:
            raise ValueError("packed reload probes failed; fresh quality was not run")
        backend = HFBackend(model, tokenizer, device, stops)
    elif label == "source_fp16":
        model, _, _ = experiment.load_hf_model(
            MODEL_ID, torch.float16, device, "multimodal_lm", MODEL_REVISION
        )
        backend = HFBackend(model, tokenizer, device, stops)
    else:
        path = _download(gguf_dir, descriptor)
        if label == "unsloth_ud_q4":
            _download(gguf_dir, MM_PROJ_F16)
            ledger = {
                "measured_artifact_bytes": UD_Q4.bytes + MM_PROJ_F16.bytes,
                "files": [UD_Q4.__dict__, MM_PROJ_F16.__dict__],
                "vision_executed": False,
            }
        model = _llama(
            path, SETTINGS["max_prompt_tokens"] + SETTINGS["max_new_tokens"], verbose=False
        )
        backend = LlamaBackend(model, tokenizer, stops, manifest["tokenizer"]["size"])
        # Render once with HF and require the GGUF tokenizer to reproduce all chat inputs.
        for item in manifest["items"]:
            if (
                item["kind"] == "task"
                and model.tokenize(item["rendered"].encode(), add_bos=False, special=True)
                != item["input_ids"]
            ):
                raise ValueError("HF/GGUF rendered prompt tokenization differs")
    try:
        for index, item in enumerate(pending, 1):
            started = time.monotonic()
            progress(
                root,
                "prompt_start",
                label=label,
                prompt=item["id"],
                remaining=len(pending) - index + 1,
            )
            try:
                result, logits = score_one(backend, item, manifest, root, label)
                path = paths_for(root, label, item)
                if label in ("source_fp16", "gguf_bf16_bridge"):
                    _atomic_npz(path.with_suffix(".npz"), logits=logits)
                    result["reference_sha256"] = file_digest(path.with_suffix(".npz"))
                result["collection"] = fingerprint(base_identity)
                result["seconds"] = time.monotonic() - started
                save_record(path, result)
                progress(
                    root,
                    "prompt_complete",
                    label=label,
                    prompt=item["id"],
                    seconds=result["seconds"],
                    kl=result["mean_teacher_kl"],
                    task_success=result.get("task_success"),
                )
            except BaseException as error:
                # Separate immutable failure attempt; completed prompts remain resumable.
                save_record(
                    run_dir / f"failure-{time.time_ns()}.json",
                    {
                        "collection": fingerprint(base_identity),
                        "prompt": item["id"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
        after = packed_residency(model) if label.startswith("b5_") else None
        if after is not None and after != before:
            raise ValueError("packed residency changed during evaluation")
        save_record(
            run_dir / "complete.json",
            {
                "identity": base_identity,
                "complete": True,
                "ledger": ledger,
                "residency": after,
                "manifest": manifest["fingerprint"],
                "boundary": "Quality diagnostics only. Timings are not a throughput benchmark.",
            },
        )
    finally:
        if isinstance(backend, LlamaBackend):
            model.close()


def summarize(root, manifest, expected_labels):
    allowed = {"source_fp16", "gguf_bf16_bridge", "unsloth_ud_q4"} | {
        f"b5_v{v}_s{s}" for v in (6, 8) for s in (0, 1, 2)
    }
    if (
        not expected_labels
        or len(set(expected_labels)) != len(expected_labels)
        or not set(expected_labels) <= allowed
    ):
        raise ValueError("expected labels must be unique registered run names")
    rows, complete, missing = [], {}, []
    for label in expected_labels:
        marker = read_record(root / "runs" / label / "complete.json")
        values = [read_record(paths_for(root, label, item)) for item in manifest["items"]]
        if marker is None or not marker.get("complete") or any(value is None for value in values):
            missing.append(label)
            continue
        if marker["manifest"] != manifest["fingerprint"]:
            raise ValueError("summary manifest mismatch")
        for item, value in zip(manifest["items"], values):
            if (value["id"], value["input_hash"], value["manifest"], value["collection"]) != (
                item["id"],
                item["input_hash"],
                manifest["fingerprint"],
                fingerprint(marker["identity"]),
            ):
                raise ValueError("summary record pairing mismatch")
        complete[label] = values
        for domain in sorted({v["domain"] for v in values}):
            subset = [v for v in values if v["domain"] == domain]
            tokens = sum(v["tokens"] for v in subset)
            row = {
                "label": label,
                "domain": domain,
                "prompts": len(subset),
                "tokens": tokens,
                "mean_teacher_kl": sum(v["tokens"] * v["mean_teacher_kl"] for v in subset) / tokens,
                "top1_agreement": sum(v["tokens"] * v["top1_agreement"] for v in subset) / tokens,
                "artifact_bytes": (marker.get("ledger") or {}).get("measured_artifact_bytes"),
            }
            if domain != "c4":
                for key in (
                    "task_success",
                    "source_task_success",
                    "json_valid",
                    "truncated",
                    "trajectory_token_agreement",
                    "exact_trajectory",
                ):
                    row[key] = float(np.mean([v[key] for v in subset]))
                row["correct_to_wrong"] = sum(
                    v["source_task_success"] and not v["task_success"] for v in subset
                )
                row["wrong_to_correct"] = sum(
                    not v["source_task_success"] and v["task_success"] for v in subset
                )
            if label == "unsloth_ud_q4":
                row["same_engine_bf16_kl"] = (
                    sum(v["tokens"] * v["same_engine_bf16"]["mean_teacher_kl"] for v in subset)
                    / tokens
                )
            rows.append(row)
    contrasts = []
    for left, right in [(f"b5_v6_s{s}", f"b5_v8_s{s}") for s in (0, 1, 2)] + [
        (f"b5_v{v}_s{s}", "unsloth_ud_q4") for s in (0, 1, 2) for v in (6, 8)
    ]:
        if left not in complete or right not in complete:
            continue
        for domain in sorted({v["domain"] for v in complete[left]}):
            a, b = [
                [r for r in complete[label] if r["domain"] == domain] for label in (left, right)
            ]
            for field in (
                ["mean_teacher_kl", "top1_agreement"]
                if domain == "c4"
                else ["mean_teacher_kl", "task_success", "trajectory_token_agreement"]
            ):
                contrasts.append(
                    {
                        "left": left,
                        "right": right,
                        "domain": domain,
                        "field": field,
                        **paired_family_interval(a, b, field),
                    }
                )
    result = {
        "protocol": PROTOCOL,
        "manifest": manifest["fingerprint"],
        "complete": not missing,
        "expected_labels": expected_labels,
        "missing": missing,
        "rows": rows,
        "paired_contrasts": contrasts,
        "provider_competitive": False,
        "independent_confirmation": False,
        "interpretation": "Fresh C4 plus authored diagnostic families. Cross-engine/common-FP16 scores include engine and precision differences; inspect the BF16 bridge. Seeds replicate rotation/quantization on the fixed calibration selection, not independent calibration corpora. No automatic promotion.",
    }
    save_record(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="original preparation run, with b5_v6_s0/prepared.json",
    )
    parser.add_argument(
        "--phase",
        choices=("freeze", "source", "packed", "bridge", "unsloth", "summary"),
        required=True,
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--arm", choices=("b5_v6", "b5_v8"), default="b5_v6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gguf-dir", type=Path, default=Path("/content/unsloth-qwen35-4b-gguf"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heartbeat-seconds", type=float, default=60)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed not in (0, 1, 2) or args.heartbeat_seconds <= 0:
        parser.error("seeds are 0/1/2 and heartbeat must be positive")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "settings": SETTINGS,
                    "phases": ["freeze", "source", "packed", "bridge", "unsloth", "summary"],
                    "existing_seed0_quantizations": 0,
                    "replication_backbones": 2,
                    "historical_aggregate_reuse": False,
                    "automatic_promotion": False,
                },
                indent=2,
            )
        )
        return
    enable_default_logging()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(0)
    root = args.output_dir.resolve()
    separate(root, args.evidence_root)
    if args.artifact_root:
        separate(root, args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    with (
        writer_lock(root / ".runner.lock"),
        _Heartbeat(f"fresh/{args.phase}", seconds=args.heartbeat_seconds),
    ):
        if args.phase == "freeze":
            # Fail before artifact scans/C4 selection, not at the first model load.
            stop_ids_for_source()
            progress(root, "verify_original_artifacts_before_capture")
            for arm in ("b5_v6", "b5_v8"):
                original = args.evidence_root / f"{arm}_s0"
                prepared = read_record(original / "prepared.json")
                if prepared is None:
                    raise ValueError(f"missing original artifact preparation: {original}")
                verify_prepared(original, prepared, prepared["identity"])
        manifest = freeze(root, args.evidence_root)
        if args.phase == "freeze":
            progress(
                root, "frozen", prompts=len(manifest["items"]), fingerprint=manifest["fingerprint"]
            )
        elif args.phase == "summary":
            if not args.expect:
                parser.error("summary requires explicit --expect labels")
            print(json.dumps(summarize(root, manifest, args.expect), indent=2))
        else:
            label = {
                "source": "source_fp16",
                "packed": f"{args.arm}_s{args.seed}",
                "bridge": "gguf_bf16_bridge",
                "unsloth": "unsloth_ud_q4",
            }[args.phase]
            run_collection(
                root,
                manifest,
                label,
                artifact_root=args.artifact_root,
                seed=args.seed,
                gguf_dir=args.gguf_dir,
                device=args.device,
            )


if __name__ == "__main__":
    main()
