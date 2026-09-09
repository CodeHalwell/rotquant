#!/usr/bin/env python3
"""Resume public-task evaluation of frozen W5 artifacts; never fit new weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import public_task_suite as suite
from scripts import run_qwen35_fresh_eval as fresh
from scripts.run_qwen35_packed_validation import read_record, save_record, writer_lock

PROTOCOL = "qwen35-public-tasks-v1"
REFERENCE_LABELS = ("source_fp16", "gguf_bf16_bridge", "unsloth_ud_q4")
ALL_LABELS = (*REFERENCE_LABELS, *(f"b5_v{v}_s{s}" for s in (0, 1, 2) for v in (6, 8)))
SETTINGS = {"caps": suite.CAPS, "max_prompt_tokens": suite.MAX_PROMPT_TOKENS,
            "enable_thinking": False, "do_sample": False, "repetition_penalty": 1.0,
            "selection_seed": suite.SELECTION_SEED, "generation_seed": 0,
            "stop_policy": "source-config-plus-tokenizer-eos-v1", "gguf_input_policy": "frozen-hf"}


def runtime():
    value = fresh.fresh_runtime()
    value.update(tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                 cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
                 float32_matmul_precision=torch.get_float32_matmul_precision())
    return value


def configure_runtime():
    fresh.enable_default_logging()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fresh.set_seed(0)


def artifact_path(label, seed0, replicas):
    return Path(seed0 if label.endswith("_s0") else replicas) / label


def preparation_binding(path, label):
    prepared = read_record(path / "prepared.json")
    if prepared is None or prepared.get("status") != "prepared":
        raise ValueError(f"missing completed checkpoint preparation: {path}")
    arm, seed = label.rsplit("_s", 1)
    identity, config = prepared["identity"], prepared["identity"]["config"]
    if (identity["arm"] != arm or identity["seed"] != int(seed)
            or config["model"] != fresh.MODEL_ID or config["model_revision"] != fresh.MODEL_REVISION):
        raise ValueError("checkpoint model/arm/seed mismatch")
    return {"prepared_sha256": fresh.file_digest(path / "prepared.json"),
            "manifest_sha256": prepared["export"]["manifest_sha256"],
            "artifact_bytes": prepared["export"]["artifact_bytes"],
            "recipe": fresh.recipe_signature(identity)}, prepared


def freeze_identity(samples, labels, seed0, replicas, scorer):
    if labels not in [list(ALL_LABELS), list(REFERENCE_LABELS) + ["b5_v6_s0", "b5_v8_s0"]]:
        raise ValueError("register both recipes with all three seeds or seed 0; provider controls are mandatory")
    bindings = {}
    for label in labels:
        if label.startswith("b5_"):
            bindings[label], _ = preparation_binding(artifact_path(label, seed0, replicas), label)
            arm = label.rsplit("_s", 1)[0]
            if bindings[label]["recipe"] != bindings[arm + "_s0"]["recipe"]:
                raise ValueError("replication recipe differs from seed 0")
    return {"protocol": PROTOCOL, "source": fresh.source_identity(), "runtime": runtime(),
            "model": fresh.MODEL_ID, "model_revision": fresh.MODEL_REVISION, "settings": SETTINGS,
            "samples_per_benchmark": samples, "labels": labels, "artifacts": bindings,
            "datasets": suite.DATASETS, "scorer": scorer.identity}


def validate_manifest(manifest):
    if manifest is None or suite.fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest.get("fingerprint"):
        raise ValueError("missing/corrupt frozen manifest")
    items = manifest["items"]
    if not items or len({v["id"] for v in items}) != len(items):
        raise ValueError("empty/duplicate manifest")
    for name, spec in manifest["datasets"].items():
        if sum(v["benchmark"] == name for v in items) != spec["selected"]:
            raise ValueError("incomplete frozen benchmark")
    for item in items:
        if (item["id"].startswith(".") or Path(item["id"]).name != item["id"]
                or item["input_hash"] != fresh._input_hash(np.array(item["input_ids"]))
                or not 0 < len(item["input_ids"]) <= suite.MAX_PROMPT_TOKENS
                or item["max_new_tokens"] != suite.CAPS[item["benchmark"]]):
            raise ValueError("invalid frozen task or token hash")


def freeze(root, seed0, replicas, samples, labels, scorer):
    fresh.separate(root, seed0)
    fresh.separate(root, replicas)
    identity = freeze_identity(samples, labels, seed0, replicas, scorer)
    old = read_record(root / "manifest.json")
    if old is not None:
        validate_manifest(old)
        if old["identity"] != identity:
            raise ValueError("frozen scope/code/runtime/artifact changed; use a new output root")
        print("Resume frozen public inputs; no retokenization", flush=True)
        return old
    tokenizer = fresh.load_tokenizer()
    items, receipts = suite.load_items(samples)
    for item in items:
        item["rendered"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        item["input_ids"] = tokenizer(item["rendered"], add_special_tokens=False).input_ids
        item["input_hash"] = fresh._input_hash(np.array(item["input_ids"]))
        if len(item["input_ids"]) > suite.MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt too long: {item['id']}; do not truncate/filter the frozen suite")
        # Exercise every selected oracle before spending GPU time.
        suite.score(item, "This is a preflight response.", ifeval=scorer)
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(fresh.MODEL_ID, revision=fresh.MODEL_REVISION)
    output_size = getattr(config, "text_config", config).vocab_size
    manifest = {"identity": identity, "datasets": receipts, "items": items,
                "tokenizer": fresh.tokenizer_identity(tokenizer, output_size),
                "stop_ids": fresh.stop_ids_for_source(tokenizer),
                "boundary": "Custom zero-shot public subsets, not leaderboard parity. No pretraining/semantic contamination guarantee. IFEval's train-named split is evaluated, never used to train. No generated code/tool execution."}
    manifest["fingerprint"] = suite.fingerprint(manifest)
    validate_manifest(manifest)
    save_record(root / "manifest.json", manifest)
    fresh.progress(root, "frozen", prompts=len(items), labels=labels,
                   prompt_tokens=max(len(i["input_ids"]) for i in items), fingerprint=manifest["fingerprint"])
    return manifest


class HFGenerator(fresh.HFBackend):
    @torch.no_grad()
    def generate_task(self, ids, cap):
        from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

        class FiniteScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if not torch.isfinite(scores).all():
                    raise ValueError("non-finite generation logits")
                return scores

        value = torch.tensor([ids], device=self.device)
        out = self.model.generate(
            input_ids=value, attention_mask=torch.ones_like(value),
            logits_processor=LogitsProcessorList([FiniteScores()]),
            generation_config=GenerationConfig(max_new_tokens=cap, do_sample=False,
                use_cache=True, eos_token_id=self.stop_ids, pad_token_id=self.stop_ids[0],
                repetition_penalty=1.0),
        )
        return out[0, len(ids):].tolist()


class GGUFGenerator(fresh.LlamaBackend):
    def generate_task(self, ids, cap):
        self.model.reset()
        self.model.eval(ids)
        output = []
        for _ in range(cap):
            logits = self.model.scores[self.model.n_tokens - 1]
            if not np.isfinite(logits).all():
                raise ValueError("non-finite GGUF generation logits")
            token = int(np.argmax(logits))
            output.append(token)
            if token in self.stop_ids or len(output) == cap:
                break
            self.model.eval([token])
        return output


def validate_record(record, item, identity):
    required = {"id": item["id"], "input_hash": item["input_hash"],
                "collection": suite.fingerprint(identity), "manifest": identity["manifest"],
                "benchmark": item["benchmark"]}
    if any(record.get(k) != v for k, v in required.items()):
        raise ValueError("saved prompt pairing/identity mismatch")
    tokens = record["continuation"]
    if not tokens or len(tokens) > item["max_new_tokens"] or any(type(v) is not int or v < 0 for v in tokens):
        raise ValueError("invalid saved generation")
    if any(t in identity["stop_ids"] for t in tokens[:-1]):
        raise ValueError("generation continued past stop token")
    if record["truncated"] != (tokens[-1] not in identity["stop_ids"]):
        raise ValueError("stop/truncation inconsistency")
    if record["truncated"] and len(tokens) != item["max_new_tokens"]:
        raise ValueError("generation ended early without EOS")


def collect_prompts(root, manifest, label, identity, backend, scorer):
    """Durable per-prompt results; a failed prompt never becomes a success/skip."""
    directory = root / "runs" / label
    completed = 0
    for item in manifest["items"]:
        path = directory / (item["id"] + ".json")
        old = read_record(path)
        if old is not None:
            validate_record(old, item, identity)
            completed += 1
            continue
        started = time.monotonic()
        fresh.progress(root, "prompt_start", label=label, prompt=item["id"],
                       completed=completed, total=len(manifest["items"]))
        try:
            tokens = backend.generate_task(item["input_ids"], item["max_new_tokens"])
            unmapped = sorted(set(tokens) - set(backend.tokenizer.get_vocab().values()))
            text = "" if unmapped else backend.tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            truncated = not tokens or tokens[-1] not in backend.stop_ids
            value = {"id": item["id"], "benchmark": item["benchmark"], "input_hash": item["input_hash"],
                     "manifest": manifest["fingerprint"], "collection": suite.fingerprint(identity),
                     "continuation": tokens, "output_text": text, "unmapped_ids": unmapped,
                     **suite.score(item, text, truncated=truncated, ifeval=scorer),
                     "seconds": time.monotonic() - started}
            validate_record(value, item, identity)
            save_record(path, value)
            completed += 1
            fresh.progress(root, "prompt_complete", label=label, prompt=item["id"],
                           completed=completed, total=len(manifest["items"]), seconds=value["seconds"],
                           generated_tokens=len(tokens), task_success=value["task_success"], truncated=truncated)
        except BaseException as error:
            save_record(directory / f"failure-{time.time_ns()}.json",
                        {"collection": suite.fingerprint(identity), "prompt": item["id"],
                         "error_type": type(error).__name__, "error": str(error)})
            raise


def run_collection(root, manifest, label, seed0, replicas, gguf_dir, scorer, device="cuda"):
    if label not in manifest["identity"]["labels"]:
        raise ValueError("arm not registered in the frozen experiment")
    tokenizer = fresh.load_tokenizer()
    if fresh.tokenizer_identity(tokenizer, manifest["tokenizer"]["size"]) != manifest["tokenizer"]:
        raise ValueError("tokenizer changed")
    stops = fresh.stop_ids_for_source(tokenizer)
    if stops != manifest["stop_ids"]:
        raise ValueError("source stopping policy changed")
    identity = {"manifest": manifest["fingerprint"], "source": fresh.source_identity(),
                "runtime": runtime(), "label": label, "stop_ids": stops, "device": device,
                "scorer": scorer.identity}
    if identity["source"] != manifest["identity"]["source"] or identity["runtime"] != manifest["identity"]["runtime"]:
        raise ValueError("code/runtime changed since freeze; do not mix runs")
    ledger = None
    if label.startswith("b5_"):
        artifact = artifact_path(label, seed0, replicas)
        fresh.separate(root, artifact)
        binding, prepared = preparation_binding(artifact, label)
        if binding != manifest["identity"]["artifacts"][label]:
            raise ValueError("frozen artifact changed")
        fresh.progress(root, "artifact_hash_audit", label=label)
        ledger = fresh.verify_prepared(artifact, prepared, prepared["identity"])
        identity["artifact"] = binding
    elif label != "source_fp16":
        identity["llama_build"] = fresh.llama_build_identity()
        other = "unsloth_ud_q4" if label == "gguf_bf16_bridge" else "gguf_bf16_bridge"
        bridge_identity = read_record(root / "runs" / other / "identity.json")
        if bridge_identity is not None and bridge_identity.get("llama_build") != identity["llama_build"]:
            raise ValueError("provider and BF16 bridge must use the same engine binary")
        descriptor = fresh.BF16 if label == "gguf_bf16_bridge" else fresh.UD_Q4
        identity.update(gguf=descriptor.__dict__, gguf_revision=fresh.GGUF_REVISION, gguf_input_policy="frozen-hf")
    identity = json.loads(json.dumps(identity))
    directory = root / "runs" / label
    old = read_record(directory / "identity.json")
    if old is not None and old != identity:
        raise ValueError("collection runtime/build changed; choose a new root")
    save_record(directory / "identity.json", identity)
    done = read_record(directory / "complete.json")
    pending = []
    for item in manifest["items"]:
        record = read_record(directory / (item["id"] + ".json"))
        if record is None:
            pending.append(item)
        else:
            validate_record(record, item, identity)
    if done is not None:
        if pending or done.get("identity") != identity or not done.get("complete"):
            raise ValueError("completion marker does not match prompt evidence")
        verify_gates(directory, label, identity)
        print(f"Resume {label}: all {len(manifest['items'])} prompts verified; no model load", flush=True)
        return
    fresh.set_seed(0)
    fresh.progress(root, "model_load", label=label, pending=len(pending))
    model = None
    try:
        if label.startswith("b5_"):
            from safetensors.torch import load_file

            model = fresh.load_packed_model(artifact / "checkpoint", device=device, dtype=torch.float16, fallback=False)
            before = fresh.packed_residency(model)
            if not before["passed"] or not before["one_shared_vocabulary_owner"] or before["backbone_fallback_cache_bytes"]:
                raise ValueError("packed residency failed")
            expected = load_file(str(artifact / "packed_probes.safetensors"))
            probes = fresh.capture_probes(model, fresh.probe_inputs(expected), device,
                                         **fresh.probe_settings(prepared["identity"]["config"]))
            parity = fresh.compare_probes(probes, expected, reload=True)
            save_record(directory / "reload_probe_report.json", {"identity": identity, "parity": parity})
            if not parity["passed"]:
                raise ValueError("packed reload probe failed; do not relax its tolerance")
            backend = HFGenerator(model, tokenizer, device, stops)
        elif label == "source_fp16":
            model, _, _ = fresh.experiment.load_hf_model(fresh.MODEL_ID, torch.float16, device, "multimodal_lm", fresh.MODEL_REVISION)
            backend = HFGenerator(model, tokenizer, device, stops)
        else:
            import llama_cpp

            if not llama_cpp.llama_supports_gpu_offload():
                raise ValueError("GGUF engine has no GPU offload; install the pinned CUDA build")
            path = fresh._download(gguf_dir, descriptor)
            if label == "unsloth_ud_q4":
                fresh._download(gguf_dir, fresh.MM_PROJ_F16)
                ledger = {"measured_artifact_bytes": fresh.UD_Q4.bytes + fresh.MM_PROJ_F16.bytes,
                          "vision_executed": False}
            model = fresh._llama(path, suite.MAX_PROMPT_TOKENS + max(suite.CAPS.values()), verbose=False)
            backend = GGUFGenerator(model, tokenizer, stops, manifest["tokenizer"]["size"])
            audit = fresh.audit_gguf_inputs(model, tokenizer, manifest, "frozen-hf")
            save_record(directory / "tokenizer_audit.json", {"identity": identity, **audit})
            fresh.progress(root, "tokenizer_audit", label=label, passed=audit["passed"], native_mismatches=len(audit["native_mismatches"]))
            if not audit["passed"]:
                raise ValueError("GGUF token-axis/byte audit failed")
        collect_prompts(root, manifest, label, identity, backend, scorer)
        after = fresh.packed_residency(model) if label.startswith("b5_") else None
        if after is not None and after != before:
            raise ValueError("packed residency changed")
        save_record(directory / "complete.json", {"complete": True, "identity": identity,
                    "ledger": ledger, "residency": after, "prompts": len(manifest["items"])})
    finally:
        if model is not None and label in ("gguf_bf16_bridge", "unsloth_ud_q4"):
            model.close()


def verify_gates(directory, label, identity):
    if label.startswith("b5_"):
        gate = read_record(directory / "reload_probe_report.json")
        marker = read_record(directory / "complete.json")
        residency = (marker or {}).get("residency") or {}
        passed = (gate or {}).get("parity", {}).get("passed") and residency.get("passed") and residency.get("one_shared_vocabulary_owner") and residency.get("backbone_fallback_cache_bytes") == 0
    elif label != "source_fp16":
        gate = read_record(directory / "tokenizer_audit.json")
        passed = (gate or {}).get("passed") and gate.get("policy") == "frozen-hf" and gate.get("manifest") == identity["manifest"]
    else:
        return
    if not passed or gate.get("identity") != identity:
        raise ValueError("missing/failed/mismatched collection gate")


def wilson(correct, total):
    if total == 0:
        return None
    p, z = correct / total, 1.959963984540054
    center = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    denominator = 1 + z * z / total
    return [(center - radius) / denominator, (center + radius) / denominator]


def summarize(root, manifest, scorer):
    validate_manifest(manifest)
    collections, identities, rows, missing = {}, {}, [], []
    items = manifest["items"]
    for label in manifest["identity"]["labels"]:
        directory = root / "runs" / label
        identity = read_record(directory / "identity.json")
        marker = read_record(directory / "complete.json")
        values = [read_record(directory / (i["id"] + ".json")) for i in items]
        if marker is None or any(v is None for v in values):
            missing.append({"label": label, "completed_prompts": sum(v is not None for v in values), "required_prompts": len(items)})
            continue
        if (not marker.get("complete") or marker["identity"] != identity
                or identity["manifest"] != manifest["fingerprint"] or identity["label"] != label
                or identity["stop_ids"] != manifest["stop_ids"]):
            raise ValueError("completion/identity mismatch")
        if identity["source"] != manifest["identity"]["source"] or identity["runtime"] != manifest["identity"]["runtime"] or identity["scorer"] != scorer.identity:
            raise ValueError("summary provenance mismatch")
        verify_gates(directory, label, identity)
        for item, value in zip(items, values):
            validate_record(value, item, identity)
            recalculated = suite.score(item, value["output_text"], truncated=value["truncated"], ifeval=scorer)
            if any(value.get(k) != v for k, v in recalculated.items()):
                raise ValueError("saved score differs from recomputed oracle")
        collections[label] = {v["id"]: v for v in values}
        identities[label] = identity
        for name in suite.BENCHMARKS:
            subset = [v for v in values if v["benchmark"] == name]
            correct = sum(v["task_success"] for v in subset)
            row = {"label": label, "benchmark": name, "prompts": len(subset), "correct": correct,
                   "accuracy": correct / len(subset), "wilson95": wilson(correct, len(subset)),
                   "truncated": sum(v["truncated"] for v in subset),
                   "unmapped_outputs": sum(bool(v["unmapped_ids"]) for v in subset),
                   "artifact_bytes": (marker.get("ledger") or {}).get("measured_artifact_bytes")}
            if name == "ifeval":
                for mode in ("strict", "loose"):
                    checks = [b for v in subset for b in v[f"instructions_{mode}"]]
                    row[f"checker_prompt_{mode}_accuracy"] = sum(v[f"prompt_{mode}"] for v in subset) / len(subset)
                    row[f"checker_instruction_{mode}_accuracy"] = sum(checks) / len(checks)
                row["checker_note"] = "Upstream checker metrics before the separate truncation guard. Primary accuracy requires strict prompt success and a completed response."
            else:
                row["format_failures"] = sum(not v["answer_format_valid"] for v in subset)
            rows.append(row)
    if ("gguf_bf16_bridge" in identities and "unsloth_ud_q4" in identities
            and identities["gguf_bf16_bridge"]["llama_build"] != identities["unsloth_ud_q4"]["llama_build"]):
        raise ValueError("provider and BF16 bridge used different engine binaries")
    pairs = [("source_fp16", label) for label in collections if label != "source_fp16"]
    pairs += [("gguf_bf16_bridge", "unsloth_ud_q4")]
    pairs += [("unsloth_ud_q4", f"b5_v{v}_s{s}") for s in (0, 1, 2) for v in (6, 8)]
    pairs += [(f"b5_v6_s{s}", f"b5_v8_s{s}") for s in (0, 1, 2)]
    contrasts = []
    for left, right in pairs:
        if left not in collections or right not in collections:
            continue
        for name in suite.BENCHMARKS:
            ids = [i["id"] for i in items if i["benchmark"] == name]
            a = np.array([collections[left][i]["task_success"] for i in ids], dtype=float)
            b = np.array([collections[right][i]["task_success"] for i in ids], dtype=float)
            delta = b - a
            rng = np.random.default_rng(suite.SELECTION_SEED)
            # Per-example paired resampling. Seeds share prompts: never pool them.
            draws = [float(delta[rng.integers(len(ids), size=len(ids))].mean()) for _ in range(4000)]
            contrasts.append({"left": left, "right": right, "benchmark": name, "prompts": len(ids),
                              "delta": float(delta.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
                              "correct_to_wrong": int(np.sum((a == 1) & (b == 0))),
                              "wrong_to_correct": int(np.sum((a == 0) & (b == 1)))})
    report = {"protocol": PROTOCOL, "manifest": manifest["fingerprint"], "complete": not missing,
              "missing": missing, "rows": rows, "paired_contrasts": contrasts, "promoted": False,
              "interpretation": "Per-benchmark custom zero-shot scores, not official leaderboard scores, an agent benchmark, Dynamic-v3 parity, or a throughput comparison. Descriptive unadjusted intervals; no pooled seed sample size or automatic winner. Missing arms are not successes."}
    save_record(root / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("setup", "freeze", "run", "summary"), required=True)
    parser.add_argument("--seed0-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--scorer-cache", type=Path, default=Path("/content/rotquant-public-scorers"))
    parser.add_argument("--gguf-dir", type=Path, default=Path("/content/unsloth-qwen35-4b-gguf"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed0-only", action="store_true")
    parser.add_argument("--label", choices=ALL_LABELS)
    parser.add_argument("--heartbeat-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.heartbeat_seconds <= 0:
        parser.error("heartbeat must be positive")
    configure_runtime()
    root = args.output_dir.resolve()
    with writer_lock(root / ".runner.lock"), fresh._Heartbeat(args.phase, seconds=args.heartbeat_seconds):
        if args.phase == "setup":
            print(json.dumps(suite.prepare_scorers(args.scorer_cache), indent=2))
            suite.IFEvalScorer(args.scorer_cache)
            return
        scorer = suite.IFEvalScorer(args.scorer_cache)
        if args.phase == "freeze":
            if args.seed0_root is None or args.replica_root is None:
                parser.error("freeze needs original --seed0-root and --replica-root")
            labels = list(REFERENCE_LABELS) + ["b5_v6_s0", "b5_v8_s0"] if args.seed0_only else list(ALL_LABELS)
            freeze(root, args.seed0_root, args.replica_root, args.samples, labels, scorer)
            return
        manifest = read_record(root / "manifest.json")
        validate_manifest(manifest)
        if scorer.identity != manifest["identity"]["scorer"]:
            raise ValueError("scorer changed")
        if args.phase == "summary":
            result = summarize(root, manifest, scorer)
            print(json.dumps(result, indent=2))
            if not result["complete"]:
                raise SystemExit("Run incomplete: see summary.json missing list")
        else:
            if args.label is None or args.seed0_root is None or args.replica_root is None:
                parser.error("run needs --label, --seed0-root and --replica-root")
            if not torch.cuda.is_available():
                raise RuntimeError("full-model run requires CUDA; CPU unit tests do not establish fit")
            run_collection(root, manifest, args.label, args.seed0_root, args.replica_root, args.gguf_dir, scorer)


if __name__ == "__main__":
    main()
