#!/usr/bin/env python3
"""Capture or score the frozen competitive suite with HF/RotQuant models.

The ``source`` phase stores the full FP16 teacher distributions once. The
``candidate`` phase can then run separately and resume prompt-by-prompt without
holding the source and candidate models in GPU memory at the same time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.checkpoint import load_packed_model
from rotquant.eval.competition import CompetitiveEvalProtocol
from rotquant.eval.competitive_collect import (
    SourceReference,
    continuation_prediction_logits,
    load_source_reference,
    save_source_reference,
    score_candidate,
    source_reference_path,
    verify_artifact_files,
)
from rotquant.eval.competitive_run import PromptObservation, RunFailure, RunMetadata
from rotquant.eval.data_manifest import read_dataset_manifest
from rotquant.utils import write_result
from scripts.run_experiment import load_hf_model


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _dtype(name: str) -> torch.dtype:
    try:
        value = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype {name!r}") from error
    if not isinstance(value, torch.dtype):
        raise TypeError(f"torch attribute {name!r} is not a dtype")
    return value


def _inputs(token_ids: tuple[int, ...], device: str) -> dict[str, torch.Tensor]:
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _generation_kwargs(tokenizer, tokens: int) -> dict[str, Any]:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    kwargs: dict[str, Any] = {
        "max_new_tokens": tokens,
        "min_new_tokens": tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if pad_id is not None:
        kwargs["pad_token_id"] = int(pad_id)
    return kwargs


@torch.no_grad()
def _generate(model, tokenizer, prompt, tokens: int) -> tuple[int, ...]:
    output = model.generate(**prompt, **_generation_kwargs(tokenizer, tokens))
    prompt_len = prompt["input_ids"].shape[-1]
    continuation = output[0, prompt_len:]
    if continuation.numel() != tokens:
        raise ValueError(
            f"generation returned {continuation.numel()} tokens; expected {tokens}"
        )
    return tuple(int(value) for value in continuation.detach().cpu())


@torch.no_grad()
def _teacher_forced_logits(model, prompt, continuation: tuple[int, ...]):
    suffix = torch.tensor(
        [continuation], dtype=torch.long, device=prompt["input_ids"].device
    )
    input_ids = torch.cat((prompt["input_ids"], suffix), dim=-1)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    output = model(**inputs, use_cache=False)
    return continuation_prediction_logits(
        output,
        prompt_tokens=prompt["input_ids"].shape[-1],
        continuation_tokens=len(continuation),
    )


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_common(args):
    protocol = CompetitiveEvalProtocol.from_manifest(_read_json(args.protocol))
    manifest = read_dataset_manifest(args.prompt_manifest)
    if manifest.role != "evaluation":
        raise ValueError("prompt manifest must have role=evaluation")
    if manifest.fingerprint != protocol.prompt_manifest_sha256:
        raise ValueError("prompt manifest does not match the protocol")
    return protocol, manifest


def capture_source(args) -> None:
    protocol, manifest = _load_common(args)
    dtype = _dtype(args.dtype)
    model, _model_tokenizer, loader = load_hf_model(
        protocol.model_id,
        dtype,
        args.device,
        args.model_loader,
        protocol.model_revision,
    )
    tokenizer = _load_protocol_tokenizer(protocol, args.trust_remote_code)
    model.eval()
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest.items, start=1):
        path = source_reference_path(args.reference_dir, item.token_sha256)
        if path.exists() and not args.force:
            try:
                load_source_reference(
                    path,
                    expected_item_sha256=item.token_sha256,
                    expected_protocol_fingerprint=protocol.fingerprint,
                    expected_tokens=protocol.generation_tokens,
                )
                print(f"resume source {index}/{len(manifest.items)}")
                continue
            except (OSError, ValueError):
                pass
        prompt = _inputs(item.token_ids, args.device)
        continuation = _generate(
            model, tokenizer, prompt, protocol.generation_tokens
        )
        logits = _teacher_forced_logits(model, prompt, continuation)
        save_source_reference(
            path,
            SourceReference(
                item_sha256=item.token_sha256,
                protocol_fingerprint=protocol.fingerprint,
                teacher_logits=logits,
                continuation=continuation,
            ),
        )
        print(f"captured source {index}/{len(manifest.items)}")
    index_payload = {
        "schema_version": 1,
        "protocol_fingerprint": protocol.fingerprint,
        "prompt_manifest_sha256": manifest.fingerprint,
        "model_id": protocol.model_id,
        "model_revision": protocol.model_revision,
        "model_loader": loader,
        "generation_tokens": protocol.generation_tokens,
        "prompt_item_sha256": list(protocol.prompt_item_sha256),
        "storage": "one object-free FP16 NumPy archive per prompt",
    }
    write_result(str(args.reference_dir / "source_reference_index.json"), index_payload)


def _load_candidate(args, protocol):
    dtype = _dtype(args.dtype)
    if args.candidate_kind == "rotquant":
        return load_packed_model(
            args.candidate,
            device=args.device,
            dtype=dtype,
            fallback=args.fallback,
            trust_remote_code=args.trust_remote_code,
        )
    if not args.candidate_revision:
        raise ValueError("--candidate-revision is required for an HF candidate")
    if args.candidate_revision.strip().casefold() in {
        "main", "master", "latest", "head"
    }:
        raise ValueError("candidate revision must be immutable")
    model, _tokenizer, _loader = load_hf_model(
        args.candidate,
        dtype,
        args.device,
        args.model_loader,
        args.candidate_revision,
    )
    return model


def _load_protocol_tokenizer(protocol, trust_remote_code: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        protocol.tokenizer_id,
        revision=protocol.tokenizer_revision,
        trust_remote_code=trust_remote_code,
    )


def _record_path(work_dir: Path, item_sha256: str) -> Path:
    return work_dir / "prompt_records" / f"{item_sha256}.json"


def _read_completed_observation(
    path: Path, collection_fingerprint: str
) -> PromptObservation | None:
    try:
        payload = _read_json(path)
        if (
            payload.get("status") != "observation"
            or payload.get("collection_fingerprint") != collection_fingerprint
        ):
            return None
        return PromptObservation.from_manifest(payload["payload"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def score_candidate_run(args) -> None:
    protocol, manifest = _load_common(args)
    metadata = RunMetadata.from_manifest(_read_json(args.run_metadata))
    if metadata.protocol_fingerprint != protocol.fingerprint:
        raise ValueError("run metadata uses a different protocol")
    verify_artifact_files(args.candidate, metadata.artifact_files)
    collection_fingerprint = hashlib.sha256(json.dumps(
        {
            "protocol_fingerprint": protocol.fingerprint,
            "metadata": metadata.manifest(),
            "candidate_kind": args.candidate_kind,
            "candidate": args.candidate,
            "candidate_revision": args.candidate_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    source_index = _read_json(args.reference_dir / "source_reference_index.json")
    if source_index.get("protocol_fingerprint") != protocol.fingerprint:
        raise ValueError("source-reference index uses a different protocol")
    tokenizer = _load_protocol_tokenizer(protocol, args.trust_remote_code)
    try:
        model = _load_candidate(args, protocol)
    except Exception as error:
        failure = RunFailure(
            stage="load",
            error_type=type(error).__name__,
            message=str(error) or repr(error),
        )
        _atomic_jsonl(args.observations, [])
        _atomic_jsonl(args.failures, [failure.manifest()])
        print(json.dumps({
            "observations": 0,
            "failures": 1,
            "expected": len(manifest.items),
            "load_failure": failure.manifest(),
        }, indent=2))
        return
    model.eval()
    records_dir = args.work_dir / "prompt_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(manifest.items, start=1):
        record_path = _record_path(args.work_dir, item.token_sha256)
        existing = (
            None if args.force
            else _read_completed_observation(record_path, collection_fingerprint)
        )
        if existing is not None:
            print(f"resume candidate {index}/{len(manifest.items)}")
            continue
        failure_stage = "load"
        try:
            source = load_source_reference(
                source_reference_path(args.reference_dir, item.token_sha256),
                expected_item_sha256=item.token_sha256,
                expected_protocol_fingerprint=protocol.fingerprint,
                expected_tokens=protocol.generation_tokens,
            )
            prompt = _inputs(item.token_ids, args.device)
            failure_stage = "generation"
            candidate_continuation = _generate(
                model, tokenizer, prompt, protocol.generation_tokens
            )
            failure_stage = "logits"
            logits = _teacher_forced_logits(model, prompt, source.continuation)
            failure_stage = "scoring"
            observation = score_candidate(
                item_sha256=item.token_sha256,
                domain=item.domain,
                source=source,
                candidate_logits=logits,
                candidate_continuation=candidate_continuation,
            )
            write_result(
                str(record_path),
                {
                    "status": "observation",
                    "collection_fingerprint": collection_fingerprint,
                    "payload": observation.manifest(),
                },
            )
            print(f"scored candidate {index}/{len(manifest.items)}")
        except Exception as error:  # preserve the structured audit trail and resume.
            failure = RunFailure(
                stage=failure_stage,
                error_type=type(error).__name__,
                message=str(error) or repr(error),
                item_sha256=item.token_sha256,
            )
            write_result(
                str(record_path),
                {
                    "status": "failure",
                    "collection_fingerprint": collection_fingerprint,
                    "payload": failure.manifest(),
                },
            )
            print(f"failed candidate {index}/{len(manifest.items)}: {failure.message}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    observations = []
    failures = []
    for item in manifest.items:
        path = _record_path(args.work_dir, item.token_sha256)
        try:
            record = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("collection_fingerprint") != collection_fingerprint:
            continue
        if record.get("status") == "observation":
            observations.append(
                PromptObservation.from_manifest(record["payload"]).manifest()
            )
        elif record.get("status") == "failure":
            failures.append(RunFailure.from_manifest(record["payload"]).manifest())
    _atomic_jsonl(args.observations, observations)
    _atomic_jsonl(args.failures, failures)
    print(json.dumps({
        "observations": len(observations),
        "failures": len(failures),
        "expected": len(manifest.items),
        "observations_path": str(args.observations),
        "failures_path": str(args.failures),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", type=Path, required=True)
    common.add_argument("--prompt-manifest", type=Path, required=True)
    common.add_argument("--reference-dir", type=Path, required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--dtype", default="float16")
    common.add_argument(
        "--model-loader",
        choices=("auto", "causal_lm", "multimodal_lm"),
        default="auto",
    )
    common.add_argument("--trust-remote-code", action="store_true")
    common.add_argument("--force", action="store_true")

    source = subparsers.add_parser("source", parents=[common])
    source.set_defaults(handler=capture_source)

    candidate = subparsers.add_parser("candidate", parents=[common])
    candidate.add_argument("--candidate-kind", choices=("hf", "rotquant"), required=True)
    candidate.add_argument("--candidate", required=True)
    candidate.add_argument("--candidate-revision")
    candidate.add_argument(
        "--run-metadata",
        type=Path,
        required=True,
        help="artifact identity from build_competitive_run_metadata.py",
    )
    candidate.add_argument("--fallback", action="store_true")
    candidate.add_argument("--work-dir", type=Path, required=True)
    candidate.add_argument("--observations", type=Path, required=True)
    candidate.add_argument("--failures", type=Path, required=True)
    candidate.set_defaults(handler=score_candidate_run)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
