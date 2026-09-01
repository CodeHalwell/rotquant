#!/usr/bin/env python3
"""Measure the released Unsloth Qwen3.5-4B Dynamic GGUF against BF16.

The comparison is deliberately engine-normalized: both the BF16 teacher and
UD-Q4_K_XL candidate run through one pinned llama.cpp build on the exact C4
token sequences used by the RotQuant W4A8 development stage. Full-vocabulary
teacher logits are persisted prompt-by-prompt, so an interrupted Colab can
resume without loading both models at once.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotquant.eval.statistics import bootstrap_report
from rotquant.utils import write_result
from scripts.run_experiment import build_calib_loader

MODEL_ID = "unsloth/Qwen3.5-4B"
MODEL_REVISION = "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
GGUF_REPO = "unsloth/Qwen3.5-4B-GGUF"
GGUF_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
LLAMA_CPP_PYTHON_REVISION = "3691546f1c9e0c1bf93323dff02230bd959cf562"
LLAMA_CPP_ENGINE_REVISION = "4df29be4f4c3673f428170fda944a5b19f743bb8"
PROTOCOL = "qwen35-4b-unsloth-gguf-kl-v1"
ROTQUANT_W4A8_INPUT_HASHES = (
    "cf0fb236893f847ff99cb0f571d38604fcb9446ee9df3b5d780a6fa7bb8e6996",
    "f798e40153c544c8469d6c036999956e3178fdefc533a10b882348409f643f85",
    "76dfffdaf3002f54dc29343d066c32626f61fb6aa530e337155d6f46c812dc1b",
    "a68977746b8c8a843f193c2b41b4401a92ea7bda637cd894196d4d6b5dd3d0ef",
)


@dataclass(frozen=True)
class ReleasedFile:
    name: str
    bytes: int
    sha256: str


BF16 = ReleasedFile(
    "Qwen3.5-4B-BF16.gguf",
    8_424_393_632,
    "9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a",
)
UD_Q4 = ReleasedFile(
    "Qwen3.5-4B-UD-Q4_K_XL.gguf",
    2_912_109_728,
    "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7",
)
MM_PROJ_F16 = ReleasedFile(
    "mmproj-F16.gguf",
    672_423_616,
    "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_released_file(path: Path, expected: ReleasedFile) -> None:
    if not path.is_file():
        raise ValueError(f"released artifact is missing: {path}")
    if path.stat().st_size != expected.bytes:
        raise ValueError(
            f"{expected.name} has {path.stat().st_size} bytes; expected "
            f"{expected.bytes}"
        )
    digest = _hash_file(path)
    if digest != expected.sha256:
        raise ValueError(
            f"{expected.name} SHA-256 is {digest}; expected {expected.sha256}"
        )


def _download(directory: Path, expected: ReleasedFile) -> Path:
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(
        repo_id=GGUF_REPO,
        filename=expected.name,
        revision=GGUF_REVISION,
        local_dir=directory,
    ))
    verify_released_file(path, expected)
    return path


def _input_hash(token_ids: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(token_ids, dtype=np.int64).tobytes()
    ).hexdigest()


def build_prompt_manifest(
    *, batches: int, prompt_len: int, skip: int
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Recreate the exact held-out C4 token batches from the W4A8 stage."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    prompts = build_calib_loader(
        tokenizer,
        batches,
        prompt_len,
        "cpu",
        skip=skip,
        revision=C4_REVISION,
    )
    arrays = [
        batch["input_ids"][0].detach().cpu().to(torch.int64).numpy().copy()
        for batch in prompts
    ]
    manifest = {
        "protocol": PROTOCOL,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_loader": type(tokenizer).__name__,
        "dataset": "allenai/c4",
        "dataset_revision": C4_REVISION,
        "split": "train",
        "selection_scheme": "eligible-skip-before-exclusion-v2",
        "skip": skip,
        "batches": batches,
        "prompt_len": prompt_len,
        "prompts": [
            {
                "index": index,
                "source_row": int(batch.source_row),
                "input_hash": _input_hash(token_ids),
                "tokens": len(token_ids),
            }
            for index, (batch, token_ids) in enumerate(zip(prompts, arrays, strict=True))
        ],
    }
    hashes = tuple(prompt["input_hash"] for prompt in manifest["prompts"])
    matches_completed_rotquant = (
        batches == 4 and prompt_len == 512 and skip == 4096
    )
    if matches_completed_rotquant and hashes != ROTQUANT_W4A8_INPUT_HASHES:
        raise RuntimeError(
            "default Unsloth prompts do not match the completed RotQuant "
            "W4A8 input hashes"
        )
    manifest["completed_rotquant_w4a8_input_hashes_match"] = (
        matches_completed_rotquant and hashes == ROTQUANT_W4A8_INPUT_HASHES
    )
    manifest["fingerprint"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return manifest, arrays


def _llama(model_path: Path, prompt_len: int, *, verbose: bool):
    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise RuntimeError(
            "llama-cpp-python is required. Install the pinned revision "
            f"{LLAMA_CPP_PYTHON_REVISION} with CMAKE_ARGS=-DGGML_CUDA=on."
        ) from error
    return Llama(
        model_path=str(model_path),
        n_ctx=prompt_len,
        n_batch=prompt_len,
        n_ubatch=min(prompt_len, 512),
        n_gpu_layers=-1,
        logits_all=True,
        verbose=verbose,
    )


def evaluate_logits(model, token_ids: np.ndarray) -> np.ndarray:
    """Return full-vocabulary logits predicting tokens 1..N-1."""

    values = [int(value) for value in token_ids]
    model.reset()
    model.eval(values)
    raw_scores = getattr(model, "scores", None)
    if raw_scores is None:
        raw_scores = getattr(model, "_scores", None)
    if raw_scores is None:
        raise RuntimeError("llama-cpp-python did not expose logits_all scores")
    scores = np.asarray(raw_scores)
    if scores.ndim != 2 or scores.shape[0] < len(values):
        raise ValueError(
            f"llama.cpp returned logits with shape {scores.shape} for "
            f"{len(values)} tokens"
        )
    result = np.asarray(scores[:len(values) - 1], dtype=np.float32).copy()
    if result.shape[0] != len(values) - 1 or result.shape[1] < 2:
        raise ValueError("llama.cpp returned an invalid full-logit matrix")
    return result


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def distribution_metrics(
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
    targets: np.ndarray,
    *,
    device: str,
    chunk_tokens: int = 16,
) -> dict[str, list[float] | list[bool]]:
    """Compute exact full-distribution metrics in bounded token chunks."""

    if teacher_logits.shape != candidate_logits.shape:
        raise ValueError("teacher and candidate logit shapes differ")
    if teacher_logits.ndim != 2 or len(targets) != teacher_logits.shape[0]:
        raise ValueError("targets must contain one ID per logit row")
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    output: dict[str, list[Any]] = {
        "teacher_kl": [],
        "top1_matches": [],
        "source_nll": [],
        "candidate_nll": [],
    }
    scoring_device = torch.device(device)
    for start in range(0, len(targets), chunk_tokens):
        stop = min(start + chunk_tokens, len(targets))
        teacher = torch.as_tensor(
            teacher_logits[start:stop], device=scoring_device, dtype=torch.float32
        )
        candidate = torch.as_tensor(
            candidate_logits[start:stop], device=scoring_device, dtype=torch.float32
        )
        target = torch.as_tensor(
            targets[start:stop], device=scoring_device, dtype=torch.long
        )
        teacher_log_prob = torch.log_softmax(teacher, dim=-1)
        candidate_log_prob = torch.log_softmax(candidate, dim=-1)
        teacher_prob = teacher_log_prob.exp()
        kl = torch.sum(
            teacher_prob * (teacher_log_prob - candidate_log_prob), dim=-1
        ).clamp_min(0.0)
        rows = torch.arange(stop - start, device=scoring_device)
        output["teacher_kl"].extend(float(value) for value in kl.cpu())
        output["top1_matches"].extend(
            bool(value) for value in teacher.argmax(-1).eq(candidate.argmax(-1)).cpu()
        )
        output["source_nll"].extend(
            float(value) for value in (-teacher_log_prob[rows, target]).cpu()
        )
        output["candidate_nll"].extend(
            float(value) for value in (-candidate_log_prob[rows, target]).cpu()
        )
    return output


def _valid_source_reference(path: Path, manifest_fingerprint: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                payload["manifest_fingerprint"].tobytes().decode("ascii")
                == manifest_fingerprint
                and payload["source_sha256"].tobytes().decode("ascii")
                == BF16.sha256
                and payload["logits"].ndim == 2
            )
    except (OSError, KeyError, ValueError, UnicodeDecodeError):
        return False


def _capture_source(
    model_path: Path,
    prompts: list[np.ndarray],
    manifest: dict[str, Any],
    reference_dir: Path,
    *,
    force: bool,
    verbose: bool,
) -> None:
    pending = [
        index for index in range(len(prompts))
        if force or not _valid_source_reference(
            reference_dir / f"prompt-{index:03d}.npz", manifest["fingerprint"]
        )
    ]
    if not pending:
        print("resume Unsloth BF16 source: all prompt logits are present", flush=True)
        return
    model = _llama(model_path, manifest["prompt_len"], verbose=verbose)
    try:
        for index in pending:
            logits = evaluate_logits(model, prompts[index])
            _atomic_npz(
                reference_dir / f"prompt-{index:03d}.npz",
                manifest_fingerprint=np.frombuffer(
                    manifest["fingerprint"].encode("ascii"), dtype=np.uint8
                ),
                source_sha256=np.frombuffer(
                    BF16.sha256.encode("ascii"), dtype=np.uint8
                ),
                input_hash=np.frombuffer(
                    manifest["prompts"][index]["input_hash"].encode("ascii"),
                    dtype=np.uint8,
                ),
                logits=logits.astype(np.float16),
            )
            print(f"captured Unsloth BF16 source {index + 1}/{len(prompts)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _score_candidate(
    model_path: Path,
    prompts: list[np.ndarray],
    manifest: dict[str, Any],
    reference_dir: Path,
    record_dir: Path,
    *,
    force: bool,
    verbose: bool,
    scoring_device: str,
    collection_fingerprint: str,
) -> list[dict[str, Any]]:
    record_dir.mkdir(parents=True, exist_ok=True)
    model = _llama(model_path, manifest["prompt_len"], verbose=verbose)
    records = []
    try:
        for index, tokens in enumerate(prompts):
            record_path = record_dir / f"prompt-{index:03d}.json"
            if record_path.exists() and not force:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if (
                    record.get("manifest_fingerprint") == manifest["fingerprint"]
                    and record.get("collection_fingerprint")
                    == collection_fingerprint
                ):
                    records.append(record)
                    print(f"resume Unsloth candidate {index + 1}/{len(prompts)}", flush=True)
                    continue
            with np.load(
                reference_dir / f"prompt-{index:03d}.npz", allow_pickle=False
            ) as source:
                teacher = np.asarray(source["logits"])
            candidate = evaluate_logits(model, tokens)
            values = distribution_metrics(
                teacher,
                candidate,
                tokens[1:],
                device=scoring_device,
            )
            record = {
                "manifest_fingerprint": manifest["fingerprint"],
                "collection_fingerprint": collection_fingerprint,
                "input_hash": manifest["prompts"][index]["input_hash"],
                **values,
            }
            write_result(str(record_path), record)
            records.append(record)
            print(f"scored Unsloth candidate {index + 1}/{len(prompts)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def _aggregate(records: list[dict[str, Any]], *, draws: int, seed: int) -> dict[str, Any]:
    teacher_kl = [value for record in records for value in record["teacher_kl"]]
    matches = [value for record in records for value in record["top1_matches"]]
    source_nll = [value for record in records for value in record["source_nll"]]
    candidate_nll = [
        value for record in records for value in record["candidate_nll"]
    ]
    nll_delta = [right - left for left, right in zip(
        source_nll, candidate_nll, strict=True
    )]
    token_kl = torch.tensor(teacher_kl, dtype=torch.float32)
    return {
        "prompts": len(records),
        "tokens": len(teacher_kl),
        "mean_teacher_kl": float(token_kl.mean()),
        "median_teacher_kl": float(torch.quantile(token_kl, 0.5)),
        "p95_teacher_kl": float(torch.quantile(token_kl, 0.95)),
        "max_teacher_kl": float(token_kl.max()),
        "top1_agreement": sum(bool(value) for value in matches) / len(matches),
        "source_nll": float(np.mean(source_nll)),
        "candidate_nll": float(np.mean(candidate_nll)),
        "nll_delta": float(np.mean(nll_delta)),
        "input_hashes": [record["input_hash"] for record in records],
        "paired_token_bootstrap": bootstrap_report(
            {
                "mean_teacher_kl": teacher_kl,
                "top1_agreement": [float(value) for value in matches],
                "nll_delta": nll_delta,
            },
            draws=draws,
            seed=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--skip", type=int, default=4096)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=1701)
    parser.add_argument(
        "--scoring-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batches < 1 or args.prompt_len < 2 or args.skip < 0:
        raise ValueError("batches/prompt-len must be positive and skip non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, prompts = build_prompt_manifest(
        batches=args.batches, prompt_len=args.prompt_len, skip=args.skip
    )
    write_result(str(args.output_dir / "prompt_manifest.json"), manifest)

    references = args.output_dir / "bf16_references"
    if not all(
        _valid_source_reference(
            references / f"prompt-{index:03d}.npz", manifest["fingerprint"]
        )
        for index in range(len(prompts))
    ) or args.force:
        bf16_path = _download(args.artifact_dir, BF16)
        _capture_source(
            bf16_path, prompts, manifest, references,
            force=args.force, verbose=args.verbose,
        )
    else:
        print("resume Unsloth BF16 source without re-downloading BF16", flush=True)

    candidate_path = _download(args.artifact_dir, UD_Q4)
    projector_path = _download(args.artifact_dir, MM_PROJ_F16)
    collection_fingerprint = hashlib.sha256(json.dumps(
        {
            "protocol": PROTOCOL,
            "prompt_manifest_fingerprint": manifest["fingerprint"],
            "candidate_sha256": UD_Q4.sha256,
            "projector_sha256": MM_PROJ_F16.sha256,
            "llama_cpp_python_revision": LLAMA_CPP_PYTHON_REVISION,
            "llama_cpp_engine_revision": LLAMA_CPP_ENGINE_REVISION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    records = _score_candidate(
        candidate_path,
        prompts,
        manifest,
        references,
        args.output_dir / "candidate_records",
        force=args.force,
        verbose=args.verbose,
        scoring_device=args.scoring_device,
        collection_fingerprint=collection_fingerprint,
    )
    metrics = _aggregate(
        records, draws=args.bootstrap_draws, seed=args.bootstrap_seed
    )
    result = {
        "protocol": PROTOCOL,
        "scope": (
            "same-engine BF16-GGUF to UD-Q4_K_XL text-backbone KL; F16 "
            "projector counted in deployed candidate bytes but not executed"
        ),
        "engine": "llama.cpp via llama-cpp-python",
        "llama_cpp_python_revision": LLAMA_CPP_PYTHON_REVISION,
        "llama_cpp_engine_revision": LLAMA_CPP_ENGINE_REVISION,
        "gguf_repo": GGUF_REPO,
        "gguf_revision": GGUF_REVISION,
        "source": BF16.__dict__,
        "candidate": {
            "files": [UD_Q4.__dict__, MM_PROJ_F16.__dict__],
            "executed_file": candidate_path.name,
            "counted_projector": projector_path.name,
            "complete_artifact_bytes": UD_Q4.bytes + MM_PROJ_F16.bytes,
        },
        "prompt_manifest_fingerprint": manifest["fingerprint"],
        "collection_fingerprint": collection_fingerprint,
        "completed_rotquant_w4a8_input_hashes_match": manifest[
            "completed_rotquant_w4a8_input_hashes_match"
        ],
        "metrics": metrics,
        "comparison_warning": (
            "Compare with RotQuant only when input_hashes match. This nearest "
            "released Unsloth Q4 artifact is not within the registered 1% "
            "matched-byte gate of the current RotQuant W4 artifact."
        ),
    }
    write_result(str(args.output_dir / "unsloth_ud_q4_kl.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
