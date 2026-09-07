"""Bounded packed-checkpoint conformance checks, separate from task accuracy.

These are reference-runtime gates, not evidence of fused-kernel performance.
Probe tensors are small safetensors files, never pickled model objects.
"""
from __future__ import annotations

import math
import os
import uuid
from pathlib import Path

import torch

from .checkpoint import MANIFEST_NAME, verify_checkpoint
from .linear import QuantLinear
from .vocabulary import PackedEmbedding, PackedOutputHead

PROTOTYPE_PARITY = {"max_abs_error": 0.125, "mean_abs_error": 0.005,
                    "mean_kl": 0.0001, "top1_agreement_min": 0.99}
RELOAD_PARITY = {"max_abs_error": 0.002, "mean_abs_error": 0.0002,
                 "mean_kl": 0.000001, "top1_agreement_min": 1.0}


@torch.inference_mode()
def capture_probes(model, prompts, device, *, prompt_tokens=64, logit_positions=4,
                   new_tokens=8):
    """Capture selected logits and short greedy traces on exact saved inputs."""
    if min(prompt_tokens, logit_positions, new_tokens) < 1 or not prompts:
        raise ValueError("probes require nonempty prompts and positive limits")
    tensors = {}
    model.eval()
    for index, prompt in enumerate(prompts):
        inputs = {key: value[..., :prompt_tokens].to(device)
                  for key, value in prompt.items() if torch.is_tensor(value)}
        if "input_ids" not in inputs or inputs["input_ids"].ndim != 2:
            raise ValueError("probes require batched input_ids")
        for key, value in inputs.items():
            tensors[f"p{index}.input.{key}"] = value.detach().cpu().contiguous().clone()
        output = model(**inputs, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        selected = logits[..., -logit_positions:, :].detach().float().cpu().contiguous()
        if not torch.isfinite(selected).all():
            raise ValueError("nonfinite probe logits")
        tensors[f"p{index}.logits"] = selected
        del output, logits
        tensors[f"p{index}.generated"] = model.generate(
            **inputs, max_new_tokens=new_tokens, do_sample=False,
        ).detach().cpu().contiguous()
    return tensors


def probe_inputs(tensors):
    prefixes = sorted({key.split(".")[0] for key in tensors}, key=lambda s: int(s[1:]))
    if not prefixes:
        raise ValueError("empty probe file")
    return [{key.split(".input.", 1)[1]: value for key, value in tensors.items()
             if key.startswith(prefix + ".input.")} for prefix in prefixes]


def save_probes(path, tensors):
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compare_probes(actual, expected, *, reload=False):
    """Fail closed on mismatched inputs/shapes; report numeric and greedy parity."""
    if set(actual) != set(expected) or not actual:
        raise ValueError("probe tensor keys differ or are empty")
    thresholds = RELOAD_PARITY if reload else PROTOTYPE_PARITY
    abs_sum = kl_sum = max_error = 0.0
    elements = positions = matches = generations = exact = 0
    for key in sorted(expected):
        a, b = actual[key], expected[key]
        if ".input." in key:
            if not torch.equal(a, b):
                raise ValueError("probe inputs differ")
        elif key.endswith(".generated"):
            generations += 1
            exact += int(torch.equal(a, b))
        elif key.endswith(".logits"):
            if a.shape != b.shape or a.numel() == 0:
                raise ValueError("probe logit shapes differ or are empty")
            a, b = a.double(), b.double()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite probe logits")
            difference = (a - b).abs()
            max_error = max(max_error, difference.max().item())
            abs_sum += difference.sum().item()
            elements += difference.numel()
            la, lb = a.log_softmax(-1), b.log_softmax(-1)
            kl_sum += (lb.exp() * (lb - la)).sum().item()
            positions += a.numel() // a.shape[-1]
            matches += a.argmax(-1).eq(b.argmax(-1)).sum().item()
        else:
            raise ValueError(f"unexpected probe key: {key}")
    if not positions or not generations:
        raise ValueError("probe logits and generated sequences are required")
    measures = {"max_abs_error": max_error, "mean_abs_error": abs_sum / elements,
                "mean_kl": max(0.0, kl_sum / positions), "top1_agreement": matches / positions,
                "positions": positions, "generation_probes": generations,
                "exact_generation_probes": exact}
    guards = {key: measures[key] <= thresholds[key]
              for key in ("max_abs_error", "mean_abs_error", "mean_kl")}
    guards["top1"] = measures["top1_agreement"] >= thresholds["top1_agreement_min"]
    if reload:
        guards["exact_generation"] = exact == generations
    return {**measures, "thresholds": thresholds, "guards": guards,
            "passed": all(guards.values()), "comparison": "reload" if reload else "prototype",
            "boundary": "Bounded numerical probes, not task accuracy or full-suite equivalence."}


def packed_residency(model):
    """Inspect live model ownership and caches; do not conflate this with VRAM."""
    embeddings = [m for m in model.modules() if isinstance(m, PackedEmbedding)]
    heads = [m for m in model.modules() if isinstance(m, PackedOutputHead)]
    if len(embeddings) != 1 or len(heads) != 1 or embeddings[0].owner is not heads[0].owner:
        raise ValueError("requires one shared packed vocabulary owner")
    owner = embeddings[0].owner
    if model.get_input_embeddings() is not embeddings[0] or model.get_output_embeddings() is not heads[0]:
        raise ValueError("packed vocabulary aliases do not match model accessors")
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    if not layers or any(m.fallback or m._fp_cache is not None for m in layers):
        raise ValueError("packed validation forbids dense backbone fallback caches")
    shape = (owner.vocab_size, owner.hidden_size)
    if any(tuple(p.shape) == shape for p in model.parameters()):
        raise ValueError("dense vocabulary parameter remains live")
    if any(q.scales.dtype != torch.float16 for q in owner.chunks):
        raise ValueError("vocabulary scale storage was recast")
    return {"passed": True, "one_shared_vocabulary_owner": True,
            "dense_vocabulary_parameters": 0, "backbone_fallback_cache_bytes": 0,
            "quantized_layers": len(layers), "vocabulary_payload_bytes": owner.packed_bytes(),
            "vocabulary_fingerprint": owner.fingerprint, "projection_mode": owner.projection_mode,
            "execution_dtype": str(owner.execution_dtype),
            "boundary": "Model-owned persistent state only; transient dequantization still occurs."}


def audit_artifact(path, *, expected_manifest_sha256=None):
    """Reconcile actual file/tensor bytes and reject extra or dense vocabulary state."""
    from safetensors import safe_open

    path = Path(path)
    manifest = verify_checkpoint(path, expected_manifest_sha256=expected_manifest_sha256)
    vocabulary = manifest.get("tied_vocabulary")
    if vocabulary is None:
        raise ValueError("vocabulary validation requires a v3 shared-owner checkpoint")
    expected = set(manifest["files_sha256"]) | {MANIFEST_NAME}
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("artifact has unrecorded/missing files or symlinks")
    files = {name: (path / name).stat().st_size for name in sorted(actual)}
    payload = 0
    tensor_counts = {}
    shape = vocabulary["shape"]
    aliases = {vocabulary["embedding"] + ".weight", vocabulary["head"] + ".weight"}
    packed_keys = []
    for chunk in vocabulary["chunks"]:
        packed_keys.extend([chunk["packed"]["tensor"], chunk["scales"]])
    if len(set(packed_keys)) != len(packed_keys):
        raise ValueError("vocabulary chunks reuse code or scale tensor keys")
    codebooks = {q["codebook"]["centroids"] for q in vocabulary["chunks"]}
    if len(codebooks) != 1:
        raise ValueError("vocabulary chunks do not share one serialized codebook")
    element_sizes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
                     "F16": 2, "BF16": 2, "F32": 4, "I32": 4, "U32": 4,
                     "F64": 8, "I64": 8, "U64": 8}
    vocabulary_tensor_bytes = 0
    for filename in (manifest["model_state"], manifest["packed_state"]):
        with safe_open(path / filename, framework="pt") as handle:
            keys = handle.keys()
            tensor_counts[filename] = len(keys)
            for key in keys:
                item = handle.get_slice(key)
                if filename == manifest["model_state"] and (key in aliases or item.get_shape() == shape):
                    raise ValueError("dense vocabulary is serialized in model state")
                size = math.prod(item.get_shape()) * element_sizes[item.get_dtype()]
                payload += size
                if filename == manifest["packed_state"] and key in set(packed_keys) | codebooks:
                    vocabulary_tensor_bytes += size
    total = sum(files.values())
    if payload > total:
        raise ValueError("tensor payload exceeds file bytes")
    return {"passed": True, "measured_artifact_bytes": total, "files": files,
            "tensor_counts": tensor_counts, "tensor_payload_bytes": payload,
            "container_and_auxiliary_file_bytes": total - payload,
            "vocabulary_codes_scales_codebook_bytes": vocabulary_tensor_bytes,
            "vocabulary_rotation_bytes": shape[1], "dense_vocabulary_serialized": False,
            "boundary": "All files in the self-contained checkpoint, including retained vision tensors and tokenizer/processor files; text-only execution is validated."}
