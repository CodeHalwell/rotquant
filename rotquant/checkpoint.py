"""Safe, reloadable checkpoints for models patched with :class:`QuantLinear`.

The ordinary module state (embeddings, norms, unquantized projections, biases,
rotations, and optional adapters) is stored separately from packed quantizer
state.  The latter lives in dataclasses rather than registered buffers, so a
plain ``state_dict`` is incomplete.  A small JSON manifest describes how to
rebuild each ``QuantLinear`` before the ordinary state is loaded.

No pickle is used.  The artifact consists of Transformers configuration files,
``rotquant_model.safetensors``, ``rotquant_packed.safetensors``, and
``rotquant_config.json``.  Cached fallback weights are deliberately excluded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .adapters import resolve_model_adapter
from .codebooks import ScalarCodebook
from .format import (
    CURRENT_PACKING,
    FORMAT_NAME,
    FORMAT_VERSION,
    validate_checkpoint_manifest,
)
from .linear import QuantLinear
from .pack import PackedTensor
from .quantize import QuantizedWeight
from .rotate import (
    ButterflyRotation,
    DenseOrthogonal,
    Identity,
    LearnedRotation,
    RandomizedHadamard,
    Rotation,
)

MANIFEST_NAME = "rotquant_config.json"
MODEL_STATE_NAME = "rotquant_model.safetensors"
PACKED_STATE_NAME = "rotquant_packed.safetensors"


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import load_model, save_file, save_model
    except ImportError as exc:  # pragma: no cover - exercised without eval extra
        raise RuntimeError(
            "packed checkpoint support requires safetensors; install "
            "rotquant[eval] or safetensors"
        ) from exc
    return safe_open, load_model, save_file, save_model


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _resolve_dtype(value: str | torch.dtype | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = value or "float16"
    if name.startswith("torch."):
        name = name.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unknown torch dtype: {value!r}")
    return dtype


def _get_parent(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _rotation_spec(rotation: Rotation) -> dict[str, Any]:
    spec: dict[str, Any] = {"dim": rotation.dim}
    if isinstance(rotation, Identity):
        spec["kind"] = "identity"
    elif isinstance(rotation, RandomizedHadamard):
        spec.update(kind="fwht", block=rotation.block)
    elif isinstance(rotation, ButterflyRotation):
        spec.update(kind="butterfly", block=rotation.block)
    elif isinstance(rotation, DenseOrthogonal):
        spec["kind"] = "dense"
    elif isinstance(rotation, LearnedRotation):
        spec["kind"] = "learned"
    else:
        raise TypeError(
            f"unsupported activation rotation in checkpoint: {type(rotation).__name__}"
        )
    return spec


def _build_rotation(spec: dict[str, Any]) -> Rotation:
    kind = spec["kind"]
    dim = int(spec["dim"])
    if kind == "identity":
        return Identity(dim)
    if kind == "fwht":
        return RandomizedHadamard(dim, block=int(spec["block"]), seed=0)
    if kind == "butterfly":
        return ButterflyRotation(dim, block=int(spec["block"]), seed=0)
    if kind == "dense":
        return DenseOrthogonal(dim, seed=0)
    if kind == "learned":
        return LearnedRotation(dim, seed=0)
    raise ValueError(f"unsupported checkpoint rotation kind: {kind!r}")


def _packed_spec(
    tensors: dict[str, torch.Tensor],
    key: str,
    packed: PackedTensor | None,
) -> dict[str, Any] | None:
    if packed is None:
        return None
    tensors[key] = packed.data.detach().to("cpu").contiguous()
    return {
        "tensor": key,
        "shape": list(packed.shape),
        "bits": packed.bits,
        "numel": packed.numel,
    }


def _tensor_spec(
    tensors: dict[str, torch.Tensor],
    key: str,
    tensor: torch.Tensor | None,
    *,
    clone: bool = False,
) -> str | None:
    if tensor is None:
        return None
    value = tensor.detach().to("cpu").contiguous()
    # Codebooks may be shared by every layer. Safetensors forbids shared
    # storage, and these vectors are tiny, so give each entry independent data.
    tensors[key] = value.clone() if clone else value
    return key


def _quantized_weight_spec(
    index: int,
    qweight: QuantizedWeight,
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    prefix = f"layer_{index:05d}"
    residual_codebook = None
    if qweight.residual_codebook is not None:
        residual_codebook = {
            "name": qweight.residual_codebook.name,
            "centroids": _tensor_spec(
                tensors,
                f"{prefix}.residual_codebook",
                qweight.residual_codebook.centroids,
                clone=True,
            ),
        }
    return {
        "packed": _packed_spec(tensors, f"{prefix}.packed", qweight.packed),
        "scales": _tensor_spec(tensors, f"{prefix}.scales", qweight.scales),
        "codebook": {
            "name": qweight.codebook.name,
            "centroids": _tensor_spec(
                tensors,
                f"{prefix}.codebook",
                qweight.codebook.centroids,
                clone=True,
            ),
        },
        "group_size": qweight.group_size,
        "out_features": qweight.out_features,
        "in_features": qweight.in_features,
        "residual_packed": _packed_spec(
            tensors, f"{prefix}.residual_packed", qweight.residual_packed
        ),
        "residual_scales": _tensor_spec(
            tensors, f"{prefix}.residual_scales", qweight.residual_scales
        ),
        "residual_codebook": residual_codebook,
        "sketch": _packed_spec(tensors, f"{prefix}.sketch", qweight.sketch),
        "sketch_row_norms": _tensor_spec(
            tensors, f"{prefix}.sketch_row_norms", qweight.sketch_row_norms
        ),
        "sketch_k": qweight.sketch_k,
        "sketch_seed": qweight.sketch_seed,
        "scale_group_size": qweight.scale_group_size,
        "scale_bits_main": qweight.scale_bits_main,
        "scale_bits_residual": qweight.scale_bits_residual,
    }


def _read_packed(handle, spec: dict[str, Any] | None) -> PackedTensor | None:
    if spec is None:
        return None
    return PackedTensor(
        data=handle.get_tensor(spec["tensor"]),
        shape=tuple(spec["shape"]),
        bits=int(spec["bits"]),
        numel=int(spec["numel"]),
    )


def _read_tensor(handle, key: str | None) -> torch.Tensor | None:
    return None if key is None else handle.get_tensor(key)


def _read_codebook(
    handle, spec: dict[str, Any] | None
) -> ScalarCodebook | None:
    if spec is None:
        return None
    return ScalarCodebook(
        handle.get_tensor(spec["centroids"]), name=spec.get("name", "scalar")
    )


def _read_quantized_weight(handle, spec: dict[str, Any]) -> QuantizedWeight:
    packed = _read_packed(handle, spec["packed"])
    codebook = _read_codebook(handle, spec["codebook"])
    if packed is None or codebook is None:
        raise ValueError("primary packed tensor and codebook are required")
    return QuantizedWeight(
        packed=packed,
        scales=_read_tensor(handle, spec.get("scales")),
        codebook=codebook,
        group_size=int(spec["group_size"]),
        out_features=int(spec["out_features"]),
        in_features=int(spec["in_features"]),
        residual_packed=_read_packed(handle, spec.get("residual_packed")),
        residual_scales=_read_tensor(handle, spec.get("residual_scales")),
        residual_codebook=_read_codebook(handle, spec.get("residual_codebook")),
        sketch=_read_packed(handle, spec.get("sketch")),
        sketch_row_norms=_read_tensor(handle, spec.get("sketch_row_norms")),
        sketch_k=int(spec.get("sketch_k", 0)),
        sketch_seed=int(spec.get("sketch_seed", 0)),
        scale_group_size=spec.get("scale_group_size"),
        scale_bits_main=float(spec.get("scale_bits_main", 16.0)),
        scale_bits_residual=float(spec.get("scale_bits_residual", 16.0)),
    )


def _first_float_dtype(model: nn.Module) -> torch.dtype:
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def save_packed_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    base_model_revision: str | None = None,
    model_loader: str = "auto",
    tokenizer=None,
    processor=None,
    deployment_metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save a self-contained, pickle-free packed model artifact.

    ``model`` must already contain ``QuantLinear`` modules.  The cached
    dequantized fallback weights are ordinary Python attributes and are not
    written. Optional deployment metadata must be a plain JSON object and is
    embedded in the manifest. The manifest is saved last so a partially written
    directory is never mistaken for a complete checkpoint.
    """
    _, _, save_file, save_model = _require_safetensors()
    serialized_deployment = None
    if deployment_metadata is not None:
        if not isinstance(deployment_metadata, dict):
            raise TypeError("deployment_metadata must be a JSON object")
        # Reject tensors and other process-local values before writing files,
        # while detaching the artifact from a caller-owned mutable object.
        serialized_deployment = json.loads(json.dumps(deployment_metadata))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"checkpoint directory is not empty: {output}; choose a new path "
            "or pass overwrite=True"
        )
    output.mkdir(parents=True, exist_ok=True)

    modules = []
    packed_tensors: dict[str, torch.Tensor] = {}
    for index, (name, module) in enumerate(
        item for item in model.named_modules() if isinstance(item[1], QuantLinear)
    ):
        if module._log_scale_multiplier is not None:
            raise ValueError(f"cannot export uncommitted scale training state: {name}")
        modules.append(
            {
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
                "rotation": _rotation_spec(module.act_rotation),
                "qweight": _quantized_weight_spec(
                    index, module.qweight, packed_tensors
                ),
                "lora_rank": module.lora_rank,
                "lora_alpha": module.lora_alpha,
                "lora_dtype": (
                    _dtype_name(module.lora_A.dtype)
                    if module.lora_A is not None
                    else None
                ),
            }
        )
    if not modules:
        raise ValueError("model contains no QuantLinear modules to export")

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "save_pretrained"):
        raise TypeError(
            "packed checkpoint export requires model.config.save_pretrained"
        )
    config.save_pretrained(output)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "save_pretrained"):
        generation_config.save_pretrained(output)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    if processor is not None:
        processor.save_pretrained(output)

    save_model(
        model,
        str(output / MODEL_STATE_NAME),
        metadata={"format": "pt", "rotquant_format": str(FORMAT_VERSION)},
    )
    save_file(
        packed_tensors,
        str(output / PACKED_STATE_NAME),
        metadata={"format": "pt", "rotquant_format": str(FORMAT_VERSION)},
    )

    manifest: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "packing": CURRENT_PACKING.to_manifest(),
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "model_loader": model_loader,
        "torch_dtype": _dtype_name(_first_float_dtype(model)),
        "model_state": MODEL_STATE_NAME,
        "packed_state": PACKED_STATE_NAME,
        "quantized_modules": modules,
    }
    adapter_name = getattr(model, "_rotquant_adapter_name", None)
    adapter = resolve_model_adapter(model, adapter_name)
    manifest["architecture"] = {
        "adapter": adapter.name,
        "model_type": adapter.model_type(model),
    }
    if serialized_deployment is not None:
        manifest["deployment"] = serialized_deployment
    validate_checkpoint_manifest(manifest)
    with (output / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    artifact_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    return {
        "path": str(output),
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "quantized_modules": len(modules),
        "artifact_bytes": artifact_bytes,
        "fallback_cache_serialized": False,
    }


def _resolve_model_class(config, model_loader: str):
    from transformers import AutoModelForCausalLM

    if model_loader == "auto":
        model_loader = (
            "multimodal_lm"
            if getattr(config, "vision_config", None) is not None
            else "causal_lm"
        )
    if model_loader == "causal_lm":
        return AutoModelForCausalLM
    if model_loader == "multimodal_lm":
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:  # pragma: no cover - version dependent
            raise RuntimeError(
                "this checkpoint requires AutoModelForMultimodalLM; upgrade "
                "Transformers to a release that provides it"
            ) from exc
        return AutoModelForMultimodalLM
    raise ValueError(f"unsupported model_loader in checkpoint: {model_loader!r}")


def load_packed_model(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
    fallback: bool = False,
    trust_remote_code: bool = False,
) -> nn.Module:
    """Reconstruct a packed model as an ordinary Transformers model instance.

    The returned object supports the model's normal ``forward`` and ``generate``
    methods.  ``fallback=False`` preserves compressed persistent storage but
    transiently dequantizes weights because RotQuant has no fused packed kernel.
    ``fallback=True`` caches dequantized weights for faster quality checks and
    therefore forfeits the runtime-memory reduction.
    """
    safe_open, load_model, _, _ = _require_safetensors()
    checkpoint = Path(checkpoint_dir)
    manifest_path = checkpoint / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"not a complete RotQuant checkpoint: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_checkpoint_manifest(manifest)

    from transformers import AutoConfig

    resolved_dtype = _resolve_dtype(dtype or manifest.get("torch_dtype"))
    config = AutoConfig.from_pretrained(
        checkpoint, trust_remote_code=trust_remote_code
    )
    model_cls = _resolve_model_class(config, manifest.get("model_loader", "auto"))
    model = model_cls.from_config(
        config, trust_remote_code=trust_remote_code, dtype=resolved_dtype
    )

    packed_path = checkpoint / manifest["packed_state"]
    with safe_open(packed_path, framework="pt", device="cpu") as packed_handle:
        for module_spec in manifest["quantized_modules"]:
            name = module_spec["name"]
            parent, attr = _get_parent(model, name)
            source = getattr(parent, attr)
            if not isinstance(source, nn.Linear):
                raise TypeError(
                    f"checkpoint expects nn.Linear at {name}, found "
                    f"{type(source).__name__}"
                )
            if (
                source.in_features != int(module_spec["in_features"])
                or source.out_features != int(module_spec["out_features"])
            ):
                raise ValueError(f"linear shape mismatch while restoring {name}")
            qweight = _read_quantized_weight(
                packed_handle, module_spec["qweight"]
            )
            rotation = _build_rotation(module_spec["rotation"])
            bias = (
                torch.empty(source.out_features, dtype=resolved_dtype)
                if module_spec["has_bias"]
                else None
            )
            qlinear = QuantLinear(
                qweight,
                act_rotation=rotation,
                bias=bias,
                fallback=fallback,
                fallback_dtype=resolved_dtype,
            )
            rank = int(module_spec.get("lora_rank", 0))
            if rank:
                qlinear.enable_lora(rank, float(module_spec["lora_alpha"]))
                qlinear.commit_lora(
                    _resolve_dtype(module_spec.get("lora_dtype") or resolved_dtype)
                )
            setattr(parent, attr, qlinear)

    missing, unexpected = load_model(
        model, checkpoint / manifest["model_state"], strict=True, device="cpu"
    )
    if missing or unexpected:  # strict=True already raises; keeps type checkers honest.
        raise RuntimeError(
            f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    generation_path = checkpoint / "generation_config.json"
    if generation_path.exists():
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(checkpoint)
    architecture = manifest.get("architecture") or {}
    if architecture.get("adapter"):
        model.__dict__["_rotquant_adapter_name"] = architecture["adapter"]
    model.to(device=device, dtype=resolved_dtype).eval()
    return model


def checkpoint_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read and minimally validate checkpoint metadata without loading weights."""
    path = Path(checkpoint_dir) / MANIFEST_NAME
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_checkpoint_manifest(manifest)
    return manifest
