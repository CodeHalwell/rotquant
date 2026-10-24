"""Separate, content-bound selected latents for assignment-changing training."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .linear import QuantLinear
from .rotate import RandomizedHadamard
from .ternary_state import content_digest, save_training_state


def validate_ternary_module(module):
    weight = module.qweight
    if not isinstance(module.act_rotation, RandomizedHadamard):
        raise TypeError("global ternary QAT requires signed Hadamard rotation")
    values = weight.codebook.centroids.detach().float().cpu().reshape(-1)
    if weight.packed.bits != 2 or not torch.equal(values, torch.tensor([-1., 0., 1.])):
        raise ValueError("global ternary QAT requires exactly three legal two-bit symbols")
    if weight.residual_packed is not None or weight.sketch is not None:
        raise ValueError("global ternary QAT does not accept residual/sketch weights")
    if module.activation_bits is not None or module.lora_A is not None or module._log_scale_multiplier is not None:
        raise ValueError("global ternary QAT requires the plain packed ternary representation")
    if weight.scales.dtype != torch.float16 or weight.scale_bits_main != 16:
        raise ValueError("global ternary QAT requires FP16 group scales")
    if not torch.isfinite(weight.scales).all() or not (weight.scales > 0).all():
        raise ValueError("global ternary QAT requires finite positive scales")
    words = weight.packed.data
    if (words & (words >> 1) & 0x55555555).any():
        raise ValueError("ternary checkpoint contains reserved code 3")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_selected_latents(student, packed, contract, block_name, path):
    from .ternary import TernaryLinear
    modules = {name: {"latent_weight": selected_latent(module),
                      "log_scales": module.log_scales.detach().cpu().clone()}
               for name, module in student.named_modules() if isinstance(module, TernaryLinear)}
    payload = {"version": 1, "contract": contract, "block_name": block_name,
               "packed_digest": content_digest(packed), "modules": modules}
    save_training_state(Path(path), payload)
    return {"path": str(path), "sha256": file_digest(path),
            "packed_digest": payload["packed_digest"], "training_only": True}


def selected_latent(module):
    if module.fixed_codes is not None:
        return module.quantized_weight().detach().cpu().clone()
    return module.latent_weight.detach().cpu().clone()


def load_selected_latents(receipt, packed=None, contract=None):
    path = Path(receipt["path"])
    if file_digest(path) != receipt["sha256"]:
        raise ValueError("selected ternary latent artifact hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1 or payload.get("packed_digest") != receipt["packed_digest"]:
        raise ValueError("selected ternary latent artifact contract mismatch")
    if contract is not None and payload["contract"] != contract:
        raise ValueError("selected ternary latent source/data/config identity mismatch")
    if packed is not None and content_digest(packed) != payload["packed_digest"]:
        raise ValueError("selected ternary latents belong to a different packed checkpoint")
    return payload


def ternary_sites(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, QuantLinear) and module.qweight.codebook.name == "ternary"}


def retained_site_latents(model, receipts):
    from .ternary import packed_ternary_state
    result, identities = {}, []
    for receipt in receipts:
        payload = load_selected_latents(receipt)
        prefix = payload["block_name"]
        packed = {}
        for name, latent in payload["modules"].items():
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in result:
                raise ValueError("duplicate selected ternary latent site")
            module = model.get_submodule(full_name)
            validate_ternary_module(module)
            packed[name] = packed_ternary_state(module)
            result[full_name] = latent
        if content_digest(packed) != payload["packed_digest"]:
            raise ValueError("selected ternary latents belong to a different packed checkpoint")
        identities.append({"sha256": receipt["sha256"], "contract": payload["contract"]})
    return result, identities
