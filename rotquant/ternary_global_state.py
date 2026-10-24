"""Atomic global optimiser resume and protected representation identities."""
from __future__ import annotations

from dataclasses import asdict

import torch

from .linear import QuantLinear
from .ternary_state import _quantized_source_state, content_digest, source_state_digest
from .vocabulary import PackedVocabulary


def vocabulary_state(model):
    return {name: {"chunks": [_quantized_source_state(chunk) for chunk in module.chunks],
                   "config": asdict(module.config), "dtype": module.execution_dtype,
                   "projection_mode": module.projection_mode, "source": module.source_digest}
            for name, module in model.named_modules() if isinstance(module, PackedVocabulary)}


def protected_digest(model, sites):
    prefixes = tuple(name + "." for name in sites)
    ordinary = {name: value for name, value in model.state_dict().items()
                if not name.startswith(prefixes)}
    packed = {name: source_state_digest(module) for name, module in model.named_modules()
              if isinstance(module, QuantLinear) and name not in sites}
    return content_digest({"ordinary": ordinary, "packed": packed, "vocabulary": vocabulary_state(model)})


def global_contract(model, train, validation, config, latent_identities, audit_identity):
    from .ternary_global_data import calls_identity
    settings = asdict(config)
    for key in ("checkpoint_dir", "resume", "checkpoint_interval"):
        settings.pop(key)
    return {"version": 1, "objective": "response-union-ce-kl-global-assignment-qat-v1",
            "source": content_digest({"model": source_state_digest(model),
                                      "vocabulary": vocabulary_state(model)}),
            "config": content_digest(settings),
            "train": calls_identity(train), "validation": calls_identity(validation),
            "selected_latents": latent_identities, "dense_audit": audit_identity}


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("global QAT resume CUDA RNG topology differs")
        torch.cuda.set_rng_state_all(state["cuda"])
