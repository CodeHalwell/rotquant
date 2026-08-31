"""The packed checkpoint schema is an executable compatibility contract."""
from __future__ import annotations

import copy

import pytest

from rotquant.format import (
    CURRENT_PACKING,
    FORMAT_NAME,
    FORMAT_VERSION,
    FormatValidationError,
    packed_word_count,
    validate_checkpoint_manifest,
)


def _manifest() -> dict:
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "packing": CURRENT_PACKING.to_manifest(),
        "model_state": "rotquant_model.safetensors",
        "packed_state": "rotquant_packed.safetensors",
        "architecture": {"adapter": "dense-decoder", "model_type": "llama"},
        "quantized_modules": [
            {
                "name": "model.proj",
                "in_features": 4,
                "out_features": 3,
                "has_bias": False,
                "lora_rank": 0,
                "rotation": {"kind": "identity", "dim": 4},
                "qweight": {
                    "packed": {
                        "tensor": "layer_00000.packed",
                        "shape": [12],
                        "bits": 4,
                        "numel": 12,
                    },
                    "codebook": {
                        "name": "gaussian",
                        "centroids": "layer_00000.codebook",
                    },
                    "group_size": 4,
                    "in_features": 4,
                    "out_features": 3,
                },
            }
        ],
    }


def test_current_manifest_contract_is_valid() -> None:
    validate_checkpoint_manifest(_manifest())
    assert CURRENT_PACKING.optimized_profile_bits == tuple(range(1, 9))
    assert packed_word_count(12, 4) == 2


def test_legacy_v1_manifest_without_explicit_packing_is_valid() -> None:
    manifest = _manifest()
    manifest["format_version"] = 1
    del manifest["packing"]
    del manifest["architecture"]
    validate_checkpoint_manifest(manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(format_version=3), "unsupported version"),
        (
            lambda value: value["packing"].update(bit_order="msb_first"),
            "packing contract",
        ),
        (
            lambda value: value["quantized_modules"][0]["qweight"]["packed"].update(
                bits=0
            ),
            "bits must be",
        ),
        (
            lambda value: value["quantized_modules"][0]["qweight"]["packed"].update(
                numel=11
            ),
            "shape contains 12",
        ),
        (lambda value: value.update(model_state="../weights"), "not a path"),
    ],
)
def test_manifest_contract_rejects_binary_ambiguity(mutation, message) -> None:
    manifest = copy.deepcopy(_manifest())
    mutation(manifest)
    with pytest.raises(FormatValidationError, match=message):
        validate_checkpoint_manifest(manifest)


def test_manifest_contract_rejects_duplicate_module_names() -> None:
    manifest = _manifest()
    manifest["quantized_modules"].append(
        copy.deepcopy(manifest["quantized_modules"][0])
    )
    with pytest.raises(FormatValidationError, match="must be unique"):
        validate_checkpoint_manifest(manifest)
