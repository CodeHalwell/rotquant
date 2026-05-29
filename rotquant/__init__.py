"""rotquant -- rotation-based weight compression evaluation toolkit."""
from .utils import BitBudget, set_seed, environment_record, get_logger
from .rotate import (
    build_rotation, Identity, RandomizedHadamard, DenseOrthogonal,
    LearnedRotation, fwht,
)
from .codebooks import (
    lloyd_max_gaussian, quantizer_mse, uniform_signed, normal_float,
    turboquant_mse_bound,
    ScalarCodebook, build_scalar_codebook, E8LatticeCodebook, nearest_e8,
)
from .pack import pack_indices, unpack_indices, PackedTensor
from .quantize import Quantizer, QuantConfig, QuantizedWeight
from .linear import QuantLinear
from .calibrate import collect_hessians, HessianAccumulator
from .patch import patch_model, PatchConfig, PATCH_MODES

__all__ = [
    "BitBudget", "set_seed", "environment_record", "get_logger",
    "build_rotation", "Identity", "RandomizedHadamard", "DenseOrthogonal",
    "LearnedRotation", "fwht",
    "lloyd_max_gaussian", "quantizer_mse", "uniform_signed", "normal_float",
    "turboquant_mse_bound",
    "ScalarCodebook", "build_scalar_codebook", "E8LatticeCodebook", "nearest_e8",
    "pack_indices", "unpack_indices", "PackedTensor",
    "Quantizer", "QuantConfig", "QuantizedWeight",
    "QuantLinear",
    "collect_hessians", "HessianAccumulator",
    "patch_model", "PatchConfig", "PATCH_MODES",
]
