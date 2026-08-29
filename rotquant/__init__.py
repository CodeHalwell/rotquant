"""rotquant -- rotation-based weight compression evaluation toolkit."""
from .utils import BitBudget, set_seed, environment_record, get_logger
from .rotate import (
    build_rotation, Identity, RandomizedHadamard, DenseOrthogonal,
    LearnedRotation, ButterflyRotation, fwht,
)
from .codebooks import (
    lloyd_max_gaussian, quantizer_mse, uniform_signed, normal_float,
    turboquant_mse_bound,
    ScalarCodebook, build_scalar_codebook, E8LatticeCodebook, nearest_e8,
)
from .pack import pack_indices, unpack_indices, PackedTensor
from .quantize import Quantizer, QuantConfig, QuantizedWeight
from .linear import QuantLinear
from .calibrate import (
    collect_hessians, collect_activations, HessianAccumulator, ActivationResult,
)
from .patch import patch_model, PatchConfig, PATCH_MODES
from .train_rotation import (
    activation_reconstruction_error, select_butterfly_checkpoint,
    train_layer_rotation, RotationTrainConfig,
)
from .block_train import (
    BlockCall, TeacherCall, BlockRotationTrainConfig, collect_block_calls,
    collect_teacher_calls,
    find_transformer_blocks, train_and_patch_blocks,
)

__all__ = [
    "BitBudget", "set_seed", "environment_record", "get_logger",
    "build_rotation", "Identity", "RandomizedHadamard", "DenseOrthogonal",
    "LearnedRotation", "ButterflyRotation", "fwht",
    "lloyd_max_gaussian", "quantizer_mse", "uniform_signed", "normal_float",
    "turboquant_mse_bound",
    "ScalarCodebook", "build_scalar_codebook", "E8LatticeCodebook", "nearest_e8",
    "pack_indices", "unpack_indices", "PackedTensor",
    "Quantizer", "QuantConfig", "QuantizedWeight",
    "QuantLinear",
    "collect_hessians", "collect_activations", "HessianAccumulator",
    "ActivationResult",
    "patch_model", "PatchConfig", "PATCH_MODES",
    "train_layer_rotation", "RotationTrainConfig",
    "activation_reconstruction_error", "select_butterfly_checkpoint",
    "BlockCall", "TeacherCall", "BlockRotationTrainConfig",
    "collect_block_calls", "collect_teacher_calls",
    "find_transformer_blocks", "train_and_patch_blocks",
]
