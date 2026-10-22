"""Stable first-order gradients for Qwen's Python GDN fallback normalisation.

Only backward intermediates are promoted. Forward retains the original
expression and dtype, so this policy changes neither deployed logits nor
quantisation precision. Fused FLA kernels are not modified.
"""
from __future__ import annotations

import functools
import types
from contextlib import contextmanager

import torch

_FALLBACKS = {
    "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
    "recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule",
}
_QWEN_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"


class _StableL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, dim, eps):
        inverse = torch.rsqrt((value * value).sum(dim=dim, keepdim=True) + eps)
        ctx.save_for_backward(value, inverse)
        ctx.dim = dim
        return value * inverse

    @staticmethod
    def backward(ctx, gradient):
        value, inverse = ctx.saved_tensors
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        work, inv, grad = value.to(dtype), inverse.to(dtype), gradient.to(dtype)
        projection = (grad * work).sum(dim=ctx.dim, keepdim=True)
        result = grad * inv - work * inv.pow(3) * projection
        return result.to(value.dtype), None, None


def stable_l2norm(value, dim=-1, eps=1e-6):
    """Original low-precision forward, analytically equivalent stable backward."""
    return _StableL2Norm.apply(value, dim, eps)


def _private_fallback(function, expected_name, token):
    if not isinstance(function, types.FunctionType):
        return None
    if function.__module__ != _QWEN_MODULE or function.__name__ != expected_name:
        return None
    if function.__globals__.get("l2norm") is stable_l2norm:
        return None
    if "l2norm" not in function.__code__.co_names:
        raise RuntimeError("Qwen fallback normalisation changed; verify its numerical contract")
    namespace = {**function.__globals__, "l2norm": stable_l2norm}
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    functools.update_wrapper(result, function)
    result.__kwdefaults__ = function.__kwdefaults__
    result._rotquant_original = function
    result._rotquant_numerics_token = token
    return result


def _restore_functions(modules, token):
    for module in modules:
        for attribute in _FALLBACKS:
            function = getattr(module, attribute, None)
            if getattr(function, "_rotquant_numerics_token", None) is token:
                setattr(module, attribute, function._rotquant_original)


@contextmanager
def stable_qwen_fallbacks(model):
    """Apply a reversible, model-local policy, including copied/replaced blocks.

    Nested contexts are supported. The scope must contain the backward pass;
    the normal single-process study worker does not run concurrent model edits.
    """
    token = object()
    originals, names = [], []
    try:
        for name, module in model.named_modules():
            for attribute, expected in _FALLBACKS.items():
                original = getattr(module, attribute, None)
                replacement = _private_fallback(original, expected, token)
                if replacement is not None:
                    setattr(module, attribute, replacement)
                    originals.append(module)
                    names.append(f"{name}.{attribute}")
        yield {"policy": "source_forward_fp32_l2_backward_v1",
               "patched_fallbacks": names, "fused_kernels_modified": False}
    finally:
        _restore_functions(originals, token)
        # QAT replaces blocks with copies; their function handles are shared
        # immutable Python functions and retain this context's token.
        _restore_functions(model.modules(), token)


def _fp32_gdn_changes(module, implementation):
    """Avoid optional kernels whose training contracts require FP16/BF16."""
    norm = implementation.Qwen3_5RMSNormGated(
        module.head_v_dim, eps=module.layer_norm_epsilon)
    norm.weight = module.norm.weight
    return {
        "chunk_gated_delta_rule": implementation.torch_chunk_gated_delta_rule,
        "recurrent_gated_delta_rule": implementation.torch_recurrent_gated_delta_rule,
        "causal_conv1d_fn": None,
        "causal_conv1d_update": implementation.torch_causal_conv1d_update,
        "norm": norm,
    }


def _training_changes(module):
    if type(module).__module__ != _QWEN_MODULE:
        return []
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    if isinstance(module, implementation.Qwen3_5GatedDeltaNet):
        if module.activation != "silu":
            raise ValueError("FP32 Qwen fallback requires the source SiLU gating contract")
        return [(module, name, value) for name, value in _fp32_gdn_changes(module, implementation).items()]
    if isinstance(module, implementation.Qwen3_5Attention):
        return [(module.config, "_attn_implementation", "eager")]
    return []


@contextmanager
def qwen_fp32_training(model):
    """Use Qwen's torch reference operators only during FP32 block training.

    Source kernels and norm modules are restored before deployed validation.
    This also avoids FP32 input rejection by optional FLA/flash-attention ops.
    """
    restored = []
    try:
        for module in list(model.modules()):
            for owner, name, replacement in _training_changes(module):
                restored.append((owner, name, getattr(owner, name)))
                setattr(owner, name, replacement)
        yield
    finally:
        for owner, name, value in reversed(restored):
            setattr(owner, name, value)
