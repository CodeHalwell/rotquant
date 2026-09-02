"""Model patching: walk a HF model, replace ``nn.Linear`` with ``QuantLinear``,
and enforce the rotation-consistency rules.

The consistency invariant: every rotated weight must have its matching activation
rotation, and the inverse transform is fused into dequant -- no mixed bases. Three
patch modes are exposed for E7:

* ``consistent``    -- weight and activation share one rotation per layer (correct).
* ``fused_inverse`` -- same as consistent, recording that the inverse is folded
  into dequant (the production path); behaviourally identical to ``consistent`` for
  a single linear, kept distinct for bookkeeping/plots.
* ``mismatched``    -- the weight is rotated but the activation is rotated by a
  *different* (or absent) basis, deliberately breaking consistency to surface the
  cross-layer drift the trap predicts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ._internal import cpu_staging_linear, get_parent, rotate_hessian
from .adapters import resolve_model_adapter
from .linear import QuantLinear
from .quantize import QuantConfig
from .rotate import (
    ButterflyRotation,
    Identity,
    LearnedRotation,
    Rotation,
    build_rotation,
)
from .utils import get_logger

logger = get_logger(__name__)

PATCH_MODES = ("consistent", "fused_inverse", "mismatched")


@dataclass
class PatchConfig:
    quant: QuantConfig
    # False is a true source-model baseline: no Linear modules are replaced.
    # This is distinct from rotation="none", which still quantises the model.
    enabled: bool = True
    rotation: str = "fwht"            # none | fwht | dense | learned
    block: int = 128
    mode: str = "consistent"          # see PATCH_MODES
    # kwargs for rotquant.train_rotation.RotationTrainConfig (e.g.
    # {"steps": 200, "lr": 1e-3}). Only meaningful with rotation="learned":
    # theta is optimised per layer (data-free, alternating minimisation) before
    # quantisation. None leaves theta at its ~identity init (and warns).
    train_rotation: dict | None = None
    include: Sequence[str] | None = None
    # Explicit registry adapter, or None to resolve from model.config.model_type.
    adapter: str | None = None
    # Substrings of layer names to leave in fp16. The default keeps the output
    # head (and its tied embedding) unquantised -- the convention every baseline
    # (GPTQ/AWQ/QuIP#/AQLM) follows, so results stay comparable. Pass () to
    # quantise everything.
    exclude: Sequence[str] = ("lm_head", "embed_out")
    fallback: bool = False
    seed: int = 0
    share_rotations: bool = False
    # Optional signed, symmetric, per-token activation quantization. This is a
    # correctness path today; fused W4A8 kernels can consume the same semantics.
    activation_bits: int | None = None
    # Optional Unsloth-Dynamic-style, model-specific mixed-precision search.
    # ``dynamic`` contains serializable search settings. ``layer_quant`` is
    # populated by the runner with the resulting per-projection recipe and is
    # intentionally excluded from normal config construction/representation.
    dynamic: dict | None = None
    layer_quant: dict[str, QuantConfig] = field(
        default_factory=dict, init=False, repr=False)


def quant_config_for(cfg: PatchConfig, name: str) -> QuantConfig:
    """Resolve a projection's selected quantizer configuration."""

    return cfg.layer_quant.get(name, cfg.quant)


def shared_rotation_key(name: str) -> str | None:
    """Return the known same-input projection site containing ``name``."""

    sibling_groups = (
        ("q_proj", "k_proj", "v_proj"),
        ("gate_proj", "up_proj"),
    )
    for siblings in sibling_groups:
        for suffix in siblings:
            marker = f".{suffix}"
            if name.endswith(marker):
                return f"{name[:-len(marker)]}.{'_'.join(siblings)}"
    return None


def make_rotations(in_features: int, cfg: PatchConfig, layer_seed: int,
                   device: torch.device | str | None = None):
    """Return (weight_rotation, act_rotation) honouring the consistency mode.

    ``device`` should be the target weight's device: the rotation is applied to
    that weight *before* the module tree is ever moved, so its buffers must be
    built where the weight lives.
    """
    weight_rot = build_rotation(cfg.rotation, in_features, block=cfg.block,
                                seed=layer_seed, device=device)
    if cfg.mode in ("consistent", "fused_inverse"):
        act_rot: Rotation = weight_rot           # matched basis -- the invariant
    elif cfg.mode == "mismatched":
        # Deliberately break it: rotate the weight but leave activations un-rotated.
        act_rot = Identity(in_features)
    else:
        raise ValueError(f"unknown patch mode: {cfg.mode}; pick from {PATCH_MODES}")
    return weight_rot, act_rot


def _rotate_hessian(rotation: Rotation, hessian: torch.Tensor) -> torch.Tensor:
    """Return ``R H R^T`` (see :func:`rotquant._internal.rotate_hessian`)."""

    return rotate_hessian(rotation, hessian)


def patch_model(model: nn.Module, cfg: PatchConfig,
                hessians: Mapping[str, torch.Tensor] | None = None,
                activations: dict[str, torch.Tensor] | None = None,
                activation_means: Mapping[str, torch.Tensor] | None = None,
                stats_out: dict | None = None) -> nn.Module:
    """Replace targeted ``nn.Linear`` layers with ``QuantLinear`` in-place.

    ``stats_out``: optional dict filled with patching side-info (currently
    per-run rotation-training aggregates under ``"rotation_train"``).
    """
    if not cfg.enabled:
        logger.info("Quantization disabled; evaluating the source model unchanged")
        return model
    if cfg.mode not in PATCH_MODES:
        raise ValueError(f"unknown patch mode: {cfg.mode}")
    if cfg.mode == "mismatched":
        logger.warning("patch mode 'mismatched' active -- consistency invariant "
                       "intentionally violated (E7 only)")
    learned_kind = cfg.rotation in ("learned", "cayley", "stiefel")
    butterfly_kind = cfg.rotation in ("butterfly", "learned_butterfly", "structured")
    if cfg.train_rotation is not None and not (learned_kind or butterfly_kind):
        raise ValueError(
            "patch.train_rotation requires rotation='learned' or 'butterfly'")
    if not isinstance(cfg.share_rotations, bool):
        raise TypeError("share_rotations must be boolean")
    if cfg.activation_bits is not None and (
        isinstance(cfg.activation_bits, bool)
        or not isinstance(cfg.activation_bits, int)
        or not 2 <= cfg.activation_bits <= 16
    ):
        raise ValueError("activation_bits must be None or an integer in [2, 16]")
    if learned_kind and cfg.train_rotation is None:
        logger.warning(
            "rotation='learned' starts at ~identity (theta init 1e-3): without "
            "patch.train_rotation (e.g. {steps: 200}) this arm measures a "
            "no-rotation control, not a learned rotation.")
    if (cfg.train_rotation or {}).get("objective") == "activation" \
            and cfg.mode not in ("consistent", "fused_inverse"):
        raise ValueError(
            "activation-aware rotation training requires a consistent mode"
        )
    hessians = hessians or {}
    activations = activations or {}
    activation_means = activation_means or {}

    include_terms = tuple(cfg.include) if cfg.include is not None else None
    exclude_terms = tuple(cfg.exclude or ())
    adapter = resolve_model_adapter(model, cfg.adapter)
    support = adapter.inspect(model)
    if stats_out is not None:
        stats_out["model_support"] = support.to_dict()
    targets = [
        (name, module)
        for name, module in adapter.iter_quantizable_modules(model)
        if (include_terms is None or any(term in name for term in include_terms))
        and not any(term in name for term in exclude_terms)
    ]
    if stats_out is not None:
        stats_out["patched_modules"] = 0
    if not targets:
        logger.warning(
            "patch_model found NO nn.Linear or adapter-specific quantizable "
            "modules through adapter=%s "
            "(include=%s, exclude=%s). The model is still full-precision!",
            adapter.name, include_terms, exclude_terms)
        return model

    model.__dict__["_rotquant_adapter_name"] = adapter.name

    train_stats: list = []
    shared_rotations: dict[str, tuple[Rotation, Rotation]] = {}
    trained_shared_sites: set[str] = set()
    for i, (name, source_module) in enumerate(targets):
        linear = adapter.to_linear(source_module)
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"adapter {adapter.name!r} returned {type(linear).__name__} "
                f"from to_linear(); expected nn.Linear"
            )
        layer_quant = quant_config_for(cfg, name)
        source_device = linear.weight.device
        source_dtype = linear.weight.dtype
        stage_on_cpu = source_device.type == "mps" and cfg.fallback
        work_linear = cpu_staging_linear(linear) if stage_on_cpu else linear
        source_hessian = hessians.get(name)

        # MPS scale-search/packing is staged on CPU, but the structured trainer
        # consists of supported dense/elementwise ops and benefits substantially
        # from running next to the original MPS weight. Dense Cayley training is
        # intentionally left on the staging device because it is not a practical
        # large-model MPS path.
        train_on_source = (stage_on_cpu and butterfly_kind
                           and cfg.train_rotation is not None)
        rotation_device = source_device if train_on_source \
            else work_linear.weight.device

        share_key = shared_rotation_key(name) if cfg.share_rotations else None
        cached_pair = (
            shared_rotations.get(share_key) if share_key is not None else None
        )
        reused_shared_rotation = cached_pair is not None
        if cached_pair is not None:
            weight_rot, act_rot = cached_pair
            if weight_rot.dim != work_linear.in_features:
                raise ValueError(f"shared rotation dimension mismatch at {name}")
        else:
            weight_rot, act_rot = make_rotations(
                work_linear.in_features,
                cfg,
                layer_seed=cfg.seed + i,
                device=rotation_device,
            )
            if share_key is not None:
                act_rot.enable_activation_cache(True)
                shared_rotations[share_key] = (weight_rot, act_rot)
        should_train_rotation = (
            isinstance(weight_rot, (LearnedRotation, ButterflyRotation))
            and cfg.train_rotation is not None
            and (share_key is None or share_key not in trained_shared_sites)
        )
        if should_train_rotation:
            from .train_rotation import (
                RotationTrainConfig,
                select_butterfly_checkpoint,
                select_butterfly_checkpoint_hessian,
                train_layer_rotation,
            )
            train_cfg = RotationTrainConfig(**cfg.train_rotation)
            train_weight = linear.weight if train_on_source else work_linear.weight
            layer_acts = activations.get(name)
            if train_cfg.objective == "activation" and layer_acts is None:
                raise ValueError(
                    f"activation-aware rotation training requires captured "
                    f"activations for {name}"
                )
            selection_weight = train_weight
            site_members = [name]
            if share_key is not None:
                members = [
                    (member_index, member_name, member_source)
                    for member_index, (member_name, member_source) in enumerate(targets)
                    if shared_rotation_key(member_name) == share_key
                ]
                member_configs = [
                    quant_config_for(cfg, member_name)
                    for _, member_name, _ in members
                ]
                if any(member_cfg != layer_quant for member_cfg in member_configs):
                    raise ValueError(
                        f"joint learned rotation at {share_key} requires one "
                        "quantization config across sibling projections"
                    )
                member_weights = []
                for _, member_name, member_source in members:
                    member_linear = adapter.to_linear(member_source)
                    if member_linear.in_features != weight_rot.dim:
                        raise ValueError(
                            f"shared rotation dimension mismatch at {member_name}")
                    member_weights.append(member_linear.weight.detach().to(
                        device=train_weight.device, dtype=train_weight.dtype))
                selection_weight = torch.cat(member_weights, dim=0)
                train_weight = selection_weight
                site_members = [member_name for _, member_name, _ in members]
                trained_shared_sites.add(share_key)
            stats = train_layer_rotation(
                weight_rot, train_weight, layer_quant,
                train_cfg, activations=layer_acts,
                hessian=source_hessian)
            stats["site_members"] = site_members
            if train_on_source:
                weight_rot.to(device=work_linear.weight.device, dtype=torch.float32)
                selection_weight = selection_weight.to(
                    device=work_linear.weight.device, dtype=torch.float32)
            if isinstance(weight_rot, ButterflyRotation) \
                    and train_cfg.objective == "activation":
                if layer_acts is None:
                    raise RuntimeError(
                        "activation-objective butterfly training reached "
                        "checkpoint selection without captured activations"
                    )
                reference_rot = ButterflyRotation(
                    work_linear.in_features, block=cfg.block,
                    seed=cfg.seed + i, device=work_linear.weight.device)
                if train_cfg.selection_tokens:
                    selection_acts = layer_acts[
                        train_cfg.max_tokens:
                        train_cfg.max_tokens + train_cfg.selection_tokens]
                    if selection_acts.shape[0] < train_cfg.selection_tokens:
                        raise ValueError(
                            f"layer {name} has only {layer_acts.shape[0]} captured "
                            "tokens; need max_tokens + selection_tokens")
                    selection_tokens = train_cfg.selection_tokens
                else:
                    selection_acts = layer_acts
                    selection_tokens = train_cfg.max_tokens
                selection = select_butterfly_checkpoint(
                    weight_rot, reference_rot, selection_weight, layer_quant,
                    selection_acts, max_tokens=selection_tokens,
                    min_improvement=train_cfg.selection_min_improvement)
                stats.update(selection)
                stats["selection_tokens"] = selection_tokens
            elif isinstance(weight_rot, ButterflyRotation) \
                    and train_cfg.objective == "hessian":
                # The proxy objective (RMS assignment, no error compensation)
                # is not the deployed quantizer.  Gate the trained rotation
                # against seeded FWHT under the exact packed weight, including
                # GPTQ, so a worse-than-FWHT basis is never deployed silently.
                reference_rot = ButterflyRotation(
                    work_linear.in_features, block=cfg.block,
                    seed=cfg.seed + i, device=work_linear.weight.device)
                # Score both arms on the error that survives the deployed
                # mean bias correction; otherwise the gate ranks a component
                # that apply_mean_bias_correction cancels.
                selection = select_butterfly_checkpoint_hessian(
                    weight_rot, reference_rot, selection_weight, layer_quant,
                    source_hessian,
                    min_improvement=train_cfg.selection_min_improvement,
                    activation_mean=(
                        activation_means.get(name)
                        if layer_quant.bias_correction in {"mean", "length_mean"}
                        else None))
                stats.update(selection)
                stats["selection_tokens"] = 0
            train_stats.append(stats)
            logger.info("trained %s rotation for %s: %.6f -> %.6f%s",
                        stats["objective"], name,
                        stats["initial_mse"], stats["final_mse"],
                        (" (final-quantizer accepted)"
                         if stats.get("selection_accepted") else
                         " (restored FWHT)"
                         if "selection_accepted" in stats else ""))
        elif reused_shared_rotation and cfg.train_rotation is not None:
            logger.debug("reusing jointly trained rotation for %s", name)
        H = source_hessian
        if H is not None:
            # Hessians may have been offloaded to CPU by collect_hessians;
            # bring them back next to the weight for the rotation + GPTQ solve.
            H = H.to(work_linear.weight.device)
        if H is not None and cfg.rotation not in ("none", "identity"):
            # Rotate the Hessian into the same basis as the rotated weight:
            # H' = R H R^T so GPTQ sees the consistent input statistics.
            H = _rotate_hessian(weight_rot, H)
        qlin = QuantLinear.from_linear(work_linear, layer_quant,
                                       weight_rotation=weight_rot,
                                       act_rotation=act_rot, H=H,
                                       fallback=cfg.fallback,
                                       fallback_dtype=source_dtype,
                                       activation_bits=cfg.activation_bits)
        if layer_quant.bias_correction in {"mean", "length_mean"}:
            activation_mean = activation_means.get(name)
            if activation_mean is None:
                raise ValueError(
                    f"mean bias correction requires activation mean for {name}"
                )
            qlin.apply_mean_bias_correction(
                work_linear.weight, activation_mean
            )
        if stage_on_cpu:
            qlin = qlin.to(device=source_device, dtype=source_dtype)
        parent, attr = get_parent(model, name)
        adapter.replace_quantized_module(
            parent, attr, source_module, qlin
        )
        if stats_out is not None:
            stats_out["patched_modules"] = i + 1
        if i == 0 or (i + 1) % 8 == 0:
            logger.info("patched %d/%d layers (last: %s)", i + 1, len(targets), name)

    if train_stats:
        agg = {
            "layers": len(train_stats),
            "projections": sum(len(s.get("site_members", (None,)))
                               for s in train_stats),
            "steps": train_stats[0]["steps"],
            "objective": train_stats[0]["objective"],
            "tokens": train_stats[0]["tokens"],
            "selection_tokens": train_stats[0].get("selection_tokens", 0),
            "mean_best_step": (
                sum(s["best_step"] for s in train_stats) / len(train_stats)
            ),
            "mean_initial_mse": (
                sum(s["initial_mse"] for s in train_stats) / len(train_stats)
            ),
            "mean_final_mse": (
                sum(s["final_mse"] for s in train_stats) / len(train_stats)
            ),
            "mean_sign_flip_rate": (
                sum(s.get("sign_flip_rate", 0.0) for s in train_stats)
                / len(train_stats)
            ),
        }
        agg["mean_relative_improvement"] = (
            (agg["mean_initial_mse"] - agg["mean_final_mse"])
            / max(agg["mean_initial_mse"], 1e-12)
        )
        selected = [s for s in train_stats if "selection_accepted" in s]
        if selected:
            agg["selection_acceptance_rate"] = (
                sum(float(s["selection_accepted"]) for s in selected) / len(selected))
            agg["mean_selection_reference_mse"] = (
                sum(s["selection_reference_mse"] for s in selected) / len(selected))
            # Rejected candidates deploy the reference, not their candidate MSE.
            agg["mean_selection_deployed_mse"] = sum(
                s["selection_candidate_mse"] if s["selection_accepted"]
                else s["selection_reference_mse"] for s in selected
            ) / len(selected)
        logger.info("rotation training: mean quant-MSE %.5f -> %.5f over %d layers",
                    agg["mean_initial_mse"], agg["mean_final_mse"], agg["layers"])
        if stats_out is not None:
            stats_out["rotation_train"] = agg

    if stats_out is not None:
        stats_out["shared_rotation_sites"] = len(shared_rotations)

    logger.info("Patched %d modules (adapter=%s, rotation=%s, mode=%s)",
                len(targets), adapter.name, cfg.rotation, cfg.mode)
    return model
