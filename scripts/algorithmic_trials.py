"""Predeclared algorithmic trial profiles for the Colab research matrix.

The profiles change one algorithmic factor at a time around a fixed FWHT,
group-128 protocol. They intentionally contain no model or evaluation settings;
the notebook supplies those once so every arm uses the same data and gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AlgorithmicTrial:
    name: str
    track: str
    overrides: tuple[tuple[str, Any], ...]
    research_only: bool = False

    def sets(self) -> list[tuple[str, Any]]:
        return list(self.overrides)


def _weight_profile(
    name: str,
    track: str,
    *,
    bits: int,
    codebook: str = "gaussian",
    scale: str = "mse_search",
    bias_correction: str = "none",
    vector_dim: int = 2,
    research_only: bool = False,
) -> AlgorithmicTrial:
    return AlgorithmicTrial(
        name=name,
        track=track,
        research_only=research_only,
        overrides=(
            ("patch.enabled", True),
            ("patch.dynamic", None),
            ("quant.bits", bits),
            ("quant.codebook", codebook),
            ("quant.scale", scale),
            ("quant.group_size", 128),
            ("quant.error_comp", "none"),
            ("quant.bias_correction", bias_correction),
            ("quant.vector_dim", vector_dim),
            # Nine scale candidates keep vector screening bounded. Scalar arms
            # use the same grid when budget-matched against a vector arm.
            ("quant.mse_search_grid", 9 if codebook == "vector" else 41),
        ),
    )


def _dynamic_profile(
    name: str,
    *,
    target_bpw: float,
    global_kl_batches: int,
    local_weight: float,
    global_kl_weight: float,
    rules: list[dict[str, Any]] | None = None,
    codebook: str = "gaussian",
    candidate_bits: list[int] | None = None,
    scale: str = "mse_search",
    allocation: str = "greedy",
    research_only: bool = False,
) -> AlgorithmicTrial:
    dynamic = {
        "candidate_bits": candidate_bits or [2, 3, 4],
        "target_bpw": target_bpw,
        "max_tokens": 64,
        "local_weight": local_weight,
        "global_kl_weight": global_kl_weight,
        "global_kl_batches": global_kl_batches,
        "global_kl_temperature": 1.0,
        "allocation": allocation,
        "rules": rules or [],
    }
    return AlgorithmicTrial(
        name=name,
        track="allocation",
        research_only=research_only,
        overrides=(
            ("patch.enabled", True),
            ("patch.dynamic", dynamic),
            ("quant.bits", max(dynamic["candidate_bits"])),
            ("quant.codebook", codebook),
            ("quant.scale", scale),
            ("quant.group_size", 128),
            ("quant.error_comp", "none"),
            ("quant.bias_correction", "none"),
            ("quant.vector_dim", 2),
            ("quant.mse_search_grid", 9 if codebook == "vector" else 41),
        ),
    )


def algorithmic_trial_matrix() -> tuple[AlgorithmicTrial, ...]:
    """Return the ordered screening matrix used by the Colab notebook."""

    profiles = [
        AlgorithmicTrial(
            "source_fp16", "control", (("patch.enabled", False),)
        ),
        _weight_profile(
            "gaussian_w4_mse", "codebook_scale", bits=4
        ),
        _weight_profile(
            "gaussian_w3_mse", "codebook_scale", bits=3
        ),
    ]
    for bits in (1, 2, 3):
        profiles.extend([
            _weight_profile(
                f"scalar_w{bits}_rms",
                "low_bit_vector",
                bits=bits,
                scale="rms",
            ),
            _weight_profile(
                f"vector_d2_w{bits}_rms",
                "low_bit_vector",
                bits=bits,
                codebook="vector",
                scale="rms",
                research_only=True,
            ),
        ])
    for bits in (3, 4):
        profiles.extend([
            _weight_profile(
                f"calibrated_w{bits}_mse",
                "codebook_scale",
                bits=bits,
                codebook="calibrated",
            ),
            _weight_profile(
                f"spherical_w{bits}_mse",
                "codebook_scale",
                bits=bits,
                codebook="spherical",
            ),
            _weight_profile(
                f"length_w{bits}_mse",
                "codebook_scale",
                bits=bits,
                bias_correction="length",
            ),
            _weight_profile(
                f"turboquant_scale_w{bits}",
                "codebook_scale",
                bits=bits,
                scale="turboquant",
            ),
        ])
    profiles.extend([
        _dynamic_profile(
            "dynamic_random_3p625",
            target_bpw=3.625,
            global_kl_batches=0,
            local_weight=1.0,
            global_kl_weight=0.0,
            allocation="random",
        ),
        _dynamic_profile(
            "dynamic_local_3p625",
            target_bpw=3.625,
            global_kl_batches=0,
            local_weight=1.0,
            global_kl_weight=0.0,
        ),
        _dynamic_profile(
            "dynamic_teacher_3p625",
            target_bpw=3.625,
            global_kl_batches=2,
            local_weight=0.25,
            global_kl_weight=1.0,
        ),
        _dynamic_profile(
            "dynamic_guarded_teacher_3p625",
            target_bpw=3.625,
            global_kl_batches=2,
            local_weight=0.25,
            global_kl_weight=1.0,
            rules=[
                {"match": "q_proj", "min_bits": 3},
                {"match": "down_proj", "min_bits": 3},
            ],
        ),
        _dynamic_profile(
            "dynamic_scalar_teacher_2p75",
            target_bpw=2.75,
            global_kl_batches=1,
            local_weight=0.5,
            global_kl_weight=1.0,
            candidate_bits=[2, 3],
            scale="rms",
        ),
        _dynamic_profile(
            "dynamic_vector_2p75",
            target_bpw=2.75,
            global_kl_batches=1,
            local_weight=0.5,
            global_kl_weight=1.0,
            codebook="vector",
            candidate_bits=[2, 3],
            scale="rms",
            research_only=True,
        ),
    ])
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise RuntimeError("algorithmic trial names must be unique")
    return tuple(profiles)


def trial_by_name(name: str) -> AlgorithmicTrial:
    try:
        return next(
            profile for profile in algorithmic_trial_matrix()
            if profile.name == name
        )
    except StopIteration as exc:
        raise KeyError(f"unknown algorithmic trial: {name}") from exc


__all__ = ["AlgorithmicTrial", "algorithmic_trial_matrix", "trial_by_name"]
