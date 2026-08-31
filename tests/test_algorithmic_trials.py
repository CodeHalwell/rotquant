"""The Colab algorithm matrix must remain complete and one-factor-at-a-time."""
from scripts.algorithmic_trials import algorithmic_trial_matrix, trial_by_name


def test_algorithmic_matrix_covers_every_research_track() -> None:
    profiles = algorithmic_trial_matrix()
    assert [profile.name for profile in profiles[:3]] == [
        "source_fp16", "gaussian_w4_mse", "gaussian_w3_mse"
    ]
    assert {profile.track for profile in profiles} == {
        "control", "low_bit_vector", "codebook_scale", "allocation"
    }
    assert {profile.name for profile in profiles} >= {
        "source_fp16",
        "scalar_w1_rms",
        "vector_d2_w1_rms",
        "scalar_w3_rms",
        "vector_d2_w3_rms",
        "gaussian_w4_mse",
        "calibrated_w4_mse",
        "length_w4_mse",
        "dynamic_teacher_3p625",
        "dynamic_random_3p625",
        "dynamic_scalar_teacher_2p75",
        "dynamic_vector_2p75",
    }


def test_vector_and_scalar_low_bit_arms_are_exactly_matched() -> None:
    for bits in (1, 2, 3):
        scalar = dict(trial_by_name(f"scalar_w{bits}_rms").overrides)
        vector = dict(trial_by_name(f"vector_d2_w{bits}_rms").overrides)
        assert scalar["quant.bits"] == vector["quant.bits"] == bits
        assert scalar["quant.scale"] == vector["quant.scale"] == "rms"
        assert scalar["quant.group_size"] == vector["quant.group_size"] == 128
        assert vector["quant.codebook"] == "vector"


def test_teacher_allocator_uses_held_out_global_signal() -> None:
    profile = trial_by_name("dynamic_teacher_3p625")
    dynamic = dict(profile.overrides)["patch.dynamic"]
    assert dynamic["global_kl_batches"] > 0
    assert dynamic["global_kl_weight"] > dynamic["local_weight"]
    assert dynamic["target_bpw"] == 3.625


def test_matched_allocation_controls_share_candidate_formats() -> None:
    random_control = dict(
        trial_by_name("dynamic_random_3p625").overrides
    )["patch.dynamic"]
    local = dict(trial_by_name("dynamic_local_3p625").overrides)["patch.dynamic"]
    assert random_control["candidate_bits"] == local["candidate_bits"]
    assert random_control["target_bpw"] == local["target_bpw"] == 3.625
    assert random_control["allocation"] == "random"

    scalar = dict(
        trial_by_name("dynamic_scalar_teacher_2p75").overrides
    )
    vector = dict(trial_by_name("dynamic_vector_2p75").overrides)
    assert scalar["patch.dynamic"] == vector["patch.dynamic"]
    assert scalar["quant.codebook"] == "gaussian"
    assert vector["quant.codebook"] == "vector"
