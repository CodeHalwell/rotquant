"""Dependency-free pilot controls, checked before installing any packages."""
import math


def selected_contexts(values=None):
    values = list(values) if values is not None else [128, 512, 2048]
    if not values or len(values) != len(set(values)) or any(v not in (128, 512, 2048) for v in values):
        raise ValueError("Select unique contexts from 128, 512, 2048")
    return sorted(values)


def phase_minutes(context, decode_steps=32, repetitions=3, *, profiling=False):
    """Conservative bound, not a speed prediction; still capped by the session.

    Observed reference prefill is 36.4 tok/s. Budget at 16 tok/s plus the
    existing 2 tok/s decode floor, with two minutes for loading/verification.
    Diagnostic event synchronization gets an additional factor of two.
    """
    validate_controls(context, decode_steps, repetitions, 2., 16384.)
    seconds = 120 + (repetitions + 1) * (context / 16 + decode_steps / 2)
    return min(30, max(3, math.ceil(seconds / 60)) * (2 if profiling else 1))


def validate_controls(context, decode, repetitions, minimum_tps, maximum_vram):
    if context not in (128, 512, 2048) or not 8 <= decode <= 64 or not 2 <= repetitions <= 5:
        raise ValueError("Pilot permits contexts 128/512/2048, 8..64 decode steps, 2..5 measured repetitions")
    if (not math.isfinite(minimum_tps) or not 0 < minimum_tps <= 1000
            or not math.isfinite(maximum_vram) or not 1024 <= maximum_vram <= 131072):
        raise ValueError("Invalid throughput/VRAM budget guard")
