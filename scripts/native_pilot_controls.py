"""Dependency-free pilot controls, checked before installing any packages."""
import math


def validate_controls(context, decode, repetitions, minimum_tps, maximum_vram):
    if context not in (128, 512, 2048) or not 8 <= decode <= 64 or not 2 <= repetitions <= 5:
        raise ValueError("Pilot permits contexts 128/512/2048, 8..64 decode steps, 2..5 measured repetitions")
    if (not math.isfinite(minimum_tps) or not 0 < minimum_tps <= 1000
            or not math.isfinite(maximum_vram) or not 1024 <= maximum_vram <= 131072):
        raise ValueError("Invalid throughput/VRAM budget guard")
