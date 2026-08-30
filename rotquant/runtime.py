"""Fail-closed capability registry for RotQuant runtime kernels."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .format import OPTIMIZED_PROFILE_BITS
from .native import (
    NativeEncodedMatrix,
    reference_dequantize,
    reference_streaming_matmul,
)

KernelFunction = Callable[..., np.ndarray]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """One implementation and the exact layouts it accepts."""

    name: str
    backend: str
    operation: str
    bits: frozenset[int]
    group_sizes: frozenset[int] | None
    implementation: KernelFunction
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.backend or not self.operation:
            raise ValueError("kernel name, backend and operation must not be empty")
        if not self.bits or not self.bits.issubset(OPTIMIZED_PROFILE_BITS):
            raise ValueError("kernel bits must be a non-empty subset of 1..8")
        if self.group_sizes is not None and (
            not self.group_sizes or any(size < 1 for size in self.group_sizes)
        ):
            raise ValueError("kernel group sizes must be positive")
        if not callable(self.implementation):
            raise TypeError("kernel implementation must be callable")

    def supports(
        self, *, backend: str, operation: str, bits: int, group_size: int
    ) -> bool:
        return (
            self.backend == backend
            and self.operation == operation
            and bits in self.bits
            and (self.group_sizes is None or group_size in self.group_sizes)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "backend": self.backend,
            "operation": self.operation,
            "bits": sorted(self.bits),
            "group_sizes": (
                None if self.group_sizes is None else sorted(self.group_sizes)
            ),
            "priority": self.priority,
        }


class KernelRegistry:
    """Runtime kernel registry with no implicit backend fallback."""

    def __init__(self) -> None:
        self._kernels: dict[str, KernelSpec] = {}

    def register(self, kernel: KernelSpec, *, replace: bool = False) -> None:
        if not isinstance(kernel, KernelSpec):
            raise TypeError("kernel must be a KernelSpec")
        if kernel.name in self._kernels and not replace:
            raise ValueError(f"kernel is already registered: {kernel.name}")
        self._kernels[kernel.name] = kernel

    def resolve(
        self,
        *,
        backend: str,
        operation: str,
        bits: int,
        group_size: int,
    ) -> KernelSpec:
        matches = [
            kernel
            for kernel in self._kernels.values()
            if kernel.supports(
                backend=backend,
                operation=operation,
                bits=bits,
                group_size=group_size,
            )
        ]
        if not matches:
            raise RuntimeError(
                "no RotQuant kernel for "
                f"backend={backend!r}, operation={operation!r}, "
                f"bits={bits}, group_size={group_size}"
            )
        return max(matches, key=lambda kernel: kernel.priority)

    def capabilities(self) -> list[dict[str, object]]:
        return [
            kernel.to_dict()
            for kernel in sorted(
                self._kernels.values(),
                key=lambda item: (
                    item.backend,
                    item.operation,
                    -item.priority,
                    item.name,
                ),
            )
        ]


KERNELS = KernelRegistry()


def _reference_matmul_kernel(
    encoded: NativeEncodedMatrix, values: np.ndarray
) -> np.ndarray:
    return reference_streaming_matmul(values, encoded)


KERNELS.register(
    KernelSpec(
        name="numpy-reference-dequantize",
        backend="reference",
        operation="dequantize",
        bits=frozenset(OPTIMIZED_PROFILE_BITS),
        group_sizes=None,
        implementation=reference_dequantize,
    )
)
KERNELS.register(
    KernelSpec(
        name="numpy-reference-matmul",
        backend="reference",
        operation="matmul",
        bits=frozenset(OPTIMIZED_PROFILE_BITS),
        group_sizes=None,
        implementation=_reference_matmul_kernel,
    )
)


def run_kernel(
    operation: str,
    encoded: NativeEncodedMatrix,
    *args: Any,
    backend: str = "reference",
    registry: KernelRegistry = KERNELS,
) -> np.ndarray:
    """Resolve and execute one kernel for an encoded matrix.

    ``backend`` is never silently replaced with ``reference``. A production
    benchmark requesting Metal or CUDA therefore fails if that kernel has not
    been registered, instead of accidentally timing a dense fallback.
    """

    kernel = registry.resolve(
        backend=backend,
        operation=operation,
        bits=encoded.layout.bits,
        group_size=encoded.layout.group_size,
    )
    return kernel.implementation(encoded, *args)


def runtime_capabilities(
    registry: KernelRegistry = KERNELS,
) -> list[dict[str, object]]:
    return registry.capabilities()


__all__ = [
    "KERNELS",
    "KernelRegistry",
    "KernelSpec",
    "run_kernel",
    "runtime_capabilities",
]
