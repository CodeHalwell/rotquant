"""Runtime dispatch must be explicit and must never hide a dense fallback."""
from __future__ import annotations

import numpy as np
import pytest

from rotquant.native import NativeEncodedMatrix, NativeLayout, pack_native_blocks
from rotquant.runtime import (
    KernelRegistry,
    KernelSpec,
    run_kernel,
    runtime_capabilities,
)


def _encoded(bits: int = 4) -> NativeEncodedMatrix:
    layout = NativeLayout(bits, 8)
    indices = np.arange(24, dtype=np.uint16).reshape(3, 8) % (1 << bits)
    scales = np.ones((3, 1), dtype=np.float16)
    return NativeEncodedMatrix(
        qdata=pack_native_blocks(indices, scales, layout),
        codebook=np.linspace(-1, 1, 1 << bits, dtype=np.float32),
        layout=layout,
        in_features=8,
        out_features=3,
    )


@pytest.mark.parametrize("bits", range(1, 9))
def test_reference_runtime_dispatch_covers_every_profile(bits):
    encoded = _encoded(bits)
    weight = run_kernel("dequantize", encoded)
    values = np.ones((2, 8), dtype=np.float32)
    output = run_kernel("matmul", encoded, values)
    np.testing.assert_allclose(output, values @ weight.T, rtol=1e-6, atol=1e-6)


def test_runtime_never_silently_falls_back_to_reference():
    with pytest.raises(RuntimeError, match="backend='metal'"):
        run_kernel("matmul", _encoded(), np.ones((1, 8)), backend="metal")


def test_kernel_registry_selects_highest_priority_compatible_kernel():
    registry = KernelRegistry()
    low = KernelSpec(
        name="low",
        backend="cpu",
        operation="dequantize",
        bits=frozenset({4}),
        group_sizes=frozenset({8}),
        implementation=lambda encoded: np.zeros((1,), dtype=np.float32),
        priority=1,
    )
    high = KernelSpec(
        name="high",
        backend="cpu",
        operation="dequantize",
        bits=frozenset({4}),
        group_sizes=frozenset({8}),
        implementation=lambda encoded: np.ones((1,), dtype=np.float32),
        priority=2,
    )
    registry.register(low)
    registry.register(high)
    selected = registry.resolve(
        backend="cpu", operation="dequantize", bits=4, group_size=8
    )
    assert selected.name == "high"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(high)


def test_runtime_capabilities_are_machine_readable():
    capabilities = runtime_capabilities()
    assert {item["operation"] for item in capabilities} == {
        "dequantize",
        "matmul",
    }
    assert all(item["bits"] == list(range(1, 9)) for item in capabilities)
