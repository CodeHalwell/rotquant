from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rotquant.pack import unpack_indices
from rotquant.ternary import TernaryConfig, TernaryLinear
from rotquant.ternary_w2a8 import (
    grouped_q8,
    grouped_w2a8_model,
    native_w2a8_reference,
    rounded_signed_fwht,
    ternary_w2a8_forward,
)


def test_q8_rounding_ties_zero_groups_and_bounds():
    values = torch.tensor([[.5, 1.5, 2.5, 127.], [0., 0., 0., 0.]])
    codes, scales = grouped_q8(values, group_size=4)
    assert codes[0, 0].tolist() == [0, 2, 2, 127]
    assert codes[1].count_nonzero() == 0 and scales[1, 0] == 1


def test_w2a8_grouped_integer_dot_matches_independent_scalar_oracle():
    torch.manual_seed(17)
    module = TernaryLinear(nn.Linear(256, 3, bias=False), TernaryConfig(128, 128)).to_quant_linear()
    values = torch.randn(2, 256)
    actual = ternary_w2a8_forward(module, values)
    rotated = rounded_signed_fwht(values, module.act_rotation.signs)
    weights = unpack_indices(module.qweight.packed).reshape(3, 256).to(torch.int64) - 1
    result = torch.zeros(2, 3)
    for row in range(3):
        for token in range(2):
            for group in range(2):
                start = group * 128
                activation = rotated[token, start:start + 128]
                scale = activation.abs().max() / 127
                integers = (activation / scale).round().to(torch.int64)
                dot = (integers * weights[row, start:start + 128]).sum()
                multiplier = scale * module.qweight.scales[row, group].float()
                result[token, row] = result[token, row] + dot.float() * multiplier
    assert torch.equal(actual, result.half().float())


def test_w2a8_reserved_codes_and_nonfinite_activations_fail_closed():
    from pytest import raises
    module = TernaryLinear(nn.Linear(128, 3), TernaryConfig(128, 128)).to_quant_linear()
    with raises(ValueError, match="finite"):
        ternary_w2a8_forward(module, torch.full((1, 128), float("nan")))
    module.qweight.packed.data[0] |= 3
    with raises(ValueError, match="reserved"):
        ternary_w2a8_forward(module, torch.zeros(1, 128))


def test_numpy_native_wire_oracle_matches_torch_and_permutations():
    from rotquant.native_v3 import encode_native_v3
    torch.manual_seed(37)
    module = TernaryLinear(nn.Linear(256, 7, bias=False), TernaryConfig(128, 128)).to_quant_linear()
    matrix = encode_native_v3(module.qweight)
    values = torch.randn(3, 256)
    signs = module.act_rotation.signs.numpy()
    expected = native_w2a8_reference(matrix, signs, values.numpy())
    assert np.array_equal(ternary_w2a8_forward(module, values).numpy(), expected)
    rows = np.arange(7, dtype=np.int32)[::-1].copy()
    columns = np.arange(256, dtype=np.int32)[::-1].copy()
    permuted = native_w2a8_reference(matrix, signs, values.numpy(), rows=rows, columns=columns)
    unpermuted = native_w2a8_reference(matrix, signs, values.numpy()[:, columns])
    assert np.array_equal(permuted, unpermuted[:, rows])


def test_w2a8_context_restores_forward_without_changing_packed_storage():
    from pytest import raises

    from rotquant.ternary_state import source_state_digest
    torch.manual_seed(17)
    module = TernaryLinear(nn.Linear(128, 7), TernaryConfig(128, 128)).to_quant_linear()
    model = nn.Sequential(module)
    values = torch.randn(2, 128)
    original, digest = model(values), source_state_digest(model)
    with raises(RuntimeError, match="stop"), grouped_w2a8_model(model) as receipt:
        assert receipt["sites"] == ["0"]
        assert torch.equal(model(values), ternary_w2a8_forward(module, values))
        raise RuntimeError("stop")
    assert torch.equal(model(values), original) and source_state_digest(model) == digest
