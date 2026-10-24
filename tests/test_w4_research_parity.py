"""Exact-tie native conformance keeps strict reload and numerical guards."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.w4_research_worker import Worker, native_argmax_diagnostics, parity_row


def probe(actual_row, reference_row):
    reference = np.tile(np.array([4., 0., -2., -3.]), (16, 1))
    reference[5] = reference_row
    actual = reference.copy()
    actual[5] = actual_row
    return actual, reference


@pytest.mark.parametrize("actual_row,reference_row", [
    ([21.21875, 21.21875, -1., -2.], [21.21875, 21.234375, -1., -2.]),
    ([21.21875, 21.234375, -1., -2.], [21.234375, 21.234375, -1., -2.]),
])
def test_native_exact_tie_intersection_preserves_strict_disagreement(actual_row, reference_row):
    actual, reference = probe(actual_row, reference_row)
    row = parity_row(actual, reference)
    assert row["passed"] and row["top1_guard"] == "argmax_set_agreement"
    assert row["metrics"]["top1_agreement"] == 15 / 16
    assert row["metrics"]["argmax_set_agreement"] == 1.
    mismatch, = row["top1_mismatches"]
    assert mismatch["position"] == 5 and mismatch["argmax_sets_intersect"]
    assert mismatch["actual_logits"] == actual[5, mismatch["token_ids"]].tolist()
    assert mismatch["reference_logits"] == reference[5, mismatch["token_ids"]].tolist()
    assert set(mismatch["actual_argmax_ids"]) & set(mismatch["reference_argmax_ids"])
    assert sorted([mismatch["actual_top2_margin"], mismatch["reference_top2_margin"]]) == [0., .015625]


def test_identical_ties_remain_exact_and_record_no_mismatch():
    logits = np.array([[1., 1., 0.], [0., 0., 0.]])
    row = parity_row(logits, logits)
    assert row["passed"]
    assert row["metrics"]["top1_agreement"] == row["metrics"]["argmax_set_agreement"] == 1.
    assert row["top1_mismatches"] == []


@pytest.mark.parametrize("actual_row,reference_row", [
    ([0., 0., .0009765625, .0009765625], [.0009765625, .0009765625, 0., 0.]),
    ([21.234375, 21.21875, -1., -2.], [21.21875, 21.234375, -1., -2.]),
    ([1. + 2**-23, 1., -1., -2.], [1., 1. + 2**-23, -1., -2.]),
])
def test_disjoint_ties_and_positive_margin_inversions_still_fail(actual_row, reference_row):
    row = parity_row(*probe(actual_row, reference_row))
    assert not row["passed"]
    assert row["metrics"]["max_abs_error"] < .125
    assert row["metrics"]["mean_abs_error"] < .005
    assert row["metrics"]["mean_kl"] < 1e-4
    assert row["metrics"]["argmax_set_agreement"] == 15 / 16
    assert row["top1_mismatches"][0]["argmax_sets_intersect"] is False


@pytest.mark.parametrize("failed_guard", ["max_abs_error", "mean_abs_error", "mean_kl"])
def test_exact_argmax_intersection_cannot_bypass_numerical_guards(failed_guard):
    reference = np.zeros((16, 256))
    reference[:, :2] = 21.
    actual = reference.copy()
    if failed_guard == "max_abs_error":
        actual[0, -1] += .126
    elif failed_guard == "mean_abs_error":
        actual += .006
    else:
        actual[:, 0] += .1
    row = parity_row(actual, reference)
    limits = {"max_abs_error": .125, "mean_abs_error": .005, "mean_kl": 1e-4}
    assert row["metrics"]["argmax_set_agreement"] == 1.
    assert row["metrics"][failed_guard] > limits[failed_guard]
    assert all(row["metrics"][key] <= limit for key, limit in limits.items() if key != failed_guard)
    assert not row["passed"]


def test_reload_stays_strict_even_when_all_numerical_limits_pass():
    actual, reference = probe([1., 1., -1., -2.], [1., 1.0009765625, -1., -2.])
    row = parity_row(actual, reference, reload=True)
    assert row["metrics"]["max_abs_error"] < .002
    assert row["metrics"]["mean_abs_error"] < .0001
    assert row["metrics"]["mean_kl"] < 1e-6
    assert row["top1_guard"] == "top1_agreement" and not row["passed"]
    assert "argmax_set_agreement" not in row["metrics"]
    assert parity_row(reference, reference, reload=True)["passed"]


@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_logits_are_rejected(reload, value):
    actual, reference = probe([1., 0., -1., -2.], [1., 0., -1., -2.])
    actual[5, 0] = value
    with pytest.raises(ValueError, match="invalid conformance logits"):
        parity_row(actual, reference, reload=reload)
    with pytest.raises(ValueError, match="invalid native parity logits"):
        native_argmax_diagnostics(actual, reference)


def test_worker_persists_raw_tie_details_and_still_raises_on_inversion(tmp_path):
    actual, reference = probe([21.21875, 21.21875, -1., -2.], [21.21875, 21.234375, -1., -2.])
    np.savez(tmp_path / "probes.npz", selection=reference)
    worker = Worker.__new__(Worker)
    worker.args = SimpleNamespace(probes=tmp_path / "probes.npz")
    worker.output = tmp_path
    worker.data = {"documents": {"selection": [{"id": "selection", "input_ids": [1, 2]}]}}
    worker.parity(lambda _: actual)
    report = json.loads((tmp_path / "parity.json").read_text())
    assert report["passed"]
    assert report["protocol"] == "rotquant-deployment-parity-v2"
    assert "does not assert identical greedy generation" in report["boundary"]
    assert report["rows"][0]["top1_mismatches"][0]["actual_argmax_ids"] == [0, 1]
    actual[5, 0] = 21.234375
    with pytest.raises(ValueError, match="failed parity"):
        worker.parity(lambda _: actual)
    assert json.loads((tmp_path / "parity.json").read_text())["passed"] is False
