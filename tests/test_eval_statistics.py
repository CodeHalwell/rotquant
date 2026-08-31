import pytest

from rotquant.eval.statistics import bootstrap_mean_interval, bootstrap_report


def test_bootstrap_interval_is_seeded_and_contains_sample_mean():
    values = [0.0, 1.0, 2.0, 3.0]
    first = bootstrap_mean_interval(values, draws=500, seed=4)
    repeated = bootstrap_mean_interval(values, draws=500, seed=4)
    assert first == repeated
    assert first[0] <= 1.5 <= first[1]


def test_bootstrap_report_labels_each_metric():
    report = bootstrap_report(
        {"kl": [0.1, 0.2, 0.3], "agreement": [1.0, 0.0, 1.0]},
        draws=100,
        seed=2,
    )
    assert set(report["intervals"]) == {"kl", "agreement"}
    assert report["confidence"] == pytest.approx(0.95)
