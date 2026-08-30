from scripts.benchmark_quantizer_variants import benchmark_variants


def test_quantizer_variant_benchmark_is_budget_matched_and_reports_tradeoff():
    payload = benchmark_variants(
        bits=(2,), dimension=8, rows=32, probes=8, seed=5)
    results = payload["results"]["2"]
    rates = {metrics["effective_bpw"] for metrics in results.values()}
    assert len(rates) == 1
    assert abs(results["spherical_length"]["global_self_dot_ratio"] - 1) < 1e-3
    assert results["gaussian"]["weight_nmse"] > 0
