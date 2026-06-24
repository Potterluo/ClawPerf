"""Tests for runner math: percentile interpolation and per-user aggregation."""

from __future__ import annotations

from clawperf.runner import _percentile, _percentiles


def test_percentile_single_value():
    assert _percentile([5.0], 0.5) == 5.0
    assert _percentile([5.0], 0.99) == 5.0


def test_percentile_empty():
    assert _percentile([], 0.5) == 0.0


def test_percentile_two_values_interpolates_median():
    """The old int-floor method returned the upper value (2.0) for [1, 2].

    Linear interpolation gives the true midpoint (1.5)."""
    assert _percentile([1.0, 2.0], 0.50) == 1.5


def test_percentile_known_quantiles():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    # Q2 (median of 1..100) = 55 via linear interpolation
    assert _percentile(vals, 0.50) == 55.0
    # min/max edges
    assert _percentile(vals, 0.0) == 10.0
    assert _percentile(vals, 1.0) == 100.0


def test_percentiles_dict_keys():
    pct = _percentiles([1.0, 2.0, 3.0, 4.0])
    assert set(pct) == {"avg", "min", "P50", "P75", "P90", "P99", "max", "N"}
    assert pct["N"] == 4
    assert pct["min"] == 1.0
    assert pct["max"] == 4.0


def test_percentiles_empty_returns_empty():
    assert _percentiles([]) == {}


def test_percentiles_never_indexes_out_of_range():
    """Regression: every quantile must stay within the sample range."""
    for n in range(1, 6):
        vals = list(range(n))
        pct = _percentiles([float(v) for v in vals])
        assert pct["min"] == 0.0
        assert pct["max"] == float(n - 1)
        for key in ("P50", "P75", "P90", "P99"):
            assert pct["min"] <= pct[key] <= pct["max"]
