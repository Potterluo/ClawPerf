"""Tests for SLO mode: verdict logic, sweep planning, config validation."""

from __future__ import annotations

from clawperf.config import BenchmarkConfig
from clawperf.runner import BenchmarkRunner


def _runner(**kw):
    defaults = dict(endpoint="http://x", model="m", mode="slo",
                    slo_ttft_ms=500, slo_tpot_ms=30)
    defaults.update(kw)
    return BenchmarkRunner(BenchmarkConfig(**defaults))


def test_slo_verdict_ok():
    r = _runner()
    assert r._slo_verdict(p_ttft=100, p_tpot=10, err_rate=0.0, timed_out=False) is True


def test_slo_verdict_ttft_exceeds():
    r = _runner()
    assert r._slo_verdict(p_ttft=600, p_tpot=10, err_rate=0.0, timed_out=False) is False


def test_slo_verdict_tpot_exceeds():
    r = _runner()
    assert r._slo_verdict(p_ttft=100, p_tpot=40, err_rate=0.0, timed_out=False) is False


def test_slo_verdict_timeout():
    r = _runner()
    assert r._slo_verdict(p_ttft=10, p_tpot=1, err_rate=0.0, timed_out=True) is False


def test_slo_verdict_none_metrics_fail():
    """If TTFT/TPOT couldn't be measured (all errors), the SLO is not met."""
    r = _runner()
    assert r._slo_verdict(p_ttft=None, p_tpot=None, err_rate=1.0, timed_out=False) is False


def test_slo_verdict_error_rate():
    r = _runner(slo_error_rate=0.05)
    # 3% errors OK
    assert r._slo_verdict(p_ttft=100, p_tpot=10, err_rate=0.03, timed_out=False) is True
    # 10% errors fail
    assert r._slo_verdict(p_ttft=100, p_tpot=10, err_rate=0.10, timed_out=False) is False


def test_slo_verdict_only_tpot_configured():
    """If only TPOT SLO is set, TTFT is unchecked."""
    r = _runner(slo_ttft_ms=None)
    assert r._slo_verdict(p_ttft=99999, p_tpot=10, err_rate=0.0, timed_out=False) is True


def test_next_n_geometric():
    r = _runner(slo_step_strategy="geometric")
    assert r._next_n(1) == 2
    assert r._next_n(4) == 8
    assert r._next_n(5) == 10  # max(n+1, n*2)


def test_next_n_linear():
    r = _runner(slo_step_strategy="linear")
    assert r._next_n(4) == 5
    assert r._next_n(10) == 11


def test_slo_label():
    r = _runner(slo_error_rate=0.01)
    label = r._slo_label()
    assert "TTFT P99<=" in label and "500" in label
    assert "TPOT P99<=" in label and "30" in label
    assert "err<=1.0%" in label


def test_config_validate_slo_ok():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", mode="slo",
                          slo_ttft_ms=500, slo_tpot_ms=30)
    assert cfg.validate() == []


def test_config_validate_slo_no_targets():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", mode="slo")
    problems = cfg.validate()
    assert any("slo-ttft-ms" in p or "slo-tpot-ms" in p for p in problems)


def test_config_validate_slo_bad_range():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", mode="slo",
                          slo_ttft_ms=500, slo_min_users=10, slo_max_users=5)
    problems = cfg.validate()
    assert any("slo_max_users" in p for p in problems)


def test_config_validate_slo_bad_percentile():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", mode="slo",
                          slo_ttft_ms=500, slo_percentile=1.5)
    problems = cfg.validate()
    assert any("slo_percentile" in p for p in problems)
