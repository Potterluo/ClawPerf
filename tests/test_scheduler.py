"""Tests for user arrival schedulers — verify intervals, not absolute times."""

from __future__ import annotations

import pytest

from clawperf.config import BenchmarkConfig
from clawperf.scheduler import burst_scheduler, steady_scheduler, poisson_scheduler, get_scheduler


@pytest.mark.asyncio
async def test_burst_scheduler():
    """Burst: all intervals are 0 (users start simultaneously)."""
    results = []
    async for uid, interval in burst_scheduler(5):
        results.append((uid, interval))
    assert len(results) == 5
    assert all(d == 0.0 for _, d in results)
    assert [uid for uid, _ in results] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_steady_scheduler():
    """Steady: first user interval=0, rest interval=param."""
    results = []
    async for uid, interval in steady_scheduler(4, interval=2.0):
        results.append((uid, interval))
    assert len(results) == 4
    assert results[0] == (0, 0.0)   # first user starts immediately
    assert results[1] == (1, 2.0)   # wait 2s before user 1
    assert results[2] == (2, 2.0)   # wait 2s before user 2
    assert results[3] == (3, 2.0)   # wait 2s before user 3
    # Total arrival timeline: 0s, 2s, 4s, 6s (cumulative)


@pytest.mark.asyncio
async def test_poisson_scheduler_intervals():
    """Poisson: first interval=0, rest are exponential intervals."""
    results = []
    async for uid, interval in poisson_scheduler(10, lambda_rate=1.0):
        results.append((uid, interval))
    assert len(results) == 10
    assert results[0] == (0, 0.0)   # first user starts immediately
    # All other intervals should be positive (not cumulative)
    for uid, interval in results[1:]:
        assert interval > 0


def test_get_scheduler_burst():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", num_users=3, user_arrival="burst")
    scheduler = get_scheduler(cfg)
    assert scheduler is not None


def test_get_scheduler_steady():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", num_users=3, user_arrival="steady:1")
    scheduler = get_scheduler(cfg)
    assert scheduler is not None


def test_get_scheduler_invalid():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", user_arrival="burst")
    cfg.arrival_mode = "unknown"
    with pytest.raises(ValueError, match="Unknown arrival mode"):
        get_scheduler(cfg)