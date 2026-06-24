"""Tests for per-user aggregation using real wall-clock timestamps.

Covers the regression where ``duration_s`` was derived from the last turn's
e2e latency (nonsense) instead of the actual request window.
"""

from __future__ import annotations

from clawperf.config import BenchmarkConfig
from clawperf.context import UserContext
from clawperf.runner import BenchmarkRunner


def _runner():
    cfg = BenchmarkConfig(endpoint="http://x", model="m", num_users=1)
    r = BenchmarkRunner(cfg)
    # Pretend the benchmark window started at t=100 (monotonic offset).
    r._bench_start_time = 100.0
    r._setup_start_time = 90.0
    r._user_contexts[0] = UserContext(
        user_id=0, system_prefix="s", user_prefix_tokens=10, user_prefix_content="p",
        input_tokens_per_turn=10, max_context_tokens=128000,
        compaction_prefix_increment=100, max_turns=10,
    )
    return r


def _turn(uid, tid, start_off, end_off, out_tok=1000, in_tok=5000, success=True):
    return {
        "user_id": uid, "turn_id": tid, "success": success,
        "output_tokens": out_tok, "input_tokens": in_tok,
        "wall_start_ts": start_off, "wall_end_ts": end_off,
        "ttft_ms": 50.0, "e2e_latency_ms": 500.0, "tpot_ms": 5.0,
    }


def test_per_user_duration_uses_wall_clock():
    r = _runner()
    # Two turns spanning 0s -> 3s window (not the e2e latency).
    r._turn_records = [
        _turn(0, 1, start_off=0.0, end_off=1.0),
        _turn(0, 2, start_off=1.0, end_off=3.0),
    ]
    agg = r._compute_user_aggregate(0)
    assert agg["duration_s"] == 3.0          # full window, not last-turn latency
    assert agg["throughput_tok_s"] == 2000 / 3.0  # 2000 output tokens / 3s


def test_per_user_single_turn():
    r = _runner()
    r._turn_records = [_turn(0, 1, start_off=0.0, end_off=0.5)]
    agg = r._compute_user_aggregate(0)
    assert agg["duration_s"] == 0.5
    assert agg["throughput_tok_s"] == 1000 / 0.5


def test_per_user_no_successful_turns():
    r = _runner()
    r._turn_records = [{
        "user_id": 0, "turn_id": 1, "success": False,
        "error_type": "context_overflow", "context_tokens": 999,
    }]
    agg = r._compute_user_aggregate(0)
    # No wall-clock window derivable -> duration/throughput absent.
    assert "duration_s" not in agg
    assert "throughput_tok_s" not in agg
    assert agg["error_count"] == 1
