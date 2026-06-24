"""Tests for the JSONL history append feature."""

from __future__ import annotations

import json
from pathlib import Path

from clawperf.config import BenchmarkConfig
from clawperf.runner import BenchmarkRunner


def _result_fixture():
    return {
        "config": {"model": "m", "endpoint": "http://x", "num_users": 1},
        "summary": {"total_compactions": 0, "prefix_cache_token_hit_rate": 0.8},
        "users": [
            {"user_id": 0, "aggregate": {"total_output_tokens": 1000, "duration_s": 2.5}},
        ],
        "system_metrics": [],
        "timeline": [],
    }


def _runner(history_path: str) -> BenchmarkRunner:
    cfg = BenchmarkConfig(endpoint="http://x", model="m", history=history_path)
    r = BenchmarkRunner(cfg)
    # Minimal state so _finalize-style fields resolve.
    r._bench_start_time = 10.0
    return r


def test_history_appends_one_line_per_call(tmp_path: Path):
    hist = tmp_path / "h.jsonl"
    r = _runner(str(hist))
    r._append_history(_result_fixture(), setup_time_s=1.0, bench_time_s=2.0)
    r._append_history(_result_fixture(), setup_time_s=1.5, bench_time_s=3.0)
    lines = hist.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["config"]["model"] == "m"
    assert rec0["timing"]["bench_time_s"] == 2.0
    # Per-user aggregates present, but no heavy per-turn arrays.
    assert rec0["users"][0]["aggregate"]["total_output_tokens"] == 1000
    assert "turns" not in rec0["users"][0]
    # output_file recorded as an absolute path.
    assert "output_file" in rec0
    assert "timestamp" in rec0


def test_history_disabled_when_empty(tmp_path: Path):
    r = _runner("")  # empty string disables
    r._append_history(_result_fixture(), 1.0, 2.0)
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere


def test_history_creates_parent_dirs(tmp_path: Path):
    hist = tmp_path / "nested" / "dir" / "h.jsonl"
    r = _runner(str(hist))
    r._append_history(_result_fixture(), 1.0, 2.0)
    assert hist.exists()


def test_history_record_is_single_valid_json_line(tmp_path: Path):
    hist = tmp_path / "h.jsonl"
    r = _runner(str(hist))
    r._append_history(_result_fixture(), 1.0, 2.0)
    # Each line must be independently parseable (no trailing commas, no newlines inside).
    line = hist.read_text(encoding="utf-8").rstrip("\n")
    json.loads(line)  # raises if malformed
