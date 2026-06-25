"""Tests for BenchmarkConfig parsing and serialization."""

from __future__ import annotations

import pytest

from clawperf.config import BenchmarkConfig


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig(endpoint="http://localhost:8000/v1/chat/completions", model="test")
        assert cfg.num_users == 1
        assert cfg.user_arrival == "burst"
        assert cfg.arrival_mode == "burst"
        assert cfg.arrival_param == 0.0
        assert cfg.max_turns == 100
        assert cfg.backend == "vllm"

    def test_burst_arrival(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="m", user_arrival="burst")
        assert cfg.arrival_mode == "burst"
        assert cfg.arrival_param == 0.0

    def test_steady_arrival(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="m", user_arrival="steady:2")
        assert cfg.arrival_mode == "steady"
        assert cfg.arrival_param == 2.0

    def test_poisson_arrival(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="m", user_arrival="poisson:0.5")
        assert cfg.arrival_mode == "poisson"
        assert cfg.arrival_param == 0.5

    def test_invalid_arrival(self):
        with pytest.raises(ValueError, match="Invalid user_arrival"):
            BenchmarkConfig(endpoint="http://x", model="m", user_arrival="invalid")

    def test_tokenizer_defaults_to_model(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="qwen3-32b", tokenizer="")
        assert cfg.tokenizer == "qwen3-32b"

    def test_tokenizer_explicit(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="m", tokenizer="custom-tokenizer")
        assert cfg.tokenizer == "custom-tokenizer"

    def test_to_dict(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="m")
        d = cfg.to_dict()
        assert d["endpoint"] == "http://x"
        assert d["model"] == "m"
        assert "num_users" in d

    def test_validate_ok_defaults(self):
        cfg = BenchmarkConfig(endpoint="http://x", model="m")
        assert cfg.validate() == []

    def test_validate_base_exceeds_window(self):
        cfg = BenchmarkConfig(
            endpoint="http://x", model="m",
            system_prefix_tokens=50000, user_prefix_tokens=50000,
            input_tokens_per_turn=5000, max_context_tokens=100000,
        )
        problems = cfg.validate()
        assert any("Base context" in p and "overflow" in p for p in problems)

    def test_validate_bad_increment_and_counts(self):
        cfg = BenchmarkConfig(
            endpoint="http://x", model="m",
            compaction_prefix_increment=0, num_users=0, max_turns=0,
        )
        problems = cfg.validate()
        assert any("compaction_prefix_increment" in p for p in problems)
        assert any("num_users" in p for p in problems)
        assert any("max_turns" in p for p in problems)
