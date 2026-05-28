"""Tests for system metrics parsing and prefix cache delta computation."""

from __future__ import annotations

import pytest

from clawperf.system_metrics import parse_prometheus_metrics, BACKEND_MAP, SystemMetricsPoller


def test_parse_prometheus_simple():
    text = "vllm:gpu_cache_usage_perc 0.45\nvllm:prefix_cache_queries_total 123.0\n"
    result = parse_prometheus_metrics(text)
    assert "vllm:gpu_cache_usage_perc" in result
    assert result["vllm:gpu_cache_usage_perc"] == 0.45
    assert "vllm:prefix_cache_queries_total" in result


def test_parse_prometheus_with_labels():
    text = 'vllm:num_requests_running{model="qwen"} 5\n'
    result = parse_prometheus_metrics(text)
    assert 'vllm:num_requests_running{model="qwen"}' in result
    assert "vllm:num_requests_running" in result


def test_parse_prometheus_comments_and_empty():
    text = "# HELP some metric\n# TYPE some counter\n\nvllm:gpu_cache_usage_perc 0.5\n"
    result = parse_prometheus_metrics(text)
    assert len(result) == 1
    assert "vllm:gpu_cache_usage_perc" in result


def test_backend_map_keys():
    assert "vllm" in BACKEND_MAP
    assert "sglang" in BACKEND_MAP
    assert "mindie" in BACKEND_MAP
    for backend, mapping in BACKEND_MAP.items():
        assert "kv_cache_usage" in mapping
        assert "num_running" in mapping
        assert "num_waiting" in mapping


def test_vllm_prefix_cache_metrics():
    vllm = BACKEND_MAP["vllm"]
    assert "prefix_cache_queries" in vllm
    assert "prefix_cache_evictions" in vllm
    assert "prefix_cache_hit_tokens" in vllm
    assert "prefix_cache_query_tokens" in vllm
    assert "external_prefix_cache_queries" in vllm
    assert "external_prefix_cache_hit_tokens" in vllm


def test_prefix_cache_delta_token_hit_rate():
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    start = {
        "prefix_cache_hit_tokens": 800000,
        "prefix_cache_query_tokens": 1000000,
        "prefix_cache_evictions": 2,
        "external_prefix_cache_hit_tokens": 300000,
        "external_prefix_cache_query_tokens": 500000,
    }
    end = {
        "prefix_cache_hit_tokens": 1800000,
        "prefix_cache_query_tokens": 2000000,
        "prefix_cache_evictions": 5,
        "external_prefix_cache_hit_tokens": 1200000,
        "external_prefix_cache_query_tokens": 1500000,
    }
    delta = poller.compute_prefix_cache_delta(start, end)
    assert delta is not None
    assert delta["prefix_cache_hit_tokens_delta"] == 1000000
    assert delta["prefix_cache_query_tokens_delta"] == 1000000
    assert delta["prefix_cache_evictions_delta"] == 3
    assert delta["prefix_cache_token_hit_rate"] == 1.0
    assert delta["external_prefix_cache_hit_tokens_delta"] == 900000
    assert delta["external_prefix_cache_query_tokens_delta"] == 1000000
    assert delta["external_prefix_cache_token_hit_rate"] == 0.9


def test_prefix_cache_delta_zero_hit_rate():
    """0% hit rate should return 0.0, not omit the key."""
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    start = {"prefix_cache_hit_tokens": 0, "prefix_cache_query_tokens": 10000,
             "external_prefix_cache_hit_tokens": 0, "external_prefix_cache_query_tokens": 10000}
    end = {"prefix_cache_hit_tokens": 0, "prefix_cache_query_tokens": 100000,
           "external_prefix_cache_hit_tokens": 0, "external_prefix_cache_query_tokens": 100000}
    delta = poller.compute_prefix_cache_delta(start, end)
    assert delta is not None
    assert delta["prefix_cache_token_hit_rate"] == 0.0
    assert delta["external_prefix_cache_token_hit_rate"] == 0.0


def test_prefix_cache_delta_none_inputs():
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    assert poller.compute_prefix_cache_delta(None, None) is None
    assert poller.compute_prefix_cache_delta({"prefix_cache_hit_tokens": 10}, None) is None


def test_prefix_cache_delta_zero_query_tokens():
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    start = {"prefix_cache_hit_tokens": 500, "prefix_cache_query_tokens": 1000,
             "external_prefix_cache_hit_tokens": 500, "external_prefix_cache_query_tokens": 1000}
    end = {"prefix_cache_hit_tokens": 500, "prefix_cache_query_tokens": 1000,
           "external_prefix_cache_hit_tokens": 500, "external_prefix_cache_query_tokens": 1000}
    delta = poller.compute_prefix_cache_delta(start, end)
    assert delta is not None
    assert "prefix_cache_token_hit_rate" not in delta
    assert "external_prefix_cache_token_hit_rate" not in delta


def test_prefix_cache_delta_no_external_keys():
    """Backend that doesn't expose external prefix cache metrics."""
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    start = {"prefix_cache_hit_tokens": 500, "prefix_cache_query_tokens": 1000}
    end = {"prefix_cache_hit_tokens": 8000, "prefix_cache_query_tokens": 10000}
    delta = poller.compute_prefix_cache_delta(start, end)
    assert delta is not None
    assert delta["prefix_cache_token_hit_rate"] == 7500 / 9000
    assert delta["external_prefix_cache_hit_tokens_delta"] == 0
    assert delta["external_prefix_cache_query_tokens_delta"] == 0
    assert "external_prefix_cache_token_hit_rate" not in delta