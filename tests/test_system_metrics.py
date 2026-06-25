"""Tests for system metrics parsing and prefix cache delta computation."""

from __future__ import annotations

from clawperf.system_metrics import (
    BACKEND_MAP,
    SystemMetricsPoller,
    match_metrics,
    parse_prometheus_metrics,
)


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


def test_parse_prometheus_multi_instance_sums():
    """Multiple labeled instances (e.g. per model/engine) must be SUMMED in the
    base form — otherwise prefix-cache deltas are undercounted (regression)."""
    text = (
        'vllm:prefix_cache_hit_tokens_total{model="A"} 100\n'
        'vllm:prefix_cache_hit_tokens_total{model="B"} 50\n'
        'vllm:prefix_cache_query_tokens_total{model="A"} 1000\n'
        'vllm:prefix_cache_query_tokens_total{model="B"} 500\n'
    )
    result = parse_prometheus_metrics(text)
    assert result["vllm:prefix_cache_hit_tokens_total"] == 150.0
    assert result["vllm:prefix_cache_query_tokens_total"] == 1500.0
    # Fully-qualified instances still preserved individually.
    assert result['vllm:prefix_cache_hit_tokens_total{model="A"}'] == 100.0


def test_parse_prometheus_no_label_not_doubled():
    """A label-free metric must not be double-counted into its base form."""
    text = "vllm:gpu_cache_usage_perc 0.45\n"
    result = parse_prometheus_metrics(text)
    assert result["vllm:gpu_cache_usage_perc"] == 0.45


def test_parse_prometheus_rejects_nan_inf():
    """NaN/Inf would poison delta/hit-rate math via NaN propagation — skip them."""
    text = (
        "vllm:gpu_cache_usage_perc NaN\n"
        "vllm:num_requests_running inf\n"
        "vllm:num_requests_waiting -inf\n"
    )
    result = parse_prometheus_metrics(text)
    assert result == {}  # all non-finite values dropped


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
    assert "kv_cache_usage" in vllm
    assert "num_running" in vllm
    assert "num_waiting" in vllm
    assert "prefix_cache_evictions" in vllm
    assert "prefix_cache_hit_tokens" in vllm
    assert "prefix_cache_query_tokens" in vllm
    assert "external_prefix_cache_hit_tokens" in vllm
    assert "external_prefix_cache_query_tokens" in vllm
    # Real vLLM names must be the first candidate for the token counters.
    assert vllm["prefix_cache_hit_tokens"][0] == "vllm:prefix_cache_hits_total"
    assert vllm["prefix_cache_query_tokens"][0] == "vllm:prefix_cache_queries_total"


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


def test_prefix_cache_delta_counter_reset():
    """If the backend restarted mid-run, counters go backwards. The delta must
    be flagged and the hit rate must NOT be computed (a negative delta would
    otherwise produce a bogus rate)."""
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    start = {"prefix_cache_hit_tokens": 1000, "prefix_cache_query_tokens": 2000,
             "external_prefix_cache_hit_tokens": 100, "external_prefix_cache_query_tokens": 200}
    end = {"prefix_cache_hit_tokens": 100, "prefix_cache_query_tokens": 200,
           "external_prefix_cache_hit_tokens": 10, "external_prefix_cache_query_tokens": 20}
    delta = poller.compute_prefix_cache_delta(start, end)
    assert delta is not None
    assert delta.get("prefix_cache_counter_reset") is True
    assert delta.get("external_prefix_cache_counter_reset") is True
    assert "prefix_cache_token_hit_rate" not in delta
    assert "external_prefix_cache_token_hit_rate" not in delta


# --- Real vLLM metric-name regression tests ------------------------------------
# ClawPerf used to look for vllm:prefix_cache_hit_tokens_total / *_query_tokens_total
# which DON'T EXIST in real vLLM — every run reported 0 hit/query tokens. Real
# vLLM exposes vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total
# (token counters, labeled with model_name + engine), and external cache metrics
# carry NO vllm: prefix.

REAL_VLLM_METRICS = """\
# HELP vllm:kv_cache_usage_perc Fraction of used KV cache blocks.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="Qwen2.5-7B"} 0.42
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen2.5-7B",engine="0"} 3
vllm:num_requests_waiting{model_name="Qwen2.5-7B",engine="0"} 0
# HELP vllm:prefix_cache_queries_total Total prompt tokens queried against the prefix cache.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="Qwen2.5-7B",engine="0"} 50000
vllm:prefix_cache_queries_total{model_name="Qwen2.5-7B",engine="1"} 30000
# HELP vllm:prefix_cache_hits_total Total prompt tokens reused via the prefix cache.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="Qwen2.5-7B",engine="0"} 40000
vllm:prefix_cache_hits_total{model_name="Qwen2.5-7B",engine="1"} 25000
# HELP external_prefix_cache_queries_total External prefix cache queried tokens.
# TYPE external_prefix_cache_queries_total counter
external_prefix_cache_queries_total{model_name="Qwen2.5-7B",engine="0"} 8000
# HELP external_prefix_cache_hits_total External prefix cache reused tokens.
# TYPE external_prefix_cache_hits_total counter
external_prefix_cache_hits_total{model_name="Qwen2.5-7B",engine="0"} 2000
"""


def test_real_vllm_metrics_parsed_and_mapped():
    """The exact names a real vLLM exposes must map to our internal fields."""
    raw = parse_prometheus_metrics(REAL_VLLM_METRICS)
    # Multi-engine counters are summed into the base form.
    assert raw["vllm:prefix_cache_queries_total"] == 80000.0
    assert raw["vllm:prefix_cache_hits_total"] == 65000.0

    mapped = match_metrics(raw, BACKEND_MAP["vllm"])
    assert mapped["prefix_cache_query_tokens"] == 80000.0
    assert mapped["prefix_cache_hit_tokens"] == 65000.0
    assert mapped["external_prefix_cache_query_tokens"] == 8000.0
    assert mapped["external_prefix_cache_hit_tokens"] == 2000.0
    assert mapped["kv_cache_usage"] == 0.42
    assert mapped["num_running"] == 3


def test_real_vllm_delta_hit_rate():
    """End-to-end: start/end snapshots from real vLLM names produce a hit rate."""
    poller = SystemMetricsPoller("http://localhost:8000/metrics", 5, "vllm")
    start = match_metrics(parse_prometheus_metrics(REAL_VLLM_METRICS), BACKEND_MAP["vllm"])
    # Bump the counters for the "end" snapshot.
    end_text = REAL_VLLM_METRICS.replace(
        'vllm:prefix_cache_queries_total{model_name="Qwen2.5-7B",engine="0"} 50000',
        'vllm:prefix_cache_queries_total{model_name="Qwen2.5-7B",engine="0"} 90000',
    ).replace(
        'vllm:prefix_cache_hits_total{model_name="Qwen2.5-7B",engine="0"} 40000',
        'vllm:prefix_cache_hits_total{model_name="Qwen2.5-7B",engine="0"} 75000',
    )
    end = match_metrics(parse_prometheus_metrics(end_text), BACKEND_MAP["vllm"])
    delta = poller.compute_prefix_cache_delta(start, end)
    # delta: queries 90000+30000 - 80000 = 40000; hits 75000+25000 - 65000 = 35000
    assert delta["prefix_cache_query_tokens_delta"] == 40000
    assert delta["prefix_cache_hit_tokens_delta"] == 35000
    assert delta["prefix_cache_token_hit_rate"] == 35000 / 40000


def test_vllm_metric_name_without_total_suffix():
    """Older vLLM builds drop the _total suffix — must still match."""
    raw = parse_prometheus_metrics(
        'vllm:prefix_cache_queries{model_name="m"} 100\n'
        'vllm:prefix_cache_hits{model_name="m"} 30\n'
    )
    mapped = match_metrics(raw, BACKEND_MAP["vllm"])
    assert mapped["prefix_cache_query_tokens"] == 100
    assert mapped["prefix_cache_hit_tokens"] == 30
