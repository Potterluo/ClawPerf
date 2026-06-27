"""Tests for the prefix-cache reset helper and base-URL derivation."""

from __future__ import annotations

from clawperf.system_metrics import RESET_PATHS, _base_url


def test_base_url_strips_v1_paths():
    assert _base_url("http://h:8921/v1/chat/completions") == "http://h:8921"
    assert _base_url("http://h:8921/v1/completions") == "http://h:8921"
    assert _base_url("http://h:8921/v1/") == "http://h:8921"
    assert _base_url("https://h/v1/chat/completions/") == "https://h"


def test_reset_paths_known_backends():
    assert RESET_PATHS["vllm"] == "/reset_prefix_cache"
    assert RESET_PATHS["sglang"] == "/flush_cache"
    assert RESET_PATHS["mindie"] is None


async def test_reset_returns_false_for_mindie():
    """MindIE has no known reset endpoint — should skip cleanly, not error."""
    from clawperf.system_metrics import reset_prefix_cache
    ok = await reset_prefix_cache("http://h:8921/v1/chat/completions", "mindie")
    assert ok is False


async def test_reset_returns_false_on_connection_error():
    """Unreachable server must not raise — it warns and returns False."""
    from clawperf.system_metrics import reset_prefix_cache
    ok = await reset_prefix_cache(
        "http://127.0.0.1:1/v1/chat/completions", "vllm"
    )
    assert ok is False
