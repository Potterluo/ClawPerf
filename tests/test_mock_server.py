"""Tests for mock server logic."""

from __future__ import annotations

from clawperf.mock_server import (
    _estimate_tokens, _generate_content, SUPPORTED_MODELS,
    PrefixCacheTrie, _messages_to_chunks,
)


def test_estimate_tokens():
    assert _estimate_tokens("hello world") == 2
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("a") == 1


def test_generate_content_short():
    messages = [{"role": "user", "content": "hello"}]
    content = _generate_content(messages, max_tokens=5)
    assert len(content) > 0


def test_generate_content_long():
    messages = [{"role": "user", "content": "test"}]
    content = _generate_content(messages, max_tokens=100)
    assert len(content) > 0


def test_supported_models():
    assert "gpt-3.5-turbo" in SUPPORTED_MODELS
    assert "qwen3-32b" in SUPPORTED_MODELS


def test_trie_grows_on_repeated_insert():
    """Trie should accumulate prefix depth across multiple inserts of growing sequences."""
    trie = PrefixCacheTrie()
    msgs1 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    msgs2 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"},
             {"role": "assistant", "content": "reply1"}, {"role": "user", "content": "more"}]

    chunks1 = _messages_to_chunks(msgs1)
    chunks2 = _messages_to_chunks(msgs2)

    trie.insert(chunks1)
    # Turn 1: trie only has first 2 chunks
    assert trie.query(chunks2) == _estimate_tokens("sys") + _estimate_tokens("hello")

    # Always insert the full Turn 2 sequence
    trie.insert(chunks2)
    # Now trie matches 3 chunks (sys + hello + reply1)
    msgs3 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"},
             {"role": "assistant", "content": "reply1"}, {"role": "user", "content": "more"},
             {"role": "assistant", "content": "reply2"}, {"role": "user", "content": "final"}]
    chunks3 = _messages_to_chunks(msgs3)
    matched = trie.query(chunks3)
    expected = _estimate_tokens("sys") + _estimate_tokens("hello") + _estimate_tokens("reply1") + _estimate_tokens("more")
    assert matched == expected