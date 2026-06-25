"""Tests for mock server logic."""

from __future__ import annotations

from clawperf.mock_server import (
    SUPPORTED_MODELS,
    PrefixCacheTrie,
    _content_to_text,
    _estimate_tokens,
    _generate_content,
    _messages_to_chunks,
)


def test_estimate_tokens():
    assert _estimate_tokens("hello world") == 2
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("a") == 1


def test_content_to_text_normalizes_none_and_list():
    """OpenAI allows content=null or a multipart array — both must not crash."""
    assert _content_to_text(None) == ""
    assert _content_to_text("hi") == "hi"
    assert _content_to_text([{"type": "text", "text": "hello "},
                             {"type": "text", "text": "world"}]) == "hello world"
    assert _content_to_text([{"type": "image_url"}]) == ""


def test_messages_to_chunks_with_odd_content():
    # None and list content used to raise TypeError; now they normalize.
    chunks = _messages_to_chunks([
        {"role": "system", "content": None},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ])
    assert len(chunks) == 2
    assert all(isinstance(c, tuple) for c in chunks)


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
    expected = sum(_estimate_tokens(c) for c in ("sys", "hello", "reply1", "more"))
    assert matched == expected


def test_trie_eviction_removes_oldest():
    """Eviction must actually drop the oldest sequence (regression: it was a no-op)."""
    trie = PrefixCacheTrie(max_prefixes=4)
    for i in range(10):
        trie.insert(_messages_to_chunks([
            {"role": "system", "content": f"sys{i}"},
            {"role": "user", "content": f"u{i}"},
        ]))
    # Cache capped at 4 sequences; the 6 oldest must be gone.
    assert len(trie._sequences) <= 4
    oldest = _messages_to_chunks([
        {"role": "system", "content": "sys0"},
        {"role": "user", "content": "u0"},
    ])
    assert trie.query(oldest) == 0  # fully evicted
    newest = _messages_to_chunks([
        {"role": "system", "content": "sys9"},
        {"role": "user", "content": "u9"},
    ])
    assert trie.query(newest) > 0   # still cached


def test_trie_eviction_preserves_shared_prefix():
    """When a sequence is evicted, shared prefix nodes that another live sequence
    references must survive (regression: old code deleted by hash, corrupting it)."""
    trie = PrefixCacheTrie(max_prefixes=2)
    shared = [{"role": "system", "content": "shared"}]
    a = shared + [{"role": "user", "content": "u1"}]
    b = shared + [{"role": "user", "content": "u1"},
                  {"role": "assistant", "content": "a1"},
                  {"role": "user", "content": "q2"}]
    trie.insert(_messages_to_chunks(a))
    trie.insert(_messages_to_chunks(b))
    # Insert a third distinct sequence -> evicts `a`, but `shared` is still used by `b`.
    trie.insert(_messages_to_chunks([
        {"role": "system", "content": "other"},
        {"role": "user", "content": "x"},
    ]))
    matched = trie.query(_messages_to_chunks(b))
    assert matched > 0  # shared prefix survived eviction


def test_trie_reinsert_refreshes_recency():
    """Re-inserting the same sequence counts as a recent use (not evicted first)."""
    def msgs(s, u):
        return [{"role": "system", "content": s}, {"role": "user", "content": u}]

    trie = PrefixCacheTrie(max_prefixes=2)
    a = _messages_to_chunks(msgs("s0", "u0"))
    trie.insert(a)
    trie.insert(_messages_to_chunks(msgs("s1", "u1")))
    # Touch `a` again so it becomes the most-recent.
    trie.insert(a)
    trie.insert(_messages_to_chunks(msgs("s2", "u2")))
    # `a` was refreshed, so s1 should have been evicted instead.
    assert trie.query(a) > 0


def test_trie_hit_rate_grows_with_prefix_reuse():
    """Multi-turn workload: hit tokens should grow turn over turn (monotonic-ish)."""
    trie = PrefixCacheTrie()
    history = []
    hits = []
    for i in range(5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
        chunks = _messages_to_chunks([{"role": "system", "content": "sys"}] + history)
        hits.append(trie.query(chunks))
        trie.insert(chunks)
    # Turn 1 has no prior cache (only system prefix shared, but it wasn't inserted yet)
    # Subsequent turns should reuse more.
    assert hits[-1] > hits[1] > hits[0]
