"""Tests for UserContext, compaction logic, and context_overflow handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clawperf.context import UserContext, CompactionEvent


def _make_tokenizer_manager(count=50000):
    tm = MagicMock()
    tm.count_chat_tokens.return_value = count
    tm.generate_random_content.return_value = "generated content"
    return tm


def test_prepare_turn_no_compaction():
    tm = _make_tokenizer_manager(count=20000)
    ctx = UserContext(
        user_id=0,
        system_prefix="sys",
        user_prefix_tokens=100,
        user_prefix_content="prefix",
        input_tokens_per_turn=500,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    result = ctx.prepare_turn(turn_id=1, current_input_content="input", tokenizer_manager=tm)
    assert result["compaction_triggered"] is False
    assert result["compaction_event"] is None
    assert result["context_overflow"] is False
    assert len(result["messages"]) > 0


def test_prepare_turn_with_compaction():
    tm = MagicMock()
    # Call 1: initial count (over limit) → triggers compaction check
    # Call 2: base context (no history) → under limit, compaction can help
    # Call 3: after compaction → under limit
    tm.count_chat_tokens.side_effect = [200000, 50000, 50000]
    tm.generate_random_content.return_value = "generated content"
    ctx = UserContext(
        user_id=0,
        system_prefix="sys",
        user_prefix_tokens=5000,
        user_prefix_content="prefix",
        input_tokens_per_turn=5000,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    result = ctx.prepare_turn(turn_id=1, current_input_content="input", tokenizer_manager=tm)
    assert result["compaction_triggered"] is True
    assert result["compaction_event"] is not None
    assert result["context_overflow"] is False  # resolved after compaction
    assert result["compaction_event"].old_prefix_len == 5000
    assert result["compaction_event"].new_prefix_len == 10000


def test_prepare_turn_compaction_still_overflows():
    """Compaction fires but context STILL exceeds limit → context_overflow=True."""
    tm = MagicMock()
    # Call 1: initial count → over limit, enters compaction logic
    # Call 2: base context (no history) → under 128000, so compaction CAN help
    # Call 3: after compaction → still over 128000
    tm.count_chat_tokens.side_effect = [200000, 50000, 150000]
    tm.generate_random_content.return_value = "generated content"
    ctx = UserContext(
        user_id=0,
        system_prefix="sys",
        user_prefix_tokens=5000,
        user_prefix_content="prefix",
        input_tokens_per_turn=5000,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    result = ctx.prepare_turn(turn_id=1, current_input_content="input", tokenizer_manager=tm)
    assert result["compaction_triggered"] is True
    assert result["context_overflow"] is True  # still over after compaction


def test_prepare_turn_base_context_exceeds_limit():
    """When base context (no history) already exceeds limit, skip compaction → overflow."""
    tm = MagicMock()
    # Call 1: initial count → over limit
    # Call 2: base context → also over limit (compaction can't help)
    tm.count_chat_tokens.side_effect = [200000, 150000]
    tm.generate_random_content.return_value = "generated content"
    ctx = UserContext(
        user_id=0,
        system_prefix="sys",
        user_prefix_tokens=5000,
        user_prefix_content="prefix",
        input_tokens_per_turn=5000,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    result = ctx.prepare_turn(turn_id=1, current_input_content="input", tokenizer_manager=tm)
    assert result["compaction_triggered"] is False  # compaction skipped
    assert result["context_overflow"] is True


def test_prepare_turn_compaction_at_exact_limit():
    """Compaction should trigger when context_tokens == max_context_tokens (>= not >)."""
    tm = MagicMock()
    tm.count_chat_tokens.side_effect = [128000, 50000, 50000]
    tm.generate_random_content.return_value = "generated content"
    ctx = UserContext(
        user_id=0,
        system_prefix="sys",
        user_prefix_tokens=5000,
        user_prefix_content="prefix",
        input_tokens_per_turn=5000,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    result = ctx.prepare_turn(turn_id=1, current_input_content="input", tokenizer_manager=tm)
    assert result["compaction_triggered"] is True


def test_append_history():
    ctx = UserContext(
        user_id=0,
        system_prefix="sys",
        user_prefix_tokens=100,
        user_prefix_content="prefix",
        input_tokens_per_turn=500,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    ctx.append_history("user msg", "assistant reply")
    assert len(ctx.history) == 1
    assert ctx.history[0] == ("user msg", "assistant reply")


def test_build_messages_structure():
    ctx = UserContext(
        user_id=0,
        system_prefix="sys_prefix",
        user_prefix_tokens=100,
        user_prefix_content="user_prefix",
        input_tokens_per_turn=500,
        max_context_tokens=128000,
        compaction_prefix_increment=5000,
        max_turns=100,
    )
    ctx.append_history("q1", "a1")
    messages = ctx._build_messages("current_input")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "sys_prefix"
    # First user message should combine prefix + first history
    assert messages[1]["role"] == "user"
    assert "user_prefix" in messages[1]["content"]
    assert "q1" in messages[1]["content"]