"""Context assembly and compaction logic.

Implements the per-turn context structure:
  [System Prefix] [User Prefix] [History] [Current Input]

And the append-mode compaction mechanism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger("clawperf")


@dataclass
class CompactionEvent:
    user_id: int
    turn: int
    time: float
    old_prefix_len: int
    new_prefix_len: int


@dataclass
class UserContext:
    """Maintains the evolving context state for a single user."""

    user_id: int
    system_prefix: str
    user_prefix_tokens: int
    user_prefix_content: str
    input_tokens_per_turn: int
    max_context_tokens: int
    compaction_prefix_increment: int
    max_turns: int

    history: List[Tuple[str, str]] = field(default_factory=list)
    compaction_events: List[CompactionEvent] = field(default_factory=list)

    def prepare_turn(
        self,
        turn_id: int,
        current_input_content: str,
        tokenizer_manager,
    ) -> dict:
        """Prepare context for a turn. May trigger compaction.

        Returns dict with: messages, context_tokens, compaction_triggered,
        compaction_event, context_overflow.
        """
        compaction_event = None
        context_overflow = False
        messages = self._build_messages(current_input_content)
        context_tokens = tokenizer_manager.count_chat_tokens(messages)

        if context_tokens >= self.max_context_tokens:
            # Check if base context (without history) already exceeds limit
            # — compaction can't help if system+prefix+input alone is too large
            base_messages = self._build_messages(current_input_content, skip_history=True)
            base_tokens = tokenizer_manager.count_chat_tokens(base_messages)
            if base_tokens >= self.max_context_tokens:
                context_overflow = True
                logger.warning(
                    "[User %02d] Base context (%d tokens) already exceeds limit (%d) "
                    "— compaction cannot help, skipping turn %d",
                    self.user_id, base_tokens, self.max_context_tokens, turn_id,
                )
            else:
                old_prefix_len = self.user_prefix_tokens
                self.history.clear()
                self.user_prefix_tokens += self.compaction_prefix_increment
                self.user_prefix_content = tokenizer_manager.generate_random_content(
                    self.user_prefix_tokens
                )
                compaction_event = CompactionEvent(
                    user_id=self.user_id,
                    turn=turn_id,
                    time=0.0,
                    old_prefix_len=old_prefix_len,
                    new_prefix_len=self.user_prefix_tokens,
                )
                self.compaction_events.append(compaction_event)
                logger.info(
                    "[User %02d] Compaction at turn %d: prefix %d → %d",
                    self.user_id, turn_id, old_prefix_len, self.user_prefix_tokens,
                )
                messages = self._build_messages(current_input_content)
                context_tokens = tokenizer_manager.count_chat_tokens(messages)

                if context_tokens >= self.max_context_tokens:
                    context_overflow = True
                    logger.warning(
                        "[User %02d] Context still exceeds limit after compaction: "
                        "%d >= %d tokens",
                        self.user_id, context_tokens, self.max_context_tokens,
                    )

        return {
            "messages": messages,
            "context_tokens": context_tokens,
            "compaction_triggered": compaction_event is not None,
            "compaction_event": compaction_event,
            "context_overflow": context_overflow,
        }

    def _build_messages(self, current_input: str, skip_history: bool = False) -> list[dict]:
        messages = []
        if self.system_prefix:
            messages.append({"role": "system", "content": self.system_prefix})

        user_parts = []
        if self.user_prefix_content:
            user_parts.append(self.user_prefix_content)

        for i, (user_msg, assistant_msg) in enumerate(self.history):
            if skip_history:
                continue
            if i == 0 and user_parts:
                user_parts.append(user_msg)
                messages.append({"role": "user", "content": "\n".join(user_parts)})
                user_parts = []
            else:
                messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        if user_parts:
            user_parts.append(current_input)
            messages.append({"role": "user", "content": "\n".join(user_parts)})
        else:
            messages.append({"role": "user", "content": current_input})

        return messages

    def append_history(self, user_message: str, assistant_reply: str):
        self.history.append((user_message, assistant_reply))