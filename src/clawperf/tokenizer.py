"""Tokenizer management — reuses EvalScope's utilities directly.

Key reused components:
  - modelscope.AutoTokenizer / transformers.AutoTokenizer for loading
  - evalscope.perf.plugin.datasets.utils.tokenize_chat_messages()
  - evalscope.perf.plugin.datasets.utils.gen_prompt_decode_to_target_len()
  - OpenaiPlugin._count_input_tokens() / _count_output_tokens()
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

logger = logging.getLogger("clawperf")


class TokenizerManager:
    """Manages tokenizer loading, token counting, and content generation.

    Delegates to EvalScope's utilities wherever possible.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_tokenizer()
        return self._tokenizer

    def _load_tokenizer(self):
        """Load tokenizer — same approach as EvalScope's OpenaiPlugin."""
        try:
            from modelscope import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=True
            )
            logger.info("Loaded tokenizer from ModelScope: %s", self.tokenizer_path)
        except Exception:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path, trust_remote_code=True
                )
                logger.info("Loaded tokenizer from HuggingFace: %s", self.tokenizer_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer from '{self.tokenizer_path}'. Error: {e}"
                )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ── Token counting (reuses EvalScope) ──

    def count_tokens(self, text: str) -> int:
        """Count tokens in a plain text string."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_chat_tokens(self, messages: list[dict]) -> int:
        """Count tokens for chat-formatted messages using EvalScope's utility."""
        from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages
        return len(tokenize_chat_messages(self.tokenizer, messages, add_generation_prompt=True))

    def count_input_tokens_from_request(self, request_json_str: str) -> int:
        """Count input tokens from a serialized request JSON — same as OpenaiPlugin._count_input_tokens."""
        import json
        request = json.loads(request_json_str)
        if 'messages' in request:
            from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages
            return len(tokenize_chat_messages(self.tokenizer, request['messages']))
        elif 'prompt' in request:
            prompt = request['prompt']
            if isinstance(prompt, list):
                return len(prompt)
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return 0

    def count_output_tokens(self, text: str) -> int:
        """Count output tokens — same as OpenaiPlugin._count_output_tokens."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ── Content generation (reuses EvalScope) ──

    def generate_random_content(self, target_tokens: int) -> str:
        """Generate random content of exactly target_tokens length.

        Uses EvalScope's gen_prompt_decode_to_target_len when available.
        """
        if target_tokens <= 0:
            return ""

        vocab = self.tokenizer.get_vocab()
        special_ids = set()
        if hasattr(self.tokenizer, "all_special_ids"):
            special_ids = set(self.tokenizer.all_special_ids)
        valid_ids = [tid for tid in vocab.values() if tid not in special_ids]
        if not valid_ids:
            valid_ids = list(vocab.values())

        token_ids = [random.choice(valid_ids) for _ in range(target_tokens)]

        # Use EvalScope's utility for exact length matching
        try:
            from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len
            text, _, _ = gen_prompt_decode_to_target_len(
                self.tokenizer, token_ids, target_tokens
            )
            return text
        except ImportError:
            pass

        # Fallback: decode then adjust
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return self._adjust_to_target_length(text, target_tokens)

    def generate_content_from_file(self, file_path: str, target_tokens: int) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Content file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._adjust_to_target_length(content, target_tokens)

    def _adjust_to_target_length(self, text: str, target_tokens: int) -> str:
        current_count = self.count_tokens(text)
        for _ in range(20):
            if current_count == target_tokens:
                break
            if current_count > target_tokens:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                text = self.tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
            else:
                vocab = self.tokenizer.get_vocab()
                special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
                valid_ids = [tid for tid in vocab.values() if tid not in special_ids]
                if not valid_ids:
                    break
                extra = [random.choice(valid_ids) for _ in range(target_tokens - current_count)]
                existing = self.tokenizer.encode(text, add_special_tokens=False)
                text = self.tokenizer.decode(existing + extra, skip_special_tokens=True)
            current_count = self.count_tokens(text)
        return text
