"""Configuration for ClawPerfBench.

Wraps EvalScope's Arguments where possible, extends with
ClawPerfBench-specific context/compaction/scheduling parameters.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class BenchmarkConfig:
    """All configurable parameters for a benchmark run."""

    # ── User configuration ──
    num_users: int = 1
    user_arrival: str = "burst"  # "burst", "steady:<interval>", "poisson:<lambda>"

    # ── Context configuration ──
    system_prefix_tokens: int = 15000
    system_prefix_source: str = "random"  # "random" or file path
    user_prefix_tokens: int = 5000
    input_tokens_per_turn: int = 5000
    output_tokens_per_turn: int = 1000
    max_context_tokens: int = 128000
    compaction_prefix_increment: int = 5000

    # ── Run configuration ──
    max_turns: int = 100

    # ── API configuration ──
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    tokenizer: str = ""  # defaults to model if empty
    ignore_eos: bool = True
    request_timeout: int = 600

    # ── System metrics configuration ──
    metrics_endpoint: Optional[str] = None
    metrics_interval: int = 5
    backend: str = "vllm"  # "vllm", "sglang", "mindie"

    # ── Output configuration ──
    output: str = "results.json"
    verbose: bool = False  # per-turn detailed logging

    # ── Derived fields ──
    arrival_mode: str = ""
    arrival_param: float = 0.0

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model
        self._parse_arrival_mode()

    def _parse_arrival_mode(self):
        if self.user_arrival == "burst":
            self.arrival_mode = "burst"
            self.arrival_param = 0.0
        elif self.user_arrival.startswith("steady:"):
            self.arrival_mode = "steady"
            self.arrival_param = float(self.user_arrival.split(":")[1])
        elif self.user_arrival.startswith("poisson:"):
            self.arrival_mode = "poisson"
            self.arrival_param = float(self.user_arrival.split(":")[1])
        else:
            raise ValueError(
                f"Invalid user_arrival format: {self.user_arrival!r}. "
                "Expected 'burst', 'steady:<interval>', or 'poisson:<lambda>'."
            )

    def to_evalscope_args(self):
        """Build an EvalScope Arguments object from this config.

        Reuses EvalScope's connection/timeout/stream/model settings.
        Note: EvalScope's number/parallel are lists after validation.
        """
        from evalscope.perf.arguments import Arguments

        args = Arguments(
            model=self.model,
            url=self.endpoint,
            tokenizer_path=self.tokenizer,
            stream=True,
            max_tokens=self.output_tokens_per_turn,
            number=[self.num_users],
            parallel=[self.num_users],
            total_timeout=self.request_timeout,
            api="openai",
            no_test_connection=True,
        )
        if self.api_key:
            args.headers["Authorization"] = f"Bearer {self.api_key}"

        # Handle ignore_eos via extra_args
        if self.ignore_eos:
            extra = dict(args.extra_args) if args.extra_args else {}
            extra["ignore_eos"] = True
            args.extra_args = extra

        return args

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
