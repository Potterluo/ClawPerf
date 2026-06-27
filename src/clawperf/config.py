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

    # ── Mode ──
    # "scenario" = multi-turn long-context workload (default).
    # "hitrate"  = controlled prefix-cache hit-rate test (prefill + measure).
    mode: str = "scenario"

    # ── Hit-rate mode configuration (ignored in scenario mode) ──
    num_requests: int = 100        # total measure-phase requests
    input_len: int = 1024          # total prompt length (prefix + boundary + suffix)
    output_len: int = 128          # generation length per request
    prefix_len: int = 0            # shared-prefix length (0 = derive from hit_rate)
    hit_rate: Optional[float] = None  # target fraction that's shared (0..1); derives prefix_len
    prefix_num: int = 1            # number of DISTINCT prefixes (requests-per-prefix = N//prefix_num)
    prefill: bool = True           # inject prefixes into cache before measuring
    concurrency: int = 1           # in-flight requests during measure phase
    seed: int = 0                  # reproducibility seed for prompt construction

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
    metrics_samples: bool = False  # collect periodic time-series (else start+end only)
    reset_cache: bool = False  # evict prefix cache before start snapshot (clean baseline)
    backend: str = "vllm"  # "vllm", "sglang", "mindie"

    # ── Output configuration ──
    output: str = ""  # defaults to timestamped filename if empty
    history: str = "clawperf_history.jsonl"  # JSONL append log; "" disables
    verbose: bool = False  # per-turn detailed logging

    # ── Derived fields ──
    arrival_mode: str = ""
    arrival_param: float = 0.0

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model
        if not self.output:
            from datetime import datetime
            self.output = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._parse_arrival_mode()

    def _parse_arrival_mode(self):
        if self.user_arrival == "burst":
            self.arrival_mode = "burst"
            self.arrival_param = 0.0
            return

        # Forms other than 'burst' must have a colon: 'steady:<n>' / 'poisson:<n>'.
        if ":" not in self.user_arrival:
            raise ValueError(
                f"Invalid user_arrival format: {self.user_arrival!r}. "
                "Expected 'burst', 'steady:<seconds>', or 'poisson:<lambda>'."
            )

        prefix, _, raw_param = self.user_arrival.partition(":")
        try:
            param = float(raw_param)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid user_arrival format: {self.user_arrival!r}. "
                "Expected 'burst', 'steady:<seconds>', or 'poisson:<lambda>' "
                f"with a numeric value after the colon (got {raw_param!r})."
            )

        if prefix == "steady":
            if param < 0:
                raise ValueError(
                    f"steady arrival interval must be >= 0, got {param}."
                )
            self.arrival_mode = "steady"
            self.arrival_param = param
        elif prefix == "poisson":
            # random.expovariate(lambda) divides by lambda — lambda==0 crashes.
            if param <= 0:
                raise ValueError(
                    f"poisson arrival lambda must be > 0, got {param}."
                )
            self.arrival_mode = "poisson"
            self.arrival_param = param
        else:
            raise ValueError(
                f"Invalid user_arrival format: {self.user_arrival!r}. "
                "Expected 'burst', 'steady:<seconds>', or 'poisson:<lambda>'."
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
            apply_chat_template=False,  # Disable to support tokenizers without chat_template
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
        """Serialize the *public* config (excludes derived internal fields)."""
        d = dataclasses.asdict(self)
        # Internal derived state — not part of the user-facing config.
        d.pop("arrival_mode", None)
        d.pop("arrival_param", None)
        return d

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty if OK).

        Called by the runner at startup so misconfigurations surface immediately
        instead of silently failing every turn.
        """
        problems: list[str] = []
        # If the configured base token budget already meets/exceeds the window,
        # every turn will overflow (base = system + user_prefix + input, before
        # chat-template overhead or history).
        base = (
            self.system_prefix_tokens
            + self.user_prefix_tokens
            + self.input_tokens_per_turn
        )
        if base >= self.max_context_tokens:
            problems.append(
                f"Base context ({self.system_prefix_tokens} system + "
                f"{self.user_prefix_tokens} user-prefix + "
                f"{self.input_tokens_per_turn} input = {base} tokens) already "
                f">= max_context_tokens ({self.max_context_tokens}). Every turn "
                "will overflow — reduce prefix/input sizes or raise "
                "--max-context-tokens."
            )
        if self.compaction_prefix_increment <= 0:
            problems.append(
                "compaction_prefix_increment must be > 0, otherwise compaction "
                "can never provide relief."
            )
        if self.num_users < 1:
            problems.append("num_users must be >= 1.")
        if self.max_turns < 1:
            problems.append("max_turns must be >= 1.")

        # Hit-rate mode validation (only applies in that mode).
        if self.mode == "hitrate":
            from clawperf.hitrate import BOUNDARY_TOKENS

            if self.num_requests < 1:
                problems.append("hitrate: num_requests must be >= 1.")
            if self.input_len < 1:
                problems.append("hitrate: input_len must be >= 1.")
            if self.output_len < 1:
                problems.append("hitrate: output_len must be >= 1.")
            if self.prefix_num < 1:
                problems.append("hitrate: prefix_num must be >= 1.")
            if self.prefix_num > self.num_requests:
                problems.append(
                    f"hitrate: prefix_num ({self.prefix_num}) must be <= "
                    f"num_requests ({self.num_requests})."
                )
            if self.hit_rate is not None and not (0.0 < self.hit_rate < 1.0):
                problems.append("hitrate: hit_rate must be in (0, 1).")
            # Resolve prefix_len for the boundary check.
            plen = self.prefix_len
            if plen == 0 and self.hit_rate is not None:
                plen = int(self.input_len * self.hit_rate)
            if plen > 0 and self.input_len - plen - BOUNDARY_TOKENS < 1:
                problems.append(
                    f"hitrate: input_len ({self.input_len}) too small for "
                    f"prefix_len ({plen}) + boundary ({BOUNDARY_TOKENS}); "
                    "need room for a unique suffix."
                )
        elif self.mode not in ("scenario",):
            problems.append(f"unknown mode {self.mode!r} (expected 'scenario' or 'hitrate').")
        return problems
