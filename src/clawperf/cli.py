"""CLI argument parser for ClawPerf."""

from __future__ import annotations

import argparse
import asyncio
import sys

from clawperf.config import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawperf",
        description=(
            "ClawPerf - Performance testing tool for LLM Serving backends. "
            "Simulates multi-user, multi-turn, long-context workloads against "
            "vLLM, SGLang, and MindIE backends. "
            "Built on EvalScope's perf infrastructure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── User configuration ──
    g = parser.add_argument_group("User Configuration")
    g.add_argument("--num-users", type=int, default=1, help="Total concurrent users.")
    g.add_argument(
        "--user-arrival", type=str, default="burst",
        help="'burst', 'steady:<seconds>', or 'poisson:<lambda>'.",
    )

    # ── Context configuration ──
    g = parser.add_argument_group("Context Configuration")
    g.add_argument("--system-prefix-tokens", type=int, default=15000)
    g.add_argument("--system-prefix-source", type=str, default="random")
    g.add_argument("--user-prefix-tokens", type=int, default=5000)
    g.add_argument("--input-tokens-per-turn", type=int, default=5000)
    g.add_argument("--output-tokens-per-turn", type=int, default=1000)
    g.add_argument("--max-context-tokens", type=int, default=128000)
    g.add_argument("--compaction-prefix-increment", type=int, default=5000)

    # ── Run configuration ──
    g = parser.add_argument_group("Run Configuration")
    g.add_argument("--max-turns", type=int, default=100)

    # ── API configuration ──
    g = parser.add_argument_group("API Configuration")
    g.add_argument("--endpoint", type=str, required=True)
    g.add_argument("--model", type=str, required=True)
    g.add_argument("--api-key", type=str, default="")
    g.add_argument("--tokenizer", type=str, default="")
    g.add_argument("--ignore-eos", action="store_true", default=True)
    g.add_argument("--no-ignore-eos", action="store_false", dest="ignore_eos")
    g.add_argument("--request-timeout", type=int, default=600)

    # ── System metrics ──
    g = parser.add_argument_group("System Metrics")
    g.add_argument("--metrics-endpoint", type=str, default=None)
    g.add_argument("--metrics-interval", type=int, default=5)
    g.add_argument("--backend", type=str, default="vllm", choices=["vllm", "sglang", "mindie"])

    # ── Output ──
    g = parser.add_argument_group("Output")
    g.add_argument("--output", type=str, default="",
                   help="Output JSON file path (default: timestamped results_<timestamp>.json)")
    g.add_argument("-v", "--verbose", action="store_true", default=False,
                   help="Print per-turn progress lines (default: tqdm progress bar)")

    return parser


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        num_users=args.num_users,
        user_arrival=args.user_arrival,
        system_prefix_tokens=args.system_prefix_tokens,
        system_prefix_source=args.system_prefix_source,
        user_prefix_tokens=args.user_prefix_tokens,
        input_tokens_per_turn=args.input_tokens_per_turn,
        output_tokens_per_turn=args.output_tokens_per_turn,
        max_context_tokens=args.max_context_tokens,
        compaction_prefix_increment=args.compaction_prefix_increment,
        max_turns=args.max_turns,
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        tokenizer=args.tokenizer,
        ignore_eos=args.ignore_eos,
        request_timeout=args.request_timeout,
        metrics_endpoint=args.metrics_endpoint,
        metrics_interval=args.metrics_interval,
        backend=args.backend,
        output=args.output,
        verbose=args.verbose,
    )


def main():
    config = parse_args()

    # Configure logging BEFORE importing the runner so that the setup phase
    # (tokenizer load + content generation) produces visible, non-frozen output.
    from clawperf.logging_setup import setup_logging
    from clawperf.runner import BenchmarkRunner

    setup_logging(verbose=config.verbose)
    runner = BenchmarkRunner(config)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        # Re-establish logging (the event loop + handlers were torn down).
        setup_logging(verbose=config.verbose)
        print("\n[ClawPerf] Interrupted. Saving partial results...")
        try:
            asyncio.run(runner.shutdown_and_save())
        except Exception as e:
            print(f"[ClawPerf] Failed to save partial results: {e}", file=sys.stderr)