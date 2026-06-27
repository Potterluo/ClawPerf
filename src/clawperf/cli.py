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

    # ── Mode ──
    g = parser.add_argument_group("Mode")
    g.add_argument("--mode", type=str, default="scenario",
                   choices=["scenario", "hitrate"],
                   help="'scenario' (default): multi-turn long-context workload. "
                        "'hitrate': controlled prefix-cache hit-rate test "
                        "(prefill + measure, reports actual vs target hit rate).")

    # ── Hit-rate mode configuration ──
    g = parser.add_argument_group("Hit-Rate Mode (only with --mode hitrate)")
    g.add_argument("--num-requests", type=int, default=100,
                   help="Total measure-phase requests.")
    g.add_argument("--input-len", type=int, default=1024,
                   help="Total prompt length (prefix + boundary + suffix), in tokens.")
    g.add_argument("--output-len", type=int, default=128,
                   help="Generation length per request.")
    g.add_argument("--prefix-len", type=int, default=0,
                   help="Shared-prefix length in tokens. 0 = derive from --hit-rate.")
    g.add_argument("--hit-rate", type=float, default=None,
                   help="Target hit rate as a fraction in (0,1); derives prefix_len "
                        "= hit_rate * input_len. Mutually exclusive with --prefix-len.")
    g.add_argument("--prefix-num", type=int, default=1,
                   help="Number of DISTINCT prefixes. requests-per-prefix = "
                        "num_requests // prefix_num. 1 = all share one prefix.")
    g.add_argument("--prefill", action="store_true", default=True,
                   help="Inject prefixes into the cache before measuring (default on).")
    g.add_argument("--no-prefill", action="store_false", dest="prefill",
                   help="Skip the prefill phase (measure cold + natural reuse only).")
    g.add_argument("--concurrency", type=int, default=1,
                   help="In-flight requests during the measure phase.")
    g.add_argument("--seed", type=int, default=0,
                   help="Reproducibility seed for prompt construction.")

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
    g.add_argument("--metrics-endpoint", type=str, default=None,
                   help="Prometheus /metrics URL. Only start+end snapshots are taken by default.")
    g.add_argument("--metrics-interval", type=int, default=5,
                   help="Polling interval (seconds) for periodic time-series; only used with --metrics-samples.")
    g.add_argument("--metrics-samples", action="store_true", default=False,
                   help="Collect periodic metrics samples throughout the run (extra /metrics calls). "
                        "Off by default — only start and end snapshots are taken.")
    g.add_argument("--reset-cache", action="store_true", default=False,
                   help="Evict the server's prefix cache before the start snapshot (POST "
                        "/reset_prefix_cache for vLLM, /flush_cache for SGLang) so the measured "
                        "hit rate reflects only this benchmark's prefixes.")
    g.add_argument("--backend", type=str, default="vllm", choices=["vllm", "sglang", "mindie"])

    # ── Output ──
    g = parser.add_argument_group("Output")
    g.add_argument("--output", type=str, default="",
                   help="Output JSON file path (default: timestamped results_<timestamp>.json)")
    g.add_argument("--history", type=str, default="clawperf_history.jsonl",
                   help="Append a one-line record (config + summary + per-user aggregates) "
                        "to this JSONL file on every run, accumulating results across runs. "
                        "Pass an empty string to disable.")
    g.add_argument("-v", "--verbose", action="store_true", default=False,
                   help="Print per-turn progress lines (default: tqdm progress bar)")

    return parser


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Validate prefix-len / hit-rate mutual exclusivity (argparse can't easily).
    if getattr(args, "prefix_len", 0) and getattr(args, "hit_rate", None) is not None:
        parser.error("--prefix-len and --hit-rate are mutually exclusive; specify one.")
    return BenchmarkConfig(
        mode=args.mode,
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        prefix_len=args.prefix_len,
        hit_rate=args.hit_rate,
        prefix_num=args.prefix_num,
        prefill=args.prefill,
        concurrency=args.concurrency,
        seed=args.seed,
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
        metrics_samples=args.metrics_samples,
        reset_cache=args.reset_cache,
        backend=args.backend,
        output=args.output,
        history=args.history,
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
    except RuntimeError as e:
        # Pre-flight / hard config errors: print a clean message, not a traceback.
        print(f"\n[ClawPerf] {e}", file=sys.stderr)
        sys.exit(1)