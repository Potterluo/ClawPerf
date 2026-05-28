# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClawPerfBench is a performance benchmarking tool for LLM serving backends (vLLM, SGLang, MindIE) with multi-turn long-context workloads. It extends EvalScope's perf infrastructure with context compaction, user arrival scheduling, system metrics polling, and per-user/per-turn metrics.

## Commands

```bash
# Install with all dependencies
pip install -e ".[dev,mock-server]"

# Run tests
pytest
pytest tests/test_config.py -v          # single test file
pytest -k "test_parse_arrival" -v       # single test by name

# Lint
ruff check

# Run benchmark (requires a live LLM endpoint)
clawperf --endpoint <url> --model <name> --num-users 4 --num-turns 3

# Run mock server for local testing
clawperf-mock-server --port 8080
```

## Architecture

**Data flow**: CLI args → `BenchmarkConfig` → `BenchmarkRunner` → (TokenizerManager + UserContext + Scheduler + EvalScope HTTP client) → JSON results

Key modules and their roles:

- **`cli.py`** — argparse entry point. Parses all benchmark options, creates `BenchmarkConfig`, launches `BenchmarkRunner` via `asyncio.run()`.
- **`config.py`** — `BenchmarkConfig` dataclass. `__post_init__` parses arrival mode strings ("burst", "steady:2", "poisson:0.5"). `to_evalscope_args()` bridges to EvalScope's `Arguments` for HTTP/streaming/model settings.
- **`runner.py`** — `BenchmarkRunner` orchestrator. Initializes EvalScope components, generates content via `TokenizerManager`, creates `UserContext` per user, schedules arrivals, launches async user loops, and finalizes results.
- **`context.py`** — `UserContext` manages per-user context state (system prefix, user prefix, history). `prepare_turn()` builds the messages list, checks overflow, triggers append-mode compaction (clear history → grow user prefix). `CompactionEvent` tracks each compaction.
- **`scheduler.py`** — Three async generators: burst (all at t=0), steady (fixed interval), poisson (exponential). `get_scheduler()` selects based on config.
- **`system_metrics.py`** — `SystemMetricsPoller` polls Prometheus endpoints with backend-specific metric mappings (vllm/sglang/mindie).
- **`tokenizer.py`** — `TokenizerManager` wraps ModelScope/HuggingFace tokenizers. Provides token counting, random content generation, and content adjustment to exact target token counts.
- **`mock_server.py`** — FastAPI mock LLM server implementing OpenAI-compatible `/v1/chat/completions` with configurable TTFT/TPOT delays.

**EvalScope dependency**: ClawPerf reuses `AioHttpClient`, `OpenaiPlugin`, `BenchmarkData`, `MetricsAccumulator`, and tokenizer utilities from `evalscope.perf`. The multi-turn orchestration, compaction, and scheduling are ClawPerf-specific additions layered on top.

## Key Design Details

- Compaction is append-mode: when context exceeds `max_context_tokens`, history is cleared and the user prefix absorbs a summary, growing over subsequent turns.
- Arrival mode is parsed from strings: `"burst"`, `"steady:<interval>"`, `"poisson:<lambda>"` — the parsing happens in `BenchmarkConfig.__post_init__`.
- System metrics polling is backend-aware: metric names differ between vLLM, SGLang, and MindIE, mapped in `system_metrics.py`.
- All benchmark logic is async (asyncio); the runner, schedulers, and HTTP calls use async generators and tasks.
- Results are written as JSON to a configurable output path; a PrettyTable summary is printed to stdout.