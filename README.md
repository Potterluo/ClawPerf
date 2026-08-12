# ClawPerfBench

[![PyPI Version](https://img.shields.io/pypi/v/clawperf.svg)](https://pypi.org/project/clawperf/)
[![Python Versions](https://img.shields.io/pypi/pyversions/clawperf.svg)](https://pypi.org/project/clawperf/)
[![License](https://img.shields.io/pypi/l/clawperf.svg)](https://github.com/ucm-system/ClawPerf/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ucm-system/ClawPerf.svg)](https://github.com/ucm-system/ClawPerf)

Performance benchmarking tool for LLM Serving backends with multi-turn long-context workloads.

[中文文档](README_CN.md)

Built on [EvalScope](https://github.com/modelscope/evalscope)'s perf infrastructure, adding:

- **Multi-turn context model**: System Prefix + User Prefix + History + Current Input
- **Append-mode compaction**: Clear history, grow user prefix when context reaches limits
- **User arrival scheduling**: Burst, steady, or Poisson arrival patterns
- **System metrics polling**: Prometheus endpoint support for vLLM, SGLang, MindIE
- **Per-user + per-turn metrics**: TTFT, TPOT, ITL with compaction tracking
- **Prefix cache simulation**: Trie-based HBM + external prefix cache hit rate tracking in mock server

![ClawPerf Benchmark Output](docs/benchmark_result.jpg)

## Installation

```bash
pip install clawperf
```

For the mock server used in testing:

```bash
pip install clawperf[mock-server]
```

For development:

```bash
pip install clawperf[dev]
```

Install from source (recommended for development):

```bash
git clone https://github.com/ucm-system/ClawPerf.git
cd ClawPerf
uv sync --extra dev --extra mock-server
```

## Quick Start

### Run a benchmark

```bash
clawperf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen3-32b \
  --num-users 5 \
  --user-arrival steady:2 \
  --max-turns 10 \
  --output results.json
```

### Start mock server (for testing)

```bash
clawperf-mock-server --port 8080
```

### End-to-end test with mock server

```bash
# Start mock server
clawperf-mock-server --port 8080

# Run benchmark against it
clawperf \
  --endpoint http://localhost:8080/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --num-users 4 \
  --max-turns 5 \
  --max-context-tokens 200000 \
  --metrics-endpoint http://localhost:8080/metrics \
  --backend vllm \
  --verbose
```

### Hit-rate test mode (controlled prefix-cache hit rate)

Instead of a multi-turn scenario, run a controlled prefix-cache hit-rate test:
specify input/output length and a target hit rate, and ClawPerf constructs
prompts with a known shared-prefix / unique-suffix split, prefills the prefixes,
then measures the **actual** hit rate from the server's Prometheus counters.

```bash
clawperf --mode hitrate \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen3-32b --tokenizer qwen3-32b \
  --num-requests 100 --input-len 1024 --output-len 128 \
  --hit-rate 0.5 \        # target 50% (or --prefix-len 512)
  --prefix-num 10 \       # 10 distinct prefixes -> 10 requests reuse each
  --concurrency 20 \
  --metrics-endpoint http://localhost:8000/metrics --backend vllm \
  --reset-cache
```

How it works (borrowed from aisbench / vLLM `prefix_repetition`):
- Each request = `[shared prefix] + [3 boundary tokens] + [unique suffix]`. The
  boundary tokens force the cache to stop at exactly `prefix_len`, so the hit is
  precisely the shared portion.
- `--prefix-num` distinct prefixes are assigned round-robin and **shuffled** so
  reuse happens under concurrency (not back-to-back duplicates).
- `--prefill` (default on) injects each distinct prefix with `output_len=1`
  before measuring, so even the first request per prefix hits.
- The summary prints **TARGET vs MEASURED** hit rate (measured from
  `vllm:prefix_cache_hits_total`/`queries_total` deltas), per-engine breakdown,
  and TTFT/TPOT percentiles.

`--hit-rate` (fraction) and `--prefix-len` (absolute) are mutually exclusive;
one derives the other from `--input-len`.

### SLO capacity sweep mode (find max concurrent users)

Specify TTFT/TPOT SLO targets; ClawPerf sweeps concurrency (closed-loop, each
user sends back-to-back multi-turn requests) and finds the **max users** the
system can sustain while meeting the SLO. Reuses the scenario workload
(system/user prefix, input/output per turn, max_context, compaction).

```bash
clawperf --mode slo \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen3-32b --tokenizer qwen3-32b \
  --slo-ttft-ms 500 --slo-tpot-ms 30 \   # P99 must be ≤ these
  --slo-percentile 0.99 \                 # P99 (or 0.95/0.90)
  --slo-min-users 1 --slo-max-users 200 \
  --slo-step-strategy geometric \         # double each step (or linear)
  --slo-step-turns 5 --slo-step-warmup-turns 1 \
  --system-prefix-tokens 15000 --input-tokens-per-turn 5000 \
  --output-tokens-per-turn 1000 --max-context-tokens 128000 \
  --backend vllm --reset-cache
```

How it works:
- **Geometric ramp** (1→2→4→8→…) finds the knee region fast; at each N it runs
  `warmup + measure` turns per user and checks P{slo_percentile} TTFT/TPOT.
- **Binary refine** between the last-good and first-bad N pinpoints the exact max.
- Optional `--slo-error-rate` caps the error rate; `--slo-step-timeout-s` aborts
  a step if the server is overloaded.
- `--slo-step-reset-cache` (default on) isolates each step; turn it off to test
  sustained pressure.
- Output: a **capacity curve** (N vs P99 TTFT/TPOT/error/SLO-met) and the
  `Max sustained users` verdict.

### Agent-at-work mode (real coding-agent perf)

Run the model as a **real coding agent** — it reads/writes files and runs shell
commands via OpenAI function-calling, multi-turn, with growing context. N tasks
run concurrently; we measure per-turn TTFT/TPOT/tokens, per-task wall time, and
the **real prefix-cache hit rate** from `/metrics`. Purely performance, no
accuracy grading. Requires the backend to support tool-calling (e.g. Qwen/GLM
on vLLM).

```bash
pip install clawperf[agent]   # installs the openai SDK used by agent mode

clawperf --mode agent \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen3-32b --tokenizer qwen3-32b \
  --agent-tasks 10 \               # 10 concurrent agent task instances
  --agent-max-steps 12 \           # per-task LLM turn cap
  --agent-max-tokens 512 \         # max generation per turn
  --metrics-endpoint http://localhost:8000/metrics --backend vllm \
  --reset-cache
```

Tasks: the built-in preset bank (small coding tasks: fix a bug, add a function,
edit a config) is used by default; or supply your own with `--agent-task-file
tasks.jsonl` (one `{"prompt", "workspace": {"path":"content"}, "max_steps"}`
per line). Each task instance gets its own workspace dir (concurrent agents
never collide). Output: per-task table (steps/finished/wall/tokens) + per-turn
TTFT percentiles + measured prefix-cache hit rate.

## CLI Options

### User Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--num-users` | 1 | Total concurrent users |
| `--user-arrival` | burst | Arrival pattern: `burst`, `steady:<seconds>`, or `poisson:<lambda>` |

### Context Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--system-prefix-tokens` | 15000 | System prefix token count |
| `--system-prefix-source` | random | Source: `random` or a file path |
| `--user-prefix-tokens` | 5000 | Per-user prefix token count |
| `--input-tokens-per-turn` | 5000 | Input tokens per turn |
| `--output-tokens-per-turn` | 1000 | Output tokens per turn |
| `--max-context-tokens` | 128000 | Context window limit |
| `--compaction-prefix-increment` | 5000 | User prefix growth on compaction |

### Run Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--max-turns` | 100 | Maximum turns per user |

### API Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | (required) | LLM API endpoint URL |
| `--model` | (required) | Model name |
| `--api-key` | (empty) | API key |
| `--tokenizer` | (defaults to model) | Tokenizer path |
| `--ignore-eos` | True | Ignore EOS token |
| `--request-timeout` | 600 | Request timeout in seconds |

### System Metrics

| Option | Default | Description |
|--------|---------|-------------|
| `--metrics-endpoint` | None | Prometheus metrics URL. Only start+end snapshots are taken. |
| `--metrics-interval` | 5 | Polling interval (s) for periodic time-series; only with `--metrics-samples` |
| `--metrics-samples` | False | Collect periodic metrics throughout the run (extra `/metrics` calls) |
| `--reset-cache` | False | Evict the server's prefix cache before the start snapshot (`/reset_prefix_cache` for vLLM, `/flush_cache` for SGLang) so the measured hit rate reflects only this benchmark |
| `--backend` | vllm | Backend: `vllm`, `sglang`, or `mindie` |

> A pre-flight health check (one tiny request) runs before content generation and aborts early if the endpoint is unreachable, so you don't burn minutes producing an all-error run.

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | results.json | Output JSON file path |
| `--history` | clawperf_history.jsonl | Append a one-line record (config + summary + per-user aggregates) to this JSONL file on every run, accumulating results across runs. Pass an empty string to disable. |

## Output Format

Results are saved as JSON with:

```json
{
  "config": { ... },
  "summary": {
    "prefix_cache_token_hit_rate": 0.7981,
    "prefix_cache_hit_tokens_delta": 712012,
    "prefix_cache_query_tokens_delta": 892165,
    "total_compactions": 0,
    ...
  },
  "users": [
    {
      "user_id": 0,
      "aggregate": {
        "total_output_tokens": 3000,
        "ttft": { "avg": 150.2, "P50": 140, "P99": 200 },
        "tpot": { "avg": 3.2, "P50": 3.0, "P99": 5.0 },
        "throughput_tok_s": 12.5,
        "error_count": 0,
        "compaction_count": 2
      },
      "turns": [
        {
          "turn_id": 1,
          "success": true,
          "ttft_ms": 150.2,
          "e2e_latency_ms": 3200.5,
          "tpot_ms": 3.2,
          "input_tokens": 25000,
          "output_tokens": 1000,
          "context_tokens": 25000,
          "compaction_triggered": false,
          "wall_start_ts": 0.016,
          "wall_end_ts": 3.354
        }
      ]
    }
  ],
  "system_metrics": [ ... ],
  "timeline": [ ... ],
  "timing": {
    "setup_time_s": 7.437,
    "bench_time_s": 12.281
  }
}
```

`timing.bench_time_s` excludes one-time setup (tokenizer download + content
generation); per-turn `wall_start_ts`/`wall_end_ts` are offsets from the
benchmark start and back the per-user duration/throughput aggregates.

## Result History

Every run appends one compact JSON line to `clawperf_history.jsonl` (configurable
via `--history`, disable with `--history ""`). Each line carries the run
`timestamp`, the full `config`, the `summary`, `timing`, and per-user
`aggregate`s — but not the heavy per-turn arrays, so the file stays queryable
as runs accumulate.

Collect and compare results across runs with standard tooling:

```bash
# Latest run's hit rate
tail -n1 clawperf_history.jsonl | jq '.summary.prefix_cache_token_hit_rate'

# Throughput trend over all runs
jq -c '{users: .config.num_users, bench_s: .timing.bench_time_s,
        hit_rate: .summary.prefix_cache_token_hit_rate}' clawperf_history.jsonl
```

The full per-turn detail for any run is still in its `--output` JSON file,
referenced from each history record's `output_file` field.

## Testing Philosophy

ClawPerfBench is designed to simulate the **real workload of an Agent system** — not single-shot API calls, but sustained multi-turn conversations that push LLM serving backends to their limits.

### Why multi-turn matters

Real Agent systems (like OpenClaw) don't send one-off requests. They maintain long conversations: a system prompt, user-specific context, and growing history. Each turn re-sends the entire accumulated context, creating exponentially growing prompts. This is fundamentally different from single-request benchmarks and exposes backend behaviors that single-shot tests miss:

- **Prefix cache effectiveness**: Does the KV-block cache actually reuse tokens across turns? A single-request benchmark can't measure this.
- **Compaction under load**: When context hits the window limit, how does the system handle truncation? Does it recover gracefully or spiral into overflow?
- **Latency degradation**: As context grows from 25K to 200K tokens, TTFT and TPOT change dramatically. Per-turn metrics reveal this progression.
- **Concurrent pressure**: Multiple users with independent conversations create mixed prefix cache states — some sharing the system prefix, others diverging at user-specific paths.

### Simulating real users

Each simulated user maintains an independent conversation state with its own growing prefix and history. Users arrive according to configurable patterns (burst, steady, Poisson) — mimicking how real traffic builds up, not an artificial flood of identical requests.

### What we measure

| What | Why it matters |
|------|---------------|
| TTFT per turn | First-token latency grows with context size — the key UX metric for Agent systems |
| TPOT per turn | Generation speed should stay stable; degradation indicates compute bottlenecks |
| Prefix cache hit rate | Token-level reuse fraction across turns — the efficiency metric for KV caching |
| Compaction events | When and how often context overflows — determines conversation continuity |
| Per-user breakdown | Different users have different prefix paths; aggregate stats hide per-user variance |

## Context Model

Each user's context follows this structure:

![Context model and compaction](docs/context_model.svg)

When context reaches `--max-context-tokens`, append-mode compaction fires:

1. The base context (system + user prefix + input, without history) is checked first. If it already exceeds the limit, compaction is skipped and the turn is marked as `context_overflow` — this prevents infinite compaction loops.
2. Otherwise, history is cleared and the user prefix grows by `--compaction-prefix-increment` tokens.
3. New random content fills the enlarged user prefix.
4. If the grown base still exceeds the limit, the prefix growth is **reverted** (history cleared only) so the user isn't permanently trapped in overflow.

This simulates how real LLM serving systems handle context overflow with prefix caching.

## Prefix Cache Simulation

The mock server simulates vLLM's KV-block prefix cache using a trie:

- **HBM trie**: Represents GPU KV cache. Queried first for longest prefix match. Always updated after every request (mimicking vLLM storing all KV blocks regardless of hit/miss).
- **External trie**: Represents CPU/disk prefix cache. Queried on HBM miss. Also always updated after every request.
- **Token-level hit rate**: `prefix_cache_hit_tokens / prefix_cache_query_tokens` — the fraction of prompt tokens that reuse cached KV blocks. This is the meaningful metric; request-level (binary) hit rate is not reported.
- **Eviction**: When the trie exceeds `max_prefixes` (200), oldest leaf nodes are evicted.

## User Arrival Scheduling

![User arrival patterns](docs/arrival_patterns.svg)

- **burst**: All users start immediately
- **steady:2**: Users arrive every 2 seconds
- **poisson:0.5**: Users arrive following a Poisson process with rate 0.5

## Architecture

ClawPerf reuses EvalScope's core perf components:

- **AioHttpClient**: Async HTTP with streaming, proper timeout/connector config
- **OpenaiPlugin**: Request building, response parsing, local token counting
- **BenchmarkData**: Single-request data container (TTFT, ITL, E2E timing)
- **MetricsAccumulator**: Real-time metrics aggregation

And adds its own orchestration layer for multi-turn, multi-user workloads.

Key modules:

| Module | Role |
|--------|------|
| `cli.py` | Argparse entry point, config creation, runner launch |
| `config.py` | `BenchmarkConfig` dataclass, arrival mode parsing |
| `runner.py` | `BenchmarkRunner` orchestrator, user loop, result finalization, JSONL history |
| `context.py` | `UserContext` context assembly, compaction with infinite-loop guard |
| `scheduler.py` | Burst/steady/Poisson async generators |
| `system_metrics.py` | `SystemMetricsPoller` with backend-specific metric mappings |
| `tokenizer.py` | `TokenizerManager` wrapping ModelScope/HuggingFace tokenizers |
| `logging_setup.py` | Centralized logging routed through `tqdm.write` |
| `mock_server.py` | FastAPI mock LLM server with trie-based prefix cache simulation |

## Development

```bash
uv sync --extra dev --extra mock-server
pytest
ruff check
```

## License

Apache License 2.0