# ClawPerfBench Design Spec

## 1. Purpose

ClawPerfBench is a CLI performance testing tool that simulates OpenClaw Agent system workloads against LLM serving backends (vLLM, SGLang, MindIE). It measures performance under multi-user, multi-turn, long-context scenarios by generating realistic request patterns with precise token-level control.

## 2. Integration Strategy

**Deep reuse of evalscope.perf** — import evalscope as a runtime dependency and directly reuse its proven components:

| Reused Component | Source Module | Usage |
|---|---|---|
| `load_tokenizer()` | `evalscope.perf.plugin.datasets.utils` | Tokenizer loading from HuggingFace/modelscope paths |
| `tokenize_chat_messages()` | `evalscope.perf.plugin.datasets.utils` | Chat template rendering → token ID list |
| `gen_prompt_decode_to_target_len()` | `evalscope.perf.plugin.datasets.utils` | Iterative token-length adjustment for precise sizing |
| `BenchmarkData` | `evalscope.perf.utils.benchmark_util` | Per-request metric data container |
| `MetricsAccumulator` | `evalscope.perf.utils.benchmark_util` | Cumulative metric accumulation |
| `BenchmarkMetrics` | `evalscope.perf.utils.benchmark_util` | Computed aggregate metrics snapshot |
| `StreamedResponseHandler` | `evalscope.perf.plugin.api.default_api` | SSE stream parsing with per-chunk timestamps |
| `OpenaiPlugin` | `evalscope.perf.plugin.api.openai_api` | OpenAI-compatible API request building + streaming |

Components we **build new** (evalscope lacks these):

- `UserState` — per-user conversation prefix growth + compaction trigger logic
- `Scheduler` — burst/steady/poisson user arrival dispatcher
- `SystemMetricsPoller` — background metrics endpoint scraper per backend preset
- Multi-user global aggregation — evalscope's perf module is single-user-per-run
- Console output — multi-user live status display
- Results JSON — structured output per design doc section 7.2

## 3. Project Structure

Flat package under `clawperf/`:

```
clawperf/
  __init__.py
  cli.py          # Entry point: argparse → Config → run benchmark
  config.py       # Pydantic config model + CLI argument parsing
  tokenizer.py    # Prompt assembly using evalscope tokenizer utils
  user.py         # UserState: conversation prefix, history, compaction
  scheduler.py    # User arrival: burst, steady, poisson
  api.py          # OpenAI-compatible streaming request dispatch
  metrics.py      # Per-turn/user/global aggregation + system metrics poller
  benchmark.py    # Main orchestrator
  output.py       # Console display + JSON results serialization
```

Dependencies (`pyproject.toml`):
- `evalscope[perf]` — tokenizer, streaming, metrics infrastructure
- `aiohttp` — async HTTP client
- `pydantic` — config and result models
- `rich` — console output
- `numpy` — percentile calculations

## 4. Core Request Model

### 4.1 Context Structure (per turn)

```
[System Prefix]    Fixed size, shared across all users (e.g. 15K tokens)
[User Prefix]      Per-user, starts at user_prefix_tokens, grows on compaction
[History]          Accumulated (user_msg, model_reply) pairs
[Current Input]    This turn's user input (e.g. 5K tokens)
```

### 4.2 Compaction (Append Mode)

Before constructing turn N:
1. Estimate total context = system_prefix_len + user_prefix_len + history_len + input_tokens_per_turn
2. If total > max_context_tokens:
   - Clear history (reset to empty)
   - Increase user_prefix_len by compaction_prefix_increment
   - Mark compaction triggered for this turn
3. Construct request with new state

User prefix grows in steps: 5K → 10K → 15K → ... , driven only by compaction events.

### 4.3 Token Generation

All prefix and input content generated as random token sequences:
1. Sample random token IDs from tokenizer vocabulary
2. Decode to text (or retain as token IDs for template)
3. Use `gen_prompt_decode_to_target_len()` to iteratively adjust to exact target length

System prefix content optionally loaded from a file (`--system-prefix-source`).

## 5. User Arrival Modes

### Burst
All `num-users` users start at t=0 simultaneously. Maximum concurrency immediately.

### Steady (`steady:<interval>`)
One new user every `<interval>` seconds until `num-users` reached. Controlled ramp-up.

### Poisson (`poisson:<lambda>`)
Inter-arrival times follow exponential distribution with rate `<lambda>` users/second. Random realistic load.

All users run independently — each has its own async coroutine with a serial turn loop. No per-user concurrency limit (one request at a time per user, but multiple users concurrent).

## 6. Module Details

### config.py

Pydantic `BenchmarkConfig` model mirroring all CLI parameters from design doc section 3. CLI parsing via `argparse`, converting to `BenchmarkConfig`. Validates required fields (`--endpoint`, `--model`).

### tokenizer.py

`PromptAssembler` class:
- `generate_system_prefix(config, tokenizer)` — random tokens or file content, sized to `system_prefix_tokens`
- `generate_user_input(config, tokenizer)` — random tokens sized to `input_tokens_per_turn`
- `generate_user_prefix(config, tokenizer, target_len)` — random tokens sized to `user_prefix_tokens` or current prefix length
- `build_messages(user_state, current_input_text)` — construct the message list: `[{"role": "system", "content": sys_prefix}] + history_messages + [{"role": "user", "content": current_input}]`. History is rendered as alternating user/assistant messages. The user prefix is prepended to the first user message in history (or to current_input if no history).
- `count_tokens(tokenizer, messages)` — call `tokenize_chat_messages()` for exact input token count

### user.py

`UserState` class:
- Fields: `user_id`, `user_prefix_len`, `history: List[Tuple[str, str]]`, `current_prefix_text`, `system_prefix_text`
- `check_compaction(config)` — estimate context size, trigger compaction if over limit
- `build_turn_request(prompt_assembler, tokenizer, config)` — full pipeline: check compaction → build messages → count tokens → return (messages, input_token_count, compaction_triggered)
- `record_response(response_text)` — append (input_text, response_text) to history

### scheduler.py

`Scheduler` class:
- `create_user_tasks(config, user_states)` — returns list of `(delay_seconds, user_id)` pairs
  - burst: all delays = 0
  - steady: delay = user_index * interval
  - poisson: delays from exponential distribution
- `dispatch_users(delays, user_coroutines)` — asyncio tasks with `asyncio.sleep(delay)` before each user starts

### api.py

`RequestDispatcher` class:
- Wraps evalscope's `OpenaiPlugin` for request building and streaming
- `send_streaming_request(messages, config, tokenizer)` → returns `TurnResult` with TTFT, ITL timestamps, output tokens, latency
- Handles `ignore_eos` via `extra_body={"ignore_eos": True}` (or backend-specific param)
- Error classification: timeout, http_4xx, http_5xx, context_overflow, network
- Per-request timeout via `--request-timeout`

### metrics.py

Three-layer aggregation:

**TurnMetrics** — per-turn data: TTFT, TPOT, ITL P50/P99, E2E latency, input/output/context tokens, compaction flag, error info

**UserMetrics** — per-user aggregate: total output tokens, total time, avg TTFT/TPOT/ITL, avg throughput, error count/types

**GlobalMetrics** — global aggregate: total requests/output tokens, global avg TTFT/TPOT, P50/P95/P99 TTFT, global throughput (tok/s + req/s), error rate/distribution, total compaction count

**SystemMetricsPoller** — background asyncio task:
- Polls `--metrics-endpoint` at `--metrics-interval`
- Parses Prometheus-format metrics per `--backend` preset
- vLLM: `kv_cache_utilization`, `gpu_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`, `prompt_cache_hit_rate`
- SGLang/MindIE: analogous fields per their documentation
- Stores as time-series list in results

### benchmark.py

`run_benchmark(config)` — main orchestrator:
1. Load tokenizer via `load_tokenizer()`
2. Generate system prefix content
3. Create `UserState` instances for all users
4. Compute user arrival schedule via `Scheduler`
5. Start `SystemMetricsPoller` if `--metrics-endpoint` provided
6. Dispatch user coroutines with arrival delays
7. Each user coroutine: turn loop (1..max_turns) → build request → send → record → update state
8. Live console output per turn completion
9. On SIGINT or completion: cancel pending, aggregate completed data, write results JSON

### output.py

**Console display** using `rich`:
- Per-turn: `[User XX] Turn YY/Z | TTFT: X.Xs | TPOT: Xms/tok | Context: XK/YK | Compaction: No/Yes`
- System metrics: `[Metrics] CacheHit: XX% | KVCache: XX% | Running: N | Waiting: N`

**Results JSON** structure:
```json
{
  "config": { ... },
  "summary": { global aggregate metrics },
  "users": [
    { "user_id": N, "aggregate": { ... }, "turns": [ { "turn_id": M, ... } ] }
  ],
  "system_metrics": [ { "timestamp": ..., "cache_hit_rate": ..., ... } ],
  "timeline": [
    { "event": "user_joined", "user_id": N, "time": T },
    { "event": "compaction", "user_id": N, "turn": M, "time": T }
  ]
}
```

## 7. Error Handling

- `asyncio.TimeoutError` → error_type: `timeout`
- `aiohttp.ClientResponseError` with 4xx status → `http_4xx`
- `aiohttp.ClientResponseError` with 5xx status → `http_5xx`
- Connection errors → `network`
- If tokenizer reports input > max_context_tokens after compaction → `context_overflow`

Errors do not kill the user task — the turn is marked failed and the user continues to the next turn (history is NOT updated for failed turns).

## 8. Graceful Shutdown

On SIGINT:
1. Cancel all pending asyncio tasks
2. Stop SystemMetricsPoller
3. Aggregate only completed turns
4. Write partial results JSON
5. Exit cleanly

## 9. Testing Strategy

- Unit tests for `UserState` compaction logic (prefix growth, history reset)
- Unit tests for `Scheduler` arrival patterns (burst/steady/poisson delays)
- Unit tests for `PromptAssembler` token counting accuracy
- Integration test with a mock streaming server for full turn loop
- CLI smoke test for argument parsing

## 10. CLI Usage

```bash
clawperf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-72B \
  --tokenizer Qwen/Qwen2.5-72B \
  --num-users 10 \
  --user-arrival burst \
  --system-prefix-tokens 15000 \
  --user-prefix-tokens 5000 \
  --input-tokens-per-turn 5000 \
  --output-tokens-per-turn 1000 \
  --max-context-tokens 128000 \
  --max-turns 100 \
  --metrics-endpoint http://localhost:8000/metrics \
  --backend vllm \
  --output results.json
```