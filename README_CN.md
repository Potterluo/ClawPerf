# ClawPerfBench

[![PyPI 版本](https://img.shields.io/pypi/v/clawperf.svg)](https://pypi.org/project/clawperf/)
[![Python 版本](https://img.shields.io/pypi/pyversions/clawperf.svg)](https://pypi.org/project/clawperf/)
[![许可证](https://img.shields.io/pypi/l/clawperf.svg)](https://github.com/ucm-system/ClawPerf/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ucm-system/ClawPerf.svg)](https://github.com/ucm-system/ClawPerf)

面向 LLM Serving 后端（vLLM、SGLang、MindIE）的多轮长上下文性能基准测试工具。

[English Documentation](README.md)

基于 [EvalScope](https://github.com/modelscope/evalscope) 的 perf 基础设施构建，扩展了：

- **多轮上下文模型**：System Prefix + User Prefix + History + Current Input
- **追加模式压缩**：上下文达到上限时清空历史、增长用户前缀
- **用户到达调度**：支持 burst、steady、Poisson 到达模式
- **系统指标采集**：Prometheus 端点支持 vLLM、SGLang、MindIE
- **逐用户逐轮指标**：TTFT、TPOT、ITL 及压缩追踪
- **Prefix Cache 模拟**：基于 Trie 的 HBM + 外部 prefix cache 命中率追踪

![ClawPerf Benchmark Output](docs/benchmark_result.jpg)

[English Documentation](README.md)

## 测试理念

ClawPerfBench 的核心目标是模拟 **Agent 系统的真实负载** —— 不是单次 API 调用，而是持续的多轮对话，将 LLM Serving 后端推到极限。

### 为什么多轮测试很重要

真实的 Agent 系统（如 OpenClaw）不是发送一次性请求。它们维护长对话：系统提示、用户专属上下文、不断累积的历史。每轮都重新发送完整累积的上下文，产生指数增长的 prompt。这与单请求基准测试根本不同，能暴露后者无法发现的后端行为：

- **Prefix cache 有效性**：KV-block cache 是否真正在多轮间复用 token？单请求基准无法测量。
- **负载下的压缩**：上下文触及窗口上限时，系统如何处理截断？是优雅恢复还是陷入溢出循环？
- **延迟退化**：上下文从 25K 增长到 200K tokens，TTFT 和 TPOT 变化剧烈。逐轮指标揭示这一过程。
- **并发压力**：多个用户各有独立对话，产生混合的 prefix cache 状态 —— 有的共享系统前缀，有的在用户路径分叉。

### 模拟真实用户

每个模拟用户维护独立的对话状态，拥有自己不断增长的前缀和历史。用户按照可配置的模式（burst、steady、Poisson）到达 —— 模拟真实流量如何逐步积累，而非人造的相同请求洪流。

### 我们测量什么

| 测量指标 | 为什么重要 |
|----------|-----------|
| 每轮 TTFT | 首 token 延迟随上下文增长 —— Agent 系统的关键体验指标 |
| 每轮 TPOT | 生成速度应保持稳定；退化表明计算瓶颈 |
| Prefix cache 命中率 | 跨轮 token 级复用比例 —— KV caching 的效率指标 |
| 压缩事件 | 何时、多久发生上下文溢出 —— 决定对话连续性 |
| 逐用户细分 | 不同用户有不同前缀路径；聚合统计会掩盖逐用户差异 |

## 安装

```bash
pip install clawperf
```

用于测试的 Mock 服务器：

```bash
pip install clawperf[mock-server]
```

开发环境：

```bash
pip install clawperf[dev]
```

从源码安装（推荐开发使用）：

```bash
git clone https://github.com/ucm-system/ClawPerf.git
cd ClawPerf
uv sync --extra dev --extra mock-server
```

## 快速开始

### 运行基准测试

```bash
clawperf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen3-32b \
  --num-users 5 \
  --user-arrival steady:2 \
  --max-turns 10 \
  --output results.json
```

### 启动 Mock 服务器（用于本地测试）

```bash
clawperf-mock-server --port 8080
```

### 端到端测试

```bash
# 启动 mock 服务器
clawperf-mock-server --port 8080

# 运行基准测试
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

## CLI 参数

### 用户配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-users` | 1 | 并发用户总数 |
| `--user-arrival` | burst | 到达模式：`burst`、`steady:<秒>` 或 `poisson:<lambda>` |

### 上下文配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--system-prefix-tokens` | 15000 | 系统级共享前缀 token 数 |
| `--system-prefix-source` | random | 内容来源：`random` 或文件路径 |
| `--user-prefix-tokens` | 5000 | 每用户前缀初始 token 数 |
| `--input-tokens-per-turn` | 5000 | 每轮用户输入 token 数 |
| `--output-tokens-per-turn` | 1000 | 每轮期望模型输出 token 数 |
| `--max-context-tokens` | 128000 | 模型最大上下文窗口 |
| `--compaction-prefix-increment` | 5000 | 触发压缩时用户前缀增加量 |

### 运行配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-turns` | 100 | 每用户最大请求轮数 |

### API 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--endpoint` | 必填 | LLM API 端点 URL |
| `--model` | 必填 | 模型名称 |
| `--api-key` | 空 | API Key |
| `--tokenizer` | 同 model | HuggingFace tokenizer 名称或路径 |
| `--ignore-eos` | True | 是否忽略 EOS token |
| `--request-timeout` | 600 | 单次请求超时秒数 |

### 系统指标

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--metrics-endpoint` | 无 | Prometheus 指标端点 URL |
| `--metrics-interval` | 5 | 采集周期（秒） |
| `--backend` | vllm | 后端预设：`vllm`、`sglang` 或 `mindie` |

### 输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | results.json | 结果输出文件路径 |
| `--history` | clawperf_history.jsonl | 每次运行追加一行记录（config + summary + 每用户聚合）到该 JSONL 文件，跨运行累积结果。传空字符串禁用。 |

## 输出格式

结果保存为 JSON：

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

`timing.bench_time_s` 不含一次性 setup（tokenizer 下载 + 内容生成）；每轮的
`wall_start_ts`/`wall_end_ts` 是相对 benchmark 起点的偏移，用于计算每用户
duration/throughput 聚合。

## 结果历史

每次运行都会向 `clawperf_history.jsonl`（可用 `--history` 配置，传 `--history ""` 禁用）
追加一行紧凑记录：`timestamp` + 完整 `config` + `summary` + `timing` + 每用户
`aggregate`（不含庞大的逐轮数组，保证文件随运行累积仍可查询）。

跨运行汇总对比结果：

```bash
# 最近一次的命中率
tail -n1 clawperf_history.jsonl | jq '.summary.prefix_cache_token_hit_rate'

# 所有运行的吞吐趋势
jq -c '{users: .config.num_users, bench_s: .timing.bench_time_s,
        hit_rate: .summary.prefix_cache_token_hit_rate}' clawperf_history.jsonl
```

任意一次运行的完整逐轮明细仍在其 `--output` JSON 文件中，可从历史记录的
`output_file` 字段定位。

## 上下文模型

每个用户的上下文遵循此结构：

![上下文模型与压缩](docs/context_model.svg)

当上下文达到 `--max-context-tokens` 时，触发追加模式压缩：

1. 先检查基础上下文（系统 + 用户前缀 + 输入，不含历史）是否已超限。若已超限，跳过压缩并标记 `context_overflow` —— 这防止了无限压缩循环。
2. 否则，清空历史对话，用户前缀增长 `--compaction-prefix-increment` tokens。
3. 用新的随机内容填充增长后的用户前缀。
4. 若增长后基础上下文仍超限，则**回退前缀增长**（仅清空历史），避免用户被永久困在溢出状态。

这模拟了真实 LLM Serving 系统利用 prefix cache 处理上下文溢出的方式。

## Prefix Cache 模拟

Mock 服务器使用 Trie 模拟 vLLM 的 KV-block prefix cache：

- **HBM Trie**：代表 GPU KV 缓存。优先查询最长前缀匹配。每次请求后都会更新（模拟 vLLM 无论命中与否都存储所有 KV block）。
- **外部 Trie**：代表 CPU/磁盘 prefix cache。HBM miss 时查询。同样每次请求后都更新。
- **Token 级命中率**：`prefix_cache_hit_tokens / prefix_cache_query_tokens` —— prompt token 中复用 KV block 的比例。这是有意义的指标；不报告请求级（二元）命中率。
- **驱逐**：当 Trie 超过 `max_prefixes`（200）时，按完整请求路径 LRU 驱逐最旧条目（引用计数保证共享前缀在仍被引用时不被删除）。

## 用户到达调度

![用户到达模式](docs/arrival_patterns.svg)

- **burst**：所有用户立即开始
- **steady:2**：每 2 秒加入 1 个新用户
- **poisson:0.5**：用户按 Poisson 过程到达，平均速率 0.5 个/秒

## 架构

ClawPerf 复用 EvalScope 的核心 perf 组件：

- **AioHttpClient**：异步 HTTP 流式请求，超时/连接器配置
- **OpenaiPlugin**：请求构建、响应解析、本地 token 计数
- **BenchmarkData**：单请求数据容器（TTFT、ITL、E2E 时序）
- **MetricsAccumulator**：实时指标聚合

并在此基础上添加多轮、多用户编排层。

核心模块：

| 模块 | 角色 |
|------|------|
| `cli.py` | Argparse 入口，配置创建，启动 runner |
| `config.py` | `BenchmarkConfig` 数据类，到达模式解析 |
| `runner.py` | `BenchmarkRunner` 编排器，用户循环，结果汇总，JSONL 历史 |
| `context.py` | `UserContext` 上下文组装，带无限循环保护的压缩 |
| `scheduler.py` | Burst/steady/Poisson 异步生成器 |
| `system_metrics.py` | `SystemMetricsPoller` 后端特定的指标映射 |
| `tokenizer.py` | `TokenizerManager` 封装 ModelScope/HuggingFace tokenizer |
| `logging_setup.py` | 集中式日志，经 `tqdm.write` 输出 |
| `mock_server.py` | FastAPI Mock LLM 服务器，Trie prefix cache 模拟 |

## 开发

```bash
uv sync --extra dev --extra mock-server
pytest
ruff check
```

## 许可证

Apache License 2.0