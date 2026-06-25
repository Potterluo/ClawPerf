"""ClawPerf - Performance benchmarking tool for LLM Serving backends.

Reuses EvalScope's perf infrastructure for HTTP, streaming, and timing,
and adds multi-turn long-context workloads with append-mode compaction,
user arrival scheduling, and system metrics polling.
"""

__version__ = "0.2.4"