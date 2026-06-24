"""System metrics polling — Prometheus endpoint with backend-specific mapping."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("clawperf")

VLLM_METRICS = {
    "kv_cache_usage": "vllm:gpu_cache_usage_perc",
    "num_running": "vllm:num_requests_running",
    "num_waiting": "vllm:num_requests_waiting",
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_evictions": "vllm:prefix_cache_evictions_total",
    "prefix_cache_hit_tokens": "vllm:prefix_cache_hit_tokens_total",
    "prefix_cache_query_tokens": "vllm:prefix_cache_query_tokens_total",
    "external_prefix_cache_queries": "external_prefix_cache_queries_total",
    "external_prefix_cache_hit_tokens": "external_prefix_cache_hit_tokens_total",
    "external_prefix_cache_query_tokens": "external_prefix_cache_query_tokens_total",
}
SGLANG_METRICS = {
    "cache_hit_rate": "sglang:cache_hit_rate",
    "kv_cache_usage": "sglang:kv_cache_usage",
    "num_running": "sglang:num_running_requests",
    "num_waiting": "sglang:num_waiting_requests",
}
MINDIE_METRICS = {
    "cache_hit_rate": "mindie:cache_hit_rate",
    "kv_cache_usage": "mindie:kv_cache_usage_ratio",
    "num_running": "mindie:num_running_requests",
    "num_waiting": "mindie:num_waiting_requests",
}
BACKEND_MAP = {"vllm": VLLM_METRICS, "sglang": SGLANG_METRICS, "mindie": MINDIE_METRICS}


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus exposition text.

    Stores the fully-qualified (labeled) key AND an aggregated base form that
    SUMS across all labeled instances of the same metric. Summing is correct
    for the cumulative counters used by ``compute_prefix_cache_delta`` and for
    running/waiting gauges; it fixes an undercount when vLLM emits one series
    per model/engine. (Ratio gauges are only displayed, never differenced.)
    """
    result: Dict[str, float] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            result[parts[0]] = value
            base = parts[0].split("{")[0]
            if base == parts[0]:
                # No labels: the qualified key already holds the value.
                continue
            # Labeled instance: aggregate into the base form by summing.
            result[base] = result.get(base, 0.0) + value
    return result


class SystemMetricsPoller:
    def __init__(self, endpoint: str, interval: int, backend: str):
        self.endpoint = endpoint.rstrip("/")
        self.interval = interval
        self.metrics_map = BACKEND_MAP.get(backend, VLLM_METRICS)
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._samples: List[Dict] = []

    async def start(self):
        self._running = True
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    async def _poll_loop(self):
        while self._running:
            try:
                sample = await self._poll_once()
                if sample:
                    self._samples.append(sample)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Metrics poll error: %s", e)
            await asyncio.sleep(self.interval)

    async def _poll_once(self) -> Optional[Dict]:
        try:
            async with self._session.get(self.endpoint) as resp:
                if resp.status != 200:
                    logger.warning("Metrics endpoint returned status %d", resp.status)
                    return None
                text = await resp.text()
        except Exception as e:
            logger.warning("Metrics endpoint request failed: %s", e)
            return None
        raw = parse_prometheus_metrics(text)
        sample = {"timestamp": time.time()}
        for our_name, their_name in self.metrics_map.items():
            if their_name in raw:
                sample[our_name] = raw[their_name]
            else:
                base = their_name.split("{")[0]
                for key in raw:
                    if base in key:
                        sample[our_name] = raw[key]
                        break
        return sample

    async def snapshot(self) -> Optional[Dict]:
        """Take a single metrics snapshot (for start/end of benchmark)."""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        result = await self._poll_once()
        return result

    def get_samples(self) -> List[Dict]:
        return self._samples

    def compute_prefix_cache_delta(
        self, start: Optional[Dict], end: Optional[Dict]
    ) -> Optional[Dict]:
        """Compute token-level prefix cache hit rate from start/end counter snapshots."""
        if not start or not end:
            return None

        hit_tok_start = start.get("prefix_cache_hit_tokens", 0) or 0
        hit_tok_end = end.get("prefix_cache_hit_tokens", 0) or 0
        query_tok_start = start.get("prefix_cache_query_tokens", 0) or 0
        query_tok_end = end.get("prefix_cache_query_tokens", 0) or 0
        evictions_start = start.get("prefix_cache_evictions", 0) or 0
        evictions_end = end.get("prefix_cache_evictions", 0) or 0

        ext_hit_tok_start = start.get("external_prefix_cache_hit_tokens", 0) or 0
        ext_hit_tok_end = end.get("external_prefix_cache_hit_tokens", 0) or 0
        ext_query_tok_start = start.get("external_prefix_cache_query_tokens", 0) or 0
        ext_query_tok_end = end.get("external_prefix_cache_query_tokens", 0) or 0

        result = {
            "prefix_cache_hit_tokens_delta": hit_tok_end - hit_tok_start,
            "prefix_cache_query_tokens_delta": query_tok_end - query_tok_start,
            "prefix_cache_evictions_delta": evictions_end - evictions_start,
            "external_prefix_cache_hit_tokens_delta": ext_hit_tok_end - ext_hit_tok_start,
            "external_prefix_cache_query_tokens_delta": ext_query_tok_end - ext_query_tok_start,
        }

        delta_query_tokens = query_tok_end - query_tok_start
        delta_hit_tokens = hit_tok_end - hit_tok_start
        ext_delta_query_tokens = ext_query_tok_end - ext_query_tok_start
        ext_delta_hit_tokens = ext_hit_tok_end - ext_hit_tok_start

        if delta_query_tokens > 0:
            result["prefix_cache_token_hit_rate"] = delta_hit_tokens / delta_query_tokens
        if ext_delta_query_tokens > 0:
            result["external_prefix_cache_token_hit_rate"] = ext_delta_hit_tokens / ext_delta_query_tokens

        return result