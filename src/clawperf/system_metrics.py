"""System metrics polling — Prometheus endpoint with backend-specific mapping."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("clawperf")

VLLM_METRICS = {
    # Real vLLM metric names. Each value is a list of candidate names tried in
    # order so the poller is robust across vLLM versions:
    #   - hit/query token counters: vllm:prefix_cache_{hits,queries}_total
    #     (older builds dropped the _total suffix; some pre-V1 used *_tokens_total)
    #   - external cache metrics have NO vllm: prefix (per aisbench reference)
    #   - GPU KV usage: kv_cache_usage_perc (V1) vs gpu_cache_usage_perc (legacy)
    "kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "num_running": ["vllm:num_requests_running"],
    "num_waiting": ["vllm:num_requests_waiting"],
    "prefix_cache_hit_tokens": [
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_hit_tokens_total",
    ],
    "prefix_cache_query_tokens": [
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_query_tokens_total",
    ],
    "prefix_cache_evictions": [
        "vllm:prefix_cache_evictions_total",
        "vllm:prefix_cache_evictions",
    ],
    "external_prefix_cache_hit_tokens": [
        "external_prefix_cache_hits_total",
        "external_prefix_cache_hits",
        "vllm:external_prefix_cache_hits_total",
    ],
    "external_prefix_cache_query_tokens": [
        "external_prefix_cache_queries_total",
        "external_prefix_cache_queries",
        "vllm:external_prefix_cache_queries_total",
    ],
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
            # Reject non-finite values (NaN/Inf): a backend may emit "NaN" for
            # an undefined ratio; storing it would poison every delta/hit-rate
            # computation with NaN propagation.
            if not math.isfinite(value):
                continue
            result[parts[0]] = value
            base = parts[0].split("{")[0]
            if base == parts[0]:
                # No labels: the qualified key already holds the value.
                continue
            # Labeled instance: aggregate into the base form by summing.
            result[base] = result.get(base, 0.0) + value
    return result


def match_metrics(raw: Dict[str, float], metrics_map: Dict) -> Dict[str, float]:
    """Map raw Prometheus keys to our internal names using candidate lists.

    Each metrics_map value may be a single name (str) or an ordered list of
    candidate names; the first match wins, making the poller robust across
    backend versions that rename or drop the ``_total`` suffix.
    """
    sample: Dict[str, float] = {}
    for our_name, candidates in metrics_map.items():
        if isinstance(candidates, str):
            candidates = [candidates]
        for cand in candidates:
            if cand in raw:
                sample[our_name] = raw[cand]
                break
        else:
            # Substring fallback for labeled variants whose summed base form
            # wasn't produced (e.g. an unexpected suffix).
            for cand in candidates:
                base = cand.split("{")[0]
                for key in raw:
                    if base in key:
                        sample[our_name] = raw[key]
                        break
                if our_name in sample:
                    break
    return sample


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
        sample.update(match_metrics(raw, self.metrics_map))
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

        # If any counter went backwards, the backend restarted mid-benchmark
        # (counters reset to 0). A negative delta is meaningless and would
        # produce a bogus hit rate, so flag it and skip the rate computation.
        if delta_query_tokens < 0 or delta_hit_tokens < 0:
            result["prefix_cache_counter_reset"] = True
            logger.warning(
                "Prefix cache counters decreased during the run "
                "(query %d -> %d, hit %d -> %d) — backend likely restarted; "
                "hit rate not computed.",
                query_tok_start, query_tok_end, hit_tok_start, hit_tok_end,
            )
        elif delta_query_tokens > 0:
            result["prefix_cache_token_hit_rate"] = delta_hit_tokens / delta_query_tokens

        if ext_delta_query_tokens < 0 or ext_delta_hit_tokens < 0:
            result["external_prefix_cache_counter_reset"] = True
            logger.warning(
                "External prefix cache counters decreased during the run "
                "(query %d -> %d, hit %d -> %d) — backend likely restarted; "
                "hit rate not computed.",
                ext_query_tok_start, ext_query_tok_end,
                ext_hit_tok_start, ext_hit_tok_end,
            )
        elif ext_delta_query_tokens > 0:
            result["external_prefix_cache_token_hit_rate"] = (
                ext_delta_hit_tokens / ext_delta_query_tokens
            )

        return result
