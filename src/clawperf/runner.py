"""Main benchmark runner — reuses EvalScope's perf infrastructure."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import signal
import statistics
import time
from typing import Dict, List, Optional

from prettytable import PrettyTable
from tqdm import tqdm

from clawperf.config import BenchmarkConfig
from clawperf.context import UserContext
from clawperf.scheduler import get_scheduler
from clawperf.system_metrics import SystemMetricsPoller
from clawperf.tokenizer import TokenizerManager

logger = logging.getLogger("clawperf")


def classify_error(bd) -> str:
    """Classify a BenchmarkData error into a standard error type."""
    if bd is None:
        return "network"
    if not bd.success:
        if bd.status_code:
            if 400 <= bd.status_code < 500:
                return "http_4xx"
            if 500 <= bd.status_code < 600:
                return "http_5xx"
        err_str = str(bd.error or "")
        if "timeout" in err_str.lower() or "TimeoutError" in err_str:
            return "timeout"
        return "network"
    return ""


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear' method).

    Robust for any sample size: returns the single value when N==1, interpolates
    between adjacent samples otherwise, and never indexes out of range.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    # Position in [0, n-1] with linear interpolation between neighbors.
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _percentiles(values: list[float]) -> dict:
    """Compute avg, P50, P75, P99, min, max from a list of values."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "avg": statistics.mean(s),
        "min": s[0],
        "P25": _percentile(values, 0.25),
        "P50": _percentile(values, 0.50),
        "P75": _percentile(values, 0.75),
        "P90": _percentile(values, 0.90),
        "P99": _percentile(values, 0.99),
        "max": s[-1],
        "N": n,
    }


def _fmt_val(v: float, unit: str = "", precision: int = 2) -> str:
    if v is None:
        return "N/A"
    s = f"{v:.{precision}f}"
    if unit:
        s += f" {unit}"
    return s


class BenchmarkRunner:
    """Main orchestrator — delegates HTTP/metrics to EvalScope."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.tokenizer_manager = TokenizerManager(config.tokenizer)
        self.system_poller: Optional[SystemMetricsPoller] = None

        self._api_plugin = None
        self._http_client = None
        self._accumulator = None

        self._system_prefix_content: str = ""
        self._user_prefix_contents: Dict[int, str] = {}
        self._turn_input_content: str = ""
        self._user_contexts: Dict[int, UserContext] = {}
        self._turn_records: List[Dict] = []
        self._timeline_events: List[Dict] = []

        self._shutdown = False
        # Two-phase timing: setup (tokenizer + content generation) is separated
        # from the actual benchmark window so reported Duration reflects only
        # request execution, not one-time startup cost.
        self._setup_start_time: float = 0.0
        self._bench_start_time: float = 0.0
        self._user_tasks: List[asyncio.Task] = []
        self._completed_turns: int = 0
        self._error_count: int = 0  # incremental error count for progress bar
        self._total_turns: int = 0
        self._pbar = None
        self._signal_installed = False
        self._metrics_start: Optional[Dict] = None
        self._metrics_end: Optional[Dict] = None
        self._prefix_cache_delta: Optional[Dict] = None

    async def run(self):
        """Execute the full benchmark."""
        self._setup_start_time = time.monotonic()
        self._setup_signal_handler()
        self._print_banner()

        # Surface misconfigurations early (e.g. base context already exceeding
        # the window) instead of letting every turn silently overflow.
        for problem in self.config.validate():
            logger.warning("CONFIG: %s", problem)

        # Dispatch to the controlled hit-rate test mode.
        if self.config.mode == "hitrate":
            await self._run_hitrate()
            return

        # 1. Initialize tokenizer
        _ = self.tokenizer_manager.tokenizer

        # 2. Initialize EvalScope components
        from evalscope.perf.core.http_client import AioHttpClient
        from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
        from evalscope.perf.utils.benchmark_util import MetricsAccumulator

        es_args = self.config.to_evalscope_args()
        es_args.parallel = es_args.parallel[0] if isinstance(es_args.parallel, list) else es_args.parallel
        es_args.number = es_args.number[0] if isinstance(es_args.number, list) else es_args.number

        self._api_plugin = OpenaiPlugin(es_args)
        self._http_client = AioHttpClient(es_args, self._api_plugin)
        # evalscope's logger.py runs logging.basicConfig(force=True) at import,
        # which overrides our earlier quieting — re-apply now that evalscope is
        # imported and its objects constructed, so its ERROR tracebacks don't
        # break the progress bar in non-verbose mode.
        from clawperf.logging_setup import quiet_third_party
        quiet_third_party(self.config.verbose)
        self._accumulator = MetricsAccumulator(
            concurrency=self.config.num_users,
            rate=-1,
        )

        # Pre-flight health check: send one tiny request and fail fast if the
        # endpoint is unreachable/wrong — avoids burning minutes of content
        # generation only to produce an all-error run. (Borrowed from llmperf.)
        await self._preflight_check()

        # 3. Generate content (off the event loop so it stays responsive;
        #    logging is now configured so each phase prints a live status line).
        await self._generate_content()

        # 4. Initialize user contexts
        self._total_turns = self.config.num_users * self.config.max_turns
        for uid in range(self.config.num_users):
            self._user_contexts[uid] = UserContext(
                user_id=uid,
                system_prefix=self._system_prefix_content,
                user_prefix_tokens=self.config.user_prefix_tokens,
                user_prefix_content=self._user_prefix_contents[uid],
                input_tokens_per_turn=self.config.input_tokens_per_turn,
                max_context_tokens=self.config.max_context_tokens,
                compaction_prefix_increment=self.config.compaction_prefix_increment,
                max_turns=self.config.max_turns,
            )

        # 5. Start system metrics poller. Only start+end snapshots are taken by
        #    default; the periodic poll loop (extra /metrics calls) runs only
        #    when --metrics-samples is set.
        if self.config.metrics_endpoint:
            self.system_poller = SystemMetricsPoller(
                endpoint=self.config.metrics_endpoint,
                interval=self.config.metrics_interval,
                backend=self.config.backend,
            )
            if self.config.metrics_samples:
                await self.system_poller.start()
            self._metrics_start = await self.system_poller.snapshot()
            logger.info("Metrics start snapshot: %s", self._snapshot_summary(self._metrics_start))

        # Optionally evict the server's prefix cache so the measured hit rate
        # reflects only this benchmark's prefixes (not residual traffic). Done
        # AFTER the start snapshot so counters are unaffected (reset evicts KV
        # blocks, not cumulative counters); the delta still isolates our run.
        if self.config.reset_cache:
            from clawperf.system_metrics import reset_prefix_cache
            await reset_prefix_cache(self.config.endpoint, self.config.backend)

        setup_time = time.monotonic() - self._setup_start_time
        logger.info("Setup complete in %.2fs — starting benchmark", setup_time)

        # Benchmark window starts here (excludes setup).
        self._bench_start_time = time.monotonic()

        # 6. Schedule users
        logger.info("Starting benchmark: %d users, arrival=%s", self.config.num_users, self.config.user_arrival)
        scheduler = get_scheduler(self.config)

        # Start tqdm progress bar (non-verbose mode). Logging already routes
        # through tqdm.write() via logging_setup, so no handler swapping needed.
        if not self.config.verbose:
            self._pbar = tqdm(
                total=self._total_turns,
                desc="Benchmark",
                unit="turn",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        async for uid, delay in scheduler:
            if self._shutdown:
                break
            if delay > 0:
                await self._interruptible_sleep(delay)
            if self._shutdown:
                break

            await self._add_timeline("user_joined", uid, time.monotonic() - self._bench_start_time)
            task = asyncio.create_task(self._run_user_loop(uid), name=f"user-{uid}")
            self._user_tasks.append(task)

        # 7. Wait for completion (always await so no task is orphaned on shutdown)
        if self._user_tasks:
            results = await asyncio.gather(*self._user_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    logger.error("User task %d failed: %s", i, result)

        # Snapshot metrics after all benchmark requests
        if self.system_poller and self.config.metrics_endpoint:
            self._metrics_end = await self.system_poller.snapshot()
            logger.info("Metrics end snapshot: %s", self._snapshot_summary(self._metrics_end))

        # 8. Close progress bar
        if self._pbar:
            self._pbar.close()

        # 9. Cleanup & save
        await self._finalize()

    async def _preflight_check(self):
        """Send one minimal request to confirm the endpoint is reachable and
        the model responds. Aborts early (RuntimeError) on failure so the user
        doesn't wait through content generation + a full all-error run."""
        logger.info("Pre-flight check: probing %s ...", self.config.endpoint)
        messages = [{"role": "user", "content": "hi"}]
        try:
            request_body = self._api_plugin.build_request(messages)
        except Exception as e:
            raise RuntimeError(f"Pre-flight: failed to build request: {e}") from e
        if request_body is None:
            raise RuntimeError("Pre-flight: build_request returned None — check --endpoint/--model.")
        try:
            bd = await asyncio.wait_for(
                self._http_client.post(request_body), timeout=min(30, self.config.request_timeout)
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Pre-flight: request to {self.config.endpoint} timed out — is the server up?"
            )
        except Exception as e:
            raise RuntimeError(f"Pre-flight: request to {self.config.endpoint} failed: {e}") from e
        if not bd.success:
            raise RuntimeError(
                f"Pre-flight: server rejected the probe (status={bd.status_code}, "
                f"error={bd.error}). Check --endpoint/--model/--api-key."
            )
        logger.info("Pre-flight check OK.")

    async def _run_hitrate(self):
        """Controlled prefix-cache hit-rate test: prefill prefixes, then measure.

        Prompt shape per request: [shared prefix][3 boundary tokens][unique suffix].
        The prefill phase injects each distinct prefix into the KV cache (output_len=1);
        the measure phase fires all requests and records TTFT/TPOT/throughput. The
        actual hit rate is read from the server's Prometheus counters (start/end delta).
        """
        from clawperf.hitrate import build_hitrate_requests, target_hit_rate

        # 1. Initialize tokenizer + EvalScope client + metrics poller (shared infra).
        _ = self.tokenizer_manager.tokenizer
        from evalscope.perf.core.http_client import AioHttpClient
        from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
        from evalscope.perf.utils.benchmark_util import MetricsAccumulator

        es_args = self.config.to_evalscope_args()
        es_args.parallel = es_args.parallel[0] if isinstance(es_args.parallel, list) else es_args.parallel
        es_args.number = es_args.number[0] if isinstance(es_args.number, list) else es_args.number
        es_args.max_tokens = self.config.output_len
        es_args.parallel = self.config.concurrency
        self._api_plugin = OpenaiPlugin(es_args)
        self._http_client = AioHttpClient(es_args, self._api_plugin)
        from clawperf.logging_setup import quiet_third_party
        quiet_third_party(self.config.verbose)
        self._accumulator = MetricsAccumulator(concurrency=self.config.concurrency, rate=-1)

        await self._preflight_check()

        # 2. Resolve prefix_len and build requests.
        prefix_len = self.config.prefix_len
        if prefix_len == 0 and self.config.hit_rate is not None:
            prefix_len = int(self.config.input_len * self.config.hit_rate)
        if prefix_len <= 0:
            raise RuntimeError(
                "hitrate: specify either --prefix-len or --hit-rate (in (0,1))."
            )
        requests = await asyncio.to_thread(
            build_hitrate_requests,
            num_requests=self.config.num_requests,
            input_len=self.config.input_len,
            prefix_len=prefix_len,
            prefix_num=self.config.prefix_num,
            tokenizer_manager=self.tokenizer_manager,
            seed=self.config.seed,
        )
        target = target_hit_rate(prefix_len, self.config.input_len)
        logger.info(
            "Hit-rate test: %d requests, input=%d, prefix=%d (target %.1f%%), "
            "%d distinct prefixes, output=%d, concurrency=%d",
            len(requests), self.config.input_len, prefix_len, target * 100,
            self.config.prefix_num, self.config.output_len, self.config.concurrency,
        )

        # 3. Optional cache reset (clean baseline).
        if self.config.reset_cache:
            from clawperf.system_metrics import reset_prefix_cache
            await reset_prefix_cache(self.config.endpoint, self.config.backend)

        # 4. Set up the metrics poller (created here so --metrics-samples covers
        #    the whole test). The START snapshot is taken AFTER prefill below so
        #    the prefill phase's cold queries don't dilute the measured hit rate.
        if self.config.metrics_endpoint:
            self.system_poller = SystemMetricsPoller(
                endpoint=self.config.metrics_endpoint,
                interval=self.config.metrics_interval,
                backend=self.config.backend,
            )
            if self.config.metrics_samples:
                await self.system_poller.start()

        setup_time = time.monotonic() - self._setup_start_time
        logger.info("Setup complete in %.2fs — starting hit-rate test", setup_time)
        self._bench_start_time = time.monotonic()

        # 5. Prefill phase: inject each distinct prefix (output_len=1).
        if self.config.prefill:
            logger.info("Prefill: injecting %d distinct prefixes ...", self.config.prefix_num)
            await self._prefill_prefixes(requests)

        # 6. Start metrics snapshot — AFTER prefill so the delta reflects only
        #    the measure phase (prefill's cold queries would otherwise dilute
        #    the measured hit rate). Matches aisbench's per-stage snapshots.
        if self.system_poller and self.config.metrics_endpoint:
            self._metrics_start = await self.system_poller.snapshot()
            logger.info("Metrics start (post-prefill): %s", self._snapshot_summary(self._metrics_start))

        # 7. Measure phase: fire all requests, concurrency-limited.
        logger.info("Measure: sending %d requests ...", len(requests))
        self._hitrate_records = await self._measure_requests(requests)

        # 8. End metrics snapshot.
        if self.system_poller and self.config.metrics_endpoint:
            self._metrics_end = await self.system_poller.snapshot()
            logger.info("Metrics end: %s", self._snapshot_summary(self._metrics_end))

        # 8. Compute prefix-cache delta (reuses the scenario-mode logic).
        if self.system_poller:
            self._prefix_cache_delta = self.system_poller.compute_prefix_cache_delta(
                self._metrics_start, self._metrics_end
            )

        bench_time_s = time.monotonic() - self._bench_start_time
        # 9. Cleanup & save (hitrate-specific finalize).
        await self._finalize_hitrate(prefix_len, target, bench_time_s, setup_time)

    async def _prefill_prefixes(self, requests):
        """Send each distinct prefix once with output_len=1 to inject into cache."""
        seen = set()
        prefill_msgs = []
        for r in requests:
            if r.prefix_idx in seen:
                continue
            seen.add(r.prefix_idx)
            prefill_msgs.append(r.prefill_prompt)
        sem = asyncio.Semaphore(self.config.concurrency)

        async def _one(prompt: str):
            async with sem:
                body = self._api_plugin.build_request(
                    [{"role": "user", "content": prompt}]
                )
                if body is None:
                    return
                # short output just to force prefill of the prompt into KV cache
                body = dict(body)
                body["max_tokens"] = 1
                try:
                    await self._http_client.post(body)
                except Exception as e:
                    logger.warning("Prefill request failed: %s", e)

        await asyncio.gather(*[_one(p) for p in prefill_msgs])

    async def _measure_requests(self, requests) -> List[Dict]:
        """Fire all measure-phase requests (concurrency-limited) and record metrics."""
        sem = asyncio.Semaphore(self.config.concurrency)
        records: List[Dict] = []

        if not self.config.verbose:
            self._pbar = tqdm(
                total=len(requests), desc="HitRate", unit="req",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        async def _one(r):
            async with sem:
                wall_start = time.monotonic()
                body = self._api_plugin.build_request(
                    [{"role": "user", "content": r.measure_prompt}]
                )
                if body is None:
                    rec = {"index": r.index, "success": False, "error_type": "build_request_failed"}
                else:
                    bd = await self._http_client.post(body)
                    wall_end = time.monotonic()
                    rec = self._build_hitrate_record(r, bd, wall_start, wall_end)
                records.append(rec)
                self._advance_progress(rec)

        await asyncio.gather(*[_one(r) for r in requests])
        if self._pbar:
            self._pbar.close()
        return records

    def _build_hitrate_record(self, req, bd, wall_start, wall_end) -> dict:
        rec = {
            "index": req.index,
            "prefix_idx": req.prefix_idx,
            "success": bd.success,
            "input_tokens": req.total_len,
            "prefix_len": req.prefix_len,
            "wall_start_ts": round(wall_start - self._bench_start_time, 3),
            "wall_end_ts": round(wall_end - self._bench_start_time, 3),
        }
        if bd.success:
            rec["ttft_ms"] = bd.first_chunk_latency * 1000 if bd.first_chunk_latency is not None else None
            rec["e2e_latency_ms"] = bd.query_latency * 1000 if bd.query_latency is not None else None
            rec["tpot_ms"] = bd.time_per_output_token * 1000 if bd.time_per_output_token is not None else None
            rec["output_tokens"] = bd.completion_tokens
        else:
            rec["error"] = bd.error
            rec["error_type"] = classify_error(bd)
            rec["status_code"] = bd.status_code
        return rec

    async def _finalize_hitrate(self, prefix_len: int, target: float, bench_time_s: float, setup_time_s: float):
        """Save results + print the hit-rate summary (measured vs target)."""
        if self.system_poller:
            await self.system_poller.stop()
        if self._http_client:
            await self._http_client.client.close()

        success = [r for r in self._hitrate_records if r.get("success")]
        errors = [r for r in self._hitrate_records if not r.get("success")]
        total_out = sum(r.get("output_tokens", 0) or 0 for r in success)
        total_in = sum(r.get("input_tokens", 0) or 0 for r in success)

        measured = None
        if self._prefix_cache_delta:
            measured = self._prefix_cache_delta.get("prefix_cache_token_hit_rate")

        summary = {
            "mode": "hitrate",
            "num_requests": len(self._hitrate_records),
            "success_count": len(success),
            "error_count": len(errors),
            "input_len": self.config.input_len,
            "prefix_len": prefix_len,
            "target_hit_rate": target,
            "measured_hit_rate": measured,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "bench_time_s": round(bench_time_s, 3),
        }
        # percentiles
        for key, src in (("ttft", "ttft_ms"), ("e2e_latency", "e2e_latency_ms"), ("tpot", "tpot_ms")):
            vals = [r[src] for r in success if r.get(src) is not None]
            if vals:
                summary[key] = _percentiles(vals)
        if self._prefix_cache_delta:
            summary["prefix_cache"] = self._prefix_cache_delta

        result = {
            "config": self.config.to_dict(),
            "summary": summary,
            "requests": self._hitrate_records,
            "timing": {"setup_time_s": round(setup_time_s, 3), "bench_time_s": round(bench_time_s, 3)},
        }
        if self.system_poller:
            result["system_metrics"] = self.system_poller.get_samples()

        import os
        out_dir = os.path.dirname(self.config.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(self.config.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        self._append_history(result, setup_time_s, bench_time_s)
        self._print_hitrate_summary(summary, bench_time_s, setup_time_s, target, measured)
        logger.info("Results saved to: %s", self.config.output)

    def _print_hitrate_summary(self, summary, bench_time_s, setup_time_s, target, measured):
        success = [r for r in self._hitrate_records if r.get("success")]
        total_out = sum(r.get("output_tokens", 0) or 0 for r in success)
        total_in = sum(r.get("input_tokens", 0) or 0 for r in success)

        print("\n" + "=" * 70, flush=True)
        print("ClawPerf - Hit-Rate Test Complete", flush=True)
        print("=" * 70, flush=True)

        ct = PrettyTable()
        ct.field_names = ["Metric", "Value"]
        ct.align["Metric"] = "l"
        ct.align["Value"] = "r"
        ct.add_row(["Setup Time", f"{setup_time_s:.2f} s"])
        ct.add_row(["Duration", f"{bench_time_s:.2f} s"])
        ct.add_row(["Total Requests", str(len(self._hitrate_records))])
        ct.add_row(["Success Requests", str(len(success))])
        ct.add_row(["Failed Requests", str(len([r for r in self._hitrate_records if not r.get("success")]))])
        ct.add_row(["Input Length", f"{self.config.input_len} tokens"])
        ct.add_row(["Prefix Length", f"{summary['prefix_len']} tokens"])
        ct.add_row(["Distinct Prefixes", str(self.config.prefix_num)])
        ct.add_row(["Output Length", f"{self.config.output_len} tokens"])
        ct.add_row(["Concurrency", str(self.config.concurrency)])
        ct.add_row(["Total Input Tokens", f"{total_in:,}"])
        ct.add_row(["Total Output Tokens", f"{total_out:,}"])
        ct.add_row(["Output Token Throughput", f"{total_out / bench_time_s:.2f} tok/s" if bench_time_s > 0 else "N/A"])
        # The headline: target vs measured hit rate.
        ct.add_row(["TARGET Hit Rate", f"{target * 100:.2f}%"])
        if measured is not None:
            ct.add_row(["MEASURED Hit Rate", f"{measured * 100:.2f}%"])
        elif self._prefix_cache_delta and self._prefix_cache_delta.get("prefix_cache_counter_reset"):
            ct.add_row(["MEASURED Hit Rate", "N/A (counter reset)"])
        else:
            ct.add_row(["MEASURED Hit Rate", "N/A (no metrics endpoint)"])
        if self._prefix_cache_delta:
            hit_tok = int(self._prefix_cache_delta.get("prefix_cache_hit_tokens_delta", 0))
            q_tok = int(self._prefix_cache_delta.get("prefix_cache_query_tokens_delta", 0))
            ct.add_row(["HBM Hit Tokens", f"{hit_tok:,}"])
            ct.add_row(["HBM Query Tokens", f"{q_tok:,}"])
        print("\n  Hit-Rate Results", flush=True)
        print(ct)

        # per-engine table (reuses scenario helper)
        if self._prefix_cache_delta:
            self._print_engine_table(
                "HBM Prefix Cache (per engine)",
                self._prefix_cache_delta.get("prefix_cache_engines", {}),
            )

        # performance percentiles
        perf = PrettyTable()
        perf.field_names = ["Metric", "Avg", "Min", "P25", "P50", "P75", "P90", "P99", "Max", "N"]
        perf.align["Metric"] = "l"
        perf.align = "r"
        for name, src in (("TTFT", "ttft_ms"), ("E2E Latency", "e2e_latency_ms"), ("TPOT", "tpot_ms")):
            vals = [r[src] for r in success if r.get(src) is not None]
            pct = _percentiles(vals)
            if pct:
                row = [name]
                for key in ("avg", "min", "P25", "P50", "P75", "P90", "P99", "max", "N"):
                    v = pct.get(key)
                    if key == "N":
                        row.append(str(int(v)))
                    elif v is not None:
                        row.append(_fmt_val(v, "ms", 2))
                    else:
                        row.append("N/A")
                perf.add_row(row)
            else:
                perf.add_row([name] + ["N/A"] * 9)
        print("\n  Performance Results", flush=True)
        print(perf)
        print("=" * 70, flush=True)

    async def _generate_content(self):
        """Generate system/user-prefix/turn-input content off the event loop."""
        logger.info("Generating system prefix (%d tokens)...", self.config.system_prefix_tokens)
        if self.config.system_prefix_source == "random":
            self._system_prefix_content = await self._gen_off_thread(
                self.tokenizer_manager.generate_random_content,
                self.config.system_prefix_tokens,
            )
        else:
            self._system_prefix_content = await self._gen_off_thread(
                self.tokenizer_manager.generate_content_from_file,
                self.config.system_prefix_source,
                self.config.system_prefix_tokens,
            )

        logger.info("Generating user prefix content (%d tokens/user)...", self.config.user_prefix_tokens)
        for uid in range(self.config.num_users):
            self._user_prefix_contents[uid] = await self._gen_off_thread(
                self.tokenizer_manager.generate_random_content,
                self.config.user_prefix_tokens,
            )
            # Yield between users so the loop stays responsive during long generation.
            await asyncio.sleep(0)

        logger.info("Generating per-turn input (%d tokens)...", self.config.input_tokens_per_turn)
        self._turn_input_content = await self._gen_off_thread(
            self.tokenizer_manager.generate_random_content,
            self.config.input_tokens_per_turn,
        )

    @staticmethod
    async def _gen_off_thread(func, *args):
        """Run a blocking tokenizer call in a worker thread."""
        return await asyncio.to_thread(func, *args)

    @staticmethod
    async def _interruptible_sleep(delay: float):
        """``asyncio.sleep`` that wakes promptly when the running task is cancelled."""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass

    async def _run_user_loop(self, user_id: int):
        ctx = self._user_contexts[user_id]

        for turn_id in range(1, self.config.max_turns + 1):
            if self._shutdown:
                break

            turn_result = ctx.prepare_turn(
                turn_id=turn_id,
                current_input_content=self._turn_input_content,
                tokenizer_manager=self.tokenizer_manager,
            )

            if turn_result["compaction_event"]:
                evt = turn_result["compaction_event"]
                evt.time = time.monotonic() - self._bench_start_time
                await self._add_timeline(
                    "compaction", user_id, evt.time,
                    turn=turn_id,
                    old_prefix_len=evt.old_prefix_len,
                    new_prefix_len=evt.new_prefix_len,
                )

            # Context overflow — skip this turn
            if turn_result["context_overflow"]:
                turn_record = {
                    "user_id": user_id,
                    "turn_id": turn_id,
                    "success": False,
                    "error_type": "context_overflow",
                    "context_tokens": turn_result["context_tokens"],
                    "compaction_triggered": turn_result["compaction_triggered"],
                }
                self._turn_records.append(turn_record)
                self._advance_progress(turn_record)
                continue

            messages = turn_result["messages"]
            context_tokens = turn_result["context_tokens"]

            request_body = self._api_plugin.build_request(messages)
            if request_body is None:
                # EvalScope returns None when request construction throws; the
                # turn must still be recorded and counted so the progress bar
                # reaches total and the failure shows up in results.
                logger.warning("[User %02d] Turn %d: failed to build request", user_id, turn_id)
                turn_record = {
                    "user_id": user_id,
                    "turn_id": turn_id,
                    "success": False,
                    "error_type": "build_request_failed",
                    "context_tokens": context_tokens,
                    "compaction_triggered": turn_result["compaction_triggered"],
                }
                self._turn_records.append(turn_record)
                self._advance_progress(turn_record)
                continue

            wall_start = time.monotonic()
            benchmark_data = await self._http_client.post(request_body)
            wall_end = time.monotonic()

            try:
                if benchmark_data.success:
                    benchmark_data.finalize(self._api_plugin)

                self._accumulator.update(benchmark_data, self._api_plugin)

                turn_record = self._build_turn_record(
                    user_id, turn_id, benchmark_data, context_tokens,
                    turn_result["compaction_triggered"],
                    wall_start, wall_end,
                )
                self._turn_records.append(turn_record)

                self._advance_progress(turn_record)

                if not benchmark_data.success:
                    continue

                ctx.append_history(self._turn_input_content, benchmark_data.generated_text)
                await asyncio.sleep(0)
            except Exception as e:
                logger.error("[User %02d] Turn %d: Error processing response: %s", user_id, turn_id, e)
                # Record the error
                error_record = {
                    "user_id": user_id,
                    "turn_id": turn_id,
                    "success": False,
                    "error_type": "processing_error",
                    "error": str(e),
                    "context_tokens": context_tokens,
                    "wall_start_ts": wall_start - self._bench_start_time,
                    "wall_end_ts": wall_end - self._bench_start_time,
                }
                self._turn_records.append(error_record)
                self._advance_progress(error_record)

    def _advance_progress(self, turn_record: dict):
        """Update progress bar or print verbose turn line."""
        self._completed_turns += 1
        # Track errors incrementally (O(1)) instead of re-scanning all records
        # every turn — the old O(n) scan made progress updates O(n^2) overall.
        if not turn_record.get("success", False):
            self._error_count += 1

        if self.config.verbose:
            self._print_verbose_turn(turn_record)
        elif self._pbar:
            self._pbar.set_postfix_str(f"err={self._error_count}")
            self._pbar.update(1)

    def _print_verbose_turn(self, t: dict):
        uid = t["user_id"]
        tid = t["turn_id"]
        max_ctx = self.config.max_context_tokens
        ctx_tok = t.get("context_tokens", 0)
        ctx_str = f"{ctx_tok//1000}K/{max_ctx//1000}K" if ctx_tok >= 1000 else f"{ctx_tok}/{max_ctx}"
        comp = "Yes" if t.get("compaction_triggered") else "No"

        if t.get("success"):
            ttft = _fmt_val(t.get("ttft_ms"), "ms", 1)
            tpot = _fmt_val(t.get("tpot_ms"), "ms", 2)
            print(
                f"[User {uid:02d}] Turn {tid}/{self.config.max_turns} | "
                f"TTFT: {ttft} | TPOT: {tpot} | Context: {ctx_str} | Compaction: {comp}",
                flush=True,
            )
        else:
            err_type = t.get("error_type", "unknown")
            print(
                f"[User {uid:02d}] Turn {tid}/{self.config.max_turns} | "
                f"ERROR [{err_type}] | Context: {ctx_str}",
                flush=True,
            )

    def _build_turn_record(
        self, user_id: int, turn_id: int, bd, context_tokens: int, compaction: bool,
        wall_start: float = 0.0, wall_end: float = 0.0,
    ) -> dict:
        # Wall-clock offsets (relative to benchmark start) for correct per-user
        # duration / throughput — not derivable from per-request latencies alone.
        bench_start = self._bench_start_time or wall_start
        record = {
            "user_id": user_id,
            "turn_id": turn_id,
            "success": bd.success,
            "context_tokens": context_tokens,
            "compaction_triggered": compaction,
            "wall_start_ts": round(wall_start - bench_start, 3) if wall_start else None,
            "wall_end_ts": round(wall_end - bench_start, 3) if wall_end else None,
        }

        # Save request details
        if bd.request:
            record["request"] = bd.request

        if bd.success:
            record["ttft_ms"] = bd.first_chunk_latency * 1000 if bd.first_chunk_latency is not None else None
            record["e2e_latency_ms"] = bd.query_latency * 1000 if bd.query_latency is not None else None
            record["tpot_ms"] = bd.time_per_output_token * 1000 if bd.time_per_output_token is not None else None
            record["input_tokens"] = bd.prompt_tokens
            record["output_tokens"] = bd.completion_tokens

            # Save generated text
            if bd.generated_text:
                record["generated_text"] = bd.generated_text

            if bd.inter_chunk_latency:
                sorted_itl = sorted(bd.inter_chunk_latency)
                record["itl_p50_ms"] = sorted_itl[int(len(sorted_itl) * 0.50)] * 1000
                record["itl_p99_ms"] = sorted_itl[min(int(len(sorted_itl) * 0.99), len(sorted_itl) - 1)] * 1000
        else:
            record["error"] = bd.error
            record["error_type"] = classify_error(bd)
            record["status_code"] = bd.status_code

            # Save error response if available
            if bd.response_messages:
                record["response"] = bd.response_messages

        return record

    def _compute_user_aggregate(self, user_id: int) -> dict:
        turns = [t for t in self._turn_records if t["user_id"] == user_id]
        success_turns = [t for t in turns if t.get("success")]
        error_turns = [t for t in turns if not t.get("success")]

        agg = {
            "total_output_tokens": sum(t.get("output_tokens", 0) or 0 for t in success_turns),
            "total_input_tokens": sum(t.get("input_tokens", 0) or 0 for t in success_turns),
            "success_count": len(success_turns),
            "error_count": len(error_turns),
            "error_types": list(set(t.get("error_type", "unknown") for t in error_turns)),
            "compaction_count": len(self._user_contexts[user_id].compaction_events),
        }

        ttft_values = [t["ttft_ms"] for t in success_turns if t.get("ttft_ms") is not None]
        if ttft_values:
            agg["ttft"] = _percentiles(ttft_values)

        e2e_values = [t["e2e_latency_ms"] for t in success_turns if t.get("e2e_latency_ms") is not None]
        if e2e_values:
            agg["e2e_latency"] = _percentiles(e2e_values)

        tpot_values = [t["tpot_ms"] for t in success_turns if t.get("tpot_ms") is not None]
        if tpot_values:
            agg["tpot"] = _percentiles(tpot_values)

        itl_p50_values = [t["itl_p50_ms"] for t in success_turns if t.get("itl_p50_ms") is not None]
        if itl_p50_values:
            agg["itl_p50"] = _percentiles(itl_p50_values)

        # Real per-user wall-clock duration: from the first turn's start to the
        # last turn's end (timestamps are offsets relative to benchmark start).
        starts = [t["wall_start_ts"] for t in success_turns if t.get("wall_start_ts") is not None]
        ends = [t["wall_end_ts"] for t in success_turns if t.get("wall_end_ts") is not None]
        if starts and ends:
            user_duration_s = max(ends) - min(starts)
            if user_duration_s > 0:
                agg["duration_s"] = user_duration_s
                agg["throughput_tok_s"] = agg["total_output_tokens"] / user_duration_s

        return agg

    async def _finalize(self):
        if self.system_poller:
            await self.system_poller.stop()

        if self._http_client:
            await self._http_client.client.close()

        # Split wall time: setup (tokenizer + content generation) vs the actual
        # benchmark request window. Reported Duration reflects only the latter.
        setup_time_s = (self._bench_start_time - self._setup_start_time) if self._setup_start_time else 0.0
        bench_time_s = (time.monotonic() - self._bench_start_time) if self._bench_start_time else 0.0

        if self._accumulator:
            es_metrics = self._accumulator.to_result()
            summary = es_metrics.create_message(api_type="openai")
        else:
            summary = {}

        summary["total_compactions"] = sum(
            len(uc.compaction_events) for uc in self._user_contexts.values()
        ) if self._user_contexts else 0

        users_data = []
        for uid in sorted(self._user_contexts.keys()):
            user_turns = [t for t in self._turn_records if t["user_id"] == uid]
            users_data.append({
                "user_id": uid,
                "aggregate": self._compute_user_aggregate(uid),
                "turns": user_turns,
            })

        sys_metrics = self.system_poller.get_samples() if self.system_poller else []

        prefix_cache_delta = None
        if self.system_poller:
            prefix_cache_delta = self.system_poller.compute_prefix_cache_delta(
                self._metrics_start, self._metrics_end
            )
        if prefix_cache_delta:
            self._prefix_cache_delta = prefix_cache_delta
            summary["prefix_cache_token_hit_rate"] = self._prefix_cache_delta.get("prefix_cache_token_hit_rate")
            summary["prefix_cache_hit_tokens_delta"] = self._prefix_cache_delta["prefix_cache_hit_tokens_delta"]
            summary["prefix_cache_query_tokens_delta"] = self._prefix_cache_delta["prefix_cache_query_tokens_delta"]
            summary["external_prefix_cache_token_hit_rate"] = self._prefix_cache_delta.get("external_prefix_cache_token_hit_rate")
            summary["external_prefix_cache_hit_tokens_delta"] = self._prefix_cache_delta["external_prefix_cache_hit_tokens_delta"]
            summary["external_prefix_cache_query_tokens_delta"] = self._prefix_cache_delta["external_prefix_cache_query_tokens_delta"]

        result = {
            "config": self.config.to_dict(),
            "summary": summary,
            "users": users_data,
            "system_metrics": sys_metrics,
            "timeline": self._timeline_events,
            "timing": {
                "setup_time_s": round(setup_time_s, 3),
                "bench_time_s": round(bench_time_s, 3),
            },
        }

        import os
        out_dir = os.path.dirname(self.config.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(self.config.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        # Append a compact one-line record (config + summary + per-user aggregates)
        # to the history JSONL so results accumulate across runs.
        self._append_history(result, setup_time_s, bench_time_s)

        self._print_final_summary(bench_time_s, setup_time_s)
        logger.info("Results saved to: %s", self.config.output)

    def _append_history(self, result: dict, setup_time_s: float, bench_time_s: float):
        """Append a queryable record to the JSONL history file (one line per run).

        Keeps the line compact: full config + summary + per-user aggregates, but
        NOT the full per-turn / timeline / metrics-sample arrays (those live in
        the ``--output`` file referenced by ``output_file``).
        """
        if not self.config.history:
            return
        import os
        from datetime import datetime

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_file": os.path.abspath(self.config.output),
            "config": result["config"],
            "summary": result["summary"],
            "timing": {"setup_time_s": setup_time_s, "bench_time_s": bench_time_s},
            # Per-user aggregates only (drop the heavy per-turn detail).
            "users": [
                {"user_id": u["user_id"], "aggregate": u["aggregate"]}
                for u in result.get("users", [])
            ],
        }
        try:
            out_dir = os.path.dirname(self.config.history)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            # Append mode — each run adds exactly one line.
            with open(self.config.history, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            logger.info("History appended to: %s", self.config.history)
        except OSError as e:
            logger.warning("Failed to append history to %s: %s", self.config.history, e)

    # ── Pretty output ──

    def _print_final_summary(self, bench_time_s: float, setup_time_s: float = 0.0):
        success_turns = [t for t in self._turn_records if t.get("success")]
        error_turns = [t for t in self._turn_records if not t.get("success")]
        total_reqs = len(self._turn_records)

        total_in_tok = sum(t.get("input_tokens", 0) or 0 for t in success_turns)
        total_out_tok = sum(t.get("output_tokens", 0) or 0 for t in success_turns)
        total_comp = sum(len(uc.compaction_events) for uc in self._user_contexts.values())

        # ── Common Metrics Table ──
        ct = PrettyTable()
        ct.field_names = ["Metric", "Value"]
        ct.align["Metric"] = "l"
        ct.align["Value"] = "r"
        ct.add_row(["Setup Time", f"{setup_time_s:.2f} s"])
        ct.add_row(["Duration", f"{bench_time_s:.2f} s"])
        ct.add_row(["Total Requests", str(total_reqs)])
        ct.add_row(["Success Requests", str(len(success_turns))])
        ct.add_row(["Failed Requests", str(len(error_turns))])
        ct.add_row(["Total Input Tokens", f"{total_in_tok:,}"])
        ct.add_row(["Prefill Token Throughput", f"{total_in_tok / bench_time_s:.2f} tok/s" if bench_time_s > 0 else "N/A"])
        ct.add_row(["Total Output Tokens", f"{total_out_tok:,}"])
        ct.add_row(["Request Throughput", f"{len(success_turns) / bench_time_s:.4f} req/s" if bench_time_s > 0 else "N/A"])
        ct.add_row(["Output Token Throughput", f"{total_out_tok / bench_time_s:.2f} tok/s" if bench_time_s > 0 else "N/A"])
        # Decode-only throughput: total output tokens / sum of per-request decode
        # times (e2e - ttft). Isolates generation speed from prefill/TTFT.
        # (Borrowed from llmperf's incremental_throughput.)
        decode_time_s = sum(
            (t.get("e2e_latency_ms", 0) - t.get("ttft_ms", 0)) / 1000
            for t in success_turns
            if t.get("e2e_latency_ms") and t.get("ttft_ms")
            and t["e2e_latency_ms"] > t["ttft_ms"]
        )
        ct.add_row([
            "Decode Throughput (excl prefill)",
            f"{total_out_tok / decode_time_s:.2f} tok/s" if decode_time_s > 0 else "N/A",
        ])
        ct.add_row(["Total Token Throughput", f"{(total_in_tok + total_out_tok) / bench_time_s:.2f} tok/s" if bench_time_s > 0 else "N/A"])
        ct.add_row(["Total Compactions", str(total_comp)])
        if self._prefix_cache_delta:
            # Always show the hit-rate row so it's not lost when only tokens print.
            def _rate_str(rate, reset):
                if reset:
                    return "N/A (counter reset)"
                if rate is not None:
                    return f"{rate * 100:.2f}%"
                return "0.00%"  # no queries -> nothing reused
            tok_rate = self._prefix_cache_delta.get("prefix_cache_token_hit_rate")
            ct.add_row([
                "HBM Prefix Cache Token Hit Rate",
                _rate_str(tok_rate, self._prefix_cache_delta.get("prefix_cache_counter_reset")),
            ])
            # Counter deltas arrive as floats from Prometheus parsing — display as ints.
            hit_tok = int(self._prefix_cache_delta.get("prefix_cache_hit_tokens_delta", 0))
            q_tok = int(self._prefix_cache_delta.get("prefix_cache_query_tokens_delta", 0))
            ct.add_row(["HBM Prefix Cache Hit Tokens", f"{hit_tok:,}"])
            ct.add_row(["HBM Prefix Cache Query Tokens", f"{q_tok:,}"])
            evictions = int(self._prefix_cache_delta.get("prefix_cache_evictions_delta", 0))
            if evictions > 0:
                ct.add_row(["HBM Prefix Cache Evictions", str(evictions)])
            ext_tok_rate = self._prefix_cache_delta.get("external_prefix_cache_token_hit_rate")
            ct.add_row([
                "External Prefix Cache Token Hit Rate",
                _rate_str(ext_tok_rate, self._prefix_cache_delta.get("external_prefix_cache_counter_reset")),
            ])
            ext_hit_tok = int(self._prefix_cache_delta.get("external_prefix_cache_hit_tokens_delta", 0))
            ext_q_tok = int(self._prefix_cache_delta.get("external_prefix_cache_query_tokens_delta", 0))
            ct.add_row(["External Prefix Cache Hit Tokens", f"{ext_hit_tok:,}"])
            ct.add_row(["External Prefix Cache Query Tokens", f"{ext_q_tok:,}"])
        if error_turns:
            err_dist = {}
            for t in error_turns:
                et = t.get("error_type", "unknown")
                err_dist[et] = err_dist.get(et, 0) + 1
            ct.add_row(["Error Distribution", str(err_dist)])

        print("\n" + "=" * 70, flush=True)
        print("ClawPerf - Benchmark Complete", flush=True)
        print("=" * 70, flush=True)
        print("\n  Common Metrics", flush=True)
        print(ct)

        # ── Per-engine prefix cache breakdown (vllm) ──
        if self._prefix_cache_delta:
            self._print_engine_table(
                "HBM Prefix Cache (per engine)",
                self._prefix_cache_delta.get("prefix_cache_engines", {}),
            )
            self._print_engine_table(
                "External Prefix Cache (per engine)",
                self._prefix_cache_delta.get("external_prefix_cache_engines", {}),
            )

        # ── Performance Table (avg/min/P25/P50/P75/P90/P99/max/N) ──
        perf = PrettyTable()
        perf.field_names = ["Metric", "Avg", "Min", "P25", "P50", "P75", "P90", "P99", "Max", "N"]
        perf.align["Metric"] = "l"
        perf.align = "r"  # default right for numbers

        ttft_pct = _percentiles([t["ttft_ms"] for t in success_turns if t.get("ttft_ms") is not None])
        e2e_pct = _percentiles([t["e2e_latency_ms"] for t in success_turns if t.get("e2e_latency_ms") is not None])
        tpot_pct = _percentiles([t["tpot_ms"] for t in success_turns if t.get("tpot_ms") is not None])
        itl_pct = _percentiles([t["itl_p50_ms"] for t in success_turns if t.get("itl_p50_ms") is not None])
        in_tok_pct = _percentiles([t["input_tokens"] for t in success_turns if t.get("input_tokens") is not None])
        out_tok_pct = _percentiles([t["output_tokens"] for t in success_turns if t.get("output_tokens") is not None])

        for name, pct, unit in [
            ("E2E Latency", e2e_pct, "ms"),
            ("TTFT", ttft_pct, "ms"),
            ("TPOT", tpot_pct, "ms"),
            ("ITL (P50)", itl_pct, "ms"),
            ("Input Tokens", in_tok_pct, ""),
            ("Output Tokens", out_tok_pct, ""),
        ]:
            if pct:
                row = [name]
                for key in ("avg", "min", "P25", "P50", "P75", "P90", "P99", "max", "N"):
                    v = pct.get(key)
                    if key == "N":
                        row.append(str(int(v)))
                    elif v is not None:
                        row.append(_fmt_val(v, unit, 2))
                    else:
                        row.append("N/A")
                perf.add_row(row)
            else:
                perf.add_row([name] + ["N/A"] * 9)

        print("\n  Performance Results", flush=True)
        print(perf)

        # ── Per-User Table ──
        ut = PrettyTable()
        ut.field_names = ["User", "In Tok", "Out Tok", "Time (s)",
                          "TTFT (ms)", "TPOT (ms)", "E2E (ms)", "Thru (tok/s)", "Comp", "Succ", "Fail"]
        ut.align["User"] = "l"
        ut.align = "r"

        user_rows = []
        for uid in sorted(self._user_contexts.keys()):
            agg = self._compute_user_aggregate(uid)
            ttft = agg.get("ttft", {})
            e2e = agg.get("e2e_latency", {})
            tpot = agg.get("tpot", {})
            ttft_avg = f"{ttft['avg']:.2f}" if ttft else "-"
            e2e_avg = f"{e2e['avg']:.2f}" if e2e else "-"
            tpot_avg = f"{tpot['avg']:.3f}" if tpot else "-"
            thru = f"{agg['throughput_tok_s']:.2f}" if agg.get("throughput_tok_s") else "-"
            dur = f"{agg['duration_s']:.2f}" if agg.get("duration_s") else "-"
            row = [
                f"User {uid+1}",
                f"{agg['total_input_tokens']:,}",
                f"{agg['total_output_tokens']:,}",
                dur,
                ttft_avg,
                tpot_avg,
                e2e_avg,
                thru,
                str(agg["compaction_count"]),
                str(agg["success_count"]),
                str(agg["error_count"]),
            ]
            ut.add_row(row)
            user_rows.append(agg)

        # ── Summary: best / worst / avg across users ──
        # For latency: Best=min, Worst=max (lower is better)
        # For throughput: Best=max, Worst=min (higher is better)
        # For duration: ambiguous, keep Best=min, Worst=max (faster completion)
        if user_rows:
            def _val(key, fn):
                vals = [float(r.get(key, 0) or 0) for r in user_rows if r.get(key)]
                if not vals:
                    return "-"
                fmt = ".2f" if max(vals) < 100 else ".1f"
                return f"{fn(vals):{fmt}}"

            def _pct(pct_key, fn):
                vals = [float(r.get(pct_key, {}).get("avg", 0)) for r in user_rows if r.get(pct_key)]
                if not vals:
                    return "-"
                fmt = ".2f" if max(vals) < 100 else ".1f"
                return f"{fn(vals):{fmt}}"

            for label, fn_latency, fn_thru in [
                ("Best", min, max),   # best latency=min, best throughput=max
                ("Worst", max, min),  # worst latency=max, worst throughput=min
                ("Avg", statistics.mean, statistics.mean),
            ]:
                ut.add_row([
                    label,
                    "", "",
                    _val("duration_s", fn_latency),
                    _pct("ttft", fn_latency),
                    _pct("tpot", fn_latency),
                    _pct("e2e_latency", fn_latency),
                    _val("throughput_tok_s", fn_thru),
                    "", "", "",
                ])

        print("\n  Per-User Summary", flush=True)
        print(ut)
        print("=" * 70, flush=True)

    @staticmethod
    def _snapshot_summary(snap: Optional[Dict]) -> str:
        """Concise one-line snapshot for logging (no huge per-engine dict)."""
        if not snap:
            return "unavailable"
        eng = snap.get("prefix_cache_engines", {})
        ext = snap.get("external_prefix_cache_engines", {})
        return (
            f"query={int(snap.get('prefix_cache_query_tokens', 0)):,} "
            f"hit={int(snap.get('prefix_cache_hit_tokens', 0)):,} "
            f"ext_query={int(snap.get('external_prefix_cache_query_tokens', 0)):,} "
            f"ext_hit={int(snap.get('external_prefix_cache_hit_tokens', 0)):,} "
            f"engines={list(eng)} ext_engines={list(ext)}"
        )

    def _print_engine_table(self, title: str, engines: dict):
        """Print a per-engine prefix-cache breakdown (query/hit tokens + rate).

        Only shown when the backend exposes engine="N" labels; a TOTAL row
        (summed across engines) is appended so both views are visible.
        """
        if not engines:
            return
        t = PrettyTable()
        t.field_names = ["Engine", "Query Tokens", "Hit Tokens", "Hit Rate"]
        t.align["Engine"] = "l"
        t.align = "r"
        total_q = 0
        total_h = 0
        for eng in sorted(engines):
            e = engines[eng]
            q = int(e.get("query_tokens_delta", 0))
            h = int(e.get("hit_tokens_delta", 0))
            total_q += q
            total_h += h
            rate = e.get("token_hit_rate")
            rate_s = f"{rate * 100:.2f}%" if rate is not None else (
                "reset" if e.get("counter_reset") else "-"
            )
            t.add_row([f"engine {eng}", f"{q:,}", f"{h:,}", rate_s])
        # TOTAL row (sum across engines; rate recomputed from the sums).
        tot_rate = (total_h / total_q) if total_q > 0 else None
        tot_rate_s = f"{tot_rate * 100:.2f}%" if tot_rate is not None else "-"
        t.add_row(["TOTAL", f"{total_q:,}", f"{total_h:,}", tot_rate_s])
        print(f"\n  {title}", flush=True)
        print(t)

    def _print_banner(self):
        print("=" * 70, flush=True)
        print("ClawPerf - LLM Serving Performance Benchmark", flush=True)
        print("  (Powered by EvalScope perf infrastructure)", flush=True)
        print("=" * 70, flush=True)
        print(f"  Model:        {self.config.model}", flush=True)
        print(f"  Endpoint:     {self.config.endpoint}", flush=True)
        print(f"  Backend:      {self.config.backend}", flush=True)
        print(f"  Users:        {self.config.num_users} (arrival: {self.config.user_arrival})", flush=True)
        print(f"  Max Turns:    {self.config.max_turns}", flush=True)
        print(f"  Context:      sys={self.config.system_prefix_tokens}, "
              f"usr={self.config.user_prefix_tokens}, "
              f"in={self.config.input_tokens_per_turn}, "
              f"out={self.config.output_tokens_per_turn}", flush=True)
        print(f"  Max Context:  {self.config.max_context_tokens} tokens", flush=True)
        print(f"  Ignore EOS:   {self.config.ignore_eos}", flush=True)
        print(f"  Tokenizer:    {self.config.tokenizer}", flush=True)
        if not self.config.metrics_endpoint:
            print("  Metrics:      NOT configured — prefix cache data will not be collected", flush=True)
        else:
            print(f"  Metrics:      {self.config.metrics_endpoint}", flush=True)
        if self.config.history:
            print(f"  History:      {self.config.history} (append)", flush=True)
        print("=" * 70, flush=True)

    async def _add_timeline(self, event: str, user_id: int, time_offset: float, **kwargs):
        entry = {"event": event, "user_id": user_id, "time": round(time_offset, 3)}
        entry.update(kwargs)
        self._timeline_events.append(entry)

    def _setup_signal_handler(self):
        """Register SIGINT handler for graceful shutdown.

        On Unix we use the loop's signal handler (runs in the event loop thread,
        can safely cancel tasks). On Windows that API is unavailable, so we fall
        back to ``signal.signal`` — the callback still flips ``_shutdown`` and the
        main loop exits at the next ``await`` checkpoint.
        """
        if self._signal_installed:
            return

        def on_sigint(*_):
            if self._shutdown:
                logger.info("SIGINT received again — forcing exit.")
                raise KeyboardInterrupt
            logger.info("SIGINT received. Initiating graceful shutdown...")
            self._shutdown = True
            for task in self._user_tasks:
                if not task.done():
                    task.cancel()
            if self._pbar:
                self._pbar.close()

        if platform.system() == "Windows":
            signal.signal(signal.SIGINT, on_sigint)
        else:
            try:
                asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_sigint)
            except NotImplementedError:
                signal.signal(signal.SIGINT, on_sigint)
        self._signal_installed = True

    async def shutdown_and_save(self):
        """Graceful shutdown path — cancel outstanding tasks and finalize partial results.

        All user tasks are awaited (with ``return_exceptions=True``) so none are
        orphaned, avoiding 'Task was destroyed but it is pending' warnings.
        """
        self._shutdown = True
        for task in self._user_tasks:
            if not task.done():
                task.cancel()
        if self._pbar:
            self._pbar.close()
        # Ensure the benchmark window is closed even if interrupted mid-setup.
        if not self._bench_start_time:
            self._bench_start_time = time.monotonic()
        if self._user_tasks:
            await asyncio.gather(*self._user_tasks, return_exceptions=True)
        await self._finalize()