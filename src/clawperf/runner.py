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


class _TqdmLogHandler(logging.Handler):
    """Redirects log output through tqdm.write() so messages don't break the progress bar."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


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


def _percentiles(values: list[float]) -> dict:
    """Compute avg, P50, P75, P99, min, max from a list of values."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "avg": statistics.mean(s),
        "min": s[0],
        "P50": s[int(n * 0.50)],
        "P75": s[int(n * 0.75)],
        "P90": s[int(n * 0.90)],
        "P99": s[min(int(n * 0.99), n - 1)],
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
        self._start_time: float = 0.0
        self._user_tasks: List[asyncio.Task] = []
        self._completed_turns: int = 0
        self._total_turns: int = 0
        self._pbar = None
        self._tqdm_handler: Optional[_TqdmLogHandler] = None
        self._saved_handlers: Optional[List[logging.Handler]] = None
        self._metrics_start: Optional[Dict] = None
        self._metrics_end: Optional[Dict] = None
        self._prefix_cache_delta: Optional[Dict] = None

    async def run(self):
        """Execute the full benchmark."""
        self._start_time = time.monotonic()
        self._setup_signal_handler()
        self._print_banner()

        # 1. Initialize tokenizer
        _ = self.tokenizer_manager.tokenizer

        # 2. Initialize EvalScope components
        from evalscope.perf.arguments import Arguments
        from evalscope.perf.core.http_client import AioHttpClient
        from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
        from evalscope.perf.utils.benchmark_util import MetricsAccumulator

        es_args = self.config.to_evalscope_args()
        es_args.parallel = es_args.parallel[0] if isinstance(es_args.parallel, list) else es_args.parallel
        es_args.number = es_args.number[0] if isinstance(es_args.number, list) else es_args.number

        self._api_plugin = OpenaiPlugin(es_args)
        self._http_client = AioHttpClient(es_args, self._api_plugin)
        self._accumulator = MetricsAccumulator(
            concurrency=self.config.num_users,
            rate=-1,
        )

        # 3. Generate content
        logger.info("Generating system prefix (%d tokens)...", self.config.system_prefix_tokens)
        if self.config.system_prefix_source == "random":
            self._system_prefix_content = self.tokenizer_manager.generate_random_content(
                self.config.system_prefix_tokens
            )
        else:
            self._system_prefix_content = self.tokenizer_manager.generate_content_from_file(
                self.config.system_prefix_source, self.config.system_prefix_tokens
            )

        logger.info("Generating user prefix content (%d tokens/user)...", self.config.user_prefix_tokens)
        for uid in range(self.config.num_users):
            self._user_prefix_contents[uid] = self.tokenizer_manager.generate_random_content(
                self.config.user_prefix_tokens
            )

        logger.info("Generating per-turn input (%d tokens)...", self.config.input_tokens_per_turn)
        self._turn_input_content = self.tokenizer_manager.generate_random_content(
            self.config.input_tokens_per_turn
        )

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

        # 5. Start system metrics poller
        if self.config.metrics_endpoint:
            self.system_poller = SystemMetricsPoller(
                endpoint=self.config.metrics_endpoint,
                interval=self.config.metrics_interval,
                backend=self.config.backend,
            )
            await self.system_poller.start()
            self._metrics_start = await self.system_poller.snapshot()
            logger.info("Metrics start snapshot: %s", self._metrics_start)

        # 6. Schedule users
        logger.info("Starting benchmark: %d users, arrival=%s", self.config.num_users, self.config.user_arrival)
        scheduler = get_scheduler(self.config)

        # Start tqdm progress bar (non-verbose mode)
        if not self.config.verbose:
            # Redirect clawperf logging through tqdm.write() so log messages
            # don't break the progress bar's single-line display.
            self._tqdm_handler = _TqdmLogHandler()
            existing = logger.handlers
            fmt = existing[0].formatter if existing else logging.Formatter(logging.BASIC_FORMAT)
            self._tqdm_handler.setFormatter(fmt)
            self._saved_handlers = existing[:]
            logger.handlers = [self._tqdm_handler]

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
                await asyncio.sleep(delay)
            if self._shutdown:
                break

            await self._add_timeline("user_joined", uid, time.monotonic() - self._start_time)
            task = asyncio.create_task(self._run_user_loop(uid), name=f"user-{uid}")
            self._user_tasks.append(task)

        # 7. Wait for completion
        if self._user_tasks and not self._shutdown:
            await asyncio.gather(*self._user_tasks, return_exceptions=True)

        # Snapshot metrics after all benchmark requests
        if self.system_poller and self.config.metrics_endpoint:
            self._metrics_end = await self.system_poller.snapshot()
            logger.info("Metrics end snapshot: %s", self._metrics_end)

        # 8. Close progress bar and restore logging
        if self._pbar:
            self._pbar.close()
        if self._tqdm_handler and self._saved_handlers is not None:
            logger.handlers = self._saved_handlers
            self._tqdm_handler = None
            self._saved_handlers = None

        # 9. Cleanup & save
        await self._finalize()

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
                evt.time = time.monotonic() - self._start_time
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
                logger.warning("[User %02d] Turn %d: failed to build request", user_id, turn_id)
                continue

            benchmark_data = await self._http_client.post(request_body)

            if benchmark_data.success:
                benchmark_data.finalize(self._api_plugin)

            self._accumulator.update(benchmark_data, self._api_plugin)

            turn_record = self._build_turn_record(
                user_id, turn_id, benchmark_data, context_tokens,
                turn_result["compaction_triggered"],
            )
            self._turn_records.append(turn_record)

            self._advance_progress(turn_record)

            if not benchmark_data.success:
                continue

            ctx.append_history(self._turn_input_content, benchmark_data.generated_text)
            await asyncio.sleep(0)

    def _advance_progress(self, turn_record: dict):
        """Update progress bar or print verbose turn line."""
        self._completed_turns += 1

        if self.config.verbose:
            self._print_verbose_turn(turn_record)
        elif self._pbar:
            success = turn_record.get("success", False)
            n_err = sum(1 for t in self._turn_records if not t.get("success"))
            self._pbar.set_postfix_str(f"err={n_err}")
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
        self, user_id: int, turn_id: int, bd, context_tokens: int, compaction: bool
    ) -> dict:
        record = {
            "user_id": user_id,
            "turn_id": turn_id,
            "success": bd.success,
            "context_tokens": context_tokens,
            "compaction_triggered": compaction,
        }

        if bd.success:
            record["ttft_ms"] = bd.first_chunk_latency * 1000 if bd.first_chunk_latency is not None else None
            record["e2e_latency_ms"] = bd.query_latency * 1000 if bd.query_latency is not None else None
            record["tpot_ms"] = bd.time_per_output_token * 1000 if bd.time_per_output_token is not None else None
            record["input_tokens"] = bd.prompt_tokens
            record["output_tokens"] = bd.completion_tokens

            if bd.inter_chunk_latency:
                sorted_itl = sorted(bd.inter_chunk_latency)
                record["itl_p50_ms"] = sorted_itl[int(len(sorted_itl) * 0.50)] * 1000
                record["itl_p99_ms"] = sorted_itl[min(int(len(sorted_itl) * 0.99), len(sorted_itl) - 1)] * 1000
        else:
            record["error"] = bd.error
            record["error_type"] = classify_error(bd)
            record["status_code"] = bd.status_code

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

        if success_turns and agg["total_output_tokens"] > 0:
            first_e2e = success_turns[0].get("e2e_latency_ms", 0) or 0
            first_ttft = success_turns[0].get("ttft_ms", 0) or 0
            last_e2e = success_turns[-1].get("e2e_latency_ms", 0) or 0
            user_duration_s = (last_e2e / 1000) if last_e2e > 0 else 0
            agg["duration_s"] = user_duration_s
            active_time_s = (last_e2e - first_ttft) / 1000 if last_e2e > first_ttft else last_e2e / 1000
            if active_time_s > 0:
                agg["throughput_tok_s"] = agg["total_output_tokens"] / active_time_s

        return agg

    async def _finalize(self):
        if self.system_poller:
            await self.system_poller.stop()

        if self._http_client:
            await self._http_client.client.close()

        wall_time_s = (time.monotonic() - self._start_time) if self._start_time > 0 else 0

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
        }

        import os
        out_dir = os.path.dirname(self.config.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(self.config.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        self._print_final_summary(wall_time_s)
        logger.info("Results saved to: %s", self.config.output)

    # ── Pretty output ──

    def _print_final_summary(self, wall_time_s: float):
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
        ct.add_row(["Duration", f"{wall_time_s:.2f} s"])
        ct.add_row(["Total Requests", str(total_reqs)])
        ct.add_row(["Success Requests", str(len(success_turns))])
        ct.add_row(["Failed Requests", str(len(error_turns))])
        ct.add_row(["Total Input Tokens", f"{total_in_tok:,}"])
        ct.add_row(["Prefill Token Throughput", f"{total_in_tok / wall_time_s:.2f} tok/s" if wall_time_s > 0 else "N/A"])
        ct.add_row(["Total Output Tokens", f"{total_out_tok:,}"])
        ct.add_row(["Request Throughput", f"{len(success_turns) / wall_time_s:.4f} req/s" if wall_time_s > 0 else "N/A"])
        ct.add_row(["Output Token Throughput", f"{total_out_tok / wall_time_s:.2f} tok/s" if wall_time_s > 0 else "N/A"])
        ct.add_row(["Total Token Throughput", f"{(total_in_tok + total_out_tok) / wall_time_s:.2f} tok/s" if wall_time_s > 0 else "N/A"])
        ct.add_row(["Total Compactions", str(total_comp)])
        if self._prefix_cache_delta:
            tok_rate = self._prefix_cache_delta.get("prefix_cache_token_hit_rate")
            if tok_rate is not None:
                ct.add_row(["HBM Prefix Cache Token Hit Rate", f"{tok_rate * 100:.2f}%"])
            hit_tok = self._prefix_cache_delta.get("prefix_cache_hit_tokens_delta", 0)
            q_tok = self._prefix_cache_delta.get("prefix_cache_query_tokens_delta", 0)
            ct.add_row(["HBM Prefix Cache Hit Tokens", f"{hit_tok:,}"])
            ct.add_row(["HBM Prefix Cache Query Tokens", f"{q_tok:,}"])
            if self._prefix_cache_delta["prefix_cache_evictions_delta"] > 0:
                ct.add_row(["HBM Prefix Cache Evictions", str(self._prefix_cache_delta["prefix_cache_evictions_delta"])])
            ext_tok_rate = self._prefix_cache_delta.get("external_prefix_cache_token_hit_rate")
            if ext_tok_rate is not None:
                ct.add_row(["External Prefix Cache Token Hit Rate", f"{ext_tok_rate * 100:.2f}%"])
            ext_hit_tok = self._prefix_cache_delta.get("external_prefix_cache_hit_tokens_delta", 0)
            ext_q_tok = self._prefix_cache_delta.get("external_prefix_cache_query_tokens_delta", 0)
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

        # ── Performance Table (avg/min/P50/P75/P90/P99/max/N) ──
        perf = PrettyTable()
        perf.field_names = ["Metric", "Avg", "Min", "P50", "P75", "P90", "P99", "Max", "N"]
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
                for key in ("avg", "min", "P50", "P75", "P90", "P99", "max", "N"):
                    v = pct.get(key)
                    if key == "N":
                        row.append(str(int(v)))
                    elif v is not None:
                        row.append(_fmt_val(v, unit, 2))
                    else:
                        row.append("N/A")
                perf.add_row(row)
            else:
                perf.add_row([name] + ["N/A"] * 8)

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
        print("=" * 70, flush=True)

    async def _add_timeline(self, event: str, user_id: int, time_offset: float, **kwargs):
        entry = {"event": event, "user_id": user_id, "time": round(time_offset, 3)}
        entry.update(kwargs)
        self._timeline_events.append(entry)

    def _setup_signal_handler(self):
        """Register SIGINT handler. Falls back to signal.signal on Windows."""
        def on_sigint():
            logger.info("SIGINT received. Initiating graceful shutdown...")
            self._shutdown = True
            for task in self._user_tasks:
                if not task.done():
                    task.cancel()
            if self._pbar:
                self._pbar.close()
            if self._tqdm_handler and self._saved_handlers is not None:
                logger.handlers = self._saved_handlers

        if platform.system() == "Windows":
            signal.signal(signal.SIGINT, lambda *_: on_sigint())
        else:
            try:
                asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_sigint)
            except NotImplementedError:
                signal.signal(signal.SIGINT, lambda *_: on_sigint())

    async def shutdown_and_save(self):
        self._shutdown = True
        for task in self._user_tasks:
            if not task.done():
                task.cancel()
        if self._pbar:
            self._pbar.close()
        if self._tqdm_handler and self._saved_handlers is not None:
            logger.handlers = self._saved_handlers
            self._tqdm_handler = None
            self._saved_handlers = None
        if self._user_tasks:
            await asyncio.gather(*self._user_tasks, return_exceptions=True)
        await self._finalize()