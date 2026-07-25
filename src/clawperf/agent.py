"""Real agent-at-work performance harness (mode 4: ``--mode agent``).

Borrows the clean agent loop + tool registry from cookllm-harness but makes it
async, streaming (for TTFT), concurrency-ready, and perf-instrumented. The model
really runs as a coding agent (read/write files, run shell) across multi-turn
growing context — this is the workload whose perf we measure.

Loop:  prompt -> stream-complete (TTFT + usage) -> (tool_calls? execute -> loop)
       -> (no tool_calls? done). Terminates on a plain-text reply or max_steps.

NOT accuracy grading — purely performance (latency, tokens, throughput, and the
real prefix-cache hit rate from the server's Prometheus counters).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clawperf")

MAX_TOOL_OUTPUT = 4000  # clip tool results so they don't blow the context


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[clipped {len(text) - limit} chars]"


# ── Tool registry ────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    ok: bool
    output: str


class AgentTools:
    """File/shell tools the agent uses, scoped to a working directory.

    Tools return strings (clipped); errors are returned as strings too so the
    model can recover — never raised into the loop.
    """

    SPECS = [
        {"type": "function", "function": {
            "name": "list_files",
            "description": "List files in the working directory (top level).",
            "parameters": {"type": "object", "properties": {}}},
        },
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file's contents by relative path.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Relative file path."}},
                "required": ["path"]}},
        },
        {"type": "function", "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}},
                "required": ["path", "content"]}},
        },
        {"type": "function", "function": {
            "name": "run_shell",
            "description": "Run a shell command in the working directory.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "Shell command to execute."}},
                "required": ["command"]}},
        },
    ]

    def __init__(self, root: str, shell_timeout: int = 30):
        self.root = os.path.abspath(root)
        self.shell_timeout = shell_timeout
        os.makedirs(self.root, exist_ok=True)

    def _safe_path(self, rel: str) -> str:
        full = os.path.abspath(os.path.join(self.root, rel))
        if not full.startswith(self.root) and full != self.root:
            raise ValueError(f"path escapes workspace: {rel}")
        return full

    async def run(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "list_files":
                return _clip("\n".join(sorted(os.listdir(self.root))) or "(empty)")
            if name == "read_file":
                with open(self._safe_path(args["path"]), encoding="utf-8") as f:
                    return _clip(f.read())
            if name == "write_file":
                p = self._safe_path(args["path"])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(args["content"])
                return f"wrote {len(args['content'])} chars to {args['path']}"
            if name == "run_shell":
                cmd = args["command"]
                proc = await asyncio.create_subprocess_shell(
                    cmd, cwd=self.root,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                try:
                    out, err = await asyncio.wait_for(
                        proc.communicate(), timeout=self.shell_timeout
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return f"error: shell command timed out after {self.shell_timeout}s"
                text = (out or b"").decode("utf-8", "replace")
                if err:
                    text += ("\n[stderr]\n" if text else "") + err.decode("utf-8", "replace")
                return _clip(text)
            return f"error: unknown tool '{name}'"
        except Exception as e:
            return f"error: tool {name} failed: {e}"


# ── Async streaming LLM client (captures TTFT + usage) ───────────────────────

@dataclass
class TurnResult:
    """One LLM turn's perf + content."""
    ttft_ms: Optional[float]        # time to first content chunk
    e2e_ms: float                   # full turn wall time
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    content: str
    tool_calls: List[Dict]           # [{id, name, arguments}]
    finish_reason: Optional[str]


class AgentClient:
    """Async streaming OpenAI-compatible client that records per-turn metrics.

    Lazily imports ``openai`` (only needed in agent mode).
    """

    def __init__(self, endpoint: str, model: str, api_key: str = "",
                 timeout: int = 600, max_tokens: int = 512):
        self.base_url = endpoint.rstrip("/").rsplit("/v1", 1)[0] + "/v1"
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client = None

    def _get(self):
        if self._client is None:
            import httpx
            from openai import AsyncOpenAI
            # trust_env=False so requests to a localhost vLLM don't get routed
            # through a corporate HTTP(S)_PROXY (which returns 502 for localhost).
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout,
                http_client=httpx.AsyncClient(trust_env=False, timeout=self.timeout),
            )
        return self._client

    async def complete(self, messages: List[Dict], tools: List[Dict]) -> TurnResult:
        client = self._get()
        t0 = time.perf_counter()
        ttft = None
        content_parts: List[str] = []
        # tool_calls arrive fragmented across chunks; accumulate per-index.
        tc_acc: Dict[int, Dict] = {}
        finish = None
        usage = None
        kwargs = dict(
            model=self.model, messages=messages, max_tokens=self.max_tokens,
            tools=tools if tools else None, stream=True,
            stream_options={"include_usage": True},
        )
        # Drop None values (e.g. tools when absent) so the backend doesn't 400.
        clean = {k: v for k, v in kwargs.items() if v is not None}
        stream = await client.chat.completions.create(**clean)
        async for chunk in stream:
            if ttft is None:
                # first chunk = request completed + first token on the wire
                ttft = (time.perf_counter() - t0) * 1000
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta and delta.content:
                content_parts.append(delta.content)
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index or 0
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
        e2e = (time.perf_counter() - t0) * 1000

        tool_calls = [
            {"id": v["id"] or f"call_{i}", "name": v["name"], "arguments": v["args"]}
            for i, v in sorted(tc_acc.items())
        ]
        return TurnResult(
            ttft_ms=ttft, e2e_ms=e2e,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish,
        )


# ── The agent loop ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an autonomous coding agent operating in a workspace. Use the "
    "provided tools (list_files, read_file, write_file, run_shell) to accomplish "
    "the task. Call tools to inspect and modify files; when the task is done, "
    "reply with a short plain-text summary (no tool call) to finish."
)


@dataclass
class AgentTurnRecord:
    turn: int
    ttft_ms: Optional[float]
    e2e_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    tool_calls: int
    tool_time_ms: float
    finish_reason: Optional[str]


@dataclass
class AgentRunResult:
    task_id: int
    steps: int
    finished: bool              # True = model gave a final text answer; False = hit step cap
    total_wall_s: float
    total_input_tokens: int
    total_output_tokens: int
    turns: List[AgentTurnRecord] = field(default_factory=list)


class Agent:
    """One agent run on one task. Async, streaming, instrumented."""

    def __init__(self, client: AgentClient, tools: AgentTools, max_steps: int = 12):
        self.client = client
        self.tools = tools
        self.max_steps = max_steps

    async def run(self, task_id: int, prompt: str) -> AgentRunResult:
        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        t0 = time.perf_counter()
        turns: List[AgentTurnRecord] = []
        finished = False
        total_in = total_out = 0

        for step in range(1, self.max_steps + 1):
            tr = await self.client.complete(messages, AgentTools.SPECS)
            total_in += tr.input_tokens or 0
            total_out += tr.output_tokens or 0

            # Echo the assistant turn back (OpenAI function-calling shape).
            asst: Dict[str, Any] = {"role": "assistant", "content": tr.content or ""}
            if tr.tool_calls:
                asst["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tr.tool_calls
                ]
            messages.append(asst)

            tool_t0 = time.perf_counter()
            for tc in tr.tool_calls:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                out = await self.tools.run(tc["name"], args)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            tool_ms = (time.perf_counter() - tool_t0) * 1000

            turns.append(AgentTurnRecord(
                turn=step, ttft_ms=tr.ttft_ms, e2e_ms=tr.e2e_ms,
                input_tokens=tr.input_tokens, output_tokens=tr.output_tokens,
                tool_calls=len(tr.tool_calls), tool_time_ms=tool_ms,
                finish_reason=tr.finish_reason,
            ))

            if not tr.tool_calls:
                finished = True
                break

        return AgentRunResult(
            task_id=task_id, steps=len(turns), finished=finished,
            total_wall_s=time.perf_counter() - t0,
            total_input_tokens=total_in, total_output_tokens=total_out,
            turns=turns,
        )
