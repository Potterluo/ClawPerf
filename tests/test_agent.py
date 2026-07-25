"""Tests for the agent loop and tools (deterministic FakeClient, no real LLM)."""

from __future__ import annotations

import os
from typing import List

import pytest

from clawperf.agent import Agent, AgentTools, TurnResult


class FakeClient:
    """Replays a canned sequence of TurnResults (no network)."""

    def __init__(self, replies: List[TurnResult]):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, tools):
        r = self.replies.pop(0)
        self.calls += 1
        return r


def _turn(content="", tool_calls=None, in_tok=10, out_tok=5):
    return TurnResult(
        ttft_ms=1.0, e2e_ms=2.0, input_tokens=in_tok, output_tokens=out_tok,
        content=content, tool_calls=tool_calls or [], finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_agent_loop_reads_then_finishes(tmp_path):
    """Agent reads a file (tool call), then gives a final text answer."""
    # workspace with a file
    (tmp_path / "hello.txt").write_text("world", encoding="utf-8")
    tools = AgentTools(str(tmp_path))
    # turn 1: model calls read_file; turn 2: model says "done"
    client = FakeClient([
        _turn(tool_calls=[{"id": "c1", "name": "read_file", "arguments": '{"path": "hello.txt"}'}]),
        _turn(content="the file says world"),
    ])
    agent = Agent(client, tools, max_steps=5)
    res = await agent.run(0, "what is in hello.txt?")
    assert res.steps == 2
    assert res.finished is True
    assert res.total_input_tokens == 20
    assert res.total_output_tokens == 10
    assert len(res.turns) == 2
    assert res.turns[0].tool_calls == 1
    assert res.turns[1].tool_calls == 0


@pytest.mark.asyncio
async def test_agent_loop_step_cap(tmp_path):
    """If the model keeps calling tools, the step cap terminates the loop."""
    tools = AgentTools(str(tmp_path))
    client = FakeClient([
        _turn(tool_calls=[{"id": f"c{i}", "name": "list_files", "arguments": "{}"}])
        for i in range(10)
    ])
    agent = Agent(client, tools, max_steps=3)
    res = await agent.run(0, "loop forever")
    assert res.steps == 3
    assert res.finished is False  # hit the cap, no final answer


@pytest.mark.asyncio
async def test_agent_write_and_run_shell(tmp_path):
    """write_file then run_shell actually execute against the workspace."""
    tools = AgentTools(str(tmp_path))
    client = FakeClient([
        _turn(tool_calls=[{"id": "c1", "name": "write_file",
                           "arguments": '{"path": "out.txt", "content": "hi"}'}]),
        _turn(tool_calls=[{"id": "c2", "name": "run_shell",
                           "arguments": '{"command": "cat out.txt"}'}]),
        _turn(content="done"),
    ])
    await Agent(client, tools, max_steps=5).run(0, "write and read back")
    assert (tmp_path / "out.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_agent_tool_path_escape_blocked(tmp_path):
    """Tool paths escaping the workspace are rejected."""
    tools = AgentTools(str(tmp_path))
    out = await tools.run("read_file", {"path": "../../etc/passwd"})
    assert "error" in out


@pytest.mark.asyncio
async def test_agent_tool_unknown(tmp_path):
    tools = AgentTools(str(tmp_path))
    out = await tools.run("nope", {})
    assert "unknown tool" in out


@pytest.mark.asyncio
async def test_agent_tool_shell_timeout(tmp_path):
    tools = AgentTools(str(tmp_path), shell_timeout=1)
    out = await tools.run("run_shell", {"command": "sleep 10"})
    assert "timed out" in out


def test_agent_config_validate():
    from clawperf.config import BenchmarkConfig

    cfg = BenchmarkConfig(endpoint="http://x", model="m", mode="agent", agent_tasks=2)
    assert cfg.validate() == []
    bad = BenchmarkConfig(endpoint="http://x", model="m", mode="agent", agent_tasks=0)
    assert any("agent_tasks" in p for p in bad.validate())


def test_agent_tasks_presets_and_materialize(tmp_path):
    from clawperf.agent_tasks import PRESET_TASKS, build_task_instances, materialize_workspace

    assert len(PRESET_TASKS) >= 3
    instances = build_task_instances(PRESET_TASKS, 6)
    assert len(instances) == 6
    assert len({t.id for t in instances}) == 6  # unique ids
    wd = materialize_workspace(PRESET_TASKS[0], str(tmp_path))
    assert os.path.isdir(wd)
    assert os.path.exists(os.path.join(wd, "calc.py"))


def test_agent_tasks_load_file(tmp_path):
    from clawperf.agent_tasks import load_tasks_from_file

    f = tmp_path / "tasks.jsonl"
    f.write_text(
        '{"prompt": "do X", "workspace": {"a.py": "print(1)"}, "max_steps": 3}\n'
        '{"prompt": "do Y"}\n',
        encoding="utf-8",
    )
    tasks = load_tasks_from_file(str(f))
    assert len(tasks) == 2
    assert tasks[0].prompt == "do X"
    assert tasks[0].workspace == {"a.py": "print(1)"}
    assert tasks[0].max_steps == 3
    assert tasks[1].max_steps == 12  # default
