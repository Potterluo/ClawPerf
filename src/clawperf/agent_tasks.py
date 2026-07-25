"""Tasks for the agent perf mode: format, presets, and workspace setup."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentTask:
    """One coding task for the agent perf benchmark.

    ``workspace`` maps relative path -> file content; it's materialized into a
    fresh temp dir per task instance so concurrent agents never collide.
    """
    id: int
    prompt: str
    workspace: Dict[str, str] = field(default_factory=dict)
    max_steps: int = 12


# ── Preset task bank ─────────────────────────────────────────────────────────
# Realistic-ish multi-turn agentic workloads (read → reason → edit → verify).
# Purely for exercising the agent loop; no accuracy grading.

PRESET_TASKS: List[AgentTask] = [
    AgentTask(
        id=0,
        prompt=(
            "In the workspace there is a Python file `calc.py` with a buggy "
            "`add` function. Read it, fix the bug so add(a,b) returns a+b, "
            "then run `python calc.py` to verify the test prints 5. When done, "
            "reply with a one-line summary."
        ),
        workspace={
            "calc.py": "# bug: subtracts instead of adding\ndef add(a, b):\n    return a - b\n\n"
                       "if __name__ == '__main__':\n    print(add(2, 3))\n",
        },
    ),
    AgentTask(
        id=1,
        prompt=(
            "Create a new file `stats.py` with a function `mean(xs)` that "
            "returns the arithmetic mean of a list of numbers. Add a main "
            "guard that prints mean([1,2,3,4]). Run `python stats.py` to "
            "verify it prints 2.5. Reply with a one-line summary when done."
        ),
        workspace={},
    ),
    AgentTask(
        id=2,
        prompt=(
            "The file `config.json` has a typo in a key (\"timout\" instead of "
            "\"timeout\"). Read it, fix the key name, write it back, and run "
            "`cat config.json` to confirm. Reply with a one-line summary."
        ),
        workspace={"config.json": '{"timout": 30, "retries": 3}\n'},
    ),
    AgentTask(
        id=3,
        prompt=(
            "List the files in the workspace, read `README.md`, then append a "
            "line '## Status' with 'benchmarked' under it. Read the file back "
            "to confirm. Reply with a one-line summary when done."
        ),
        workspace={"README.md": "# Project\n\nA small example project.\n"},
    ),
]


def load_tasks_from_file(path: str) -> List[AgentTask]:
    """Load custom tasks from a JSONL file (one task per line).

    Each line: {"prompt": "...", "workspace": {"path": "content"}, "max_steps": 12}
    """
    tasks: List[AgentTask] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(AgentTask(
                id=i,
                prompt=obj["prompt"],
                workspace=obj.get("workspace", {}),
                max_steps=obj.get("max_steps", 12),
            ))
    return tasks


def build_task_instances(
    base_tasks: List[AgentTask], count: int
) -> List[AgentTask]:
    """Materialize `count` task instances, cycling through the base bank and
    giving each a unique id (so concurrent agents have distinct workspaces)."""
    if not base_tasks:
        raise ValueError("no tasks available")
    out: List[AgentTask] = []
    for i in range(count):
        base = base_tasks[i % len(base_tasks)]
        out.append(AgentTask(
            id=i, prompt=base.prompt, workspace=dict(base.workspace),
            max_steps=base.max_steps,
        ))
    return out


def materialize_workspace(task: AgentTask, base_dir: str) -> str:
    """Write the task's workspace files into a fresh per-task subdir."""
    wd = os.path.join(base_dir, f"task_{task.id}")
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd, exist_ok=True)
    for rel, content in task.workspace.items():
        full = os.path.join(wd, rel)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    return wd
