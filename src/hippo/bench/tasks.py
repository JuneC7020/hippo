"""Benchmark tasks and their deterministic scoring.

A task names the tool(s) it should end in and substrings the call's arguments must contain.
Scoring never asks a model: `expected_tool_called` and `args_ok` come from the recorded calls,
`recall_at_k` from what `tool_search` actually returned. That separates "the retriever never
surfaced it" from "the model saw it and picked wrong" (docs/token-bottlenecks.md B8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TASKS = Path(__file__).resolve().parents[3] / "benchmarks" / "tasks" / "synthetic.json"


@dataclass
class Task:
    id: str
    prompt: str
    expected_tools: list[str]
    expected_args_contains: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            id=str(d["id"]),
            prompt=str(d["prompt"]),
            expected_tools=[str(x) for x in d.get("expected_tools") or []],
            expected_args_contains={
                str(k): str(v) for k, v in (d.get("expected_args_contains") or {}).items()
            },
        )


def load_tasks(path: Path | str | None = None) -> list[Task]:
    p = Path(path) if path else DEFAULT_TASKS
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("tasks") if isinstance(data, dict) else data
    return [Task.from_dict(t) for t in items or []]


def required_tools(tasks: Iterable[Task]) -> list[str]:
    """Every tool any task expects, in first-seen order (the benchmark subset must contain them)."""
    seen: list[str] = []
    for t in tasks:
        for name in t.expected_tools:
            if name not in seen:
                seen.append(name)
    return seen


@dataclass
class Score:
    expected_tool_called: bool
    args_ok: bool
    success: bool
    recall_at_k: float | None  # None when no tool_search happened (mode all)
    n_tool_calls: int
    n_search_calls: int
    n_load_calls: int
    n_unknown: int
    matched_call: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["matched_call"] = (
            json.dumps(self.matched_call, ensure_ascii=False, default=str)[:300]
            if self.matched_call
            else None
        )
        return d


def _contains(value: Any, needle: str) -> bool:
    return needle.lower() in json.dumps(value, ensure_ascii=False, default=str).lower()


def score_task(
    task: Task,
    calls: Sequence[dict[str, Any]],
    search_events: Sequence[dict[str, Any]],
    *,
    n_load_calls: int = 0,
    n_unknown: int = 0,
) -> Score:
    """`calls` are {tool, args} for real tools; `search_events` are tool_search trace rows."""
    expected = set(task.expected_tools)
    matched: dict[str, Any] | None = None
    args_ok = False
    for call in calls:
        if call.get("tool") not in expected:
            continue
        args = call.get("args") or {}
        ok = all(_contains(args.get(k), v) for k, v in task.expected_args_contains.items())
        if matched is None or (ok and not args_ok):
            matched, args_ok = call, ok
        if ok:
            break
    called = matched is not None
    recall: float | None = None
    if search_events:
        seen: set[str] = set()
        for ev in search_events:
            seen.update(ev.get("hits") or [])
        recall = len(expected & seen) / len(expected) if expected else 1.0
    return Score(
        expected_tool_called=called,
        args_ok=called and args_ok,
        success=called and args_ok,
        recall_at_k=recall,
        n_tool_calls=len(calls),
        n_search_calls=len(search_events),
        n_load_calls=n_load_calls,
        n_unknown=n_unknown,
        matched_call=matched,
    )
