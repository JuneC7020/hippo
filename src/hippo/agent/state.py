"""LangGraph state for the planner -> worker -> reviewer orchestrator (M3).

`AgentState` is checkpointed (SQLite) so `hippo resume TASK_ID` can continue.
Everything non-serializable (tools, tracer, llm factory) travels in `RunContext`
via LangGraph's runtime context, which is *not* checkpointed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.tools import BaseTool

from hippo.trace import Tracer

SubtaskStatus = Literal["pending", "done", "failed"]
Verdict = Literal["approve", "revise", "escalate"]
RunStatus = Literal["planning", "working", "reviewing", "done", "failed"]

MAX_WORKER_ATTEMPTS = 2  # first try + one retry, then escalate to the planner
MAX_REPLANS = 1  # one re-plan, then give the user what we have


class Subtask(TypedDict, total=False):
    id: str
    goal: str
    allowed_tools: list[str]
    budget_steps: int
    status: SubtaskStatus
    attempts: int
    result: str
    evidence: str
    confidence: float
    feedback: str  # last reviewer feedback, fed back into the retry


class Review(TypedDict, total=False):
    verdict: Verdict
    feedback: str
    subtask_id: str


class AgentState(TypedDict, total=False):
    task: str
    recalled: list[str]
    plan: list[Subtask]
    current: int  # index into plan
    review: Review | None
    replans: int
    escalation_note: str  # why the planner is being asked to re-plan
    final: str
    status: RunStatus
    history: list[str]  # human-readable log of what happened, used by finalize/resume


@dataclass
class RunContext:
    """Per-run objects. Not serialized; supplied on every (re)invocation."""

    tools: list[BaseTool]
    tracer: Tracer
    model: str
    api_key: str
    token_budget: int = 8000
    llm_factory: Callable[..., Any] | None = None  # tests inject a fake LLM here
    extra: dict[str, Any] = field(default_factory=dict)


def tool_names(tools: list[BaseTool]) -> list[str]:
    return [t.name for t in tools]
