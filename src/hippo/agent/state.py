from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str
    goal: str
    allowed_tools: list[str] = Field(default_factory=list)
    budget_tokens: int = 2000
    result: str | None = None
    evidence: str | None = None
    confidence: float | None = None


class Task(BaseModel):
    id: str
    goal: str
    subtasks: list[SubTask] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    task: str
    messages: list[dict]
    subtasks: list[dict]
    review: str
    retries: int
    summary: str
    recalled: list[str]


Role = Literal["planner", "worker", "reviewer"]
