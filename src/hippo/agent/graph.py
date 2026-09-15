"""LangGraph orchestrator.

M0: run_once() — single ChatOpenAI call.
M1: run_agent() — LLM + tool loop (MCP or local filesystem).
M3: planner -> worker -> reviewer with retry/escalation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from hippo.agent.prompts import ONESHOT_SYSTEM, TOOL_SYSTEM
from hippo.trace import Tracer

MAX_TOOL_STEPS = 12


def _with_memory(system: str, recalled: list[str] | None) -> str:
    if not recalled:
        return system
    return system + "\n\n" + "\n".join(recalled)


def run_once(
    task: str,
    *,
    model: str,
    api_key: str,
    tracer: Tracer | None = None,
    recalled: list[str] | None = None,
) -> str:
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    msg = llm.invoke(
        [
            {"role": "system", "content": _with_memory(ONESHOT_SYSTEM, recalled)},
            {"role": "user", "content": task},
        ]
    )
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    if tracer:
        tracer.emit("llm", model=model, task=task[:200], chars=len(text))
    return text


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                parts.append(text if text else str(block))
        return "\n".join(parts)
    return str(content)


async def run_agent(
    task: str,
    *,
    model: str,
    api_key: str,
    tools: list[BaseTool],
    tracer: Tracer,
    max_steps: int = MAX_TOOL_STEPS,
    recalled: list[str] | None = None,
) -> str:
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    bound = llm.bind_tools(tools) if tools else llm
    by_name = {t.name: t for t in tools}
    messages: list[Any] = [
        SystemMessage(content=_with_memory(TOOL_SYSTEM, recalled)),
        HumanMessage(content=task),
    ]
    tracer.emit("run_start", task=task[:500], model=model, n_tools=len(tools))

    for step in range(max_steps):
        msg = await bound.ainvoke(messages)
        messages.append(msg)
        calls = getattr(msg, "tool_calls", None) or []
        names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", "") for c in calls]
        tracer.emit(
            "llm",
            step=step,
            chars=len(_content_text(getattr(msg, "content", ""))),
            tool_calls=names,
        )
        if not calls:
            text = _content_text(getattr(msg, "content", ""))
            tracer.emit("run_end", step=step, chars=len(text))
            return text or "(empty response)"

        for call in calls:
            name = call["name"] if isinstance(call, dict) else call.name
            args = call["args"] if isinstance(call, dict) else call.args
            call_id = call["id"] if isinstance(call, dict) else call.id
            tool = by_name.get(name)
            if tool is None:
                result = f"unknown tool: {name}"
            else:
                try:
                    result = await tool.ainvoke(args)
                except Exception as exc:  # noqa: BLE001
                    result = f"tool error: {exc}"
            messages.append(ToolMessage(content=str(result)[:12_000], tool_call_id=call_id))

    tracer.emit("run_end", step=max_steps, reason="max_steps")
    return "Stopped after the tool-call budget. Ask me to continue with a narrower task."


def build_graph():
    """M3: return compiled LangGraph. Not wired in M0/M1."""
    raise NotImplementedError("LangGraph planner/worker/reviewer lands in M3")
