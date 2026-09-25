"""Orchestration.

M0: run_once()  - single ChatOpenAI call.
M1: run_agent() - LLM + tool loop (MCP or local filesystem).
M3: build_graph() - LangGraph planner -> worker -> reviewer with one retry, one
    re-plan, then escalation to the user. Checkpointed in SQLite for `hippo resume`.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from hippo.agent.context import compress_messages
from hippo.agent.prompts import (
    FINALIZE_SYSTEM,
    ONESHOT_SYSTEM,
    PLANNER_SYSTEM,
    REPLAN_NOTE,
    REVIEWER_SYSTEM,
    TOOL_SYSTEM,
    WORKER_SYSTEM,
)
from hippo.agent.state import (
    MAX_REPLANS,
    MAX_WORKER_ATTEMPTS,
    AgentState,
    Review,
    RunContext,
    Subtask,
    tool_names,
)
from hippo.llm import make_chat
from hippo.metrics import UsageMeter
from hippo.tools.registry import ToolRegistry
from hippo.tools.search import Exposure, build_exposure
from hippo.trace import Tracer

MAX_TOOL_STEPS = 12
MIN_SUBTASK_STEPS = 6  # planners under-estimate; an edit needs read -> edit -> test -> answer

REPEAT_NUDGE = (
    "\n\n[hippo] You already made this exact call and got this exact result. Repeating it will "
    "not help. Read the message above, then change something: list the directory or read the "
    "file to get real paths and exact text, fix the argument names, or try another tool."
)

# Every worker gets these regardless of the planner's pick: reading the repo and running its
# tests are never the privilege the planner is deciding about. Writes/commits still must be
# granted per subtask (and by --write).
BASELINE_TOOLS = frozenset(
    {
        "read_file",
        "read_text_file",
        "read_multiple_files",
        "list_directory",
        "directory_tree",
        "search_files",
        "get_file_info",
        "list_dir",
        "run_pytest",
        "git_status",
        "git_diff",
        "git_diff_unstaged",
    }
)


def _short(name: str) -> str:
    return name.split("__")[-1]


def _log_tail(result: str, head: int = 160, tail: int = 240) -> str:
    """Keep the start (what) and the end (pytest summary / error line) of a tool result."""
    flat = result.replace(REPEAT_NUDGE, " [repeated call]")
    if len(flat) <= head + tail:
        return flat
    return flat[:head] + " ... " + flat[-tail:]


def _format_tool_log(log: list[dict[str, str]] | None) -> str:
    if not log:
        return "(no tool calls)"
    lines = []
    for i, row in enumerate(log, 1):
        lines.append(f"{i}. {row['tool']}({row['args']})\n   -> {row['result']}")
    return "\n".join(lines)


MAX_SUBTASKS = 4


# --------------------------------------------------------------------------- helpers


def _with_memory(system: str, recalled: list[str] | None) -> str:
    if not recalled:
        return system
    return system + "\n\n" + "\n".join(recalled)


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


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: whole text, fenced block, or first {...} span."""
    blob = text.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", blob, flags=re.S).strip()
    for candidate in (blob, _first_brace_span(blob)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _first_brace_span(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _make_llm(ctx: RunContext):
    if ctx.llm_factory is not None:
        return ctx.llm_factory(model=ctx.model, api_key=ctx.api_key)
    return make_chat(ctx.model, api_key=ctx.api_key, base_url=ctx.base_url)


def _summarizer(llm) -> Callable[[str], str]:
    from hippo.agent.context import COMPRESS_SYSTEM

    def _run(body: str) -> str:
        out = llm.invoke([SystemMessage(content=COMPRESS_SYSTEM), HumanMessage(content=body)])
        return _content_text(getattr(out, "content", out))

    return _run


# --------------------------------------------------------------------------- M0 / M1


def run_once(
    task: str,
    *,
    model: str,
    api_key: str,
    tracer: Tracer | None = None,
    recalled: list[str] | None = None,
    base_url: str | None = None,
) -> str:
    llm = make_chat(model, api_key=api_key, base_url=base_url)
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


async def tool_loop(
    llm,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    tracer: Tracer,
    max_steps: int,
    token_budget: int | None = None,
    summarize: Callable[[str], str] | None = None,
    model: str = "gpt-4o-mini",
    label: str = "run",
    tool_log: list[dict[str, str]] | None = None,
    tools_provider: Callable[[], list[BaseTool]] | None = None,
    meter: UsageMeter | None = None,
    on_unknown: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Shared LLM/tool loop. Returns (final_text, stop_reason).

    `max_steps` counts tool-calling rounds. When they are used up the model gets
    one last turn *without* tools, so a tight budget yields a partial answer
    instead of nothing (stop_reason "budget_answer").
    If `tool_log` is given, every call is appended as {tool, args, result} (truncated).

    The bound tool set is re-read from `tools_provider()` at every step (default: the
    fixed `tools` list), which is what lets tool search grow it mid-run. `meter` records
    what each request cost (schemas included); one is created if not given so the
    compression budget can count the schema block too.
    """
    provider = tools_provider or (lambda: tools)
    meter = meter or UsageMeter(model=model)
    seen_calls: dict[str, str] = {}  # "name:args" -> result, to catch verbatim retries

    for step in range(max_steps + 1):
        last_turn = step == max_steps
        current = [] if last_turn else list(provider())
        by_name = {t.name: t for t in current}
        bound = llm.bind_tools(current) if current else llm
        overhead = meter.schema_tokens(current) if current else 0
        if last_turn:
            messages.append(
                HumanMessage(
                    content="Tool budget is exhausted. Do not call tools. Answer now with what "
                    "you have observed so far, and say what is still unverified."
                )
            )
        if token_budget and summarize:
            messages, info = compress_messages(
                messages,
                summarize=summarize,
                token_budget=token_budget,
                model=model,
                fixed_overhead=overhead,
            )
            if info:
                tracer.emit("compress", label=label, step=step, **info)

        t0 = time.perf_counter()
        msg = await bound.ainvoke(messages)
        meter.llm_ms += (time.perf_counter() - t0) * 1000
        usage = meter.record(
            msg, messages_sent=messages, bound_tools=current, step=step, label=label
        )
        messages.append(msg)
        calls = [] if last_turn else (getattr(msg, "tool_calls", None) or [])
        names = [c.get("name") if isinstance(c, dict) else getattr(c, "name", "") for c in calls]
        tracer.emit(
            "llm",
            label=label,
            step=step,
            chars=len(_content_text(getattr(msg, "content", ""))),
            tool_calls=names,
        )
        tracer.emit(
            "usage",
            label=label,
            step=step,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read=usage.cache_read,
            tool_schema_tokens=usage.tool_schema_tokens,
            n_bound_tools=usage.n_bound_tools,
            estimated=usage.estimated,
        )
        if not calls:
            text = _content_text(getattr(msg, "content", ""))
            return (text or "(empty response)"), ("budget_answer" if last_turn else "answer")

        for call in calls:
            name = call["name"] if isinstance(call, dict) else call.name
            args = call["args"] if isinstance(call, dict) else call.args
            call_id = call["id"] if isinstance(call, dict) else call.id
            tool = by_name.get(name)
            if tool is None:
                result = on_unknown(name) if on_unknown else f"unknown tool: {name}"
                tracer.emit("unknown_tool", label=label, step=step, tool=name)
            else:
                try:
                    result = await tool.ainvoke(args)
                except Exception as exc:  # noqa: BLE001
                    result = f"tool error: {exc}"
            result = str(result)[:12_000]
            args_json = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
            key = f"{name}:{args_json}"
            if seen_calls.get(key) == result:
                result += REPEAT_NUDGE
                tracer.emit("repeat_call", label=label, step=step, tool=name)
            seen_calls[key] = result
            meter.record_tool_result(name, result)
            if tool_log is not None:
                tool_log.append(
                    {"tool": name, "args": args_json[:300], "result": _log_tail(result)}
                )
            messages.append(ToolMessage(content=result, tool_call_id=call_id))

    return "Stopped after the tool-call budget.", "max_steps"  # unreachable in practice


@dataclass
class AgentRun:
    """Everything a single-agent run produced; `hippo bench` scores these."""

    text: str
    reason: str
    messages: list[BaseMessage]
    meter: UsageMeter
    tool_log: list[dict[str, str]]
    exposure: Exposure


async def run_tool_agent(
    task: str,
    *,
    tools: list[BaseTool],
    tracer: Tracer,
    llm: Any = None,
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str | None = None,
    max_steps: int = MAX_TOOL_STEPS,
    recalled: list[str] | None = None,
    token_budget: int | None = None,
    exposure: str = "all",
    retriever: str = "keyword",
    search_k: int = 5,
    always_on: Iterable[str] = (),
    registry: ToolRegistry | None = None,
    index_dir: Path | None = None,
    max_active: int | None = None,
    label: str = "single",
) -> AgentRun:
    """Single-agent tool loop with a chosen tool exposure. Returns the full run record."""
    llm = llm or make_chat(model, api_key=api_key, base_url=base_url)
    exp = build_exposure(
        tools,
        mode=exposure,
        registry=registry,
        tracer=tracer,
        retriever=retriever,
        k=search_k,
        always_on=always_on,
        index_dir=index_dir,
        max_active=max_active,
    )
    system = _with_memory(TOOL_SYSTEM + exp.system_note, recalled)
    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=task)]
    meter = UsageMeter(model=model)
    tool_log: list[dict[str, str]] = []
    tracer.emit("run_start", task=task[:500], model=model, n_tools=len(tools), exposure=exp.mode)
    text, reason = await tool_loop(
        llm,
        tools,
        messages,
        tracer=tracer,
        max_steps=max_steps,
        token_budget=token_budget,
        summarize=_summarizer(llm) if token_budget else None,
        model=model,
        label=label,
        tool_log=tool_log,
        tools_provider=exp.tools_provider,
        meter=meter,
        on_unknown=exp.on_unknown,
    )
    tracer.emit("run_end", reason=reason, chars=len(text), **meter.as_dict())
    return AgentRun(text, reason, messages, meter, tool_log, exp)


async def run_agent(
    task: str,
    *,
    model: str,
    api_key: str,
    tools: list[BaseTool],
    tracer: Tracer,
    max_steps: int = MAX_TOOL_STEPS,
    recalled: list[str] | None = None,
    token_budget: int | None = None,
    base_url: str | None = None,
    exposure: str = "all",
    retriever: str = "keyword",
    search_k: int = 5,
    always_on: Iterable[str] = (),
    registry: ToolRegistry | None = None,
    index_dir: Path | None = None,
) -> str:
    """M1 single-agent loop (`hippo run --single`)."""
    run = await run_tool_agent(
        task,
        tools=tools,
        tracer=tracer,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_steps=max_steps,
        recalled=recalled,
        token_budget=token_budget,
        exposure=exposure,
        retriever=retriever,
        search_k=search_k,
        always_on=always_on,
        registry=registry,
        index_dir=index_dir,
    )
    if run.reason != "answer":
        return run.text + "\n\n(tool budget reached; ask me to continue with a narrower task)"
    return run.text


# --------------------------------------------------------------------------- M3 nodes


def _normalize_plan(raw: dict[str, Any] | None, available: list[str], task: str) -> list[Subtask]:
    items = (raw or {}).get("subtasks") if isinstance(raw, dict) else None
    plan: list[Subtask] = []
    if isinstance(items, list):
        for i, item in enumerate(items[:MAX_SUBTASKS]):
            if not isinstance(item, dict) or not str(item.get("goal") or "").strip():
                continue
            allowed = [t for t in (item.get("allowed_tools") or []) if t in available]
            try:
                budget = int(item.get("budget_steps") or 6)
            except (TypeError, ValueError):
                budget = 6
            plan.append(
                Subtask(
                    id=str(item.get("id") or f"s{i + 1}"),
                    goal=str(item["goal"]).strip(),
                    allowed_tools=allowed,
                    budget_steps=max(MIN_SUBTASK_STEPS, min(budget, MAX_TOOL_STEPS)),
                    status="pending",
                    attempts=0,
                )
            )
    if not plan:  # unparseable or empty: one subtask that is the whole task
        plan.append(
            Subtask(
                id="s1",
                goal=task,
                allowed_tools=[],
                budget_steps=MAX_TOOL_STEPS,
                status="pending",
                attempts=0,
            )
        )
    return plan


async def planner_node(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    llm = _make_llm(ctx)
    available = tool_names(ctx.tools)
    plan_before = list(state.get("plan") or [])
    done = [s for s in plan_before if s.get("status") == "done"]
    replans = int(state.get("replans") or 0)
    note = state.get("escalation_note") or ""
    history = list(state.get("history") or [])

    system = _with_memory(PLANNER_SYSTEM, state.get("recalled"))
    user = f"TASK:\n{state['task']}\n\nAVAILABLE TOOLS:\n" + ("\n".join(available) or "(none)")
    if note:
        replans += 1
        done_txt = "\n".join(
            f"- {s['id']}: {s['goal']} -> {s.get('result', '')[:300]}" for s in done
        )
        user += "\n\n" + REPLAN_NOTE.format(note=note, done=done_txt or "(none)")

    out = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = parse_json_object(_content_text(getattr(out, "content", out)))
    new_steps = _normalize_plan(raw, available, state["task"])
    # keep finished work, renumber the new tail so ids stay unique
    for i, s in enumerate(new_steps):
        s["id"] = f"s{len(done) + i + 1}"
    plan = done + new_steps
    ctx.tracer.emit(
        "plan",
        replan=bool(note),
        n_done=len(done),
        subtasks=[
            {"id": s["id"], "goal": s["goal"][:200], "tools": s["allowed_tools"]} for s in new_steps
        ],
    )
    history.append(
        ("re-planned" if note else "planned") + f": {[s['goal'][:80] for s in new_steps]}"
    )
    return {
        "plan": plan,
        "current": len(done),
        "replans": replans,
        "escalation_note": "",
        "review": None,
        "status": "working",
        "history": history,
    }


def _parse_worker_output(text: str) -> tuple[str, str, float]:
    data = parse_json_object(text)
    if data and str(data.get("result") or "").strip():
        try:
            conf = float(data.get("confidence") or 0.5)
        except (TypeError, ValueError):
            conf = 0.5
        return str(data["result"]).strip(), str(data.get("evidence") or "").strip(), conf
    return text.strip(), "", 0.5


async def worker_node(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    plan = [dict(s) for s in state["plan"]]
    idx = int(state.get("current") or 0)
    sub = plan[idx]
    allowed = set(sub.get("allowed_tools") or [])
    if allowed:
        tools = [t for t in ctx.tools if t.name in allowed or _short(t.name) in BASELINE_TOOLS]
    else:
        tools = list(ctx.tools)
    llm = _make_llm(ctx)
    # Tool exposure: with search modes the worker binds only meta-tools plus what it loads;
    # the baseline read tools stay always-on so it never spends a search on read_file.
    always_on = ctx.always_on if ctx.always_on is not None else BASELINE_TOOLS
    exp = build_exposure(
        tools,
        mode=ctx.exposure,
        tracer=ctx.tracer,
        retriever=ctx.retriever,
        k=ctx.search_k,
        always_on=always_on,
        index_dir=ctx.index_dir,
    )

    prior = []
    for s in plan[:idx]:
        if s.get("status") != "done":
            continue
        line = f"- {s['id']} ({s['goal'][:120]}): {s.get('result', '')[:500]}"
        if s.get("evidence"):
            line += f"\n  evidence: {s['evidence'][:400]}"
        prior.append(line)
    system = _with_memory(WORKER_SYSTEM + exp.system_note, state.get("recalled"))
    user = (
        f"OVERALL TASK (for context only):\n{state['task']}\n\n"
        f"YOUR SUBTASK ({sub['id']}):\n{sub['goal']}"
    )
    if ctx.workspace_tree:
        user += (
            f"\n\nWORKSPACE FILES (relative paths; use these, do not guess):\n{ctx.workspace_tree}"
        )
    if prior:
        user += "\n\nRESULTS OF EARLIER SUBTASKS:\n" + "\n".join(prior)
    if sub.get("feedback"):
        user += f"\n\nREVIEWER FEEDBACK ON YOUR PREVIOUS ATTEMPT (fix this):\n{sub['feedback']}"

    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]
    ctx.tracer.emit(
        "worker_start",
        subtask=sub["id"],
        attempt=int(sub.get("attempts") or 0) + 1,
        n_tools=len(tools),
        exposure=exp.mode,
    )
    tool_log: list[dict[str, str]] = []
    meter = UsageMeter(model=ctx.model)
    text, reason = await tool_loop(
        llm,
        tools,
        messages,
        tracer=ctx.tracer,
        max_steps=int(sub.get("budget_steps") or 6),
        token_budget=ctx.token_budget,
        summarize=_summarizer(llm),
        model=ctx.model,
        label=sub["id"],
        tool_log=tool_log,
        tools_provider=exp.tools_provider,
        meter=meter,
        on_unknown=exp.on_unknown,
    )
    result, evidence, confidence = _parse_worker_output(text)
    if reason != "answer":
        evidence = (evidence + " | answered at tool budget").strip(" |")
        confidence = min(confidence, 0.6)
    sub.update(
        attempts=int(sub.get("attempts") or 0) + 1,
        result=result,
        evidence=evidence,
        confidence=confidence,
        tool_log=tool_log[-10:],
    )
    plan[idx] = sub
    ctx.tracer.emit(
        "worker_end",
        subtask=sub["id"],
        reason=reason,
        confidence=confidence,
        chars=len(result),
        **meter.as_dict(),
    )
    history = list(state.get("history") or [])
    history.append(f"{sub['id']} attempt {sub['attempts']}: {result[:120]}")
    return {"plan": plan, "status": "reviewing", "history": history}


async def reviewer_node(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    plan = [dict(s) for s in state["plan"]]
    idx = int(state.get("current") or 0)
    sub = plan[idx]
    llm = _make_llm(ctx)

    user = (
        f"SUBTASK GOAL:\n{sub['goal']}\n\nWORKER RESULT:\n{sub.get('result', '')}\n\n"
        f"WORKER EVIDENCE (self-reported):\n{sub.get('evidence', '') or '(none given)'}\n\n"
        f"TOOL LOG (ground truth - what the worker actually ran and saw):\n"
        f"{_format_tool_log(sub.get('tool_log'))}\n\n"
        f"WORKER CONFIDENCE: {sub.get('confidence', 0.5)}\n"
        f"ATTEMPT: {sub.get('attempts', 1)} of {MAX_WORKER_ATTEMPTS}"
    )
    out = await llm.ainvoke([SystemMessage(content=REVIEWER_SYSTEM), HumanMessage(content=user)])
    data = parse_json_object(_content_text(getattr(out, "content", out))) or {}
    verdict = str(data.get("verdict") or "approve").lower()
    feedback = str(data.get("feedback") or "").strip()
    if verdict not in {"approve", "revise", "escalate"}:
        verdict = "approve"
    if verdict == "revise" and int(sub.get("attempts") or 0) >= MAX_WORKER_ATTEMPTS:
        verdict = "escalate"
        feedback = (
            f"{sub['id']} still not acceptable after {MAX_WORKER_ATTEMPTS} attempts: {feedback}"
        )

    current = idx
    note = ""
    if verdict == "approve":
        sub["status"] = "done"
        current = idx + 1
    elif verdict == "revise":
        sub["feedback"] = feedback
    else:
        sub["status"] = "failed"
        sub["feedback"] = feedback
        note = feedback or f"{sub['id']} failed"
    plan[idx] = sub
    ctx.tracer.emit("review", subtask=sub["id"], verdict=verdict, feedback=feedback[:300])
    history = list(state.get("history") or [])
    history.append(f"review {sub['id']}: {verdict} - {feedback[:100]}")
    return {
        "plan": plan,
        "current": current,
        "review": Review(verdict=verdict, feedback=feedback, subtask_id=sub["id"]),  # type: ignore[typeddict-item]
        "escalation_note": note,
        "status": "working",
        "history": history,
    }


def route_after_review(state: AgentState) -> str:
    verdict = (state.get("review") or {}).get("verdict", "approve")
    if verdict == "revise":
        return "worker"
    if verdict == "escalate":
        return "planner" if int(state.get("replans") or 0) < MAX_REPLANS else "finalize"
    if int(state.get("current") or 0) < len(state.get("plan") or []):
        return "worker"
    return "finalize"


async def finalize_node(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    plan = state.get("plan") or []
    failed = [s for s in plan if s.get("status") == "failed"]
    pending = [s for s in plan if s.get("status") == "pending"]
    if len(plan) == 1 and plan[0].get("status") == "done":
        final = plan[0].get("result") or ""
    else:
        llm = _make_llm(ctx)
        lines = []
        for s in plan:
            lines.append(
                f"[{s['id']} {s.get('status')}] goal: {s['goal']}\n"
                f"  result: {s.get('result', '')}\n  evidence: {s.get('evidence', '')}"
            )
        user = f"TASK:\n{state['task']}\n\nSUBTASKS:\n" + "\n".join(lines)
        out = await llm.ainvoke(
            [SystemMessage(content=FINALIZE_SYSTEM), HumanMessage(content=user)]
        )
        final = _content_text(getattr(out, "content", out)).strip()
    if failed or pending:
        final += (
            "\n\n(escalated: "
            + ", ".join(f"{s['id']} {s.get('status')}" for s in failed + pending)
            + " - I could not finish these; narrow the task, "
            + "or run with --write if files must change)"
        )
    status = "failed" if (failed or pending) else "done"
    ctx.tracer.emit("finalize", status=status, n_failed=len(failed), chars=len(final))
    return {"final": final, "status": status}


def build_graph(checkpointer=None):
    """planner -> worker -> reviewer -> (worker | planner | finalize)."""
    g = StateGraph(AgentState, context_schema=RunContext)
    g.add_node("planner", planner_node)
    g.add_node("worker", worker_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("finalize", finalize_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "worker")
    g.add_edge("worker", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"worker": "worker", "planner": "planner", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def initial_state(task: str, recalled: list[str] | None = None) -> AgentState:
    return AgentState(
        task=task,
        recalled=list(recalled or []),
        plan=[],
        current=0,
        review=None,
        replans=0,
        escalation_note="",
        final="",
        status="planning",
        history=[],
    )
