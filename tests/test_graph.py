"""Planner/worker/reviewer graph with a scripted fake LLM. No network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fakes import FakeLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from hippo.agent import graph as g
from hippo.agent.state import RunContext
from hippo.trace import Tracer


class EchoArgs(BaseModel):
    text: str


def _tools() -> list[StructuredTool]:
    async def echo(text: str) -> str:
        return f"echo:{text}"

    return [
        StructuredTool.from_function(
            coroutine=echo, name="local__echo", description="echo", args_schema=EchoArgs
        )
    ]


def _ctx(tmp_path: Path, llm: FakeLLM) -> RunContext:
    return RunContext(
        tools=_tools(),
        tracer=Tracer(tmp_path / "trace.jsonl"),
        model="fake",
        api_key="x",
        token_budget=100_000,
        llm_factory=lambda **_: llm,
    )


def _plan(*goals: str) -> str:
    return json.dumps(
        {
            "subtasks": [
                {
                    "id": f"s{i + 1}",
                    "goal": goal,
                    "allowed_tools": ["local__echo"],
                    "budget_steps": 3,
                }
                for i, goal in enumerate(goals)
            ]
        }
    )


def _worker(result: str, conf: float = 0.9) -> str:
    return json.dumps({"result": result, "evidence": "saw it", "confidence": conf})


def _review(verdict: str, feedback: str = "") -> str:
    return json.dumps({"verdict": verdict, "feedback": feedback})


def _run(tmp_path: Path, llm: FakeLLM, task: str = "do the thing"):
    graph = g.build_graph()
    return asyncio.run(graph.ainvoke(g.initial_state(task), context=_ctx(tmp_path, llm)))


def test_single_subtask_approved_returns_worker_result(tmp_path: Path) -> None:
    llm = FakeLLM(
        {
            "planner": [_plan("answer it")],
            "worker": [_worker("42")],
            "reviewer": [_review("approve")],
        }
    )
    out = _run(tmp_path, llm)
    assert out["status"] == "done"
    assert out["final"] == "42"
    assert llm.calls == ["planner", "worker", "reviewer"]  # finalize skipped for 1 subtask


def test_revise_retries_once_with_feedback_then_approves(tmp_path: Path) -> None:
    llm = FakeLLM(
        {
            "planner": [_plan("check pyproject")],
            "worker": [_worker("guess", 0.4), _worker("ruff", 0.9)],
            "reviewer": [_review("revise", "you did not open the file"), _review("approve")],
        }
    )
    out = _run(tmp_path, llm)
    assert out["status"] == "done"
    assert out["final"] == "ruff"
    assert out["plan"][0]["attempts"] == 2
    assert llm.calls == ["planner", "worker", "reviewer", "worker", "reviewer"]


def test_two_revises_escalate_and_replan_once(tmp_path: Path) -> None:
    llm = FakeLLM(
        {
            "planner": [_plan("bad approach"), _plan("good approach")],
            "worker": [_worker("a", 0.2), _worker("b", 0.2), _worker("c", 0.9)],
            "reviewer": [
                _review("revise", "no"),
                _review("revise", "still no"),
                _review("approve"),
            ],
            "final answer": ["Final: c"],
        }
    )
    out = _run(tmp_path, llm)
    # the failed approach is replaced by the re-plan; the run as a whole succeeds
    assert out["status"] == "done"
    assert out["replans"] == 1
    assert [s["status"] for s in out["plan"]] == ["done"]
    assert out["final"] == "c"
    assert any(h.startswith("re-planned") for h in out["history"])
    assert llm.calls.count("planner") == 2 and llm.calls.count("worker") == 3


def test_escalate_after_replan_budget_goes_to_user(tmp_path: Path) -> None:
    llm = FakeLLM(
        {
            "planner": [_plan("x"), _plan("y")],
            "worker": [_worker("a"), _worker("b")],
            "reviewer": [
                _review("escalate", "impossible"),
                _review("escalate", "still impossible"),
            ],
            "final answer": ["could not do it"],
        }
    )
    out = _run(tmp_path, llm)
    assert out["status"] == "failed"
    assert out["replans"] == 1
    assert llm.calls.count("planner") == 2  # no third plan


def test_multi_subtask_uses_finalizer_and_passes_prior_results(tmp_path: Path) -> None:
    llm = FakeLLM(
        {
            "planner": [_plan("first", "second")],
            "worker": [_worker("one"), _worker("two")],
            "reviewer": [_review("approve"), _review("approve")],
            "final answer": ["one and two"],
        }
    )
    out = _run(tmp_path, llm)
    assert out["status"] == "done"
    assert out["final"] == "one and two"
    assert llm.calls[-1] == "final answer"


def test_unparseable_plan_falls_back_to_whole_task(tmp_path: Path) -> None:
    llm = FakeLLM(
        {"planner": ["I cannot plan"], "worker": [_worker("ok")], "reviewer": [_review("approve")]}
    )
    out = _run(tmp_path, llm, task="whole task")
    assert out["plan"][0]["goal"] == "whole task"
    assert out["final"] == "ok"


def test_worker_tool_call_then_answer(tmp_path: Path) -> None:
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "local__echo", "args": {"text": "hi"}, "id": "c1", "type": "tool_call"}
        ],
    )
    llm = FakeLLM(
        {
            "planner": [_plan("use the tool")],
            "worker": [call, _worker("echo:hi")],
            "reviewer": [_review("approve")],
        }
    )
    out = _run(tmp_path, llm)
    assert out["final"] == "echo:hi"
    kinds = [r["kind"] for r in Tracer(tmp_path / "trace.jsonl").read()]
    assert "plan" in kinds and "worker_start" in kinds and "review" in kinds


def test_repeated_identical_tool_call_gets_nudged(tmp_path: Path) -> None:
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "local__echo", "args": {"text": "same"}, "id": "c1", "type": "tool_call"}
        ],
    )
    call2 = AIMessage(
        content="",
        tool_calls=[
            {"name": "local__echo", "args": {"text": "same"}, "id": "c2", "type": "tool_call"}
        ],
    )
    llm = FakeLLM({"worker": [call, call2, _worker("done")]})
    messages = [SystemMessage(content="You are the worker of hippo"), HumanMessage(content="go")]
    tracer = Tracer(tmp_path / "t.jsonl")
    text, reason = asyncio.run(
        g.tool_loop(llm, _tools(), messages, tracer=tracer, max_steps=4, label="x")
    )
    assert reason == "answer"
    tool_msgs = [m.content for m in messages if m.__class__.__name__ == "ToolMessage"]
    assert len(tool_msgs) == 2
    assert "[hippo]" not in tool_msgs[0] and "[hippo]" in tool_msgs[1]
    assert any(r["kind"] == "repeat_call" for r in tracer.read())


def test_reviewer_sees_worker_tool_log(tmp_path: Path) -> None:
    """The reviewer prompt must carry what the worker actually ran, not just its prose."""
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "local__echo", "args": {"text": "proof"}, "id": "c1", "type": "tool_call"}
        ],
    )
    prompts: list[str] = []

    class SpyLLM(FakeLLM):
        def _reply(self, messages):
            if self._role(messages) == "reviewer":
                prompts.append(messages[-1].content)
            return super()._reply(messages)

    llm = SpyLLM(
        {
            "planner": [_plan("use the tool")],
            "worker": [call, _worker("echo:proof")],
            "reviewer": [_review("approve")],
        }
    )
    out = _run(tmp_path, llm)
    assert out["status"] == "done"
    assert prompts and "local__echo" in prompts[0] and "echo:proof" in prompts[0]
    assert out["plan"][0]["tool_log"][0]["tool"] == "local__echo"


def test_worker_always_gets_baseline_read_tools(tmp_path: Path) -> None:
    """Planner grants only the edit tool; worker must still be able to read and run tests."""

    async def noop(**_):
        return "ok"

    class Empty(BaseModel):
        pass

    def mk(name: str) -> StructuredTool:
        return StructuredTool.from_function(
            coroutine=noop, name=name, description=name, args_schema=Empty
        )

    seen: dict[str, list[str]] = {}

    class SpyLLM(FakeLLM):
        def bind_tools(self, tools):
            seen["names"] = sorted(t.name for t in tools)
            return self

    llm = SpyLLM({"worker": [_worker("edited")]})
    ctx = RunContext(
        tools=[mk("filesystem__edit_file"), mk("filesystem__read_file"), mk("local__run_pytest")],
        tracer=Tracer(tmp_path / "trace.jsonl"),
        model="fake",
        api_key="x",
        token_budget=100_000,
        llm_factory=lambda **_: llm,
    )
    state = g.initial_state("fix it")
    state["plan"] = [
        {
            "id": "s1",
            "goal": "edit",
            "allowed_tools": ["filesystem__edit_file"],
            "budget_steps": 3,
            "status": "pending",
            "attempts": 0,
        }
    ]
    state["current"] = 0

    class RT:
        context = ctx

    asyncio.run(g.worker_node(state, RT()))
    assert seen["names"] == ["filesystem__edit_file", "filesystem__read_file", "local__run_pytest"]


def test_checkpoint_resume_continues_after_interrupt(tmp_path: Path) -> None:
    """Crash inside the worker, then resume from the planner checkpoint."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    config = {"configurable": {"thread_id": "t1"}}
    db = str(tmp_path / "ckpt.db")

    class Crash(Exception):
        pass

    crashing = FakeLLM({"planner": [_plan("resumable")]})

    async def boom(messages, **_):
        raise Crash("power cut")

    crashing.ainvoke = boom  # type: ignore[method-assign]
    # planner must still answer: route planner to the scripted reply, worker to boom
    real_reply = FakeLLM._reply

    async def routed(messages, **_):
        if crashing._role(messages) == "planner":
            return real_reply(crashing, messages)
        raise Crash("power cut")

    crashing.ainvoke = routed  # type: ignore[method-assign]

    async def first_run():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = g.build_graph(saver)
            try:
                await graph.ainvoke(
                    g.initial_state("resumable"), config=config, context=_ctx(tmp_path, crashing)
                )
            except Crash:
                pass
            snap = await graph.aget_state(config)
            return snap

    snap = asyncio.run(first_run())
    assert snap.values["plan"][0]["goal"] == "resumable"
    assert "worker" in snap.next

    healthy = FakeLLM({"worker": [_worker("done after resume")], "reviewer": [_review("approve")]})

    async def second_run():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = g.build_graph(saver)
            return await graph.ainvoke(None, config=config, context=_ctx(tmp_path, healthy))

    out = asyncio.run(second_run())
    assert out["final"] == "done after resume"
    assert "planner" not in healthy.calls


def test_parse_json_object_variants() -> None:
    assert g.parse_json_object('{"a": 1}') == {"a": 1}
    assert g.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert g.parse_json_object('text before {"a": {"b": 2}} after') == {"a": {"b": 2}}
    assert g.parse_json_object("nothing here") is None


def test_route_after_review() -> None:
    plan = [{"id": "s1"}, {"id": "s2"}]
    assert g.route_after_review({"review": {"verdict": "revise"}, "plan": plan}) == "worker"
    assert (
        g.route_after_review({"review": {"verdict": "approve"}, "current": 1, "plan": plan})
        == "worker"
    )
    assert (
        g.route_after_review({"review": {"verdict": "approve"}, "current": 2, "plan": plan})
        == "finalize"
    )
    assert g.route_after_review({"review": {"verdict": "escalate"}, "replans": 0}) == "planner"
    assert g.route_after_review({"review": {"verdict": "escalate"}, "replans": 1}) == "finalize"


def test_human_message_import_used() -> None:  # keeps ruff happy about HumanMessage in helpers
    assert HumanMessage(content="x").content == "x"
