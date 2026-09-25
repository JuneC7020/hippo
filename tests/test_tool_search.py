"""Tool catalog, retrievers, meta-tools and per-step rebinding. No network."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeLLM, ScriptedToolLLM, tool_call
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from hippo.agent import graph as g
from hippo.agent.state import RunContext
from hippo.tools.catalog import KeywordRetriever, ToolCatalog, make_entry, tokenize
from hippo.tools.registry import ToolRegistry
from hippo.tools.search import build_exposure, meta_tools, normalize_exposure
from hippo.trace import Tracer


class RepoArgs(BaseModel):
    repo: str = Field(description="owner/name")
    title: str = Field(description="Issue title")


class ChannelArgs(BaseModel):
    channel: str = Field(description="Channel name")
    text: str = Field(description="Message text")


class QueryArgs(BaseModel):
    sql: str = Field(description="SQL statement")


def _mk(name: str, desc: str, schema: type[BaseModel]) -> StructuredTool:
    async def run(**kwargs):
        return f"{name} ok {kwargs}"

    return StructuredTool.from_function(
        coroutine=run, name=name, description=desc, args_schema=schema
    )


def _tools() -> list[StructuredTool]:
    return [
        _mk("github__create_issue", "Create a new issue in a GitHub repository.", RepoArgs),
        _mk("github__list_issues", "List issues in a repository.", RepoArgs),
        _mk("slack__send_message", "Post a message to a Slack channel.", ChannelArgs),
        _mk("postgres__run_query", "Run a read-only SQL query.", QueryArgs),
        _mk("filesystem__read_file", "Read a file from the workspace. (mcp:filesystem)", QueryArgs),
    ]


# --------------------------------------------------------------------------- catalog / retriever


def test_tokenize_splits_snake_and_camel_and_stems() -> None:
    assert tokenize("create_issue listIssues repositories") == [
        "create",
        "issue",
        "list",
        "issue",
        "repository",
    ]


def test_entry_summary_strips_mcp_suffix() -> None:
    e = make_entry(_tools()[4])
    assert e.server == "filesystem" and e.short == "read_file"
    assert e.summary == "Read a file from the workspace."


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("open a github issue about login", "github__create_issue"),
        ("post to the #releases slack channel", "slack__send_message"),
        ("run a sql query against postgres", "postgres__run_query"),
        ("find all issues in the repo", "github__list_issues"),
    ],
)
def test_keyword_retriever_top1(query: str, expected: str) -> None:
    cat = ToolCatalog(_tools())
    hits = cat.search(query, 3)
    assert hits and hits[0].name == expected


def test_keyword_retriever_empty_query_and_no_match() -> None:
    r = KeywordRetriever([make_entry(t) for t in _tools()])
    assert r.search("", 5) == []
    assert r.search("zzzz qqqq", 5) == []


def test_catalog_activation_is_ordered_idempotent_and_capped() -> None:
    cat = ToolCatalog(_tools(), always_on=["read_file"], max_active=2)
    assert cat.active_names == ["filesystem__read_file"]
    activated, unknown = cat.activate(["github__create_issue", "nope", "send_message"])
    assert activated == ["github__create_issue", "slack__send_message"] and unknown == ["nope"]
    cat.activate(["postgres__run_query"])  # cap 2 evicts the oldest
    assert cat.active_names == [
        "filesystem__read_file",
        "slack__send_message",
        "postgres__run_query",
    ]
    cat.activate(["slack__send_message"])  # re-activating moves it to most recent, no dup
    assert cat.active_names.count("slack__send_message") == 1
    assert "not loaded" in cat.unknown_tool_message("github__create_issue")
    assert "not in the catalog" in cat.unknown_tool_message("bogus")


def test_synthetic_catalog_recall_at_5_on_task_prompts() -> None:
    """The benchmark's own tasks should be findable from their prompts (docs B8)."""
    from hippo.bench.catalog import load_catalog, make_tools
    from hippo.bench.tasks import load_tasks

    cat = ToolCatalog(make_tools(load_catalog().tools))
    misses = []
    for task in load_tasks():
        hits = [e.name for e in cat.search(task.prompt, 5)]
        if task.expected_tools[0] not in hits:
            misses.append((task.id, hits))
    assert not misses, misses


# --------------------------------------------------------------------------- meta tools


def test_normalize_exposure_variants() -> None:
    assert normalize_exposure(None) == "all"
    assert normalize_exposure("search-schema") == "search_schema"
    assert normalize_exposure("SEARCH") == "search"
    with pytest.raises(ValueError):
        normalize_exposure("everything")


def test_mode_search_returns_summaries_without_schemas(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path / "t.jsonl")
    reg = ToolRegistry(max_calls=10)
    cat = ToolCatalog(_tools())
    meta = {t.name: t for t in meta_tools(cat, mode="search", registry=reg, tracer=tracer)}
    assert set(meta) == {"tool_search", "tool_load"}
    out = asyncio.run(meta["tool_search"].ainvoke({"query": "create github issue", "k": 2}))
    assert "github__create_issue" in out and "parameters" not in out and "tool_load" in out
    assert cat.active_names == []  # nothing bound until loaded
    out2 = asyncio.run(meta["tool_load"].ainvoke({"names": ["github__create_issue", "zz"]}))
    assert "github__create_issue" in out2 and "zz" in out2
    assert cat.active_names == ["github__create_issue"]
    kinds = [r["kind"] for r in tracer.read()]
    assert "tool_search" in kinds and "tool_load" in kinds
    assert reg.calls == 2  # meta calls count against the budget


def test_mode_search_schema_returns_parameters_and_activates(tmp_path: Path) -> None:
    cat = ToolCatalog(_tools())
    meta = {t.name: t for t in meta_tools(cat, mode="search_schema")}
    assert set(meta) == {"tool_search"}
    out = asyncio.run(meta["tool_search"].ainvoke({"query": "slack message", "k": 1}))
    assert "slack__send_message" in out and "parameters:" in out and '"channel"' in out
    assert cat.active_names == ["slack__send_message"]


def test_no_hits_message() -> None:
    cat = ToolCatalog(_tools())
    meta = {t.name: t for t in meta_tools(cat, mode="search")}
    out = asyncio.run(meta["tool_search"].ainvoke({"query": "qqqq", "k": 3}))
    assert out.startswith("No tools matched")


# --------------------------------------------------------------------------- tool_loop rebinding


def test_tool_loop_rebinds_per_step_in_search_mode(tmp_path: Path) -> None:
    tools = _tools()
    exp = build_exposure(tools, mode="search")
    llm = ScriptedToolLLM("github__create_issue", {"repo": "acme/x", "title": "t"})
    messages = [SystemMessage(content="sys" + exp.system_note), HumanMessage(content="open issue")]
    tracer = Tracer(tmp_path / "t.jsonl")
    text, reason = asyncio.run(
        g.tool_loop(
            llm,
            tools,
            messages,
            tracer=tracer,
            max_steps=6,
            tools_provider=exp.tools_provider,
            on_unknown=exp.on_unknown,
        )
    )
    assert reason == "answer" and text.startswith("done")
    # step 1: meta only; step 2: still meta (search result back); step 3: target bound
    assert llm.bound_per_step[0] == ["tool_search", "tool_load"]
    assert llm.bound_per_step[1] == ["tool_search", "tool_load"]
    assert llm.bound_per_step[2] == ["tool_search", "tool_load", "github__create_issue"]
    usage = [r for r in tracer.read() if r["kind"] == "usage"]
    assert [u["n_bound_tools"] for u in usage] == [2, 2, 3, 3]
    assert usage[0]["tool_schema_tokens"] < 400  # two meta tools, not the catalog


def test_tool_loop_mode_all_binds_everything_every_step(tmp_path: Path) -> None:
    tools = _tools()
    exp = build_exposure(tools, mode="all")
    llm = ScriptedToolLLM("slack__send_message", {"channel": "#x", "text": "hi"})
    messages = [SystemMessage(content="sys"), HumanMessage(content="post")]
    asyncio.run(
        g.tool_loop(
            llm,
            tools,
            messages,
            tracer=Tracer(tmp_path / "t.jsonl"),
            max_steps=4,
            tools_provider=exp.tools_provider,
        )
    )
    assert all(len(b) == len(tools) for b in llm.bound_per_step)


def test_unknown_tool_gets_load_hint(tmp_path: Path) -> None:
    tools = _tools()
    exp = build_exposure(tools, mode="search")
    llm = FakeLLM(
        {
            "other": [
                tool_call("github__create_issue", {"repo": "a/b", "title": "t"}),
                {"content": "gave up"},
            ]
        }
    )
    messages = [SystemMessage(content="x"), HumanMessage(content="go")]
    tracer = Tracer(tmp_path / "t.jsonl")
    asyncio.run(
        g.tool_loop(
            llm,
            tools,
            messages,
            tracer=tracer,
            max_steps=3,
            tools_provider=exp.tools_provider,
            on_unknown=exp.on_unknown,
        )
    )
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert (
        tool_msgs and "not loaded" in tool_msgs[0].content and "tool_load" in tool_msgs[0].content
    )
    assert any(r["kind"] == "unknown_tool" for r in tracer.read())


def test_run_tool_agent_search_mode_end_to_end(tmp_path: Path) -> None:
    tools = _tools()
    llm = ScriptedToolLLM("postgres__run_query", {"sql": "select 1"})
    run = asyncio.run(
        g.run_tool_agent(
            "run select 1",
            tools=tools,
            tracer=Tracer(tmp_path / "t.jsonl"),
            llm=llm,
            exposure="search",
            registry=ToolRegistry(max_calls=10),
        )
    )
    assert run.reason == "answer"
    assert run.exposure.catalog is not None and run.exposure.catalog.searches == 1
    assert run.meter.n_requests == 4
    assert run.meter.search_result_tokens > 0 and run.meter.tool_result_tokens > 0
    assert [t["tool"] for t in run.tool_log] == ["tool_search", "tool_load", "postgres__run_query"]


def test_worker_node_search_mode_keeps_baseline_tools_always_on(tmp_path: Path) -> None:
    """With exposure=search the worker binds meta-tools + baseline read tools, not the catalog."""
    seen: list[list[str]] = []

    class SpyLLM(FakeLLM):
        def bind_tools(self, tools):
            seen.append(sorted(t.name for t in tools))
            return self

    llm = SpyLLM({"worker": [{"content": '{"result": "ok", "evidence": "", "confidence": 1}'}]})
    ctx = RunContext(
        tools=_tools(),
        tracer=Tracer(tmp_path / "trace.jsonl"),
        model="fake",
        api_key="x",
        token_budget=100_000,
        llm_factory=lambda **_: llm,
        exposure="search",
    )
    state = g.initial_state("do it")
    state["plan"] = [
        {"id": "s1", "goal": "g", "allowed_tools": [], "budget_steps": 3, "status": "pending"}
    ]
    state["current"] = 0

    class RT:
        context = ctx

    asyncio.run(g.worker_node(state, RT()))
    assert seen[0] == ["filesystem__read_file", "tool_load", "tool_search"]
