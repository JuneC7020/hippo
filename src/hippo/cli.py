from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from hippo.config import Settings, load_settings
from hippo.trace import Tracer, list_trace_ids

app = typer.Typer(help="hippo: long-term memory CLI agent")
console = Console()


def _require_key(settings: Settings) -> None:
    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY is empty. Copy .env.example to .env and set it.[/red]")
        raise typer.Exit(1)


@dataclass
class RunOptions:
    model: str
    workspace: Path
    data_dir: Path
    task_id: str
    no_mcp: bool = False
    write: bool = False
    oneshot: bool = False
    single: bool = False
    no_memory: bool = False


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to execute"),
    demo: bool = typer.Option(False, "--demo", help="Use HIPPO_DEMO_MODEL instead of HIPPO_MODEL"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="Use local filesystem tools only"),
    write: bool = typer.Option(False, "--write", help="Allow write/delete/commit tools"),
    oneshot: bool = typer.Option(False, "--oneshot", help="Skip tools (M0 behaviour)"),
    single: bool = typer.Option(False, "--single", help="One agent, no planner/reviewer (M1)"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip recall and persist"),
    workspace: Path | None = typer.Option(
        None, "--workspace", "-w", help="Repository to work in (default HIPPO_WORKSPACE or .)"
    ),
) -> None:
    """Run a task: recall memory -> plan -> work -> review -> answer -> store an episode."""
    settings = load_settings()
    _require_key(settings)
    opts = RunOptions(
        model=settings.hippo_demo_model if demo else settings.hippo_model,
        workspace=Path(workspace or settings.hippo_workspace).resolve(),
        data_dir=Path(settings.hippo_data_dir),
        task_id=uuid.uuid4().hex[:12],
        no_mcp=no_mcp,
        write=write,
        oneshot=oneshot,
        single=single,
        no_memory=no_memory,
    )
    asyncio.run(_run_async(task, settings, opts, resume=False))


@app.command()
def resume(
    task_id: str = typer.Argument(..., help="Task id from `hippo tasks`"),
    demo: bool = typer.Option(False, "--demo"),
    no_mcp: bool = typer.Option(False, "--no-mcp"),
    write: bool = typer.Option(False, "--write"),
    no_memory: bool = typer.Option(False, "--no-memory"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Continue an interrupted or failed task from its last LangGraph checkpoint."""
    from hippo.store.sqlite import connect, get_task

    settings = load_settings()
    _require_key(settings)
    data_dir = Path(settings.hippo_data_dir)
    row = get_task(connect(data_dir / "hippo.db"), task_id)
    if row is None:
        console.print(f"[red]unknown task_id {task_id}[/red]  (see `hippo tasks`)")
        raise typer.Exit(1)
    if row["status"] == "done":
        console.print(f"[yellow]task {task_id} already finished[/yellow]")
        raise typer.Exit(0)
    opts = RunOptions(
        model=settings.hippo_demo_model if demo else settings.hippo_model,
        workspace=Path(workspace or settings.hippo_workspace).resolve(),
        data_dir=data_dir,
        task_id=task_id,
        no_mcp=no_mcp,
        write=write,
        no_memory=no_memory,
    )
    asyncio.run(_run_async(row["goal"], settings, opts, resume=True))


def _recall(settings: Settings, task: str, tracer: Tracer) -> tuple[Any, list[str]]:
    from hippo.memory.manager import build_memory_manager

    try:
        mem = build_memory_manager(settings)
        hits = mem.recall(task, k=5)
        block = mem.format_recall(hits)
        tracer.emit("recall", backend=mem.backend, n=len(hits))
        if block:
            console.print(f"[dim]memory backend={mem.backend}  recalled={len(hits)}[/dim]")
            return mem, [block]
        console.print(f"[dim]memory backend={mem.backend}  (empty)[/dim]")
        return mem, []
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]memory recall skipped: {exc}[/yellow]")
        tracer.emit("recall_error", error=str(exc))
        return None, []


async def _run_async(task: str, settings: Settings, opts: RunOptions, *, resume: bool) -> None:
    from hippo.agent.graph import run_once
    from hippo.config import resolve_mcp_config
    from hippo.store.sqlite import connect, upsert_task
    from hippo.tools.mcp_client import McpHub, local_fs_tools, run_pytest_tool
    from hippo.tools.registry import ToolRegistry

    api_key = settings.openai_api_key
    tracer = Tracer(opts.data_dir / "traces" / f"{opts.task_id}.jsonl")
    tracer.emit("meta", task_id=opts.task_id, workspace=str(opts.workspace), resume=resume)
    db = connect(opts.data_dir / "hippo.db")
    upsert_task(db, opts.task_id, task, "running")

    mem, recalled = (None, []) if opts.no_memory else _recall(settings, task, tracer)

    hub = McpHub(ToolRegistry(max_calls=60, allow_dangerous=opts.write), tracer)
    final_status = "done"
    try:
        if opts.oneshot:
            text = run_once(
                task, model=opts.model, api_key=api_key, tracer=tracer, recalled=recalled
            )
        else:
            mcp_cfg = resolve_mcp_config(Path(settings.hippo_mcp_config))
            tools = [] if opts.no_mcp else await hub.start(mcp_cfg, opts.workspace)
            if not tools:
                console.print(
                    "[yellow]MCP unavailable or --no-mcp: using local filesystem tools[/yellow]"
                )
                tools = local_fs_tools(opts.workspace, hub.registry, tracer)
            tools.append(run_pytest_tool(opts.workspace, hub.registry, tracer))
            if opts.single:
                text = await _run_single(task, tools, tracer, opts, api_key, recalled, settings)
            else:
                text, final_status = await _run_graph(
                    task, tools, tracer, opts, api_key, recalled, settings, resume
                )
    except KeyboardInterrupt:
        upsert_task(db, opts.task_id, task, "interrupted")
        tracer.emit("run_interrupted")
        console.print(
            f"\n[yellow]interrupted.[/yellow] continue later with: hippo resume {opts.task_id}"
        )
        raise typer.Exit(130) from None
    except Exception as exc:  # noqa: BLE001
        upsert_task(db, opts.task_id, task, "failed")
        tracer.emit("run_error", error=str(exc)[:1000])
        console.print(f"[red]run failed:[/red] {exc}")
        console.print(f"[dim]task_id={opts.task_id}  |  hippo trace {opts.task_id}[/dim]")
        if not opts.oneshot and not opts.single:
            console.print(f"[dim]retry from the last checkpoint: hippo resume {opts.task_id}[/dim]")
        raise typer.Exit(1) from exc
    finally:
        await hub.aclose()

    # Only grounded runs (ones that could look at the repo) write memory. A --oneshot answer
    # is recall + the model's guess; storing it would feed hearsay back into recall.
    if mem is not None and not opts.oneshot:
        try:
            saved = mem.persist_run(
                task=task, answer=text, task_id=opts.task_id, model=opts.model, api_key=api_key
            )
            tracer.emit("memory_write", facts=len(saved.get("facts") or []))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]memory write failed: {exc}[/yellow]")
            tracer.emit("memory_write_error", error=str(exc))

    upsert_task(db, opts.task_id, task, final_status)
    console.print(text, markup=False)
    console.print(f"[dim]task_id={opts.task_id}  |  hippo trace {opts.task_id}[/dim]")


async def _run_single(task, tools, tracer, opts, api_key, recalled, settings) -> str:
    from hippo.agent.graph import run_agent

    return await run_agent(
        task,
        model=opts.model,
        api_key=api_key,
        tools=tools,
        tracer=tracer,
        recalled=recalled,
        token_budget=settings.hippo_token_budget,
    )


async def _run_graph(
    task, tools, tracer, opts, api_key, recalled, settings, resume: bool
) -> tuple[str, str]:
    """Returns (final_text, task_status) where task_status is done|escalated."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from hippo.agent.graph import build_graph, initial_state
    from hippo.agent.state import RunContext
    from hippo.tools.mcp_client import workspace_tree

    ctx = RunContext(
        tools=tools,
        tracer=tracer,
        model=opts.model,
        api_key=api_key,
        token_budget=settings.hippo_token_budget,
        workspace_tree=workspace_tree(opts.workspace),
    )
    config = {"configurable": {"thread_id": opts.task_id}}
    async with AsyncSqliteSaver.from_conn_string(str(opts.data_dir / "checkpoints.db")) as saver:
        graph = build_graph(saver)
        if resume:
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise RuntimeError(
                    f"no checkpoint for {opts.task_id}; it never reached the planner. "
                    f"Run it again with `hippo run`."
                )
            if not snapshot.next:
                return snapshot.values.get("final") or "(task had already finished)", "done"
            console.print(f"[dim]resuming at {', '.join(snapshot.next)}[/dim]")
            tracer.emit("resume", next=list(snapshot.next))
            result = await graph.ainvoke(None, config=config, context=ctx)
        else:
            result = await graph.ainvoke(initial_state(task, recalled), config=config, context=ctx)

    plan = result.get("plan") or []
    if len(plan) > 1 or any(int(s.get("attempts") or 0) > 1 for s in plan):
        console.print(
            "[dim]"
            + " | ".join(f"{s['id']}:{s.get('status')}(x{s.get('attempts', 0)})" for s in plan)
            + f"  replans={result.get('replans', 0)}[/dim]"
        )
    status = "escalated" if result.get("status") == "failed" else "done"
    return result.get("final") or "(no final answer)", status


@app.command()
def tasks() -> None:
    """List recent tasks from SQLite."""
    settings = load_settings()
    from hippo.store.sqlite import connect, list_tasks

    rows = list_tasks(connect(Path(settings.hippo_data_dir) / "hippo.db"))
    if not rows:
        console.print("no tasks yet")
        return
    table = Table(show_header=True)
    table.add_column("id")
    table.add_column("status")
    table.add_column("goal")
    table.add_column("updated")
    for row in rows:
        table.add_row(row["id"], row["status"], row["goal"][:60], row["updated_at"][:19])
    console.print(table)


@app.command()
def memory(
    query: str | None = typer.Argument(None),
) -> None:
    """Search long-term memory (Seahorse or Chroma)."""
    if not query:
        console.print("usage: hippo memory SEARCH_QUERY")
        raise typer.Exit(2)
    from hippo.memory.manager import build_memory_manager

    settings = load_settings()
    mem = build_memory_manager(settings)
    hits = mem.recall(query, k=8)
    if not hits:
        console.print(f"no hits ({mem.backend})")
        return
    console.print(f"[dim]backend={mem.backend}[/dim]")
    # markup=False: recall lines contain "[fact]" / "[episode]" tags rich would eat
    console.print(mem.format_recall(hits), markup=False)


@app.command()
def trace(
    task_id: str | None = typer.Argument(None),
    raw: bool = typer.Option(False, "--raw", help="Full JSON per event instead of one line each"),
) -> None:
    """Show what a run did: plan, tool calls, reviews (one line per event; --raw for JSON)."""
    settings = load_settings()
    ids = list_trace_ids(Path(settings.hippo_data_dir))
    if not task_id:
        if not ids:
            console.print("no traces yet")
            raise typer.Exit(0)
        console.print("\n".join(ids[-20:]))
        return
    path = Path(settings.hippo_data_dir) / "traces" / f"{task_id}.jsonl"
    tracer = Tracer(path)
    rows = tracer.read()
    if not rows:
        console.print(f"no trace: {task_id}")
        raise typer.Exit(1)
    for row in rows:
        if raw:
            console.print(JSON.from_data(row))
        else:
            console.print(format_trace_line(row), markup=False, highlight=False)


def format_trace_line(row: dict) -> str:
    """Compact one-line rendering of a trace event (timestamps dropped)."""
    kind = row.get("kind", "?")
    ts = (row.get("ts") or "")[11:19]
    body: str
    if kind == "plan":
        subs = row.get("subtasks") or []
        body = ("re-plan" if row.get("replan") else "plan") + f" {len(subs)} subtask(s)"
        for s in subs:
            body += f"\n{'':>10}{s['id']}: {s['goal'][:90]}  tools={s.get('tools') or 'all'}"
    elif kind == "tool_call":
        args = json.dumps(row.get("args") or {}, ensure_ascii=False)
        body = f"{row.get('server')}.{row.get('tool')}({args[:110]})"
    elif kind == "tool_result":
        body = f"{'':>4}-> {row.get('tool')} {row.get('chars')} chars"
    elif kind == "worker_start":
        body = f"{row.get('subtask')} attempt {row.get('attempt')} (tools={row.get('n_tools')})"
    elif kind == "worker_end":
        body = f"{row.get('subtask')} {row.get('reason')} confidence={row.get('confidence')}"
    elif kind == "review":
        body = f"{row.get('subtask')} {row.get('verdict').upper()}: {row.get('feedback', '')[:110]}"
    elif kind == "llm":
        calls = row.get("tool_calls") or []
        body = f"{row.get('label')} step {row.get('step')}: " + (
            ", ".join(calls) if calls else f"answer ({row.get('chars')} chars)"
        )
    elif kind == "mcp":
        body = f"{row.get('event')} {row.get('server')}" + (
            f" ({row.get('tools')} tools)" if row.get("tools") is not None else ""
        )
    else:
        rest = {k: v for k, v in row.items() if k not in {"ts", "kind"}}
        body = json.dumps(rest, ensure_ascii=False)[:140]
    return f"{ts} {kind:<12} {body}"


if __name__ == "__main__":
    app()
