from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from hippo.config import load_settings
from hippo.trace import Tracer, list_trace_ids

app = typer.Typer(help="hippo: long-term memory CLI agent")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to execute"),
    demo: bool = typer.Option(False, "--demo", help="Use HIPPO_DEMO_MODEL instead of HIPPO_MODEL"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="Use local filesystem tools only"),
    write: bool = typer.Option(False, "--write", help="Allow write/delete/commit tools"),
    oneshot: bool = typer.Option(False, "--oneshot", help="Skip tools (M0 behaviour)"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip recall and persist"),
) -> None:
    """Run a task. Recalls prior memory, uses tools, then stores an episode summary."""
    settings = load_settings()
    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY is empty. Copy .env.example to .env and set it.[/red]")
        raise typer.Exit(1)
    model = settings.hippo_demo_model if demo else settings.hippo_model
    task_id = uuid.uuid4().hex[:12]
    asyncio.run(
        _run_async(
            task,
            model=model,
            api_key=settings.openai_api_key,
            workspace=Path(settings.hippo_workspace).resolve(),
            data_dir=Path(settings.hippo_data_dir),
            task_id=task_id,
            no_mcp=no_mcp,
            write=write,
            oneshot=oneshot,
            no_memory=no_memory,
        )
    )


async def _run_async(
    task: str,
    *,
    model: str,
    api_key: str,
    workspace: Path,
    data_dir: Path,
    task_id: str,
    no_mcp: bool,
    write: bool,
    oneshot: bool,
    no_memory: bool,
) -> None:
    from hippo.agent.graph import run_agent, run_once
    from hippo.memory.manager import build_memory_manager
    from hippo.store.sqlite import connect, upsert_task
    from hippo.tools.mcp_client import McpHub, local_fs_tools
    from hippo.tools.registry import ToolRegistry

    settings = load_settings()
    tracer = Tracer(data_dir / "traces" / f"{task_id}.jsonl")
    tracer.emit("meta", task_id=task_id, workspace=str(workspace))
    registry = ToolRegistry(max_calls=40, allow_dangerous=write)
    db = connect(data_dir / "hippo.db")
    upsert_task(db, task_id, task, "running")

    recalled_block: list[str] = []
    mem = None
    if not no_memory:
        try:
            mem = build_memory_manager(settings)
            hits = mem.recall(task, k=5)
            block = mem.format_recall(hits)
            if block:
                recalled_block = [block]
                console.print(f"[dim]memory backend={mem.backend}  recalled={len(hits)}[/dim]")
            else:
                console.print(f"[dim]memory backend={mem.backend}  (empty)[/dim]")
            tracer.emit("recall", backend=mem.backend, n=len(hits))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]memory recall skipped: {exc}[/yellow]")
            tracer.emit("recall_error", error=str(exc))
            mem = None

    try:
        if oneshot:
            text = run_once(
                task, model=model, api_key=api_key, tracer=tracer, recalled=recalled_block
            )
        else:
            hub = McpHub(registry, tracer)
            tools = []
            if not no_mcp:
                tools = await hub.start(Path("mcp.json"), workspace)
            if not tools:
                console.print(
                    "[yellow]MCP unavailable or --no-mcp: using local filesystem tools[/yellow]"
                )
                tools = local_fs_tools(workspace, registry, tracer)
            try:
                text = await run_agent(
                    task,
                    model=model,
                    api_key=api_key,
                    tools=tools,
                    tracer=tracer,
                    recalled=recalled_block,
                )
            finally:
                await hub.aclose()
    except Exception as exc:  # noqa: BLE001
        upsert_task(db, task_id, task, "failed")
        tracer.emit("run_error", error=str(exc)[:1000])
        console.print(f"[red]run failed:[/red] {exc}")
        console.print(f"[dim]task_id={task_id}  |  hippo trace {task_id}[/dim]")
        raise typer.Exit(1) from exc

    if mem is not None:
        try:
            saved = mem.persist_run(
                task=task, answer=text, task_id=task_id, model=model, api_key=api_key
            )
            tracer.emit("memory_write", facts=len(saved.get("facts") or []))
        except Exception as exc:  # noqa: BLE001
            console.print(
                "[yellow]memory write failed "
                f"(search still works if the key is read-only): {exc}[/yellow]"
            )
            tracer.emit("memory_write_error", error=str(exc))

    upsert_task(db, task_id, task, "done")
    console.print(text)
    console.print(f"[dim]task_id={task_id}  |  hippo trace {task_id}[/dim]")


@app.command()
def resume(task_id: str = typer.Argument(...)) -> None:
    """Resume a checkpointed task (M3)."""
    console.print(f"[yellow]resume not implemented yet[/yellow] (task_id={task_id})")
    raise typer.Exit(2)


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
) -> None:
    """Print JSONL traces for a run."""
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
        console.print(JSON.from_data(row))


if __name__ == "__main__":
    app()
