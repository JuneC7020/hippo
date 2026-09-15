from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON

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
) -> None:
    """Run a task. Default is a single-agent tool loop (MCP filesystem/git, local fallback)."""
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
) -> None:
    from hippo.agent.graph import run_agent, run_once
    from hippo.tools.mcp_client import McpHub, local_fs_tools
    from hippo.tools.registry import ToolRegistry

    tracer = Tracer(data_dir / "traces" / f"{task_id}.jsonl")
    tracer.emit("meta", task_id=task_id, workspace=str(workspace))
    registry = ToolRegistry(max_calls=40, allow_dangerous=write)

    if oneshot:
        text = run_once(task, model=model, api_key=api_key, tracer=tracer)
        console.print(text)
        console.print(f"[dim]task_id={task_id}[/dim]")
        return

    hub = McpHub(registry, tracer)
    tools = []
    if not no_mcp:
        tools = await hub.start(Path("mcp.json"), workspace)
    if not tools:
        console.print("[yellow]MCP unavailable or --no-mcp: using local filesystem tools[/yellow]")
        tools = local_fs_tools(workspace, registry, tracer)
    try:
        text = await run_agent(task, model=model, api_key=api_key, tools=tools, tracer=tracer)
    finally:
        await hub.aclose()
    console.print(text)
    console.print(f"[dim]task_id={task_id}  ·  hippo trace {task_id}[/dim]")


@app.command()
def resume(task_id: str = typer.Argument(...)) -> None:
    """Resume a checkpointed task (M3)."""
    console.print(f"[yellow]resume not implemented yet[/yellow] (task_id={task_id})")
    raise typer.Exit(2)


@app.command()
def tasks() -> None:
    """List stored tasks (M2/M3)."""
    console.print("[yellow]tasks not implemented yet[/yellow]")
    raise typer.Exit(2)


@app.command()
def memory(
    query: str | None = typer.Argument(None),
) -> None:
    """Search long-term memory (M2)."""
    if not query:
        console.print("usage: hippo memory SEARCH_QUERY")
        raise typer.Exit(2)
    console.print(f"[yellow]memory search not implemented yet[/yellow] ({query})")
    raise typer.Exit(2)


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
