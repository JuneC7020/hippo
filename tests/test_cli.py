from pathlib import Path

from typer.testing import CliRunner

from hippo.cli import app

runner = CliRunner()


def test_run_requires_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = runner.invoke(app, ["run", "hello"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_format_trace_line_is_compact() -> None:
    from hippo.cli import format_trace_line

    plan = {
        "ts": "2026-09-16T19:54:13.6+00:00",
        "kind": "plan",
        "replan": False,
        "subtasks": [{"id": "s1", "goal": "do x", "tools": ["local__run_pytest"]}],
    }
    line = format_trace_line(plan)
    assert line.startswith("19:54:13 plan") and "s1: do x" in line
    call = {"kind": "tool_call", "server": "local", "tool": "run_pytest", "args": {"path": ""}}
    assert "local.run_pytest(" in format_trace_line(call)
    review = {"kind": "review", "subtask": "s1", "verdict": "approve", "feedback": "ok"}
    assert "s1 APPROVE: ok" in format_trace_line(review)


def test_run_failure_marks_task_failed(monkeypatch, tmp_path: Path) -> None:
    """LLM errors must not leave the task stuck at 'running' or dump a traceback."""
    import hippo.agent.graph as graph

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HIPPO_MEMORY_BACKEND", "chroma")

    async def boom(*_args, **_kwargs):
        raise RuntimeError("credit_balance_exhausted")

    monkeypatch.setattr(graph, "run_agent", boom)
    result = runner.invoke(app, ["run", "--no-mcp", "--single", "hello"])
    assert result.exit_code == 1
    assert "run failed" in result.output
    assert "credit_balance_exhausted" in result.output

    from hippo.store.sqlite import connect, list_tasks

    rows = list_tasks(connect(tmp_path / ".hippo" / "hippo.db"))
    assert rows and rows[0]["status"] == "failed"
