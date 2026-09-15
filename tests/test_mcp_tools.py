import asyncio
from pathlib import Path

from hippo.tools.mcp_client import json_schema_to_model, local_fs_tools
from hippo.tools.registry import ToolRegistry
from hippo.trace import Tracer, list_trace_ids


def test_json_schema_required_and_optional() -> None:
    model = json_schema_to_model(
        "read",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "file"},
                "offset": {"type": "integer"},
            },
            "required": ["path"],
        },
    )
    obj = model(path="a.txt")
    assert obj.path == "a.txt"
    assert obj.offset is None


def test_local_list_and_read(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    tracer = Tracer(tmp_path / "t.jsonl")
    registry = ToolRegistry(max_calls=10)
    tools = {t.name: t for t in local_fs_tools(tmp_path, registry, tracer)}
    listing = asyncio.run(tools["local__list_dir"].ainvoke({"path": "."}))
    assert "hello.txt" in listing
    body = asyncio.run(tools["local__read_file"].ainvoke({"path": "hello.txt"}))
    assert body == "hi"
    kinds = [row["kind"] for row in tracer.read()]
    assert "tool_call" in kinds
    assert "tool_result" in kinds


def test_list_trace_ids(tmp_path: Path) -> None:
    folder = tmp_path / "traces"
    Tracer(folder / "abc.jsonl").emit("meta", task_id="abc")
    assert list_trace_ids(tmp_path) == ["abc"]


def test_dangerous_write_blocked() -> None:
    r = ToolRegistry(max_calls=5)
    try:
        r.check("filesystem__write_file")
        assert False, "expected permission error"
    except PermissionError:
        pass
    r2 = ToolRegistry(max_calls=5, allow_dangerous=True)
    r2.check("filesystem__write_file")
