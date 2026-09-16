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


def test_run_pytest_tool_reports_exit_code(tmp_path: Path) -> None:
    from hippo.tools.mcp_client import run_pytest_tool

    (tmp_path / "test_x.py").write_text("def test_ok():\n    assert 1 == 1\n", encoding="utf-8")
    tool = run_pytest_tool(tmp_path, ToolRegistry(max_calls=5), Tracer(tmp_path / "t.jsonl"))
    out = asyncio.run(tool.ainvoke({"path": "", "keyword": ""}))
    assert out.startswith("exit code 0")
    assert "1 passed" in out


def test_run_pytest_tool_refuses_paths_outside_workspace(tmp_path: Path) -> None:
    from hippo.tools.mcp_client import run_pytest_tool

    tool = run_pytest_tool(tmp_path, ToolRegistry(max_calls=5), Tracer(tmp_path / "t.jsonl"))
    try:
        asyncio.run(tool.ainvoke({"path": "../outside", "keyword": ""}))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_json_schema_to_model_spells_out_nested_items() -> None:
    from hippo.tools.mcp_client import json_schema_to_model

    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    }
    model = json_schema_to_model("fs__edit_file", schema)
    js = model.model_json_schema()
    item_ref = js["properties"]["edits"]["items"]
    item = js["$defs"][item_ref["$ref"].rsplit("/", 1)[-1]] if "$ref" in item_ref else item_ref
    assert set(item["properties"]) == {"oldText", "newText"}
    parsed = model(path="a.py", edits=[{"oldText": "x", "newText": "y"}])
    assert parsed.edits[0].oldText == "x"


def test_workspace_tree_lists_relative_paths_and_skips_noise(tmp_path: Path) -> None:
    from hippo.tools.mcp_client import workspace_tree

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text("# hi\n", encoding="utf-8")
    tree = workspace_tree(tmp_path)
    assert tree.splitlines() == ["pkg/", "pkg/mod.py", "README.md"]


def test_rewrite_args_git_root(tmp_path: Path) -> None:
    from hippo.tools.mcp_client import _rewrite_args

    (tmp_path / ".git").mkdir()
    nested = tmp_path / "examples" / "sandbox"
    nested.mkdir(parents=True)
    out = _rewrite_args(["--repository", "${GIT_ROOT}", "${WORKSPACE}", "."], nested)
    assert out == [
        "--repository",
        str(tmp_path.resolve()),
        str(nested.resolve()),
        str(nested.resolve()),
    ]


def test_task_scoped_client_reports_start_failure() -> None:
    from mcp.client.stdio import StdioServerParameters

    from hippo.tools.mcp_client import _TaskScopedClient

    params = StdioServerParameters(command="python", args=["-c", "import sys; sys.exit(3)"])
    try:
        asyncio.run(_TaskScopedClient(params).start())
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "MCP server" in str(exc)


def test_resolve_mcp_config_falls_back_to_packaged(tmp_path: Path, monkeypatch) -> None:
    from hippo.config import resolve_mcp_config

    monkeypatch.chdir(tmp_path)
    resolved = resolve_mcp_config(Path("mcp.json"))
    assert resolved.exists() and resolved.name == "mcp.json"


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
