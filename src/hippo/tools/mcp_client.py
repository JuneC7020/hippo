"""MCP stdio clients from mcp.json, plus a local filesystem fallback."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from hippo.tools.registry import ToolRegistry
from hippo.trace import Tracer

# hippo's own memory server: the CLI already injects recall and persists episodes directly,
# so connecting to it from inside a run would double-open the Chroma dir. Other clients use it.
SKIP_SERVERS = frozenset({"memory", "seahorse"})

# A tool parameter called "schema" (postgres servers) shadows BaseModel.schema(); pydantic warns
# every time the model is (re)built, including inside langchain's tool-call schema. The field
# works and the JSON schema keeps the real name, so silence just that message.
warnings.filterwarnings("ignore", message='Field name "schema"', category=UserWarning)


def _npx() -> str:
    return "npx.cmd" if os.name == "nt" else "npx"


def _python() -> str:
    return "python"


JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _py_type(name: str, spec: dict[str, Any]) -> Any:
    """JSON-schema fragment -> Python/pydantic type, recursing into arrays and objects.

    Flattening `edits: [{oldText, newText}]` to a bare `list` makes OpenAI's function schema
    lose `items`, and the model then invents keys. Nested models keep the real shape.
    """
    typ = spec.get("type", "string")
    if isinstance(typ, list):  # e.g. ["string", "null"]
        typ = next((t for t in typ if t != "null"), "string")
    if typ == "array":
        items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
        return list[_py_type(f"{name}_item", items)]
    if typ == "object" and isinstance(spec.get("properties"), dict):
        return json_schema_to_model(name, spec)
    return JSON_TYPES.get(typ, str)


def _plain(value: Any) -> Any:
    """Nested pydantic instances (from `_py_type`) -> JSON-ready dicts/lists for MCP and traces."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def json_schema_to_model(name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
    schema = schema or {}
    props: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for key, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        typ = _py_type(f"{name}_{key}", spec)
        desc = spec.get("description") or ""
        if key in required:
            fields[key] = (typ, Field(description=desc))
        else:
            fields[key] = (typ | None, Field(default=None, description=desc))
    if not fields:
        return create_model(f"{name}_Args")
    return create_model(f"{name}_Args", **fields)


def _result_to_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else str(content)
    return str(content)


class McpHub:
    """Holds live MCP clients for the duration of a `hippo run`."""

    def __init__(self, registry: ToolRegistry, tracer: Tracer) -> None:
        self.registry = registry
        self.tracer = tracer
        self._clients: list[Any] = []
        self._tools: list[StructuredTool] = []

    async def start(
        self,
        config_path: Path,
        workspace: Path,
        *,
        skip: frozenset[str] = SKIP_SERVERS,
    ) -> list[StructuredTool]:
        if not config_path.exists():
            self.tracer.emit("mcp", event="no_config", path=str(config_path))
            return []
        data = json.loads(config_path.read_text(encoding="utf-8"))
        servers: dict[str, Any] = data.get("mcpServers") or {}
        from mcp.client.stdio import StdioServerParameters

        for name, spec in servers.items():
            if name in skip:
                self.tracer.emit("mcp", event="skip", server=name)
                continue
            command = spec.get("command") or ""
            args = list(spec.get("args") or [])
            if command in {"npx", "npx.cmd"}:
                command = _npx()
            if command == "python":
                command = _python()
            args = _rewrite_args(args, workspace)
            params = StdioServerParameters(command=command, args=args, cwd=str(workspace))
            try:
                client = await _connect(params)
            except Exception as exc:  # noqa: BLE001
                self.tracer.emit("mcp", event="connect_failed", server=name, error=str(exc))
                continue
            self._clients.append(client)
            try:
                listing = await client.list_tools()
            except Exception as exc:  # noqa: BLE001
                self.tracer.emit("mcp", event="list_failed", server=name, error=str(exc))
                continue
            tools = getattr(listing, "tools", listing) or []
            self.tracer.emit("mcp", event="connected", server=name, tools=len(tools))
            for tool in tools:
                self._tools.append(self._wrap(name, client, tool))
        return list(self._tools)

    def _wrap(self, server: str, client: Any, tool: Any) -> StructuredTool:
        original = tool.name
        qualified = f"{server}__{original}"
        description = (tool.description or original) + f" (mcp:{server})"
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        model = json_schema_to_model(qualified, schema)
        registry = self.registry
        tracer = self.tracer

        async def _call(**kwargs: Any) -> str:
            payload = {k: _plain(v) for k, v in kwargs.items() if v is not None}
            registry.check(qualified)
            tracer.emit("tool_call", server=server, tool=original, args=payload)
            try:
                result = await client.call_tool(original, payload)
                text = _result_to_text(result)[:12_000]
                tracer.emit("tool_result", server=server, tool=original, chars=len(text))
                return text
            except Exception as exc:  # noqa: BLE001
                tracer.emit("tool_error", server=server, tool=original, error=str(exc))
                return f"tool error: {exc}"

        return StructuredTool.from_function(
            coroutine=_call,
            name=qualified,
            description=description,
            args_schema=model,
        )

    async def aclose(self) -> None:
        for client in reversed(self._clients):
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()


CONNECT_TIMEOUT_S = 90


class _TaskScopedClient:
    """Run an MCP client's context managers inside one dedicated asyncio task.

    The MCP SDK is anyio-based: its stdio transport opens cancel scopes that must be
    entered and exited by the same task. Entering them from the agent's main task and
    closing them later (or after a failed connect) raises "Attempted to exit cancel
    scope in a different task" and can cancel unrelated awaits in the main task. So
    the whole lifecycle lives in `_runner`; the main task only sends requests.
    """

    def __init__(self, params: Any) -> None:
        import asyncio

        self._params = params
        self._session: Any = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._error: BaseException | None = None
        self._task: asyncio.Task | None = None

    async def _open(self):
        """Yield an object with list_tools/call_tool, on mcp 1.x or 2.x."""
        try:
            from mcp import Client  # mcp >= 2
        except ImportError:
            Client = None  # noqa: N806
        if Client is not None:
            async with Client(self._params) as client:
                yield client
            return
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def _runner(self) -> None:
        try:
            async for session in self._open():
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - includes CancelledError/ExceptionGroup
            self._error = exc
        finally:
            self._session = None
            self._ready.set()

    async def start(self) -> _TaskScopedClient:
        import asyncio

        self._task = asyncio.create_task(self._runner())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=CONNECT_TIMEOUT_S)
        except TimeoutError:
            await self.aclose()
            raise RuntimeError(f"MCP server did not initialize within {CONNECT_TIMEOUT_S}s")
        if self._session is None:
            err = self._error
            await self.aclose()
            raise RuntimeError(f"MCP server failed to start: {_first_error(err)}")
        return self

    async def list_tools(self) -> Any:
        return await self._session.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._session.call_tool(name, arguments)

    async def aclose(self) -> None:
        import asyncio

        self._stop.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except BaseException:  # noqa: BLE001
                pass


def _first_error(exc: BaseException | None) -> str:
    """Unwrap anyio ExceptionGroups to the first leaf message."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}" if exc else "unknown error"


async def _connect(params: Any) -> Any:
    return await _TaskScopedClient(params).start()


def _git_root(workspace: Path) -> Path:
    """Top-level of the git repo containing `workspace`, else `workspace` itself."""
    for candidate in (workspace, *workspace.parents):
        if (candidate / ".git").exists():
            return candidate
    return workspace


def _rewrite_args(args: list[str], workspace: Path) -> list[str]:
    ws = workspace.resolve()
    out: list[str] = []
    for a in args:
        if a in {".", "./", "${WORKSPACE}"}:
            out.append(str(ws))
        elif a == "${GIT_ROOT}":
            out.append(str(_git_root(ws)))
        else:
            out.append(a)
    return out


TREE_SKIP = frozenset(
    {
        ".git",
        ".hippo",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def workspace_tree(workspace: Path, *, max_depth: int = 3, max_entries: int = 150) -> str:
    """Shallow relative file listing given to workers so they don't burn steps guessing paths."""
    root = workspace.resolve()
    lines: list[str] = []
    truncated = False

    def walk(d: Path, depth: int) -> None:
        nonlocal truncated
        if truncated:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for p in entries:
            if p.name in TREE_SKIP or p.name.startswith(".") and p.is_dir():
                continue
            if len(lines) >= max_entries:
                truncated = True
                return
            rel = p.relative_to(root).as_posix()
            lines.append(rel + ("/" if p.is_dir() else ""))
            if p.is_dir() and depth < max_depth:
                walk(p, depth + 1)

    walk(root, 1)
    if truncated:
        lines.append(f"... (truncated at {max_entries} entries)")
    return "\n".join(lines)


PYTEST_TIMEOUT_S = 180


def run_pytest_tool(workspace: Path, registry: ToolRegistry, tracer: Tracer) -> StructuredTool:
    """`python -m pytest` inside the workspace. Fixed command, not a shell: safe without --write."""
    import asyncio
    import sys

    root = workspace.resolve()

    class PytestArgs(BaseModel):
        path: str = Field(default="", description="Test file/dir relative to workspace ('' = all)")
        keyword: str = Field(default="", description="pytest -k expression (optional)")

    async def run_pytest(path: str = "", keyword: str = "") -> str:
        registry.check("local__run_pytest")
        tracer.emit(
            "tool_call", server="local", tool="run_pytest", args={"path": path, "k": keyword}
        )
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"]
        if path:
            target = (root / path).resolve()
            target.relative_to(root)
            cmd.append(str(target))
        if keyword:
            cmd += ["-k", keyword]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=PYTEST_TIMEOUT_S)
            text = out.decode("utf-8", errors="replace")
            text = text[-8000:]  # keep the tail: summary + last failure
            text = f"exit code {proc.returncode}\n{text}"
        except TimeoutError:
            proc.kill()
            text = f"pytest timed out after {PYTEST_TIMEOUT_S}s"
        tracer.emit("tool_result", server="local", tool="run_pytest", chars=len(text))
        return text

    return StructuredTool.from_function(
        coroutine=run_pytest,
        name="local__run_pytest",
        description=(
            "Run the workspace's pytest suite (or a subset) and return exit code plus output tail. "
            "Use it to reproduce a failure before changing code and to verify after."
        ),
        args_schema=PytestArgs,
    )


def local_fs_tools(workspace: Path, registry: ToolRegistry, tracer: Tracer) -> list[StructuredTool]:
    """Read-only workspace tools used when MCP servers are unavailable."""
    root = workspace.resolve()

    def _safe(rel: str) -> Path:
        path = (root / rel).resolve()
        path.relative_to(root)
        return path

    class ListArgs(BaseModel):
        path: str = Field(default=".", description="Directory relative to workspace")

    class ReadArgs(BaseModel):
        path: str = Field(description="File path relative to workspace")

    class SearchArgs(BaseModel):
        query: str = Field(description="Substring to find")
        glob: str = Field(default="*.py", description="Glob under workspace")

    async def list_dir(path: str = ".") -> str:
        registry.check("local__list_dir")
        tracer.emit("tool_call", server="local", tool="list_dir", args={"path": path})
        target = _safe(path)
        if not target.is_dir():
            return f"not a directory: {path}"
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        text = "\n".join(names[:400])
        tracer.emit("tool_result", server="local", tool="list_dir", chars=len(text))
        return text or "(empty)"

    async def read_file(path: str) -> str:
        registry.check("local__read_file")
        tracer.emit("tool_call", server="local", tool="read_file", args={"path": path})
        target = _safe(path)
        if not target.is_file():
            return f"not a file: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")[:12_000]
        tracer.emit("tool_result", server="local", tool="read_file", chars=len(text))
        return text

    async def search_files(query: str, glob: str = "*.py") -> str:
        registry.check("local__search_files")
        tracer.emit(
            "tool_call",
            server="local",
            tool="search_files",
            args={"query": query, "glob": glob},
        )
        hits: list[str] = []
        for p in root.rglob(glob):
            if not p.is_file():
                continue
            try:
                body = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if query in body:
                rel = p.relative_to(root).as_posix()
                hits.append(rel)
            if len(hits) >= 40:
                break
        text = "\n".join(hits) if hits else "(no matches)"
        tracer.emit("tool_result", server="local", tool="search_files", chars=len(text))
        return text

    return [
        StructuredTool.from_function(
            coroutine=list_dir,
            name="local__list_dir",
            description="List a workspace directory",
            args_schema=ListArgs,
        ),
        StructuredTool.from_function(
            coroutine=read_file,
            name="local__read_file",
            description="Read a UTF-8 file",
            args_schema=ReadArgs,
        ),
        StructuredTool.from_function(
            coroutine=search_files,
            name="local__search_files",
            description="Find files whose contents contain query",
            args_schema=SearchArgs,
        ),
    ]
