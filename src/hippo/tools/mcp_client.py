"""MCP stdio clients from mcp.json, plus a local filesystem fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from hippo.tools.registry import ToolRegistry
from hippo.trace import Tracer

SKIP_SERVERS = frozenset({"seahorse"})  # P2 / M5


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


def json_schema_to_model(name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
    schema = schema or {}
    props: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for key, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        typ = JSON_TYPES.get(spec.get("type", "string"), str)
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
            payload = {k: v for k, v in kwargs.items() if v is not None}
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


class _SessionClient:
    """mcp 1.x adapter: stdio_client + ClientSession behind the 2.x `Client` surface."""

    def __init__(self, params: Any) -> None:
        from contextlib import AsyncExitStack

        self._params = params
        self._stack = AsyncExitStack()
        self._session: Any = None

    async def start(self) -> _SessionClient:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def list_tools(self) -> Any:
        return await self._session.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._session.call_tool(name, arguments)

    async def aclose(self) -> None:
        await self._stack.aclose()


class _ClientV2:
    """mcp 2.x `Client` with the same close method name."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list_tools(self) -> Any:
        return await self._client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._client.call_tool(name, arguments)

    async def aclose(self) -> None:
        await self._client.__aexit__(None, None, None)


async def _connect(params: Any) -> Any:
    """Open a stdio MCP client on either mcp 1.x or 2.x."""
    try:
        from mcp import Client  # mcp >= 2
    except ImportError:
        return await _SessionClient(params).start()
    client = Client(params)
    await client.__aenter__()
    return _ClientV2(client)


def _rewrite_args(args: list[str], workspace: Path) -> list[str]:
    out: list[str] = []
    for a in args:
        if a in {".", "./"}:
            out.append(str(workspace.resolve()))
        else:
            out.append(a)
    return out


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
