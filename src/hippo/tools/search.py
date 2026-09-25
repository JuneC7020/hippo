"""Tool search: two meta-tools that stand in for a large tool catalog.

Exposure modes (a benchmark variable, see docs/token-bottlenecks.md B4):

  all            bind every tool on every step (baseline)
  search         tool_search -> names + one-line summaries; tool_load -> the *next* step
                 binds those schemas. Two extra round trips, model picks exactly what to bind.
  search_schema  tool_search returns the full schemas of the top-k hits and activates them
                 at once. One extra round trip, but all k schemas get bound (and the schema
                 text sits in the history from then on).

`build_exposure` gives `tool_loop` a per-step `tools_provider` so the bound set can grow.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from hippo.tools.catalog import Retriever, ToolCatalog, ToolEntry
from hippo.tools.registry import ToolRegistry
from hippo.trace import Tracer

EXPOSURES = ("all", "search", "search_schema")
META_TOOL_NAMES = frozenset({"tool_search", "tool_load"})

SEARCH_SYSTEM_NOTE = """
Tools are not all listed. Use tool_search(query) to find tools by what they do; it returns
names and one-line summaries. Then call tool_load(names=[...]) with the exact names you need,
and they become callable on your next turn. Search with a few specific words (the object and
the action, e.g. "create github issue"), load only what you will use, then do the task."""

SEARCH_SCHEMA_SYSTEM_NOTE = """
Tools are not all listed. Use tool_search(query) to find tools by what they do; every tool it
returns is loaded immediately and callable on your next turn, with its full schema in the
result. Search with a few specific words (the object and the action, e.g. "create github
issue"), then do the task."""


def normalize_exposure(value: str | None) -> str:
    v = (value or "all").strip().lower().replace("-", "_")
    if v in {"", "all", "none"}:
        return "all"
    if v in {"search", "search_load", "load"}:
        return "search"
    if v in {"search_schema", "schema"}:
        return "search_schema"
    raise ValueError(f"unknown tool exposure: {value!r} (all|search|search-schema)")


# --------------------------------------------------------------------------- rendering


def _compact_schema(entry: ToolEntry) -> str:
    return json.dumps(entry.parameters, separators=(",", ":"), ensure_ascii=False)


def render_hits(hits: Sequence[ToolEntry], *, query: str, with_schema: bool) -> str:
    if not hits:
        return (
            f'No tools matched "{query}". Try other words for the object or the action, '
            "or a service name (github, slack, jira, stripe, ...)."
        )
    if with_schema:
        lines = [f'Loaded {len(hits)} tool(s) for "{query}"; they are callable now:']
        for e in hits:
            lines.append(f"- {e.name}: {e.summary}")
            lines.append(f"  parameters: {_compact_schema(e)}")
        return "\n".join(lines)
    lines = [f'Found {len(hits)} tool(s) for "{query}":']
    for e in hits:
        lines.append(f"- {e.name}: {e.summary}")
    lines.append("Call tool_load with the exact names you need, then call them.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- meta tools


class SearchArgs(BaseModel):
    query: str = Field(description="What the tool should do, a few specific words")
    k: int | None = Field(default=None, description="How many candidates to return (default 5)")


class LoadArgs(BaseModel):
    names: list[str] = Field(description="Exact tool names from tool_search to make callable")


def meta_tools(
    catalog: ToolCatalog,
    *,
    mode: str,
    registry: ToolRegistry | None = None,
    tracer: Tracer | None = None,
    k_default: int = 5,
) -> list[StructuredTool]:
    mode = normalize_exposure(mode)
    if mode == "all":
        return []
    with_schema = mode == "search_schema"

    def _check(name: str) -> None:
        if registry is not None:
            registry.check(name)

    def _emit(kind: str, **payload: Any) -> None:
        if tracer is not None:
            tracer.emit(kind, **payload)

    async def tool_search(query: str, k: int | None = None) -> str:
        _check("tool_search")
        kk = int(k or k_default)
        _emit("tool_call", server="hippo", tool="tool_search", args={"query": query, "k": kk})
        hits = catalog.search(query, kk)
        if with_schema:
            catalog.activate([e.name for e in hits])
        text = render_hits(hits, query=query, with_schema=with_schema)
        _emit(
            "tool_search",
            query=query,
            k=kk,
            mode=mode,
            hits=[e.name for e in hits],
            activated=[e.name for e in hits] if with_schema else [],
        )
        _emit("tool_result", server="hippo", tool="tool_search", chars=len(text))
        return text

    async def tool_load(names: list[str]) -> str:
        _check("tool_load")
        _emit("tool_call", server="hippo", tool="tool_load", args={"names": names})
        activated, unknown = catalog.activate(names)
        parts: list[str] = []
        if activated:
            parts.append("Loaded and callable on your next turn: " + ", ".join(activated))
        if unknown:
            parts.append(
                "Unknown (not in catalog, check the exact name from tool_search): "
                + ", ".join(unknown)
            )
        text = "\n".join(parts) or "Nothing to load."
        _emit("tool_load", names=names, activated=activated, unknown=unknown)
        _emit("tool_result", server="hippo", tool="tool_load", chars=len(text))
        return text

    tools = [
        StructuredTool.from_function(
            coroutine=tool_search,
            name="tool_search",
            description=(
                "Find tools by what they do. Returns matching tool names with one-line "
                "summaries" + (" and loads them for use." if with_schema else ".")
            ),
            args_schema=SearchArgs,
        )
    ]
    if not with_schema:
        tools.append(
            StructuredTool.from_function(
                coroutine=tool_load,
                name="tool_load",
                description=(
                    "Make tools callable by exact name (from tool_search). Load only the tools "
                    "you will actually use."
                ),
                args_schema=LoadArgs,
            )
        )
    return tools


# --------------------------------------------------------------------------- exposure


@dataclass
class Exposure:
    mode: str
    tools_provider: Callable[[], list[BaseTool]]
    catalog: ToolCatalog | None
    meta: list[BaseTool]
    system_note: str

    def on_unknown(self, name: str) -> str:
        if self.catalog is None:
            return f"unknown tool: {name}"
        return self.catalog.unknown_tool_message(name)


def build_exposure(
    tools: Sequence[BaseTool],
    *,
    mode: str,
    registry: ToolRegistry | None = None,
    tracer: Tracer | None = None,
    retriever: str | Retriever = "keyword",
    k: int = 5,
    always_on: Iterable[str] = (),
    index_dir: Path | None = None,
    max_active: int | None = None,
) -> Exposure:
    """Decide what `tool_loop` binds at each step for the given mode."""
    mode = normalize_exposure(mode)
    tools = list(tools)
    if mode == "all":
        return Exposure(mode, lambda: tools, None, [], "")
    catalog = ToolCatalog(
        tools,
        retriever=retriever,
        always_on=always_on,
        index_dir=index_dir,
        max_active=max_active,
    )
    meta = meta_tools(catalog, mode=mode, registry=registry, tracer=tracer, k_default=k)

    def provider() -> list[BaseTool]:
        return list(meta) + catalog.active_tools()

    note = SEARCH_SCHEMA_SYSTEM_NOTE if mode == "search_schema" else SEARCH_SYSTEM_NOTE
    return Exposure(mode, provider, catalog, list(meta), note)
