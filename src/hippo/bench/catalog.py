"""Synthetic tool universe: JSON specs -> StructuredTools with canned results.

The specs are in MCP shape (`name`, `description`, `inputSchema`) and go through the same
`json_schema_to_model` as real MCP tools, so their OpenAI function schemas cost the same
tokens a real server's would. Calling one never leaves the process: it returns the spec's
`result` if it has one, else an echo of the arguments.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from hippo.tools.mcp_client import _plain, json_schema_to_model
from hippo.tools.registry import ToolRegistry
from hippo.trace import Tracer

DEFAULT_CATALOG = Path(__file__).resolve().parents[3] / "benchmarks" / "catalog"
DEFAULT_CATALOG = DEFAULT_CATALOG / "synthetic_tools.json"


@dataclass
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    result: str | None = None

    @property
    def qualified(self) -> str:
        return f"{self.server}__{self.name}"


@dataclass
class CatalogSpec:
    servers: dict[str, str]
    tools: list[ToolSpec]
    path: Path | None = None
    by_name: dict[str, ToolSpec] = field(init=False)

    def __post_init__(self) -> None:
        self.by_name = {t.qualified: t for t in self.tools}

    def __len__(self) -> int:
        return len(self.tools)


def load_catalog(path: Path | str | None = None) -> CatalogSpec:
    p = Path(path) if path else DEFAULT_CATALOG
    data = json.loads(p.read_text(encoding="utf-8"))
    tools = [
        ToolSpec(
            server=t["server"],
            name=t["name"],
            description=t.get("description") or t["name"],
            input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
            result=t.get("result"),
        )
        for t in data.get("tools") or []
    ]
    return CatalogSpec(servers=dict(data.get("servers") or {}), tools=tools, path=p)


def select_subset(
    catalog: CatalogSpec,
    size: int,
    *,
    must_include: Iterable[str] = (),
    seed: int = 7,
) -> list[ToolSpec]:
    """`size` tools: every `must_include` first, then a seeded shuffle of the rest.

    The same seed gives the same distractors for every mode, so a mode comparison at one
    size is over an identical catalog. Order in the returned list is the catalog order.
    """
    required = [catalog.by_name[n] for n in must_include if n in catalog.by_name]
    required_names = {t.qualified for t in required}
    rest = [t for t in catalog.tools if t.qualified not in required_names]
    rng = random.Random(seed)
    rng.shuffle(rest)
    chosen = required + rest[: max(0, size - len(required))]
    order = {t.qualified: i for i, t in enumerate(catalog.tools)}
    return sorted(chosen, key=lambda t: order[t.qualified])


def make_tool(
    spec: ToolSpec,
    *,
    registry: ToolRegistry | None = None,
    tracer: Tracer | None = None,
    calls: list[dict[str, Any]] | None = None,
) -> StructuredTool:
    """A StructuredTool whose coroutine returns the canned result. `calls` collects invocations."""
    qualified = spec.qualified
    model = json_schema_to_model(qualified, spec.input_schema)
    description = spec.description + f" (mcp:{spec.server})"  # same suffix McpHub adds

    async def _call(**kwargs: Any) -> str:
        payload = {k: _plain(v) for k, v in kwargs.items() if v is not None}
        if registry is not None:
            registry.check(qualified)
        if tracer is not None:
            tracer.emit("tool_call", server=spec.server, tool=spec.name, args=payload)
        if calls is not None:
            calls.append({"tool": qualified, "args": payload})
        text = spec.result or json.dumps(
            {"ok": True, "tool": qualified, "args": payload}, ensure_ascii=False
        )
        if tracer is not None:
            tracer.emit("tool_result", server=spec.server, tool=spec.name, chars=len(text))
        return text

    return StructuredTool.from_function(
        coroutine=_call, name=qualified, description=description, args_schema=model
    )


def make_tools(
    specs: Sequence[ToolSpec],
    *,
    registry: ToolRegistry | None = None,
    tracer: Tracer | None = None,
    calls: list[dict[str, Any]] | None = None,
) -> list[StructuredTool]:
    return [make_tool(s, registry=registry, tracer=tracer, calls=calls) for s in specs]
