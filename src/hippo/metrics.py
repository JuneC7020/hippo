"""Token and cost accounting for one run.

Two numbers hippo never had: what the provider actually billed per request
(`AIMessage.usage_metadata`), and how much of every request is tool schemas
(`schema_tokens`). The second is what tool search is supposed to shrink; the first is
what proves it. Both are split out per step so a trace can show *where* tokens went.

Nothing here talks to the network. When a message carries no usage (fake LLMs, some
proxies) the meter falls back to a tiktoken estimate and says so (`estimated=True`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

FALLBACK_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _encoder(model: str):
    """tiktoken encoder for `model`, o200k_base for unknown slugs (DeepSeek etc.), None offline."""
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding(FALLBACK_ENCODING)
    except Exception:  # noqa: BLE001 - tiktoken missing or BPE files not cached
        return None


def count_text_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = _encoder(model)
    if enc is None:
        return len(text) // 4
    return len(enc.encode(text, disallowed_special=()))


def tool_to_openai_schema(tool: BaseTool) -> dict[str, Any]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(tool)


def schema_tokens_each(tools: Sequence[BaseTool], model: str = "gpt-4o-mini") -> dict[str, int]:
    """Tokens of each tool's OpenAI function definition as the provider would see it.

    Serialized compactly; providers re-encode it their own way, so treat this as a stable
    estimate for comparing configurations, not an invoice.
    """
    out: dict[str, int] = {}
    for tool in tools:
        blob = json.dumps(tool_to_openai_schema(tool), separators=(",", ":"), ensure_ascii=False)
        out[tool.name] = count_text_tokens(blob, model)
    return out


def schema_tokens(tools: Sequence[BaseTool], model: str = "gpt-4o-mini") -> int:
    """Total tokens of the tool-definitions block. 0 for no tools."""
    if not tools:
        return 0
    return sum(schema_tokens_each(tools, model).values())


@dataclass
class Prices:
    """USD per 1M tokens. Zero means unknown; cost is then reported as 0."""

    input: float = 0.0
    cached_input: float = 0.0
    output: float = 0.0

    def cost(self, input_tokens: int, cache_read: int, output_tokens: int) -> float:
        cached = min(cache_read, input_tokens)
        fresh = input_tokens - cached
        cached_price = self.cached_input or self.input
        total = fresh * self.input + cached * cached_price + output_tokens * self.output
        return total / 1_000_000

    @property
    def known(self) -> bool:
        return bool(self.input or self.output)


@dataclass
class StepUsage:
    label: str
    step: int
    input_tokens: int
    output_tokens: int
    cache_read: int
    tool_schema_tokens: int
    n_bound_tools: int
    estimated: bool
    provider_cost_usd: float | None = None


@dataclass
class UsageMeter:
    """Accumulates per-request usage and per-call tool result sizes for one run."""

    model: str = "gpt-4o-mini"
    steps: list[StepUsage] = field(default_factory=list)
    tool_result_tokens: int = 0
    search_result_tokens: int = 0
    n_tool_results: int = 0
    llm_ms: float = 0.0  # wall time spent inside model calls
    meta_tool_names: frozenset[str] = frozenset({"tool_search", "tool_load"})
    _schema_cache: dict[str, int] = field(default_factory=dict, repr=False)

    # ---- recording

    def schema_tokens(self, tools: Sequence[BaseTool]) -> int:
        """Like `schema_tokens()` but cached per tool name for the life of the run."""
        total = 0
        missing = [t for t in tools if t.name not in self._schema_cache]
        if missing:
            self._schema_cache.update(schema_tokens_each(missing, self.model))
        for t in tools:
            total += self._schema_cache[t.name]
        return total

    def record(
        self,
        message: Any,
        *,
        messages_sent: Sequence[BaseMessage],
        bound_tools: Sequence[BaseTool],
        step: int,
        label: str = "run",
    ) -> StepUsage:
        usage = getattr(message, "usage_metadata", None) or {}
        schema = self.schema_tokens(bound_tools)
        estimated = not usage
        if usage:
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            details = usage.get("input_token_details") or {}
            cache_read = int(details.get("cache_read") or 0)
        else:
            from hippo.agent.context import count_tokens

            input_tokens = count_tokens(list(messages_sent), self.model) + schema
            output_tokens = count_text_tokens(_text(getattr(message, "content", "")), self.model)
            for call in getattr(message, "tool_calls", None) or []:
                output_tokens += count_text_tokens(json.dumps(call, default=str), self.model)
            cache_read = 0
        row = StepUsage(
            label=label,
            step=step,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            tool_schema_tokens=schema,
            n_bound_tools=len(bound_tools),
            estimated=estimated,
            provider_cost_usd=_provider_cost(message),
        )
        self.steps.append(row)
        return row

    def record_tool_result(self, name: str, text: str) -> int:
        n = count_text_tokens(text, self.model)
        self.n_tool_results += 1
        if name.split("__")[-1] in self.meta_tool_names:
            self.search_result_tokens += n
        else:
            self.tool_result_tokens += n
        return n

    # ---- totals

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def cache_read(self) -> int:
        return sum(s.cache_read for s in self.steps)

    @property
    def tool_schema_tokens(self) -> int:
        """Schema tokens summed over requests, i.e. what was actually re-sent."""
        return sum(s.tool_schema_tokens for s in self.steps)

    @property
    def n_requests(self) -> int:
        return len(self.steps)

    @property
    def estimated(self) -> bool:
        return any(s.estimated for s in self.steps)

    @property
    def provider_cost_usd(self) -> float | None:
        costs = [s.provider_cost_usd for s in self.steps if s.provider_cost_usd is not None]
        return sum(costs) if costs else None

    def cost_usd(self, prices: Prices | None) -> float | None:
        """Provider-reported cost when available, else the price table, else None."""
        if self.provider_cost_usd is not None:
            return self.provider_cost_usd
        if prices is None or not prices.known:
            return None
        return prices.cost(self.input_tokens, self.cache_read, self.output_tokens)

    def bound_tools_per_step(self) -> list[int]:
        return [s.n_bound_tools for s in self.steps]

    def as_dict(self, prices: Prices | None = None) -> dict[str, Any]:
        return {
            "n_requests": self.n_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read": self.cache_read,
            "tool_schema_tokens": self.tool_schema_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "search_result_tokens": self.search_result_tokens,
            "n_bound_tools_per_step": self.bound_tools_per_step(),
            "llm_ms": round(self.llm_ms, 1),
            "estimated": self.estimated,
            "cost_usd": self.cost_usd(prices),
        }

    def step_dicts(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else str(b.get("text", "") if isinstance(b, dict) else b)
            for b in content
        )
    return str(content or "")


def _provider_cost(message: Any) -> float | None:
    """OpenRouter puts billed cost in usage when asked (`usage: {include: true}`)."""
    meta = getattr(message, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    cost = usage.get("cost") if isinstance(usage, dict) else None
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None
