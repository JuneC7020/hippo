"""Fake LLMs shared by the tests. No network."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage


class FakeLLM:
    """Routes each call by the system prompt it receives and pops a scripted reply."""

    def __init__(self, script: dict[str, list]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def bind_tools(self, _tools):
        return self

    def _role(self, messages) -> str:
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        first = system.split("\n", 1)[0].lower()
        for key in ("planner", "worker", "reviewer"):
            if f"the {key} of hippo" in first:
                return key
        if "final answer" in first:
            return "final answer"
        if "compress" in first:
            return "compress"
        return "other"

    def _reply(self, messages):
        role = self._role(messages)
        self.calls.append(role)
        queue = self.script.get(role) or []
        item = queue.pop(0) if queue else {"content": "{}"}
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=item["content"] if isinstance(item, dict) else str(item))

    def invoke(self, messages, **_):
        return self._reply(messages)

    async def ainvoke(self, messages, **_):
        return self._reply(messages)


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


class ScriptedToolLLM:
    """A fake that behaves like a competent tool user for ONE target tool.

    Looks at what is bound on each step: if the target is bound and not yet called, call it;
    if only meta-tools are bound, search; if the search result is back, load the target; once
    the target has answered, reply with text. `bound_per_step` records what it saw each turn,
    which is what the tool-search tests assert on.
    """

    def __init__(
        self,
        target: str,
        args: dict[str, Any],
        *,
        query: str | None = None,
        usage: Callable[[int], dict[str, Any]] | None = None,
    ) -> None:
        self.target = target
        self.args = args
        self.query = query or target.split("__")[-1].replace("_", " ")
        self.usage = usage
        self.bound: list[str] = []
        self.bound_per_step: list[list[str]] = []
        self.n = 0

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages, **_):
        return AIMessage(content="summary")

    async def ainvoke(self, messages, **_):
        self.n += 1
        self.bound_per_step.append(list(self.bound))
        called = any(
            (getattr(m, "tool_calls", None) or [{}])[0].get("name") == self.target for m in messages
        )
        last = str(getattr(messages[-1], "content", ""))
        cid = f"c{self.n}"
        if called:
            msg = AIMessage(content=f"done: {last[:80]}")
        elif self.target in self.bound:
            msg = tool_call(self.target, self.args, cid)
        elif "tool_load" in self.bound and last.startswith("Found"):
            msg = tool_call("tool_load", {"names": [self.target]}, cid)
        elif "tool_search" in self.bound:
            msg = tool_call("tool_search", {"query": self.query}, cid)
        else:
            msg = AIMessage(content="no suitable tool")
        if self.usage is not None:
            msg.usage_metadata = self.usage(self.n)
        return msg
