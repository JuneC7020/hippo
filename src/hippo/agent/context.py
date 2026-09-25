"""Working-context management: token counting and compression of old turns.

Design decision #2 (PLAN.md): summary compression keeps *continuity* inside a
run; vector recall provides *selective precision* across runs. This module is
only the former. It never touches long-term memory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

COMPRESS_SYSTEM = (
    "You compress an agent's working context. Summarize the conversation turns "
    "below (tool calls and their results) into a dense note the agent can keep "
    "working from: files/paths seen, key facts found, what was tried, what failed, "
    "what is still unknown. Keep concrete identifiers (paths, names, numbers). "
    "Plain text, no markdown, under 250 words."
)

KEEP_TAIL = 4  # most recent non-system messages that are never compressed


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and "text" in block:
                out.append(str(block["text"]))
            else:
                out.append(str(getattr(block, "text", block)))
        return "\n".join(out)
    return str(content)


def count_tokens(
    messages: list[BaseMessage], model: str = "gpt-4o-mini", *, fixed_overhead: int = 0
) -> int:
    """Approximate prompt tokens with tiktoken (4 chars/token fallback).

    `fixed_overhead` is what the provider adds outside the messages, mainly the
    tool-definitions block; callers get it from `hippo.metrics.schema_tokens`. Without it,
    a budget check on a run with many tools sees only a fraction of the real prompt.
    """
    from hippo.metrics import count_text_tokens

    total = fixed_overhead
    for msg in messages:
        body = _text(getattr(msg, "content", ""))
        for call in getattr(msg, "tool_calls", None) or []:
            body += " " + str(call.get("args") if isinstance(call, dict) else call)
        total += 4 + count_text_tokens(body, model)
    return total


def _render_turns(turns: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in turns:
        if isinstance(msg, AIMessage):
            calls = getattr(msg, "tool_calls", None) or []
            if calls:
                for call in calls:
                    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", "")
                    lines.append(f"ASSISTANT called {name}({args})")
            text = _text(msg.content).strip()
            if text:
                lines.append(f"ASSISTANT: {text[:1500]}")
        elif isinstance(msg, ToolMessage):
            lines.append(f"TOOL RESULT: {_text(msg.content).strip()[:2500]}")
        elif isinstance(msg, HumanMessage):
            lines.append(f"USER: {_text(msg.content).strip()[:1500]}")
        elif isinstance(msg, SystemMessage):
            lines.append(f"(context) {_text(msg.content).strip()[:800]}")
    return "\n".join(lines)


def _tail_start(messages: list[BaseMessage], keep_tail: int) -> int:
    """Index where the protected tail begins; never split an AI tool-call from its results."""
    idx = max(1, len(messages) - keep_tail)
    while idx < len(messages) and isinstance(messages[idx], ToolMessage):
        idx -= 1
    return max(1, idx)


def compress_messages(
    messages: list[BaseMessage],
    *,
    summarize: Callable[[str], str],
    token_budget: int,
    model: str = "gpt-4o-mini",
    keep_tail: int = KEEP_TAIL,
    fixed_overhead: int = 0,
) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    """If over budget, replace old turns with one summary message.

    Layout in/out: [system, human_task, ...turns...]. The system message and the
    first human message are preserved; older turns become a single
    `SystemMessage("Compressed context: ...")`; the last `keep_tail` turns stay
    verbatim. Returns (messages, info) where info is None when nothing happened.
    `fixed_overhead` (tool schemas) counts against the budget but cannot be compressed.
    """
    before = count_tokens(messages, model, fixed_overhead=fixed_overhead)
    if before <= token_budget or len(messages) <= 2 + keep_tail:
        return messages, None

    head = messages[:2]
    tail_start = _tail_start(messages, keep_tail)
    middle = messages[2:tail_start]
    tail = messages[tail_start:]
    if not middle:
        return messages, None

    # Fold a previous compression note into the new one instead of stacking them.
    prior = [m for m in middle if isinstance(m, SystemMessage)]
    body = _render_turns(middle)
    if prior:
        body = "Earlier compressed context:\n" + _text(prior[0].content) + "\n\n" + body

    summary = summarize(body).strip()
    note = SystemMessage(content="Compressed context from earlier in this run:\n" + summary)
    new_messages = head + [note] + tail
    after = count_tokens(new_messages, model, fixed_overhead=fixed_overhead)
    info = {
        "before_tokens": before,
        "after_tokens": after,
        "dropped_messages": len(middle),
        "summary_chars": len(summary),
        "fixed_overhead": fixed_overhead,
    }
    return new_messages, info
