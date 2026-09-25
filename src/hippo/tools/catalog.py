"""Tool catalog: every tool hippo could bind, an index over their descriptions, and the
small set that is currently *activated* (bound into the next LLM request).

The catalog does not talk to the model. `tools/search.py` wraps it in the two meta-tools
(`tool_search`, `tool_load`) and decides what the model sees; `tool_loop` asks the
catalog which tools to bind at every step.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.tools import BaseTool

# --------------------------------------------------------------------------- entries


@dataclass
class ToolEntry:
    name: str  # qualified, e.g. "github__create_issue"
    server: str  # "github"
    short: str  # "create_issue"
    summary: str  # first line of the description, trimmed
    description: str
    tool: BaseTool
    _schema: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def schema(self) -> dict[str, Any]:
        """OpenAI function schema (what the provider bills for)."""
        if self._schema is None:
            from hippo.metrics import tool_to_openai_schema

            self._schema = tool_to_openai_schema(self.tool)
        return self._schema

    @property
    def parameters(self) -> dict[str, Any]:
        return (self.schema.get("function") or {}).get("parameters") or {}


def _split_name(name: str) -> tuple[str, str]:
    if "__" in name:
        server, short = name.split("__", 1)
        return server, short
    return "local", name


_MCP_SUFFIX = re.compile(r"\s*\(mcp:[^)]*\)\s*$")


def _summary(description: str, short: str) -> str:
    first = (description or "").strip().splitlines()
    line = first[0].strip() if first else ""
    line = _MCP_SUFFIX.sub("", line)
    if not line:
        line = short.replace("_", " ")
    if len(line) > 140:
        line = line[:137].rstrip() + "..."
    return line


def make_entry(tool: BaseTool) -> ToolEntry:
    server, short = _split_name(tool.name)
    desc = (tool.description or "").strip()
    return ToolEntry(
        name=tool.name,
        server=server,
        short=short,
        summary=_summary(desc, short),
        description=desc,
        tool=tool,
    )


# --------------------------------------------------------------------------- retrievers


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (tool name, score) with the best first."""


_STOP = frozenset(
    "a an the of to in for on with and or is are be this that it its from by as at into "
    "using use set new all any my our your please can you i we me us them tell give".split()
)
# Query-side expansion only: a user says "find" or "post", tool names say "search" or "send".
# Each token also keeps itself, so this widens recall without moving exact matches down.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "find": ("search",),
    "look": ("search",),
    "lookup": ("search",),
    "query": ("search", "run"),
    "show": ("get", "list"),
    "fetch": ("get",),
    "read": ("get",),
    "open": ("create",),
    "make": ("create",),
    "add": ("create",),
    "new": ("create",),
    "book": ("create", "event"),
    "schedule": ("create", "event"),
    "meeting": ("event",),
    "post": ("send", "message"),
    "message": ("send",),
    "mail": ("email", "send"),
    "email": ("send",),
    "remove": ("delete",),
    "change": ("update",),
    "edit": ("update",),
    "modify": ("update",),
    "link": ("url",),
    "download": ("get", "url"),
    "ticket": ("issue",),
    "bug": ("issue",),
    "pr": ("pull", "request"),
    "start": ("trigger",),
    "build": ("pipeline",),
    "deploy": ("environment",),
    "silence": ("mute",),
    "close": ("resolve",),
}
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _stem(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; snake_case and camelCase split; light plural stemming."""
    text = _CAMEL.sub(" ", text or "")
    out: list[str] = []
    for tok in _NON_ALNUM.split(text.lower()):
        if not tok or tok in _STOP:
            continue
        out.append(_stem(tok))
    return out


def expand_query(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        out.append(tok)
        for syn in SYNONYMS.get(tok, ()):
            out.append(_stem(syn))
    return out


def _param_text(entry: ToolEntry) -> str:
    """Parameter names and descriptions; they often carry the nouns users say ("subject")."""
    try:
        props = entry.parameters.get("properties") or {}
    except Exception:  # noqa: BLE001 - odd schema, index without params
        return ""
    parts: list[str] = []
    for key, spec in props.items():
        parts.append(str(key))
        if isinstance(spec, dict) and spec.get("description"):
            parts.append(str(spec["description"]))
    return " ".join(parts)


class KeywordRetriever:
    """BM25 over name (weighted), server, description and parameters. Deterministic, no deps."""

    K1 = 1.5
    B = 0.75
    NAME_WEIGHT = 3

    def __init__(self, entries: Sequence[ToolEntry]) -> None:
        self._names = [e.name for e in entries]
        self._docs: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for e in entries:
            toks = (
                tokenize(e.short) * self.NAME_WEIGHT
                + tokenize(e.server) * 2
                + tokenize(e.description)
                + tokenize(_param_text(e))
            )
            c = Counter(toks)
            self._docs.append(c)
            df.update(c.keys())
        n = max(len(entries), 1)
        self._avgdl = sum(sum(c.values()) for c in self._docs) / n if entries else 1.0
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = expand_query(tokenize(query))
        if not q or not self._docs:
            return []
        scores: list[tuple[str, float]] = []
        for name, doc in zip(self._names, self._docs, strict=True):
            dl = sum(doc.values())
            s = 0.0
            for t in q:
                tf = doc.get(t)
                if not tf:
                    continue
                idf = self._idf.get(t, 0.0)
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self._avgdl)
                s += idf * tf * (self.K1 + 1) / denom
            if s > 0:
                scores.append((name, s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]


class EmbeddingRetriever:
    """Dense retrieval through the same ChromaStore hippo uses for memory.

    The collection is keyed by a hash of the catalog so an on-disk index is built once
    per catalog; with `persist_dir=None` it lives in memory for the process.
    """

    def __init__(self, entries: Sequence[ToolEntry], persist_dir: Path | None = None) -> None:
        from hippo.memory.chroma import ChromaStore

        digest = catalog_fingerprint(entries)
        self._store = ChromaStore(persist_dir, f"tools_{digest}")
        if self._store.count() < len(entries):
            for e in entries:
                self._store.add(
                    f"{e.short.replace('_', ' ')} ({e.server}): {e.description}",
                    id=e.name,
                    metadata={"server": e.server},
                )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        hits = self._store.search(query, k=k)
        return [(h["id"], -float(h.get("score") or 0.0)) for h in hits]


def catalog_fingerprint(entries: Iterable[ToolEntry]) -> str:
    h = hashlib.sha1()
    for e in sorted(entries, key=lambda x: x.name):
        h.update(e.name.encode())
        h.update(b"\0")
        h.update(e.description.encode("utf-8", "replace"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def make_retriever(
    kind: str | Retriever, entries: Sequence[ToolEntry], *, index_dir: Path | None = None
) -> Retriever:
    if not isinstance(kind, str):
        return kind
    kind = kind.lower()
    if kind == "keyword":
        return KeywordRetriever(entries)
    if kind == "embedding":
        return EmbeddingRetriever(entries, index_dir)
    raise ValueError(f"unknown retriever: {kind} (keyword|embedding)")


# --------------------------------------------------------------------------- catalog


class ToolCatalog:
    """All tools, an index, and the activated subset that gets bound next."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        retriever: str | Retriever = "keyword",
        always_on: Iterable[str] = (),
        index_dir: Path | None = None,
        max_active: int | None = None,
    ) -> None:
        self.entries: list[ToolEntry] = [make_entry(t) for t in tools]
        self.by_name: dict[str, ToolEntry] = {e.name: e for e in self.entries}
        self.by_short: dict[str, ToolEntry] = {}
        for e in self.entries:
            self.by_short.setdefault(e.short, e)
        self.retriever = make_retriever(retriever, self.entries, index_dir=index_dir)
        self.max_active = max_active
        self._active: list[str] = []
        self.always_on: list[str] = []
        for name in always_on:
            e = self.resolve(name)
            if e is not None and e.name not in self.always_on:
                self.always_on.append(e.name)
        self.searches = 0
        self.loads = 0
        self.search_ms = 0.0

    # ---- lookup

    def __len__(self) -> int:
        return len(self.entries)

    def resolve(self, name: str) -> ToolEntry | None:
        return self.by_name.get(name) or self.by_short.get(name.split("__")[-1])

    def is_known(self, name: str) -> bool:
        return self.resolve(name) is not None

    def all_tools(self) -> list[BaseTool]:
        return [e.tool for e in self.entries]

    def fingerprint(self) -> str:
        return catalog_fingerprint(self.entries)

    # ---- search / activate

    def search(self, query: str, k: int = 5) -> list[ToolEntry]:
        self.searches += 1
        k = max(1, min(int(k or 5), 25))
        t0 = time.perf_counter()
        ranked = self.retriever.search(query, k)
        self.search_ms += (time.perf_counter() - t0) * 1000
        return [self.by_name[n] for n, _ in ranked if n in self.by_name]

    def activate(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        """Returns (activated qualified names, unknown names). Idempotent."""
        self.loads += 1
        activated: list[str] = []
        unknown: list[str] = []
        for raw in names:
            e = self.resolve(str(raw).strip())
            if e is None:
                unknown.append(str(raw))
                continue
            if e.name in self._active:
                self._active.remove(e.name)  # move to most-recent position
            self._active.append(e.name)
            activated.append(e.name)
        if self.max_active is not None:
            while len(self._active) > self.max_active:
                self._active.pop(0)
        return activated, unknown

    def deactivate(self, names: Iterable[str]) -> None:
        for raw in names:
            e = self.resolve(str(raw))
            if e is not None and e.name in self._active:
                self._active.remove(e.name)

    def reset(self) -> None:
        self._active.clear()

    @property
    def active_names(self) -> list[str]:
        out = list(self.always_on)
        out += [n for n in self._active if n not in out]
        return out

    def active_tools(self) -> list[BaseTool]:
        return [self.by_name[n].tool for n in self.active_names]

    # ---- messages for the model

    def unknown_tool_message(self, name: str) -> str:
        e = self.resolve(name)
        if e is None:
            return (
                f"unknown tool: {name}. It is not in the catalog. "
                "Use tool_search to find the right tool name."
            )
        return (
            f"tool {e.name} exists but is not loaded. "
            f'Call tool_load with names=["{e.name}"] first, then call it.'
        )
