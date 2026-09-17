"""hippo-memory: an MCP server that exposes hippo's long-term memory to other clients.

The point: the memory hippo builds while working in your terminal is not locked inside hippo.
Cursor (or any MCP client) can search the same episodes/facts and add to them.

    python -m hippo.mcp_server                       # stdio, uses env / .env like the CLI
    python -m hippo.mcp_server --env-file /path/.env # when launched from another cwd (Cursor)

Tools: memory_search, memory_remember_fact, memory_remember_episode, memory_ingest_file,
memory_status. No LLM call is made here: the backend embeds (Chroma local model or Seahorse).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

# Chroma / sentence-transformers print progress to stdout; stdout *is* the MCP transport.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")

from mcp.server.fastmcp import FastMCP  # noqa: E402

CHUNK_CHARS = 700
MAX_CHUNKS = 40

mcp = FastMCP(
    "hippo-memory",
    instructions=(
        "Long-term memory shared with the hippo CLI agent. Call memory_search before answering "
        "questions about a repository the user has worked on; call memory_remember_fact when you "
        "learn something durable (paths, conventions, decisions)."
    ),
)

_manager: Any = None


def _mem() -> Any:
    """Lazily build the MemoryManager.

    stdout *is* the MCP transport here. Chroma (telemetry, model download progress) and
    Seahorse print to stdout on first use, which would corrupt the JSON-RPC stream and hang the
    client, so anything touching a backend runs with stdout redirected to stderr.
    """
    global _manager
    if _manager is None:
        from hippo.config import load_settings
        from hippo.memory.manager import build_memory_manager

        with redirect_stdout(sys.stderr):
            _manager = build_memory_manager(load_settings())
    return _manager


def quiet(fn):
    """Keep backend chatter off the protocol stream (stdout)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)

    return wrapper


def _format(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "(no memories match)"
    lines = []
    for h in hits:
        meta = h.get("metadata") or {}
        kind = meta.get("kind") or "memory"
        src = meta.get("source") or meta.get("task_id") or ""
        tag = f"[{kind}{' ' + str(src) if src else ''}]"
        lines.append(f"- {tag} {(h.get('text') or '').strip()}")
    return "\n".join(lines)


@quiet
def search_memory(query: str, k: int = 5) -> str:
    return _format(_mem().recall(query, k=max(1, min(int(k), 20))))


@quiet
def remember_fact(text: str, source: str = "cursor") -> str:
    text = text.strip()
    if not text:
        return "nothing to store"
    return f"stored fact {_mem().remember_fact(text, source=source)}"


@quiet
def remember_episode(summary: str, source: str = "cursor") -> str:
    summary = summary.strip()
    if not summary:
        return "nothing to store"
    return f"stored episode {_mem().remember_episode(summary, source=source)}"


@quiet
def ingest_file(path: str, max_chunks: int = 20) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        return f"not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    chunks = chunk_text(text)[: max(1, min(int(max_chunks), MAX_CHUNKS))]
    mem = _mem()
    ids = [mem.remember_fact(f"{p.name}: {c}", source=str(p)) for c in chunks]
    return f"ingested {len(ids)} chunk(s) from {p.name}"


@quiet
def status() -> str:
    mem = _mem()
    from hippo.config import load_settings

    s = load_settings()
    lines = [f"backend: {mem.backend}", f"data dir: {Path(s.hippo_data_dir).resolve()}"]
    for name, store in (("episodes", mem.episodes), ("facts", mem.facts)):
        count = getattr(store, "count", None)
        n = count() if callable(count) else "?"
        table = getattr(store, "collection", None) or getattr(store, "table_name", name)
        lines.append(f"{name}: {table} ({n} rows)")
    return "\n".join(lines)


# Async wrappers: FastMCP runs *sync* tools in a worker thread, which deadlocks Chroma
# against the stdio transport on Windows. Async tools stay on the server event loop.


@mcp.tool()
async def memory_search(query: str, k: int = 5) -> str:
    """Search hippo's long-term memory (durable facts + past episodes) for a question."""
    return search_memory(query, k)


@mcp.tool()
async def memory_remember_fact(text: str, source: str = "cursor") -> str:
    """Store one durable fact (a path, a convention, a decision). Near-duplicates are merged."""
    return remember_fact(text, source)


@mcp.tool()
async def memory_remember_episode(summary: str, source: str = "cursor") -> str:
    """Store what happened in this session: what was asked, concluded, changed or left open."""
    return remember_episode(summary, source)


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-aware chunks of about `size` chars."""
    chunks: list[str] = []
    cur = ""
    for para in text.replace("\r\n", "\n").split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > size:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > size * 2:  # a single huge paragraph
            chunks.append(cur[:size])
            cur = cur[size:]
    if cur:
        chunks.append(cur)
    return chunks


@mcp.tool()
async def memory_ingest_file(path: str, max_chunks: int = 20) -> str:
    """Read a text file (README, ADR, notes) and store it as facts tagged with its path."""
    return ingest_file(path, max_chunks)


@mcp.tool()
async def memory_status() -> str:
    """Which backend is in use, where it lives, and how many memories it holds."""
    return status()


def configure_launch(*, env_file: str | None = None, data_dir: str | None = None) -> Path:
    """Resolve HIPPO_DATA_DIR so Cursor (cwd != hippo/) still hits the same store as the CLI."""
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        data = os.environ.get("HIPPO_DATA_DIR", ".hippo")
        if not Path(data).is_absolute():
            os.environ["HIPPO_DATA_DIR"] = str((Path(env_file).parent / data).resolve())
    if data_dir:
        os.environ["HIPPO_DATA_DIR"] = str(Path(data_dir).resolve())
    resolved = Path(os.environ.get("HIPPO_DATA_DIR", ".hippo")).resolve()
    os.environ["HIPPO_DATA_DIR"] = str(resolved)
    return resolved


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="hippo-memory MCP server (stdio)")
    ap.add_argument("--env-file", help=".env to load first (for launches from another cwd)")
    ap.add_argument("--data-dir", help="override HIPPO_DATA_DIR")
    args = ap.parse_args(argv)
    data = configure_launch(env_file=args.env_file, data_dir=args.data_dir)
    print(f"hippo-memory: data dir {data}", file=sys.stderr)
    # Open the backend *before* stdio JSON-RPC starts. Chroma/onnx print on import;
    # if that hits stdout after the protocol is live, the client never sees a tool result.
    try:
        _mem()
        print(f"hippo-memory: backend {_manager.backend}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"hippo-memory: backend init failed: {exc}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
