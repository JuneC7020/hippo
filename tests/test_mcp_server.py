"""hippo-memory MCP server: chunking, in-process tools, stdio handshake."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import hippo.mcp_server as srv
from hippo.mcp_server import chunk_text


def test_chunk_text_respects_paragraphs_and_size() -> None:
    text = "\n\n".join(f"para {i} " + "x" * 200 for i in range(8))
    chunks = chunk_text(text, size=500)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0].startswith("para 0")
    assert "para 7" in chunks[-1]


def test_chunk_text_splits_one_huge_paragraph() -> None:
    chunks = chunk_text("y" * 3000, size=700)
    assert len(chunks) >= 2 and sum(len(c) for c in chunks) == 3000


def test_tools_search_remember_ingest(tmp_path: Path, monkeypatch) -> None:
    """Tool bodies without stdio: same functions Cursor would call, against a temp Chroma."""
    monkeypatch.setenv("HIPPO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HIPPO_MEMORY_BACKEND", "chroma")
    srv._manager = None
    note = tmp_path / "notes.md"
    note.write_text("# invoicely\n\nMoney is integer cents.\n", encoding="utf-8")

    stored = srv.remember_fact("invoicely stores money in integer cents")
    assert stored.startswith("stored fact")
    ep = srv.remember_episode("Onboarded invoicely; two pricing tests were failing")
    assert ep.startswith("stored episode")
    ingested = srv.ingest_file(str(note))
    assert ingested.startswith("ingested")
    hit = srv.search_memory("money unit")
    assert "integer cents" in hit and "[fact" in hit
    st = srv.status()
    assert "backend: chroma" in st and "facts:" in st
    assert srv.ingest_file(str(tmp_path / "nope")).startswith("not a file")
    assert srv.remember_fact("  ") == "nothing to store"


def test_stdio_lists_tools(tmp_path: Path) -> None:
    """JSON-RPC handshake. Opening Chroma in the server while the parent also has it
    loaded deadlocks on Windows (file lock); tool bodies are tested in-process above."""
    from mcp.client.stdio import StdioServerParameters

    from hippo.tools.mcp_client import _TaskScopedClient

    env = {
        **os.environ,
        "HIPPO_DATA_DIR": str(tmp_path / "data"),
        "HIPPO_MEMORY_BACKEND": "chroma",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "ANONYMIZED_TELEMETRY": "False",
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "hippo.mcp_server"], env=env)

    async def go() -> list[str]:
        client = await _TaskScopedClient(params).start()
        try:
            return sorted(t.name for t in (await client.list_tools()).tools)
        finally:
            await client.aclose()

    names = asyncio.run(asyncio.wait_for(go(), timeout=20))
    assert names == [
        "memory_ingest_file",
        "memory_remember_episode",
        "memory_remember_fact",
        "memory_search",
        "memory_status",
    ]


def test_configure_launch_resolves_data_dir_next_to_env(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("HIPPO_DATA_DIR=.hippo\nHIPPO_MEMORY_BACKEND=chroma\n", encoding="utf-8")
    monkeypatch.delenv("HIPPO_DATA_DIR", raising=False)
    resolved = srv.configure_launch(env_file=str(env))
    assert resolved == (tmp_path / ".hippo").resolve()
    override = srv.configure_launch(data_dir=str(tmp_path / "other"))
    assert override == (tmp_path / "other").resolve()


def test_mcp_json_lists_memory_and_hippo_skips_it() -> None:
    from hippo.tools.mcp_client import SKIP_SERVERS

    cfg = json.loads((Path(__file__).resolve().parents[1] / "mcp.json").read_text(encoding="utf-8"))
    assert "memory" in cfg["mcpServers"]
    assert "memory" in SKIP_SERVERS
