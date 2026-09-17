#!/usr/bin/env python3
"""Demo 3: the memory hippo wrote is reachable over MCP (what Cursor would call).

    python scripts/demo3_mcp_share.py

Uses HIPPO_DATA_DIR from .env (same store as `hippo memory`).
Does not hold the Chroma dir open in this process — two PersistentClients on one
folder deadlock on Windows. CLI recall runs as a child, then MCP as another.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


def _cli_memory(query: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "hippo.cli", "memory", query],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (proc.stdout or proc.stderr or "").strip()


async def _mcp_search(query: str) -> tuple[str, str]:
    from mcp.client.stdio import StdioServerParameters

    from hippo.tools.mcp_client import _result_to_text, _TaskScopedClient

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hippo.mcp_server", "--env-file", str(ROOT / ".env")],
        env=env,
        cwd=str(ROOT),
    )
    client = await _TaskScopedClient(params).start()
    try:
        status = _result_to_text(await client.call_tool("memory_status", {}))
        search = _result_to_text(await client.call_tool("memory_search", {"query": query}))
        return status, search
    finally:
        await client.aclose()


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    query = "invoicely tests discounts money"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=== CLI  hippo memory (child process, then exit) ===")
    cli = _cli_memory(query)
    print(cli or "(empty — run scripts/demo1_onboarding first)")
    print()

    print("=== MCP  memory_status + memory_search (what Cursor calls) ===")
    status, search = asyncio.run(_mcp_search(query))
    print(status)
    print()
    print(search)
    same = any(w in search.lower() for w in ("invoicely", "pricing", "pytest", "cents", "discount"))
    print()
    print("shared store:" + (" yes" if same else " no (empty store or different data dir)"))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
