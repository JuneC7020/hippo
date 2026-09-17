# Use hippo's memory from Cursor

hippo writes episodes and facts into `.hippo/chroma` (or Seahorse). The same store is an MCP
server, so Cursor can search and add to it without going through the CLI.

hippo itself **skips** the `memory` server in `mcp.json` (recall is injected directly; opening
Chroma twice from the same process is a lock). Cursor is a *different* process, so it should
connect.

## 1. Install once

From the hippo clone:

```bash
pip install -e .
```

Confirm the server starts (it speaks JSON-RPC on stdin; Ctrl-C to quit):

```bash
python -m hippo.mcp_server --env-file .env
```

You should see `hippo-memory: data dir ...` on **stderr**. Tools: `memory_search`,
`memory_remember_fact`, `memory_remember_episode`, `memory_ingest_file`, `memory_status`.

## 2. Point Cursor at it

Cursor Settings → MCP → add a server, or drop this into `.cursor/mcp.json` (user or project).
Replace the two paths with yours; `--env-file` is what makes a relative `HIPPO_DATA_DIR=.hippo`
resolve to the **hippo clone**, not Cursor's workspace.

```json
{
  "mcpServers": {
    "hippo-memory": {
      "command": "python",
      "args": [
        "-m", "hippo.mcp_server",
        "--env-file", "D:/path/to/hippo/.env"
      ]
    }
  }
}
```

If `python -m hippo.mcp_server` is not on PATH (no editable install), use the venv's python:

```json
"command": "D:/path/to/hippo/.venv/Scripts/python.exe"
```

Reload MCP. `memory_status` should report the same backend and data dir as `hippo memory`.

## 3. The 30-second demo

1. In a terminal, from the hippo clone, run demo 1 (or any `hippo run` that finishes):

   ```bash
   scripts/demo1_onboarding.sh
   ```

2. In Cursor, with the server connected, ask:

   > Use hippo-memory: were any invoicely tests failing last time, and which module handles discounts?

3. Cursor should call `memory_search` and answer from the episode/facts hippo stored — no repo
   tools required.

`scripts/demo3_mcp_share.py` does the same round-trip without Cursor, so CI and reviewers can
verify the protocol even if they never open the IDE.

## Tools

| tool | when to call |
| --- | --- |
| `memory_search` | before answering questions about a repo hippo has worked on |
| `memory_remember_fact` | a durable path / convention / decision |
| `memory_remember_episode` | end of a session: what was asked, concluded, left open |
| `memory_ingest_file` | seed from a README or ADR |
| `memory_status` | which backend, where it lives, how many rows |

Writes from Cursor show up in `hippo memory "..."` and the next `hippo run` recall. Near-duplicate
facts are merged (token Jaccard ≥ 0.8).

Do not open the **same Chroma directory from two processes at once** (`hippo run` and Cursor
calling `memory_search` in parallel). Sequential access is fine; the lock is released when the
process exits. Seahorse does not have this limitation.
