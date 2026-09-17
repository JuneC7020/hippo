# Architecture

The maintained diagram is the Mermaid block in [`../README.md`](../README.md#architecture).
Plain-text version for slides:

```
hippo CLI (Typer)  run / resume / tasks / memory / trace
  └─ Orchestrator (LangGraph, SQLite checkpoints -> hippo resume)
       ├─ Planner   : 1-4 subtasks {goal, extra tools, step budget}; re-plans once on escalation
       ├─ Worker    : bounded tool loop on ONE subtask -> {result, evidence, confidence}
       │              always has read tools + run_pytest; write tools only if granted AND --write
       └─ Reviewer  : judges on the worker's tool log -> approve | revise (1 retry) | escalate
  ├─ Tool layer (MCP client; each server in its own asyncio task)
  │    ├─ filesystem MCP  (npx @modelcontextprotocol/server-filesystem)
  │    ├─ git MCP         (python -m mcp_server_git, repo root auto-detected)
  │    ├─ hippo-memory MCP (python -m hippo.mcp_server; Cursor; CLI skips)
  │    └─ local run_pytest (fixed command, workspace-confined, time-limited)
  ├─ Memory layer
  │    ├─ working context : token budget -> old tool turns compressed to one note
  │    ├─ episodes        : what was asked / concluded / left failing   (table: episodes)
  │    └─ facts           : durable repo knowledge, deduped on write     (table: facts)
  │         backend: Seahorse (HIPPO_MEMORY_BACKEND=seahorse) or Chroma fallback
  └─ SQLite: task rows (hippo.db), LangGraph checkpoints (checkpoints.db), JSONL traces
```

Demo transcripts: `demo1.txt`, `demo2.txt`; rendered to `demo1.svg`, `demo2.svg` by
`../scripts/render_demo_svg.py`. Cursor setup: `cursor.md`. Interview deck: `slides.html`.
