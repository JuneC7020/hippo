# Architecture (M4 README diagram source)

```
hippo CLI (Typer)
  └─ Orchestrator (LangGraph)
       ├─ Planner
       ├─ Worker(s)
       └─ Reviewer
  ├─ Tool Layer (MCP client) — filesystem, git, seahorse
  ├─ Memory Layer — working / summary / episodic / semantic
  └─ SQLite checkpoints — hippo resume
```

See `../PLAN.md` for design decisions and milestones.
