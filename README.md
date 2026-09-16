# hippo

Long-term memory CLI agent (MCP tools + Seahorse memory + LangGraph). Work in progress.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # set OPENAI_API_KEY
hippo run "What Python packages does this repo declare?"
hippo trace               # list run ids
```

Flags: `--no-mcp` local filesystem tools only · `--write` allow write/delete/commit · `--single` one agent, no planner/reviewer · `--oneshot` no tools · `--no-memory` skip recall/persist.

`hippo memory "<query>"` searches Seahorse (or Chroma). `hippo tasks` lists SQLite task rows. `hippo resume TASK_ID` continues an interrupted run from its last checkpoint.

### How a run works

```
recall (Seahorse/Chroma)
  -> planner   : 1-4 subtasks, each with goal + allowed tools + step budget
  -> worker    : tool loop on ONE subtask, returns {result, evidence, confidence}
  -> reviewer  : approve | revise (worker retries once with feedback) | escalate
  -> planner   : re-plan once on escalation, keeping finished subtasks
  -> finalize  : answer from all subtask results (partial answer if escalated)
  -> persist   : episode summary + durable facts written to memory
```

Every node boundary is a LangGraph checkpoint in `.hippo/checkpoints.db` (thread = task id), so a killed
process resumes at the node it was in. Tools, tracer and the LLM travel in LangGraph's runtime context,
not in state, so nothing non-serializable is checkpointed.

Working context is compressed inside a subtask: when the prompt exceeds `HIPPO_TOKEN_BUDGET`, older
tool turns are summarized into one note and the last 4 turns stay verbatim (`agent/context.py`).
When a subtask's tool budget runs out the worker gets one final no-tools turn, so a tight budget yields
a partial answer with "unverified" flagged, not silence.

MCP servers come from `mcp.json`: filesystem via `npx @modelcontextprotocol/server-filesystem`, git via
the official Python `mcp-server-git` (`@modelcontextprotocol/server-git` does not exist on npm).
`hippo trace TASK_ID` prints the JSONL trace: plan, per-subtask tool calls, reviews, compressions.

Memory backend: `HIPPO_MEMORY_BACKEND=auto` (default) uses Seahorse when `SEAHORSE_API_KEY` is set, else local Chroma under `.hippo/chroma`. Force with `seahorse` or `chroma`.

### Seahorse notes (as of 2026-09-15)

- The key needs **WRITE** permission; tables are created on first write. A read-only key recalls nothing and writes warn instead of crashing.
- Seahorse provisions tables asynchronously and its gateway times out (408) before that finishes. The stock client retries the POST and creates duplicate tables. `SeahorseStore` therefore creates once, polls `GET /v2/tables` until `status == success`, and binds to the oldest healthy table. `python scripts/seahorse_tables.py [--prune | --drop NAME]` lists/cleans tables.
- Tables are **dense-only** (`use_sparse=False`): the tenant has no sparse (BM25) inference endpoint, and a hybrid table then rejects every insert.
- Open issue: `POST /v2/data/active-set-flush` 408s in every sync mode, so freshly inserted rows are accepted (`inserted_row_count: 1`) but never become searchable (`indexed-row-count` stays 0). Until Dnotitia fixes this, run with `HIPPO_MEMORY_BACKEND=chroma`; the agent code path is identical.

Keys go in `.env` only — never `.env.example`.

Public repo: [github.com/JuneC7020/hippo](https://github.com/JuneC7020/hippo). Use a **venv** installing into a global interpreter can upgrade langchain to 1.x.
