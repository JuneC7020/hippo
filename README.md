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

Flags: `--no-mcp` local filesystem tools only · `--write` allow write/delete/commit · `--oneshot` no tools.

Public repo: [github.com/JuneC7020/hippo](https://github.com/JuneC7020/hippo). This folder is its own git repo (not the parent 전문연 tree). Use a **venv** — installing into a global interpreter can upgrade langchain to 1.x.
