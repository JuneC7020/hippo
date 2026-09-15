"""Seahorse add/search smoke test.

Usage (from hippo/):
    python scripts/smoke_seahorse.py

Needs SEAHORSE_API_KEY in the environment or .env.
Sign up: https://console.seahorse.dnotitia.ai
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from hippo.config import load_settings  # noqa: E402


def _store(api_key: str, table: str):
    try:
        from langchain_seahorse import SeahorseVectorStore
        return SeahorseVectorStore(api_key=api_key, table_name=table)
    except ImportError:
        pass
    try:
        from seahorse_vector_store import SeahorseVectorStore
        return SeahorseVectorStore(api_key=api_key, table_name=table)
    except ImportError as e:
        raise SystemExit(
            "langchain-seahorse is not installed. pip install -e '.[dev]'\n" + str(e)
        ) from e


def main() -> None:
    settings = load_settings()
    if not settings.seahorse_api_key:
        print("SEAHORSE_API_KEY is empty. Skipping (Chroma fallback will be used in M2).")
        print("Create a key at https://console.seahorse.dnotitia.ai and put it in .env")
        sys.exit(0)

    table = settings.seahorse_table_facts
    vs = _store(settings.seahorse_api_key, table)
    vs.add_texts(["hippo smoke: hippocampus is the brain region for long-term memory."])
    hits = vs.similarity_search("long-term memory", k=1)
    print("table:", table)
    print("hits:", hits)
    if not hits:
        raise SystemExit("Seahorse search returned no hits")
    print("ok")


if __name__ == "__main__":
    main()
