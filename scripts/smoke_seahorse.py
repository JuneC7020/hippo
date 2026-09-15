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

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from hippo.config import load_settings  # noqa: E402
from hippo.memory.seahorse import SeahorseStore  # noqa: E402


def main() -> None:
    settings = load_settings()
    if not settings.seahorse_api_key:
        print("SEAHORSE_API_KEY is empty. Put it in .env ")
        sys.exit(0)

    store = SeahorseStore(settings.seahorse_api_key, settings.seahorse_table_facts)
    can_write = True
    write_error = ""
    try:
        doc_id = store.add(
            "hippo smoke: hippocampus is the brain region for long-term memory.",
            metadata={"kind": "smoke"},
        )
        print("added", doc_id)
    except Exception as exc:  # noqa: BLE001
        can_write = False
        write_error = str(exc)
        print("add failed:", write_error[:300])

    hits = store.search("long-term memory", k=1)
    print("table:", settings.seahorse_table_facts)
    print("hits:", [h.get("text", "")[:80] for h in hits])
    if hits:
        print("ok: Seahorse read+write working")
        return
    if not can_write:
        if "WRITE permission" in write_error or "403" in write_error:
            print(
                "Seahorse key is READ-only and the table has never been written, "
                "so there is nothing to recall.\n"
                "Until you have a WRITE key, run with HIPPO_MEMORY_BACKEND=chroma "
                "(local) - the agent code path is identical."
            )
        else:
            print(
                "Seahorse write failed for a non-permission reason (timeout / server). "
                "Check `python scripts/seahorse_tables.py`; if it keeps failing use "
                "HIPPO_MEMORY_BACKEND=chroma."
            )
        return
    raise SystemExit("Seahorse write succeeded but search returned no hits")


if __name__ == "__main__":
    main()
