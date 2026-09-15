"""List (and optionally prune) Seahorse tables for the key in .env.

Usage (from hippo/):
    python scripts/seahorse_tables.py               # list
    python scripts/seahorse_tables.py --prune       # delete duplicate hippo tables,
                                                    # keeping the oldest ready one per name
    python scripts/seahorse_tables.py --drop NAME   # delete every table called NAME

Duplicates appear when the gateway times out on table creation and a client
retries the POST. Prune only touches tables named in .env (episodes/facts).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from hippo.config import load_settings  # noqa: E402
from hippo.memory.seahorse import READY, _has_sparse, _open_store, list_tables  # noqa: E402


def main() -> None:
    settings = load_settings()
    if not settings.seahorse_api_key:
        print("SEAHORSE_API_KEY is empty.")
        sys.exit(0)
    vs = _open_store(settings.seahorse_api_key, settings.seahorse_table_facts)
    tables = list_tables(vs)
    for t in tables:
        sparse = "hybrid" if _has_sparse(t) else "dense"
        print(
            f"{t.get('table_name'):<20} {t.get('table_uuid')} {t.get('status'):<9} "
            f"{sparse:<6} {t.get('created_at')}"
        )
    if not tables:
        print("(no tables)")

    # Deletes are async server-side and the gateway 408s; retrying is useless.
    vs._client._max_retries = 1

    if "--drop" in sys.argv:
        name = sys.argv[sys.argv.index("--drop") + 1]
        for t in tables:
            if t.get("table_name") != name:
                continue
            print(f"delete {name} {t['table_uuid']}")
            _delete(vs, str(t["table_uuid"]))
        _after(vs)
        return

    if "--prune" not in sys.argv:
        return

    for name in (settings.seahorse_table_episodes, settings.seahorse_table_facts):
        same = [t for t in tables if t.get("table_name") == name]
        ready = [t for t in same if t.get("status") == READY]
        keep = ready[0]["table_uuid"] if ready else None
        for t in same:
            if t.get("table_uuid") == keep:
                continue
            if t.get("status") != READY:
                print(
                    f"skip {name} {t['table_uuid']} (status={t.get('status')}, still provisioning)"
                )
                continue
            print(f"delete {name} {t['table_uuid']}")
            _delete(vs, str(t["table_uuid"]))
    _after(vs)


def _delete(vs, table_uuid: str) -> None:
    try:
        vs._client.set_base_url(vs._gateway_url)
        vs._client.delete_table(table_uuid)
    except Exception as exc:  # noqa: BLE001
        # Gateway often 408s but the delete still lands; verify by re-listing.
        print("   gateway:", str(exc)[:120])


def _after(vs) -> None:
    print("after (deletes finish asynchronously; re-run to confirm):")
    remaining = list_tables(vs)
    for t in remaining:
        print(f"{t.get('table_name'):<20} {t.get('table_uuid')} {t.get('status')}")
    if not remaining:
        print("(no tables)")


if __name__ == "__main__":
    main()
