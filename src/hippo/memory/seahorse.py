"""Seahorse vector store backend.

Seahorse provisions tables asynchronously: POST /v2/tables returns only after the
table is ready, which is usually longer than the gateway timeout (408). The
`seahorse_vector_store` client retries that POST and ends up creating duplicate
tables, then binds to the newest (still `creating`) one and fails on schema.

We therefore own the table lifecycle here: create once (tolerating 408), poll
GET /v2/tables until a table with our name reports `status == "success"`, and
bind the vector store to the *oldest* healthy one so the choice is stable.

Tables are created dense-only (`use_sparse=False`). Free-tier tenants have no
sparse (BM25) inference endpoint (`INFERENCE_ENDPOINT_NOT_FOUND`), and a table
with a non-nullable `sparse_vector` column then rejects every insert.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

log = logging.getLogger(__name__)

READY = "success"
SPARSE_COLUMN = "sparse_vector"


def _open_store(api_key: str, table_name: str, *, use_sparse: bool = False):
    from seahorse_vector_store import SeahorseVectorStore

    return SeahorseVectorStore(api_key=api_key, table_name=table_name, use_sparse=use_sparse)


def _has_sparse(table: dict[str, Any]) -> bool:
    cols = ((table.get("schema") or {}).get("columns")) or []
    return any(c.get("name") == SPARSE_COLUMN for c in cols if isinstance(c, dict))


def _doc_to_hit(doc: Any) -> dict[str, Any]:
    text = getattr(doc, "page_content", None) or str(doc)
    meta = dict(getattr(doc, "metadata", None) or {})
    doc_id = getattr(doc, "id", None) or meta.get("id")
    return {"id": doc_id, "text": text, "metadata": meta}


def list_tables(vs: Any, name: str | None = None) -> list[dict[str, Any]]:
    """Tables for this key, oldest first, optionally filtered by name."""
    tables = vs._client.list_tables()
    if name is not None:
        tables = [t for t in tables if str(t.get("table_name")) == name]
    return sorted(tables, key=lambda t: str(t.get("created_at", "")))


def _pick_ready(tables: list[dict[str, Any]], *, use_sparse: bool) -> dict[str, Any] | None:
    ready = [
        t
        for t in tables
        if t.get("status") == READY and t.get("table_uuid") and _has_sparse(t) == use_sparse
    ]
    return ready[0] if ready else None


class SeahorseStore:
    def __init__(
        self,
        api_key: str,
        table_name: str,
        *,
        create_timeout_s: float = 240.0,
        poll_interval_s: float = 5.0,
        use_sparse: bool = False,
    ) -> None:
        self.api_key = api_key
        self.table_name = table_name
        self.create_timeout_s = create_timeout_s
        self.poll_interval_s = poll_interval_s
        self.use_sparse = use_sparse
        self._vs = _open_store(api_key, table_name, use_sparse=use_sparse)
        self._bound = False

    # -- table lifecycle -------------------------------------------------

    def _bind_existing(self) -> bool:
        """Bind to the oldest ready table with our name. False if none."""
        if self._bound:
            return True
        tables = list_tables(self._vs, self.table_name)
        chosen = _pick_ready(tables, use_sparse=self.use_sparse)
        if chosen is None:
            incompatible = [t for t in tables if _has_sparse(t) != self.use_sparse]
            if incompatible:
                log.warning(
                    "seahorse: %d table(s) named %r have use_sparse=%s but we need %s; "
                    "ignoring them. Drop with scripts/seahorse_tables.py --drop %s",
                    len(incompatible),
                    self.table_name,
                    not self.use_sparse,
                    self.use_sparse,
                    self.table_name,
                )
            return False
        if len(tables) > 1:
            log.warning(
                "seahorse: %d tables named %r; using oldest ready %s. "
                "Run scripts/seahorse_tables.py --prune to remove duplicates.",
                len(tables),
                self.table_name,
                chosen["table_uuid"],
            )
        self._vs._bind_table(str(chosen["table_uuid"]))
        self._bound = True
        return True

    def ensure_table(self) -> None:
        """Create the table if needed and bind. Requires a WRITE key to create."""
        if self._bind_existing():
            return
        from seahorse_vector_store.exceptions import SeahorseAPIError

        tables = list_tables(self._vs, self.table_name)
        compatible = [t for t in tables if _has_sparse(t) == self.use_sparse]
        if not compatible:
            client = self._vs._client
            saved = client._max_retries
            client._max_retries = 1  # never retry the POST: each retry = one more table
            try:
                client.set_base_url(self._vs._gateway_url)
                client.create_table(self._vs._build_create_table_payload())
            except SeahorseAPIError as exc:
                if "408" not in str(exc) and "Timeout" not in str(exc):
                    raise
                log.info("seahorse: create_table timed out at gateway; polling for readiness")
            finally:
                client._max_retries = saved

        deadline = time.monotonic() + self.create_timeout_s
        while time.monotonic() < deadline:
            if self._bind_existing():
                return
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"seahorse table {self.table_name!r} not ready after {self.create_timeout_s:.0f}s"
        )

    # -- MemoryStore -----------------------------------------------------

    def add(self, text: str, *, metadata: dict | None = None, id: str | None = None) -> str:
        self.ensure_table()
        meta = dict(metadata or {})
        if id:
            meta.setdefault("id", id)
        ids = self._vs.add_texts([text], metadatas=[meta])
        return ids[0] if ids else (id or uuid.uuid4().hex)

    def search(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        if not self._bind_existing():
            # No ready table yet: nothing has ever been written (or it is still
            # provisioning). Empty recall, not an error.
            return []
        docs = self._vs.similarity_search(query, k=k)
        return [_doc_to_hit(d) for d in docs]

    def delete(self, id: str) -> None:
        if self._bind_existing():
            self._vs.delete(ids=[id])
