"""Local Chroma fallback when SEAHORSE_API_KEY is empty."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any


def _flat_meta(metadata: dict | None) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, bool | int | float):
            out[str(key)] = value
        elif value is None:
            continue
        else:
            out[str(key)] = str(value)
    return out or {"_": True}


class ChromaStore:
    def __init__(self, persist_dir: Path, collection: str) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(collection)
        self.collection = collection

    def add(self, text: str, *, metadata: dict | None = None, id: str | None = None) -> str:
        doc_id = id or uuid.uuid4().hex
        self._col.add(ids=[doc_id], documents=[text], metadatas=[_flat_meta(metadata)])
        return doc_id

    def search(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        count = self._col.count()
        if count == 0:
            return []
        n = min(k, count)
        raw = self._col.query(query_texts=[query], n_results=n)
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            hits.append(
                {
                    "id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": dists[i] if i < len(dists) else None,
                }
            )
        return hits

    def delete(self, id: str) -> None:
        self._col.delete(ids=[id])
