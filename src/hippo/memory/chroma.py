"""Local Chroma fallback when SEAHORSE_API_KEY is empty. Wired in M2."""

from __future__ import annotations

from pathlib import Path


class ChromaStore:
    def __init__(self, persist_dir: Path, collection: str) -> None:
        self.persist_dir = persist_dir
        self.collection = collection

    def add(self, text: str, *, metadata: dict | None = None, id: str | None = None) -> str:
        raise NotImplementedError("ChromaStore.add lands in M2")

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        raise NotImplementedError("ChromaStore.search lands in M2")

    def delete(self, id: str) -> None:
        raise NotImplementedError("ChromaStore.delete lands in M2")
