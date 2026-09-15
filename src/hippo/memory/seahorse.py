"""Seahorse vector store backend. Wired in M2."""

from __future__ import annotations


class SeahorseStore:
    def __init__(self, api_key: str, table_name: str) -> None:
        self.api_key = api_key
        self.table_name = table_name

    def add(self, text: str, *, metadata: dict | None = None, id: str | None = None) -> str:
        raise NotImplementedError("SeahorseStore.add lands in M2")

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        raise NotImplementedError("SeahorseStore.search lands in M2")

    def delete(self, id: str) -> None:
        raise NotImplementedError("SeahorseStore.delete lands in M2")
