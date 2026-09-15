from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    def add(self, text: str, *, metadata: dict | None = None, id: str | None = None) -> str: ...
    def search(self, query: str, *, k: int = 5) -> list[dict]: ...
    def delete(self, id: str) -> None: ...
