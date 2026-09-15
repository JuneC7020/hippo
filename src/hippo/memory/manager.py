"""Episodic/semantic write, recall injection, context compression. M2/M3."""

from __future__ import annotations

from hippo.memory.base import MemoryStore


class MemoryManager:
    def __init__(self, episodes: MemoryStore, facts: MemoryStore) -> None:
        self.episodes = episodes
        self.facts = facts

    def remember_episode(self, summary: str, **meta: object) -> str:
        return self.episodes.add(summary, metadata=dict(meta) or None)

    def remember_fact(self, text: str, **meta: object) -> str:
        return self.facts.add(text, metadata=dict(meta) or None)

    def recall(self, query: str, *, k: int = 5) -> list[dict]:
        hits = self.facts.search(query, k=k) + self.episodes.search(query, k=k)
        return hits[:k]

    def compress(self, messages: list[dict], token_budget: int) -> tuple[str, list[dict]]:
        raise NotImplementedError("context compression lands in M3")
