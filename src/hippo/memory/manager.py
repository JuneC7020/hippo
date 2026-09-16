"""Episodic/semantic write, recall injection. In-run compression lives in agent/context.py."""

from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from hippo.memory.base import MemoryStore
from hippo.memory.chroma import ChromaStore
from hippo.memory.seahorse import SeahorseStore

SUMMARY_SYSTEM = (
    "You write long-term memory for a coding agent.\n"
    'Return JSON only: {"episode": "2-4 sentences: what was asked, what was concluded, and any '
    'outcome worth remembering next time (failing tests, errors, files changed, open questions)", '
    '"facts": ["durable repo/user fact", ...]}\n'
    "facts must still be true next week (paths, stack, conventions) and name the repo or "
    "project they belong to. Empty facts is fine.\n"
    "Only record what the TASK and ANSWER actually state. Never invent outcomes; if the answer "
    "does not say whether something worked, leave it out.\n"
    "No markdown."
)

DEDUPE_JACCARD = 0.8


def _norm_tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}


def _near_duplicate(a: str, b: str) -> bool:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= DEDUPE_JACCARD


class MemoryManager:
    def __init__(self, episodes: MemoryStore, facts: MemoryStore, backend: str) -> None:
        self.episodes = episodes
        self.facts = facts
        self.backend = backend

    def remember_episode(self, summary: str, **meta: object) -> str:
        payload = {k: v for k, v in meta.items() if v is not None}
        payload["kind"] = "episode"
        return self.episodes.add(summary, metadata=payload)

    def remember_fact(self, text: str, **meta: object) -> str:
        """Add a fact unless a near-identical one exists (returns the existing id then)."""
        for hit in self.facts.search(text, k=3):
            if _near_duplicate(text, hit.get("text") or ""):
                return hit["id"]
        payload = {k: v for k, v in meta.items() if v is not None}
        payload["kind"] = "fact"
        return self.facts.add(text, metadata=payload)

    def recall(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        """Top-k across both stores. Episodes get a reserved share (about a third of k)
        so a pile of short facts cannot crowd out the "what happened last time" story."""
        k = max(k, 1)
        facts = self.facts.search(query, k=k)
        episodes = self.episodes.search(query, k=k)
        for row in facts:
            meta = row.setdefault("metadata", {})
            meta["kind"] = meta.get("kind") or "fact"
        for row in episodes:
            meta = row.setdefault("metadata", {})
            meta["kind"] = meta.get("kind") or "episode"
        n_ep = min(len(episodes), max(1, k // 3)) if episodes else 0
        n_facts = min(len(facts), k - n_ep)
        n_ep = min(len(episodes), k - n_facts)  # give unused fact slots back to episodes
        return facts[:n_facts] + episodes[:n_ep]

    def format_recall(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return ""
        lines = ["Known memories from prior sessions:"]
        for hit in hits:
            kind = (hit.get("metadata") or {}).get("kind") or "memory"
            text = (hit.get("text") or "").strip().replace("\n", " ")
            if text:
                lines.append(f"- [{kind}] {text[:400]}")
        return "\n".join(lines)

    def persist_run(
        self,
        *,
        task: str,
        answer: str,
        task_id: str,
        model: str,
        api_key: str,
    ) -> dict[str, Any]:
        """Summarize a finished run and write episode (+ optional facts)."""
        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        raw = llm.invoke(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {
                    "role": "user",
                    "content": f"TASK:\n{task[:2000]}\n\nANSWER:\n{answer[:4000]}",
                },
            ]
        )
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        parsed = _parse_summary(content)
        episode = parsed.get("episode") or answer[:400]
        self.remember_episode(episode, task_id=task_id)
        fact_ids = []
        for fact in parsed.get("facts") or []:
            if isinstance(fact, str) and fact.strip():
                fact_ids.append(self.remember_fact(fact.strip(), task_id=task_id))
        return {"episode": episode, "facts": fact_ids}


def _parse_summary(text: str) -> dict[str, Any]:
    blob = text.strip()
    if blob.startswith("```"):
        blob = blob.strip("`")
        if blob.startswith("json"):
            blob = blob[4:].strip()
    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {"episode": blob[:500], "facts": []}


def build_memory_manager(settings: Any):
    """Seahorse when keyed (or forced), otherwise local Chroma."""
    from pathlib import Path

    backend = (getattr(settings, "hippo_memory_backend", "auto") or "auto").lower()
    key = getattr(settings, "seahorse_api_key", "")
    use_seahorse = backend == "seahorse" or (backend == "auto" and bool(key))
    if use_seahorse:
        if not key:
            raise RuntimeError("HIPPO_MEMORY_BACKEND=seahorse but SEAHORSE_API_KEY is empty")
        episodes = SeahorseStore(settings.seahorse_api_key, settings.seahorse_table_episodes)
        facts = SeahorseStore(settings.seahorse_api_key, settings.seahorse_table_facts)
        return MemoryManager(episodes, facts, backend="seahorse")
    data_dir = Path(settings.hippo_data_dir)
    episodes = ChromaStore(data_dir / "chroma", "episodes")
    facts = ChromaStore(data_dir / "chroma", "facts")
    return MemoryManager(episodes, facts, backend="chroma")
