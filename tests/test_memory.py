from pathlib import Path

from hippo.memory.chroma import ChromaStore
from hippo.memory.manager import MemoryManager, _parse_summary
from hippo.store.sqlite import connect, list_tasks, upsert_task


def test_chroma_add_search_delete(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", "facts")
    doc_id = store.add("hippo stores long-term memory in Seahorse", metadata={"kind": "fact"})
    hits = store.search("long-term memory", k=3)
    assert hits
    assert "Seahorse" in hits[0]["text"]
    store.delete(doc_id)
    assert store.search("long-term memory", k=3) == []


def test_manager_recall_merges(tmp_path: Path) -> None:
    episodes = ChromaStore(tmp_path / "chroma", "episodes")
    facts = ChromaStore(tmp_path / "chroma", "facts")
    mem = MemoryManager(episodes, facts, backend="chroma")
    mem.remember_fact("This repo is a Typer CLI named hippo")
    mem.remember_episode("User asked what the project is; we listed pyproject.toml")
    hits = mem.recall("what is hippo", k=4)
    assert hits
    text = mem.format_recall(hits)
    assert "hippo" in text.lower()


def test_manager_recall_reserves_episode_slot(tmp_path: Path) -> None:
    episodes = ChromaStore(tmp_path / "chroma", "episodes")
    facts = ChromaStore(tmp_path / "chroma", "facts")
    mem = MemoryManager(episodes, facts, backend="chroma")
    for i in range(8):
        mem.remember_fact(f"invoicely fact number {i} about pricing and tests")
    mem.remember_episode("Onboarded invoicely; two pricing tests were failing")
    hits = mem.recall("invoicely pricing tests", k=5)
    kinds = [h["metadata"]["kind"] for h in hits]
    assert len(hits) == 5
    assert kinds.count("episode") == 1 and kinds.count("fact") == 4


def test_remember_fact_dedupes_near_duplicates(tmp_path: Path) -> None:
    episodes = ChromaStore(tmp_path / "chroma", "episodes")
    facts = ChromaStore(tmp_path / "chroma", "facts")
    mem = MemoryManager(episodes, facts, backend="chroma")
    first = mem.remember_fact("invoicely: tests can be run using the pytest command")
    again = mem.remember_fact("Invoicely: tests can be run using the pytest command.")
    other = mem.remember_fact("invoicely stores money in integer cents")
    assert again == first
    assert other != first
    assert facts.search("pytest", k=10).__len__() == 2


def test_parse_summary_json() -> None:
    parsed = _parse_summary('{"episode": "did a thing", "facts": ["uses Python 3.12"]}')
    assert parsed["episode"] == "did a thing"
    assert parsed["facts"] == ["uses Python 3.12"]


def test_task_list(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    upsert_task(conn, "abc", "inspect repo", "done")
    rows = list_tasks(conn)
    assert rows[0]["id"] == "abc"
    assert rows[0]["status"] == "done"
    conn.close()
