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
