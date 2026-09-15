from pathlib import Path

from hippo.store.sqlite import connect


def test_sqlite_schema(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
    ).fetchone()
    assert row is not None
    conn.close()
