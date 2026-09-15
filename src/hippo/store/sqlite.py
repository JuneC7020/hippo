"""SQLite task metadata. LangGraph SqliteSaver lands in M3."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now() -> str:
    return datetime.now(UTC).isoformat()


def upsert_task(conn: sqlite3.Connection, task_id: str, goal: str, status: str) -> None:
    ts = now()
    conn.execute(
        """
        INSERT INTO tasks (id, goal, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            goal=excluded.goal,
            status=excluded.status,
            updated_at=excluded.updated_at
        """,
        (task_id, goal, status, ts, ts),
    )
    conn.commit()


def list_tasks(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, goal, status, created_at, updated_at FROM tasks "
        "ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
