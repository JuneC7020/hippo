from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def emit(self, kind: str, **payload: Any) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "kind": kind,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def list_trace_ids(data_dir: Path) -> list[str]:
    folder = data_dir / "traces"
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.jsonl"))
