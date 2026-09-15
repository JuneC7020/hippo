"""Table-selection logic for the Seahorse backend (no network)."""

from hippo.memory.seahorse import _has_sparse, _pick_ready


def _table(uuid: str, status: str, created: str, sparse: bool) -> dict:
    cols = [{"name": "id"}, {"name": "text"}, {"name": "dense_vector"}]
    if sparse:
        cols.append({"name": "sparse_vector"})
    return {
        "table_uuid": uuid,
        "table_name": "hippo_facts",
        "status": status,
        "created_at": created,
        "schema": {"columns": cols},
    }


def test_has_sparse_reads_schema() -> None:
    assert _has_sparse(_table("a", "success", "1", sparse=True))
    assert not _has_sparse(_table("b", "success", "1", sparse=False))
    assert not _has_sparse({"schema": None})


def test_pick_ready_prefers_oldest_compatible_ready_table() -> None:
    tables = [  # already sorted oldest-first, as list_tables() guarantees
        _table("hybrid-old", "success", "1", sparse=True),
        _table("creating", "creating", "2", sparse=False),
        _table("dense-a", "success", "3", sparse=False),
        _table("dense-b", "success", "4", sparse=False),
    ]
    chosen = _pick_ready(tables, use_sparse=False)
    assert chosen is not None and chosen["table_uuid"] == "dense-a"
    chosen = _pick_ready(tables, use_sparse=True)
    assert chosen is not None and chosen["table_uuid"] == "hybrid-old"


def test_pick_ready_none_when_nothing_ready() -> None:
    assert _pick_ready([_table("x", "creating", "1", sparse=False)], use_sparse=False) is None
    assert _pick_ready([], use_sparse=False) is None
