from hippo.tools.registry import ToolRegistry


def test_registry_allows_then_budget() -> None:
    r = ToolRegistry(allowed=["read_file"], max_calls=1)
    r.check("read_file")
    try:
        r.check("read_file")
        assert False, "expected budget error"
    except RuntimeError:
        pass


def test_registry_blocks_unknown() -> None:
    r = ToolRegistry(allowed=["read_file"])
    try:
        r.check("write_file")
        assert False, "expected permission error"
    except PermissionError:
        pass
