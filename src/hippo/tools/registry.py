"""Tool whitelist, call budget, confirmation for dangerous tools. M1."""

from __future__ import annotations

DANGEROUS = frozenset(
    {
        "write_file",
        "delete_file",
        "move_file",
        "git_commit",
        "git_push",
        "create_directory",
        "edit_file",
    }
)


class ToolRegistry:
    def __init__(
        self,
        allowed: list[str] | None = None,
        max_calls: int = 40,
        *,
        allow_dangerous: bool = False,
    ) -> None:
        self.allowed = set(allowed) if allowed else None
        self.max_calls = max_calls
        self.allow_dangerous = allow_dangerous
        self.calls = 0

    def check(self, name: str) -> None:
        short = name.split("__")[-1]
        if self.allowed is not None and name not in self.allowed and short not in self.allowed:
            raise PermissionError(f"tool not allowed: {name}")
        if self.is_dangerous(name) and not self.allow_dangerous:
            raise PermissionError(f"dangerous tool blocked (pass --write): {name}")
        if self.calls >= self.max_calls:
            raise RuntimeError("tool call budget exceeded")
        self.calls += 1

    def is_dangerous(self, name: str) -> bool:
        short = name.split("__")[-1]
        return name in DANGEROUS or short in DANGEROUS
