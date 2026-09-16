from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineItem:
    """One invoice line. `unit_cents` is the unit price in integer cents."""

    description: str
    quantity: int
    unit_cents: int

    @property
    def line_cents(self) -> int:
        return self.quantity * self.unit_cents


@dataclass
class Invoice:
    number: str
    customer: str
    items: list[LineItem] = field(default_factory=list)
    discount_percent: int = 0  # 0-100
    vat_percent: int = 20

    def add(self, item: LineItem) -> None:
        self.items.append(item)
