"""Pricing rules. All amounts are integer cents; percentages are whole numbers 0-100."""

from __future__ import annotations

from invoicely.models import Invoice


def subtotal_cents(invoice: Invoice) -> int:
    return sum(item.line_cents for item in invoice.items)


def apply_discount(amount_cents: int, discount_percent: int) -> int:
    """Return `amount_cents` reduced by `discount_percent`, rounded half up to a cent."""
    if not 0 <= discount_percent <= 100:
        raise ValueError(f"discount_percent must be 0-100, got {discount_percent}")
    discount = amount_cents * discount_percent // 100
    return amount_cents - discount


def add_vat(amount_cents: int, vat_percent: int) -> int:
    return amount_cents + (amount_cents * vat_percent + 50) // 100


def total_cents(invoice: Invoice) -> int:
    """subtotal -> discount -> VAT."""
    net = apply_discount(subtotal_cents(invoice), invoice.discount_percent)
    return add_vat(net, invoice.vat_percent)
