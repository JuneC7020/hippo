from __future__ import annotations

from invoicely.models import Invoice
from invoicely.pricing import subtotal_cents, total_cents


def _money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def render(invoice: Invoice) -> str:
    lines = [f"Invoice {invoice.number} for {invoice.customer}", ""]
    for item in invoice.items:
        lines.append(
            f"{item.description:<24} {item.quantity:>3} x {_money(item.unit_cents):>8}"
            f" = {_money(item.line_cents):>9}"
        )
    lines.append("")
    lines.append(f"{'Subtotal':<40} {_money(subtotal_cents(invoice)):>9}")
    if invoice.discount_percent:
        lines.append(f"{'Discount':<40} {invoice.discount_percent:>8}%")
    lines.append(f"{'VAT':<40} {invoice.vat_percent:>8}%")
    lines.append(f"{'Total':<40} {_money(total_cents(invoice)):>9}")
    return "\n".join(lines)
