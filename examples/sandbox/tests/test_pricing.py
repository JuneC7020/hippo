import pytest

from invoicely import Invoice, LineItem, apply_discount, subtotal_cents, total_cents


def _invoice(discount: int = 0) -> Invoice:
    inv = Invoice(number="INV-1", customer="ACME", discount_percent=discount, vat_percent=20)
    inv.add(LineItem("Widget", 3, 1999))  # 59.97
    inv.add(LineItem("Gadget", 1, 12501))  # 125.01
    return inv


def test_subtotal() -> None:
    assert subtotal_cents(_invoice()) == 18498


def test_discount_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        apply_discount(1000, 101)


def test_discount_rounds_half_up_to_cent() -> None:
    # Discounts round half up to the cent, like add_vat does.
    assert apply_discount(101, 12) == 89  # 12% of 101 = 12.12 -> 12 -> 89 (no rounding issue)
    assert apply_discount(1001, 15) == 851  # 15% of 1001 = 150.15 -> 150 -> 851
    assert apply_discount(1005, 15) == 854  # 15% of 1005 = 150.75 -> 151 -> 854  (floor gives 855)


def test_total_with_discount_and_vat() -> None:
    # 184.98 - 10% = 166.48 (166.482 -> 166.48), + 20% VAT = 199.78 (199.776 -> 199.78)
    assert total_cents(_invoice(discount=10)) == 19978
