from invoicely import Invoice, LineItem, render


def test_render_contains_lines_and_total() -> None:
    inv = Invoice(number="INV-7", customer="Globex", discount_percent=0, vat_percent=0)
    inv.add(LineItem("Consulting", 2, 50000))
    out = render(inv)
    assert "Invoice INV-7 for Globex" in out
    assert "Consulting" in out
    assert out.strip().endswith("1000.00")
