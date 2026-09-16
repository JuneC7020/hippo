# invoicely (demo sandbox)

A deliberately small invoicing library used as the target repository for hippo's demos.
It is not part of the hippo package.

- `invoicely/models.py` — `LineItem`, `Invoice` dataclasses
- `invoicely/pricing.py` — subtotal, discount, VAT
- `invoicely/render.py` — plain-text invoice rendering
- `tests/` — pytest suite. **Two tests fail on purpose, one root cause** (see demo 2 in the hippo README).

Conventions: money is handled in integer cents, never floats. Discounts are percentages (0–100).

```bash
python -m pytest -q          # from this directory
```
