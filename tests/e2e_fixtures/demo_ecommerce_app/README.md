# Demo E-Commerce App (fixture)

A single-file, dependency-free product page + checkout flow used to
self-test Sentinel-QA's own pipeline end to end (`tests/e2e_fixtures/`).

## Running it

```bash
cd tests/e2e_fixtures/demo_ecommerce_app
python -m http.server 8123
```

Then point a run's `target_url` at `http://localhost:8123`.

## Flows available

- **Add to cart**: click "Add to Cart" on the product page; the cart
  count updates.
- **Wishlist**: click "Add to Wishlist"; triggers a browser `alert()`.
- **Checkout**: click "View Cart" to navigate to the checkout section,
  then "Place Order" to see an order status message.

## Known seeded bug

The cart total (`#cart-total`) is intentionally never updated when an
item is added — see the `BUG` comment in `index.html`. A story asserting
that the cart total reflects the item price should reliably fail against
this fixture, which makes it useful for:

- Regression-testing `triage/failure_classifier.py` (this should always
  classify as `app_bug`, never `selector_drift` or `flaky_test`, since
  the failure is deterministic and the elements are all cleanly
  resolvable).
- Verifying `reporting/bug_report_writer.py` produces a stable,
  reasonable bug report against a known-truth defect — compare against
  `tests/golden_reports/`.

Every other assertion (cart count increments, checkout navigation,
order confirmation) should pass — don't "fix" the seeded bug without
also updating `tests/golden_reports/` and any test relying on it.
