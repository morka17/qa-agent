# Demo SaaS Dashboard (fixture)

A single-file, dependency-free sidebar-nav dashboard used to self-test
Sentinel-QA's pipeline against a UI shape different from
`demo_ecommerce_app` — dynamic panel switching, an async-delayed action,
and a settings form.

## Running it

```bash
cd tests/e2e_fixtures/demo_saas_dashboard
python -m http.server 8124
```

Then point a run's `target_url` at `http://localhost:8124`.

## Flows available

- **Overview**: default panel, shows an active-user count with a
  "Refresh Stats" button that updates asynchronously after ~600ms
  (useful for exercising wait/retry timing, not just instant DOM changes).
- **Reports**: a "Generate Report" button.
- **Settings**: a display-name input and "Save Changes" button.

## Known seeded bug

Clicking "Generate Report" never renders anything into `#report-output`
— see the `BUG` comment in `index.html`. A story asserting "the report
is shown after generating it" should reliably fail here. This is a
second, independent known-truth `app_bug` (distinct from
`demo_ecommerce_app`'s cart-total bug) with different surrounding
markup, useful for confirming the perception layer's resolution
strategies generalize across UI conventions rather than being tuned to
one fixture's specific structure.
