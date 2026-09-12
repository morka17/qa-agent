# Golden Bug Reports

Expected `BugReport` output for fixed, deterministic inputs — a
regression net for `reporting/bug_report_writer.py` and
`reporting/report_schema.py`. Each scenario pairs:

- `<scenario>.json` — the structural fields (severity, category, repro
  steps, labels) with no LLM-authored prose, deterministic regardless of
  which LLM produced the report.
- `<scenario>.md` — the full rendered `BugReport.as_markdown_body()`
  output, generated from a **fixed stub LLM response** (not a live
  model call) so the file is byte-stable across runs.

`test_golden_bug_reports.py` regenerates a report from the same fixed
inputs (intent, plan, evidence, classification, RCA, and stub LLM
response) used to produce these files and asserts the output matches
exactly.

## Updating a golden file

A golden file only needs to change when a deliberate behavior change
affects its output — e.g. `bug_report_writer.py`'s severity mapping
changes, or `BugReport.as_markdown_body()`'s formatting changes. To
regenerate:

1. Update the source code.
2. Run the scenario's generation logic (see the `_intent`/`_plan`/etc.
   fixtures in `test_golden_bug_reports.py`) and capture its output.
3. Overwrite the `.md`/`.json` file with the new output.
4. Re-run `pytest tests/golden_reports/` to confirm it now passes, and
   review the diff in your PR — a golden file diff is exactly the
   change reviewers most need to see.

Never hand-edit a golden file to make a failing test pass without
understanding *why* the output changed — that defeats the point of a
regression snapshot.

## Scenarios

- **`cart_total_bug`** — the seeded bug in
  `tests/e2e_fixtures/demo_ecommerce_app/`: the cart total never updates
  after adding an item. A `HIGH`-priority story classified as `app_bug`
  should always produce `HIGH` severity (see
  `bug_report_writer.determine_severity`).
