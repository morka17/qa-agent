# 🛡️ Sentinel-QA

**An autonomous E2E testing agent that reads user stories, drives real browsers, and files evidence-backed bug reports — no test scripts required.**

[![CI](https://img.shields.io/github/actions/workflow/status/morka17/qa-agent/ci.yml?branch=main&label=CI)](https://github.com/your-org/sentinel-qa/actions)
[![License](https://img.shields.io/github/license/morka17/qa-agent)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Playwright](https://img.shields.io/badge/playwright-e2e-45ba4b)](https://playwright.dev)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/chat-discord-7289da)](#)

---

## What is Sentinel-QA?

Traditional E2E testing breaks the moment a `data-testid` changes. Sentinel-QA doesn't rely on brittle selectors — it **perceives** the DOM the way a real user (or QA engineer) would: by role, by visible text, and by visual grounding when nothing else resolves.

Give it a user story or acceptance criteria. Sentinel plans the test, drives a real browser via [Playwright](https://playwright.dev), verifies the outcome, and — when something breaks — writes and files a **reproducible bug report** with video, trace, and console evidence attached automatically.

No hand-written test scripts. No selector maintenance. No "can't reproduce."

```
User Story  →  Test Plan  →  Browser Execution  →  Verification  →  Bug Report Filed
```

---

## Why Sentinel-QA

| Problem | How Sentinel solves it |
|---|---|
| Selectors break on every UI refactor | Semantic resolution: ARIA role/name → text match → vision-model grounding |
| Writing E2E tests is slow and someone has to maintain them | Tests are planned automatically from user stories / Gherkin acceptance criteria |
| Flaky tests erode trust in the suite | Built-in triage separates **real bugs** from **flaky tests** from **selector drift** |
| Bug reports without repro steps waste engineering time | Every filed bug ships with video, trace, and console/network evidence |
| Vendor lock-in to one LLM or one issue tracker | Pluggable LLM provider layer; pluggable Jira / GitHub Issues / Linear filers |

---

## How It Works

```
┌─────────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐
│  Ingestion  │ →  │ Planning │ →  │ Perception   │ →  │  Execution   │ →  │Verification│
│ (user story)│    │(step graph)│  │(DOM resolve) │    │ (Playwright) │    │ (assertions)│
└─────────────┘    └──────────┘    └─────────────┘    └──────────────┘    └─────┬─────┘
                                                                                  │
                                                          ┌───────────┐    ┌──────▼──────┐
                                                          │  Reporting │ ←  │   Triage    │
                                                          │(file a bug)│    │(classify fail)│
                                                          └───────────┘    └─────────────┘
```

1. **Ingestion** — parses a user story or Gherkin acceptance criteria into structured test intent
2. **Planning** — an LLM generates an ordered, validated step graph (actions + assertions)
3. **Perception** — resolves each step's target element semantically, not by fragile CSS selectors
4. **Execution** — drives a real browser via Playwright with self-healing retries
5. **Verification** — checks assertions, visual regressions, and console/network errors
6. **Triage** — classifies any failure as a genuine bug, a flaky test, or selector drift
7. **Reporting** — writes a structured bug report with attached evidence and files it to your tracker

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/morka17/qa-agent.git
cd qa-agent
poetry install
playwright install --with-deps chromium

# Configure
cp .env.example .env
# set your LLM provider key, target app URL, and issue tracker credentials in .env

# Run against a user story
poetry run qa_agent run \
  --story "As a user, I can add an item to my cart and see the updated total" \
  --target-url https://your-staging-app.com

# Or run against a full backlog (CSV / Jira / Linear)
poetry run qa_agent run --stories ./backlog.csv
```

A successful run produces a report in `runs/<run_id>/`:
```
runs/2026-09-01-a1b2c3/
├── plan.json          # generated step graph
├── trace.zip          # Playwright trace (open with `playwright show-trace`)
├── video.webm
├── screenshots/
└── report.md          # filed bug report (if a failure was found)
```

---

## Architecture

Full system design, sequence diagrams, and Architecture Decision Records live in [`docs/architecture.md`](docs/architecture.md) and [`docs/adr/`](docs/adr/). Key design principles:

- **Layered selector resolution** — ARIA/text matching first; LLM reasoning and vision-model grounding are fallbacks, not the default (for cost and reliability)
- **Versioned, auditable prompts** — every LLM call traces back to a reviewed prompt spec in [`docs/prompts/`](docs/prompts/), not an inlined string
- **Evidence-first bug reports** — no report is filed without attached trace/video/screenshot proof
- **Replayable runs** — every run can be replayed via `scripts/replay_trace.py` for debugging
- **Guardrails before autonomy** — destructive actions (delete, payment, irreversible submit) require an allowlist match or human approval gate

---

## Roadmap

- [x] Core perception + execution loop against fixture apps
- [x] Story → test plan pipeline
- [ ] Failure triage & root-cause analysis
- [ ] Automated bug filing (Jira / GitHub Issues / Linear)
- [ ] Multi-app concurrent orchestration + operator dashboard
- [ ] Observability (OpenTelemetry, cost tracking) & guardrails hardening
- [ ] Public benchmark suite (selector accuracy, flake rate, cost per run)

See the [full roadmap](docs/architecture.md#roadmap) for phase-by-phase detail.

---

## Contributing

Sentinel-QA is open source and contributions are welcome — whether that's a bug fix, a new selector strategy, a new issue-tracker integration, or a benchmark result.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Check [open issues](https://github.com/your-org/sentinel-qa/issues) labeled `good first issue`
3. Run `make setup && make test` before opening a PR

---

## Tech Stack

Python · FastAPI · Playwright · PostgreSQL + pgvector · SQLAlchemy (async) · Alembic · Redis/Temporal · Docker · Terraform · OpenTelemetry

---

## License

[MIT](LICENSE) — free to use, modify, and self-host.

---

<p align="center">Built for teams who'd rather review a filed bug report than write another Playwright script by hand.</p>