# Contributing to Sentinel-QA

Thanks for your interest in contributing — Sentinel-QA is built in the open, and
contributions of all sizes (bug fixes, new selector strategies, issue-tracker
integrations, docs, benchmark results) are welcome.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Adding a New Connector / Issue Tracker](#adding-a-new-connector--issue-tracker)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)

## Code of Conduct

Be respectful, assume good intent, and keep discussion focused on the work.
Harassment or disrespectful conduct of any kind will not be tolerated.

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker (for the local Postgres/Redis stack)

### Setup

```bash
git clone https://github.com/morka17/qa-agent.git
cd sentinel-qa

poetry install --with dev
poetry run playwright install --with-deps chromium

cp .env.example .env
docker compose up -d postgres redis

poetry run alembic upgrade head
pre-commit install
```

Verify everything is wired up:

```bash
make test
```

## Development Workflow

1. **Fork** the repo and create a branch off `main`:
   `git checkout -b feat/short-description` or `fix/short-description`
2. Make your change, with tests (see [Testing](#testing) below).
3. Run the full local check before pushing:
   ```bash
   make lint
   make typecheck
   make test
   ```
4. Push and open a pull request against `main`.

## Coding Standards

- **Formatting**: [Black](https://black.readthedocs.io/), enforced via pre-commit — don't hand-format.
- **Linting**: [Ruff](https://docs.astral.sh/ruff/) — run `poetry run ruff check .`
- **Typing**: All new code must be fully type-annotated. `mypy --strict` runs in CI.
- **Docstrings**: Every public module, class, and function gets a docstring explaining
  *why*, not just *what* — the "why" is what saves the next contributor time.
- **No inline LLM prompt strings**: Prompts are versioned files under `docs/prompts/`
  and loaded, not hardcoded in Python modules (see `docs/adr/0003-llm-provider-abstraction.md`).
- **Imports**: Absolute imports only (`from qa_agent.ingestion.schemas import UserStory`),
  sorted by `ruff` (isort rules).

## Testing

Tests are organized by scope:

| Directory | Scope | Runs in CI on every PR? |
|---|---|---|
| `tests/unit/` | Pure logic, no I/O, no browser | Yes |
| `tests/integration/` | Requires DB/Redis/live browser | Yes (containerized) |
| `tests/e2e_fixtures/` | Full agent loop against demo apps | Nightly only |

- New code requires new tests. Coverage must not drop below the threshold set in
  `pyproject.toml` (`fail_under`).
- Mock external HTTP calls with `respx` (see `ingestion/connectors/` tests for the pattern) —
  never hit real Jira/Linear APIs in unit tests.
- Run a single test file: `poetry run pytest tests/unit/ingestion/test_story_parser.py -v`

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(perception): add visual-grounding fallback for element resolution
fix(connectors): handle paginated Jira responses correctly
docs(readme): clarify quickstart env vars
test(planning): add coverage for plan_validator edge cases
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

## Pull Request Process

1. Fill out the PR template completely — link the issue it closes, if any.
2. Keep PRs focused: one logical change per PR. Large refactors should be
   discussed in an issue first.
3. CI must pass (lint, typecheck, unit + integration tests) before review.
4. At least one maintainer approval is required to merge.
5. Update `CHANGELOG.md` under `[Unreleased]` for any user-facing change.
6. Squash-merge is used to keep `main` history linear.

## Adding a New Connector / Issue Tracker

Sentinel-QA is designed to be pluggable at two seams:

- **Story sources** (`src/qa_agent/ingestion/connectors/`): implement a class that
  produces a `list[UserStory]` (see `csv_connector.py` for the simplest reference
  implementation, `jira_connector.py` for one with pagination and auth).
- **Bug filers** (`src/qa_agent/reporting/issue_trackers/`): implement a class that
  accepts a `BugReport` and files it against the target system.

New connectors should:
- Raise a dedicated `<Name>ConnectorError` for non-recoverable failures.
- Never let a single malformed record abort an entire batch — log and skip it.
- Include unit tests with the external API mocked.
- Be added to `docs/architecture.md`'s connector list.

## Reporting Bugs

Open a [GitHub issue](https://github.com/morka17/qa-agent/issues/new) with:

- What you expected vs. what happened
- Steps to reproduce (a minimal user story + target app is ideal)
- The run's `trace.zip` / `report.md` if the bug is in agent output, when possible
- Your environment (Python version, OS, `sentinel-qa` version)

## Proposing Features

Open a [GitHub Discussion](https://github.com/morka17/qa-agent/discussions) or
issue tagged `enhancement` before writing code for anything non-trivial — it saves
everyone time if the design gets aligned first. Check `docs/adr/` to see if a
relevant decision has already been made (and why).

---

Thanks again for contributing — every fixed selector edge case and filed bug
report format improvement makes Sentinel more trustworthy for everyone running it.