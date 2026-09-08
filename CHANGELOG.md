# Changelog

All notable changes to Sentinel-QA are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ingestion layer: `UserStory`, `AcceptanceCriterion`, and `TestIntent` schemas.
- Gherkin/BDD acceptance-criteria extraction with heuristic fallback for
  non-Gherkin, loosely structured tickets.
- `StoryParser`: canonical regex parsing of "As a... I want... so that..."
  stories, with LLM-backed extraction for free-form narratives.
- Story source connectors: Jira (REST v3), Linear (GraphQL), and CSV.
- Environment-driven configuration via `pydantic-settings` (`config/settings.py`).
- Structured JSON/human logging with automatic `run_id`/`story_id` context tagging.

### Changed
- N/A (initial development)

### Fixed
- N/A (initial development)

---

## [0.1.0] - Unreleased

Initial scaffold of the project: repository structure, roadmap, and the
ingestion pipeline described above. No execution, perception, or reporting
functionality is implemented yet — see the [roadmap](docs/architecture.md#roadmap)
for planned milestones.

[Unreleased]: https://github.com/morka17/qa-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/morka17/qa-agent/releases/tag/v0.1.0