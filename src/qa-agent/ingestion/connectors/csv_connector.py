"""
Reads user stories from a local CSV/TSV file and normalizes them into
UserStory objects.

This is the connector for the common "product manager exported a backlog
spreadsheet" workflow, and is also the one used in CI/demo fixtures since
it has no external network dependency.

Expected columns (header row required; extra columns are preserved in
`metadata` and unrecognized ones are ignored):

    title, narrative, acceptance_criteria, priority, labels, target_url, external_id

Only `title` and `narrative` are required — everything else has a
sensible default.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.ingestion.acceptance_criteria import extract_acceptance_criteria
from qa_agent.ingestion.schemas import Priority, StorySource, UserStory

logger = get_logger(__name__)

_REQUIRED_COLUMNS = {"title", "narrative"}
_KNOWN_COLUMNS = _REQUIRED_COLUMNS | {
    "acceptance_criteria",
    "priority",
    "labels",
    "target_url",
    "external_id",
}


class CSVConnectorError(Exception):
    """Raised when the CSV is missing required columns or is unparseable."""


class CSVConnector:
    """
    Reads UserStory records from a CSV/TSV file on disk.

    Example:
        connector = CSVConnector()
        stories = connector.read_stories(Path("backlog.csv"))
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def read_stories(
        self,
        path: str | Path,
        target_url: str | None = None,
        delimiter: str | None = None,
    ) -> list[UserStory]:
        path = Path(path)
        if not path.exists():
            raise CSVConnectorError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            stories = self._parse(
                fh,
                delimiter=delimiter or self._settings.csv_default_delimiter,
                default_target_url=target_url,
                source_label=str(path),
            )

        logger.info("Read %d stories from CSV %s.", len(stories), path)
        return stories

    def _parse(
        self,
        fh: TextIO,
        delimiter: str,
        default_target_url: str | None,
        source_label: str,
    ) -> list[UserStory]:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None:
            raise CSVConnectorError(f"{source_label} appears to be empty (no header row).")

        header = {name.strip().lower() for name in reader.fieldnames}
        missing = _REQUIRED_COLUMNS - header
        if missing:
            raise CSVConnectorError(
                f"{source_label} is missing required column(s): {sorted(missing)}. "
                f"Required: {sorted(_REQUIRED_COLUMNS)}."
            )

        stories: list[UserStory] = []
        for row_number, row in enumerate(reader, start=2):  # header is row 1
            normalized = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            try:
                stories.append(
                    self._row_to_story(
                        normalized,
                        row_number=row_number,
                        default_target_url=default_target_url,
                        source_label=source_label,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - we want to skip-and-log, not abort a whole batch
                logger.warning(
                    "Skipping %s row %d due to parse error: %s", source_label, row_number, exc
                )
                continue

        return stories

    @staticmethod
    def _row_to_story(
        row: dict[str, str],
        row_number: int,
        default_target_url: str | None,
        source_label: str,
    ) -> UserStory:
        title = row.get("title", "")
        narrative = row.get("narrative", "")
        if not title or not narrative:
            raise CSVConnectorError(
                f"Row {row_number} in {source_label} is missing title or narrative."
            )

        raw_priority = row.get("priority", "").lower()
        try:
            priority = Priority(raw_priority) if raw_priority else Priority.MEDIUM
        except ValueError:
            priority = Priority.MEDIUM

        labels = [label.strip() for label in row.get("labels", "").split(";") if label.strip()]

        # Acceptance criteria can be supplied pre-written in the CSV
        # (pipe-separated Given/When/Then lines); otherwise infer from
        # the narrative text.
        raw_ac = row.get("acceptance_criteria", "")
        acceptance_criteria = (
            extract_acceptance_criteria(raw_ac.replace("|", "\n"))
            if raw_ac
            else extract_acceptance_criteria(narrative)
        )

        extra_metadata: dict[str, Any] = {
            k: v for k, v in row.items() if k not in _KNOWN_COLUMNS and v
        }
        extra_metadata["source_file"] = source_label
        extra_metadata["source_row"] = row_number

        return UserStory(
            external_id=row.get("external_id") or None,
            source=StorySource.CSV,
            title=title,
            narrative=narrative,
            acceptance_criteria=acceptance_criteria,
            priority=priority,
            labels=labels,
            target_url=row.get("target_url") or default_target_url,
            metadata=extra_metadata,
        )