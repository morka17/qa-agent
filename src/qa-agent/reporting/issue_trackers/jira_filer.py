"""
Files a `BugReport` as a Jira issue via the REST API v3, and posts
occurrence-count comments on repeat detections of an already-filed bug
(deduped via `triage/dedupe_engine.py`).

Mirrors `ingestion/connectors/jira_connector.py`'s auth pattern (HTTP
Basic with an email + API token) since this is the same Jira Cloud API,
just the write side instead of the read side.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.reporting.report_schema import BugReport, IssueTrackerFilingError, Severity

logger = get_logger(__name__)

# Jira doesn't have a generic "severity" field out of the box on most
# schemes — it's commonly modeled as priority instead. Mapped here so a
# self-hosted Jira with the default priority scheme works without extra
# configuration; a team using a custom severity field can override via
# `field_overrides` in the constructor.
_SEVERITY_TO_JIRA_PRIORITY = {
    Severity.CRITICAL: "Highest",
    Severity.HIGH: "High",
    Severity.MEDIUM: "Medium",
    Severity.LOW: "Low",
    Severity.INFO: "Lowest",
}


class JiraFilerError(IssueTrackerFilingError):
    pass


class JiraFiler:
    """
    Example:
        filer = JiraFiler()
        filed_report = await filer.file(report)
    """

    def __init__(self, settings: Settings | None = None, field_overrides: dict[str, Any] | None = None) -> None:
        self._settings = settings or get_settings()
        self._field_overrides = field_overrides or {}
        if not (
            self._settings.jira_base_url
            and self._settings.jira_email
            and self._settings.jira_api_token
            and self._settings.jira_project_key
        ):
            raise JiraFilerError(
                "JiraFiler requires jira_base_url, jira_email, jira_api_token, and "
                "jira_project_key to be configured in settings."
            )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=str(self._settings.jira_base_url),
            auth=(self._settings.jira_email, self._settings.jira_api_token.get_secret_value()),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30.0,
        )

    def _build_create_payload(self, report: BugReport) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "project": {"key": self._settings.jira_project_key},
            "summary": report.title,
            "description": _markdown_to_adf(report.as_markdown_body()),
            "issuetype": {"name": "Bug"},
            "priority": {"name": _SEVERITY_TO_JIRA_PRIORITY[report.severity]},
            "labels": [label.replace(" ", "-") for label in report.labels],
        }
        fields.update(self._field_overrides)
        return {"fields": fields}

    async def file(self, report: BugReport) -> BugReport:
        payload = self._build_create_payload(report)
        async with self._client() as client:
            response = await client.post("/rest/api/3/issue", json=payload)
            if response.status_code not in (200, 201):
                raise JiraFilerError(
                    f"Jira issue creation failed with status {response.status_code}: "
                    f"{response.text[:500]}"
                )
            data = response.json()

        issue_key = data["key"]
        issue_url = f"{str(self._settings.jira_base_url).rstrip('/')}/browse/{issue_key}"
        logger.info("Filed Jira issue %s for report %s.", issue_key, report.id)

        return report.model_copy(
            update={
                "tracker_name": "jira",
                "tracker_ref": issue_key,
                "tracker_url": issue_url,
                "filed_at": datetime.now(timezone.utc),
            }
        )

    async def add_occurrence_comment(self, report: BugReport, occurrence_count: int) -> None:
        if not report.tracker_ref:
            raise JiraFilerError(
                f"Cannot add occurrence comment: report {report.id} has no tracker_ref "
                "(has it been filed yet?)."
            )
        comment_body = _markdown_to_adf(
            f"Sentinel-QA observed this failure again (occurrence #{occurrence_count}) "
            f"in run `{report.run_id}`."
        )
        async with self._client() as client:
            response = await client.post(
                f"/rest/api/3/issue/{report.tracker_ref}/comment", json={"body": comment_body}
            )
            if response.status_code not in (200, 201):
                raise JiraFilerError(
                    f"Jira comment failed with status {response.status_code}: {response.text[:500]}"
                )
        logger.info("Added occurrence comment to Jira issue %s.", report.tracker_ref)


def _markdown_to_adf(markdown_text: str) -> dict[str, Any]:
    """
    Minimal markdown-to-Atlassian-Document-Format conversion: Jira Cloud's
    API requires ADF (not plain markdown/wiki-markup) for rich text
    fields. This deliberately does NOT attempt full markdown parsing
    (headers, lists, bold) — it wraps each line as its own paragraph node,
    which renders as readable plain text in Jira. A team wanting properly
    rendered markdown should swap this for a full ADF library.
    """
    paragraphs = [
        {"type": "paragraph", "content": [{"type": "text", "text": line}]} if line.strip() else {"type": "paragraph", "content": []}
        for line in markdown_text.splitlines()
    ]
    return {"type": "doc", "version": 1, "content": paragraphs}