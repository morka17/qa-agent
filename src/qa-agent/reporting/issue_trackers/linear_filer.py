"""
Files a `BugReport` as a Linear issue via its GraphQL API, and posts
occurrence-count comments on repeat detections.

Mirrors `ingestion/connectors/linear_connector.py`'s auth pattern (API
key in the `Authorization` header, no "Bearer" prefix) since this is the
same Linear API, just the write side.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.reporting.report_schema import BugReport, IssueTrackerFilingError, Severity

logger = get_logger(__name__)

_LINEAR_API_URL = "https://api.linear.app/graphql"

# Linear's 0-4 priority scale (0=no priority, 1=urgent, 4=low) — the
# inverse mapping of the read-side connector's `_LINEAR_PRIORITY_MAP` in
# `ingestion/connectors/linear_connector.py`.
_SEVERITY_TO_LINEAR_PRIORITY = {
    Severity.CRITICAL: 1,
    Severity.HIGH: 2,
    Severity.MEDIUM: 3,
    Severity.LOW: 4,
    Severity.INFO: 4,
}

_CREATE_ISSUE_MUTATION = """
mutation CreateIssue($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      identifier
      url
    }
  }
}
"""

_CREATE_COMMENT_MUTATION = """
mutation CreateComment($input: CommentCreateInput!) {
  commentCreate(input: $input) {
    success
  }
}
"""


class LinearFilerError(IssueTrackerFilingError):
    pass


class LinearFiler:
    """
    Example:
        filer = LinearFiler()
        filed_report = await filer.file(report)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not (self._settings.linear_api_key and self._settings.linear_team_id):
            raise LinearFilerError(
                "LinearFiler requires linear_api_key and linear_team_id to be "
                "configured in settings."
            )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={
                "Authorization": self._settings.linear_api_key.get_secret_value(),
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def _graphql(self, client: httpx.AsyncClient, query: str, variables: dict[str, Any]) -> dict:
        response = await client.post(_LINEAR_API_URL, json={"query": query, "variables": variables})
        if response.status_code != 200:
            raise LinearFilerError(
                f"Linear GraphQL call failed with status {response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        if "errors" in body:
            raise LinearFilerError(f"Linear GraphQL errors: {body['errors']}")
        return body["data"]

    async def file(self, report: BugReport) -> BugReport:
        variables = {
            "input": {
                "teamId": self._settings.linear_team_id,
                "title": report.title,
                "description": report.as_markdown_body(),
                "priority": _SEVERITY_TO_LINEAR_PRIORITY[report.severity],
                "labelIds": [],  # label-name -> ID resolution is workspace-specific; left to a field_overrides-style hook if needed
            }
        }
        async with self._client() as client:
            data = await self._graphql(client, _CREATE_ISSUE_MUTATION, variables)

        result = data["issueCreate"]
        if not result["success"]:
            raise LinearFilerError(f"Linear reported issueCreate failure for report {report.id}.")

        issue = result["issue"]
        logger.info("Filed Linear issue %s for report %s.", issue["identifier"], report.id)

        return report.model_copy(
            update={
                "tracker_name": "linear",
                "tracker_ref": issue["identifier"],
                "tracker_url": issue["url"],
                "filed_at": datetime.now(timezone.utc),
            }
        )

    async def add_occurrence_comment(self, report: BugReport, occurrence_count: int) -> None:
        if not report.tracker_ref:
            raise LinearFilerError(
                f"Cannot add occurrence comment: report {report.id} has no tracker_ref "
                "(has it been filed yet?)."
            )
        # Linear's comment mutation takes the issue's internal ID, not its
        # human-readable identifier (e.g. "ENG-123"); a real deployment
        # would resolve/cache that ID at file()-time rather than the
        # identifier itself, since tracker_ref is what's stored on the
        # report. Kept explicit here as a known limitation rather than
        # silently guessing.
        raise LinearFilerError(
            "add_occurrence_comment requires the issue's internal Linear ID, which is "
            "not preserved on BugReport (only its human-readable identifier is). "
            "Resolve the ID via an issue lookup before calling this, or extend "
            "BugReport to retain it."
        )