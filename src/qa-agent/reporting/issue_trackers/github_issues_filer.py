"""
Files a `BugReport` as a GitHub Issue via the REST API, and posts
occurrence-count comments on repeat detections.

GitHub Issues has no native severity field, so severity is expressed
purely as a label (`severity:high`, already present in `report.labels`
via `bug_report_writer.py`) rather than requiring project-specific custom
field configuration the way Jira does.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.reporting.report_schema import BugReport, IssueTrackerFilingError

logger = get_logger(__name__)

_GITHUB_API_BASE = "https://api.github.com"


class GitHubIssuesFilerError(IssueTrackerFilingError):
    pass


class GitHubIssuesFiler:
    """
    Example:
        filer = GitHubIssuesFiler()
        filed_report = await filer.file(report)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not (self._settings.github_token and self._settings.github_repo):
            raise GitHubIssuesFilerError(
                "GitHubIssuesFiler requires github_token and github_repo "
                "('org/repo') to be configured in settings."
            )
        if "/" not in self._settings.github_repo:
            raise GitHubIssuesFilerError(
                f"github_repo must be in 'org/repo' form, got {self._settings.github_repo!r}."
            )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_GITHUB_API_BASE,
            headers={
                "Authorization": f"Bearer {self._settings.github_token.get_secret_value()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    def _build_create_payload(self, report: BugReport) -> dict[str, Any]:
        return {
            "title": report.title,
            "body": report.as_markdown_body(),
            "labels": [label.replace(" ", "-") for label in report.labels],
        }

    async def file(self, report: BugReport) -> BugReport:
        payload = self._build_create_payload(report)
        async with self._client() as client:
            response = await client.post(f"/repos/{self._settings.github_repo}/issues", json=payload)
            if response.status_code not in (200, 201):
                raise GitHubIssuesFilerError(
                    f"GitHub issue creation failed with status {response.status_code}: "
                    f"{response.text[:500]}"
                )
            data = response.json()

        issue_number = str(data["number"])
        issue_url = data["html_url"]
        logger.info(
            "Filed GitHub issue #%s for report %s in %s.",
            issue_number,
            report.id,
            self._settings.github_repo,
        )

        return report.model_copy(
            update={
                "tracker_name": "github",
                "tracker_ref": issue_number,
                "tracker_url": issue_url,
                "filed_at": datetime.now(timezone.utc),
            }
        )

    async def add_occurrence_comment(self, report: BugReport, occurrence_count: int) -> None:
        if not report.tracker_ref:
            raise GitHubIssuesFilerError(
                f"Cannot add occurrence comment: report {report.id} has no tracker_ref "
                "(has it been filed yet?)."
            )
        comment_body = (
            f"Sentinel-QA observed this failure again (occurrence #{occurrence_count}) "
            f"in run `{report.run_id}`."
        )
        async with self._client() as client:
            response = await client.post(
                f"/repos/{self._settings.github_repo}/issues/{report.tracker_ref}/comments",
                json={"body": comment_body},
            )
            if response.status_code not in (200, 201):
                raise GitHubIssuesFilerError(
                    f"GitHub comment failed with status {response.status_code}: {response.text[:500]}"
                )
        logger.info("Added occurrence comment to GitHub issue #%s.", report.tracker_ref)