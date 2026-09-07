"""
Fetches user stories from Jira and normalizes them into UserStory objects.

Uses the Jira REST API v3 (`/rest/api/3/search`) with JQL so callers can
scope pulls to a specific sprint, epic, or label rather than a whole
project. Authentication is HTTP Basic with an email + API token, per
Atlassian's documented auth scheme for Jira Cloud.
"""

from __future__ import annotations

from typing import Any

import httpx

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.ingestion.acceptance_criteria import extract_acceptance_criteria
from qa_agent.ingestion.schemas import Priority, StorySource, UserStory

logger = get_logger(__name__)

_DEFAULT_JQL_TEMPLATE = (
    'project = "{project_key}" AND issuetype = Story ORDER BY created DESC'
)

# Jira's default priority names, mapped onto Sentinel's coarser scale.
_JIRA_PRIORITY_MAP = {
    "highest": Priority.CRITICAL,
    "high": Priority.HIGH,
    "medium": Priority.MEDIUM,
    "low": Priority.LOW,
    "lowest": Priority.LOW,
}


class JiraConnectorError(Exception):
    """Raised for any non-recoverable Jira API failure."""


class JiraConnector:
    """
    Pulls stories from Jira Cloud and converts them into UserStory objects.

    Example:
        connector = JiraConnector()
        stories = await connector.fetch_stories(jql='sprint in openSprints()')
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not (
            self._settings.jira_base_url
            and self._settings.jira_email
            and self._settings.jira_api_token
        ):
            raise JiraConnectorError(
                "Jira connector requires jira_base_url, jira_email, and "
                "jira_api_token to be configured in settings."
            )

    async def fetch_stories(
        self,
        jql: str | None = None,
        max_results: int = 50,
        target_url: str | None = None,
    ) -> list[UserStory]:
        """
        Fetch stories matching the given JQL (defaults to all Story-type
        issues in the configured project), paginating through Jira's
        search endpoint until exhausted or `max_results` is reached.
        """
        jql = jql or _DEFAULT_JQL_TEMPLATE.format(
            project_key=self._settings.jira_project_key
        )
        stories: list[UserStory] = []
        start_at = 0
        page_size = min(max_results, 100)

        async with httpx.AsyncClient(
            base_url=str(self._settings.jira_base_url),
            auth=(
                self._settings.jira_email,
                self._settings.jira_api_token.get_secret_value(),
            ),
            headers={"Accept": "application/json"},
            timeout=30.0,
        ) as client:
            while len(stories) < max_results:
                response = await client.get(
                    "/rest/api/3/search",
                    params={
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": page_size,
                        "fields": "summary,description,priority,labels,created",
                    },
                )
                if response.status_code != 200:
                    raise JiraConnectorError(
                        f"Jira search failed with status {response.status_code}: "
                        f"{response.text[:500]}"
                    )

                payload = response.json()
                issues: list[dict[str, Any]] = payload.get("issues", [])
                if not issues:
                    break

                for issue in issues:
                    stories.append(self._issue_to_story(issue, target_url=target_url))

                start_at += len(issues)
                total = payload.get("total", 0)
                if start_at >= total:
                    break

        logger.info("Fetched %d stories from Jira (jql=%r).", len(stories), jql)
        return stories[:max_results]

    @staticmethod
    def _extract_plain_text(description_field: Any) -> str:
        """
        Jira Cloud stores descriptions as Atlassian Document Format (ADF)
        JSON, not plain text. This walks the ADF node tree and
        concatenates every text leaf, which is sufficient for downstream
        NL parsing even though it discards rich formatting.
        """
        if description_field is None:
            return ""
        if isinstance(description_field, str):
            return description_field

        text_parts: list[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "text" and "text" in node:
                    text_parts.append(node["text"])
                for child in node.get("content", []) or []:
                    _walk(child)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(description_field)
        return "\n".join(text_parts)

    def _issue_to_story(self, issue: dict[str, Any], target_url: str | None) -> UserStory:
        fields = issue.get("fields", {})
        narrative = self._extract_plain_text(fields.get("description"))
        priority_name = (fields.get("priority") or {}).get("name", "medium").lower()

        return UserStory(
            external_id=issue.get("key"),
            source=StorySource.JIRA,
            title=fields.get("summary", "Untitled story"),
            narrative=narrative or fields.get("summary", ""),
            acceptance_criteria=extract_acceptance_criteria(narrative),
            priority=_JIRA_PRIORITY_MAP.get(priority_name, Priority.MEDIUM),
            labels=fields.get("labels", []) or [],
            target_url=target_url,
            metadata={
                "jira_id": issue.get("id"),
                "jira_key": issue.get("key"),
                "jira_created": fields.get("created"),
            },
        )