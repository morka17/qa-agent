"""
Fetches user stories (issues) from Linear and normalizes them into
UserStory objects.

Linear only exposes a GraphQL API, so this connector issues a single
paginated query rather than a REST call. Authentication is a personal or
workspace API key sent as the `Authorization` header (no "Bearer" prefix,
per Linear's API docs).
"""

from __future__ import annotations

from typing import Any

import httpx

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.ingestion.acceptance_criteria import extract_acceptance_criteria
from qa_agent.ingestion.schemas import Priority, StorySource, UserStory

logger = get_logger(__name__)

_LINEAR_API_URL = "https://api.linear.app/graphql"

# Linear uses a numeric 0-4 priority scale (0 = no priority, 1 = urgent,
# 4 = low). Mapped onto Sentinel's coarser scale.
_LINEAR_PRIORITY_MAP: dict[int, Priority] = {
    0: Priority.MEDIUM,
    1: Priority.CRITICAL,
    2: Priority.HIGH,
    3: Priority.MEDIUM,
    4: Priority.LOW,
}

_ISSUES_QUERY = """
query TeamIssues($teamId: String!, $first: Int!, $after: String) {
  team(id: $teamId) {
    issues(first: $first, after: $after, filter: { state: { type: { neq: "canceled" } } }) {
      nodes {
        id
        identifier
        title
        description
        priority
        labels {
          nodes { name }
        }
        createdAt
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""


class LinearConnectorError(Exception):
    """Raised for any non-recoverable Linear API failure."""


class LinearConnector:
    """
    Pulls issues from a configured Linear team and converts them into
    UserStory objects.

    Example:
        connector = LinearConnector()
        stories = await connector.fetch_stories(max_results=100)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not (self._settings.linear_api_key and self._settings.linear_team_id):
            raise LinearConnectorError(
                "Linear connector requires linear_api_key and "
                "linear_team_id to be configured in settings."
            )

    async def fetch_stories(
        self,
        max_results: int = 50,
        target_url: str | None = None,
    ) -> list[UserStory]:
        stories: list[UserStory] = []
        cursor: str | None = None
        page_size = min(max_results, 50)

        async with httpx.AsyncClient(
            headers={
                "Authorization": self._settings.linear_api_key.get_secret_value(),
                "Content-Type": "application/json",
            },
            timeout=30.0,
        ) as client:
            while len(stories) < max_results:
                response = await client.post(
                    _LINEAR_API_URL,
                    json={
                        "query": _ISSUES_QUERY,
                        "variables": {
                            "teamId": self._settings.linear_team_id,
                            "first": page_size,
                            "after": cursor,
                        },
                    },
                )
                if response.status_code != 200:
                    raise LinearConnectorError(
                        f"Linear query failed with status {response.status_code}: "
                        f"{response.text[:500]}"
                    )

                body = response.json()
                if "errors" in body:
                    raise LinearConnectorError(f"Linear GraphQL errors: {body['errors']}")

                issues_conn = body["data"]["team"]["issues"]
                nodes: list[dict[str, Any]] = issues_conn["nodes"]
                if not nodes:
                    break

                for issue in nodes:
                    stories.append(self._issue_to_story(issue, target_url=target_url))

                page_info = issues_conn["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                cursor = page_info["endCursor"]

        logger.info(
            "Fetched %d stories from Linear (team=%s).",
            len(stories),
            self._settings.linear_team_id,
        )
        return stories[:max_results]

    def _issue_to_story(self, issue: dict[str, Any], target_url: str | None) -> UserStory:
        narrative = issue.get("description") or issue.get("title", "")
        priority_value = issue.get("priority", 0) or 0
        labels = [node["name"] for node in (issue.get("labels", {}).get("nodes", []) or [])]

        return UserStory(
            external_id=issue.get("identifier"),
            source=StorySource.LINEAR,
            title=issue.get("title", "Untitled story"),
            narrative=narrative,
            acceptance_criteria=extract_acceptance_criteria(narrative),
            priority=_LINEAR_PRIORITY_MAP.get(priority_value, Priority.MEDIUM),
            labels=labels,
            target_url=target_url,
            metadata={
                "linear_id": issue.get("id"),
                "linear_identifier": issue.get("identifier"),
                "linear_created_at": issue.get("createdAt"),
            },
        )