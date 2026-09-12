"""Unit tests for qa_agent.ingestion.connectors.*, using httpx.MockTransport
so no real network calls are made against Jira/Linear."""

import tempfile
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from qa_agent.config.settings import Settings
from qa_agent.ingestion.connectors.csv_connector import CSVConnector, CSVConnectorError
from qa_agent.ingestion.connectors.jira_connector import JiraConnector, JiraConnectorError
from qa_agent.ingestion.connectors.linear_connector import LinearConnector, LinearConnectorError
from qa_agent.ingestion.schemas import Priority, StorySource


@pytest.mark.asyncio
class TestJiraConnector:
    async def test_requires_credentials(self):
        with pytest.raises(JiraConnectorError):
            JiraConnector(settings=Settings())

    async def test_maps_issue_to_story(self):
        settings = Settings(
            jira_base_url="https://example.atlassian.net",
            jira_email="bot@example.com",
            jira_api_token=SecretStr("token"),
            jira_project_key="QA",
        )
        connector = JiraConnector(settings=settings)

        raw_issue = {
            "id": "1",
            "key": "QA-1",
            "fields": {
                "summary": "Add to cart",
                "description": {
                    "content": [
                        {"content": [{"type": "text", "text": "As a user, I want to add to cart."}]}
                    ]
                },
                "priority": {"name": "High"},
                "labels": ["cart"],
                "created": "2026-01-01T00:00:00Z",
            },
        }
        story = connector._issue_to_story(raw_issue, target_url=None)

        assert story.source == StorySource.JIRA
        assert story.external_id == "QA-1"
        assert story.priority == Priority.HIGH
        assert story.labels == ["cart"]
        assert "add to cart" in story.narrative.lower()

    async def test_fetch_stories_paginates_via_mock_transport(self):
        import respx

        settings = Settings(
            jira_base_url="https://example.atlassian.net",
            jira_email="bot@example.com",
            jira_api_token=SecretStr("token"),
            jira_project_key="QA",
        )
        connector = JiraConnector(settings=settings)

        with respx.mock(base_url="https://example.atlassian.net") as mock:
            mock.get("/rest/api/3/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "issues": [
                            {
                                "id": "1",
                                "key": "QA-1",
                                "fields": {
                                    "summary": "s",
                                    "description": None,
                                    "priority": {"name": "Medium"},
                                    "labels": [],
                                },
                            }
                        ],
                        "total": 1,
                    },
                )
            )
            stories = await connector.fetch_stories()

        assert len(stories) == 1
        assert stories[0].external_id == "QA-1"


@pytest.mark.asyncio
class TestLinearConnector:
    async def test_requires_credentials(self):
        with pytest.raises(LinearConnectorError):
            LinearConnector(settings=Settings())

    async def test_maps_issue_to_story(self):
        settings = Settings(linear_api_key=SecretStr("key"), linear_team_id="team-1")
        connector = LinearConnector(settings=settings)

        raw_issue = {
            "id": "internal-1",
            "identifier": "ENG-42",
            "title": "Fix checkout bug",
            "description": "As a user, I want checkout to work.",
            "priority": 1,
            "labels": {"nodes": [{"name": "bug"}]},
            "createdAt": "2026-01-01T00:00:00Z",
        }
        story = connector._issue_to_story(raw_issue, target_url="https://shop.example.com")

        assert story.source == StorySource.LINEAR
        assert story.external_id == "ENG-42"
        assert story.priority == Priority.CRITICAL  # Linear priority 1 == urgent
        assert story.labels == ["bug"]
        assert story.target_url == "https://shop.example.com"


class TestCSVConnector:
    def test_reads_valid_csv(self):
        content = (
            "title,narrative,priority,labels,target_url\n"
            'Add to cart,"As a customer, I want to add an item to my cart.",high,cart;checkout,'
            "https://shop.example.com\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            stories = CSVConnector().read_stories(path)
        finally:
            path.unlink()

        assert len(stories) == 1
        assert stories[0].title == "Add to cart"
        assert stories[0].priority == Priority.HIGH
        assert stories[0].labels == ["cart", "checkout"]

    def test_missing_required_column_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("title\nOnly a title\n")  # missing 'narrative'
            path = Path(f.name)

        try:
            with pytest.raises(CSVConnectorError):
                CSVConnector().read_stories(path)
        finally:
            path.unlink()

    def test_missing_file_raises(self):
        with pytest.raises(CSVConnectorError):
            CSVConnector().read_stories("/nonexistent/path.csv")

    def test_skips_malformed_row_without_aborting_batch(self):
        content = (
            "title,narrative\n"
            "Good row,This has a narrative\n"
            ",Missing title\n"  # malformed: empty title
            "Another good row,Also has a narrative\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            stories = CSVConnector().read_stories(path)
        finally:
            path.unlink()

        assert len(stories) == 2
        assert [s.title for s in stories] == ["Good row", "Another good row"]
