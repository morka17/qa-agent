"""Unit tests for qa_agent.reporting.issue_trackers.*, using respx so no
real Jira/GitHub/Linear API calls are made."""

import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from qa_agent.config.settings import Settings
from qa_agent.reporting.issue_trackers.github_issues_filer import (
    GitHubIssuesFiler,
    GitHubIssuesFilerError,
)
from qa_agent.reporting.issue_trackers.jira_filer import JiraFiler, JiraFilerError
from qa_agent.reporting.issue_trackers.linear_filer import LinearFiler, LinearFilerError
from qa_agent.reporting.report_schema import BugReport, Severity
from qa_agent.triage.failure_classifier import FailureCategory


def _report() -> BugReport:
    return BugReport(
        title="Checkout fails with server error",
        severity=Severity.HIGH,
        category=FailureCategory.APP_BUG,
        summary="s",
        description="d",
        expected_behavior="e",
        actual_behavior="a",
        run_id="run-xyz",
        labels=["sentinel-qa", "category:app_bug", "severity:high"],
    )


@pytest.mark.asyncio
class TestJiraFiler:
    def _settings(self) -> Settings:
        return Settings(
            jira_base_url="https://mycompany.atlassian.net",
            jira_email="bot@mycompany.com",
            jira_api_token=SecretStr("token"),
            jira_project_key="QA",
        )

    def test_requires_credentials(self):
        with pytest.raises(JiraFilerError):
            JiraFiler(settings=Settings())

    async def test_files_issue_and_populates_tracker_fields(self):
        settings = self._settings()
        filer = JiraFiler(settings=settings)

        with respx.mock(base_url="https://mycompany.atlassian.net") as mock:
            route = mock.post("/rest/api/3/issue").mock(
                return_value=httpx.Response(201, json={"key": "QA-42", "id": "10001"})
            )
            filed = await filer.file(_report())

        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["fields"]["project"]["key"] == "QA"
        assert body["fields"]["priority"]["name"] == "High"

        assert filed.tracker_name == "jira"
        assert filed.tracker_ref == "QA-42"
        assert filed.tracker_url == "https://mycompany.atlassian.net/browse/QA-42"
        assert filed.filed_at is not None

    async def test_non_2xx_response_raises(self):
        settings = self._settings()
        filer = JiraFiler(settings=settings)

        with respx.mock(base_url="https://mycompany.atlassian.net") as mock:
            mock.post("/rest/api/3/issue").mock(return_value=httpx.Response(400, text="bad request"))
            with pytest.raises(JiraFilerError):
                await filer.file(_report())

    async def test_occurrence_comment_requires_tracker_ref(self):
        settings = self._settings()
        filer = JiraFiler(settings=settings)
        with pytest.raises(JiraFilerError):
            await filer.add_occurrence_comment(_report(), occurrence_count=2)


@pytest.mark.asyncio
class TestGitHubIssuesFiler:
    def _settings(self) -> Settings:
        return Settings(github_token=SecretStr("gh-token"), github_repo="myorg/myrepo")

    def test_requires_credentials(self):
        with pytest.raises(GitHubIssuesFilerError):
            GitHubIssuesFiler(settings=Settings())

    def test_rejects_malformed_repo(self):
        with pytest.raises(GitHubIssuesFilerError):
            GitHubIssuesFiler(settings=Settings(github_token=SecretStr("t"), github_repo="not-org-slash-repo"))

    async def test_files_issue(self):
        settings = self._settings()
        filer = GitHubIssuesFiler(settings=settings)

        with respx.mock(base_url="https://api.github.com") as mock:
            mock.post("/repos/myorg/myrepo/issues").mock(
                return_value=httpx.Response(
                    201, json={"number": 77, "html_url": "https://github.com/myorg/myrepo/issues/77"}
                )
            )
            filed = await filer.file(_report())

        assert filed.tracker_name == "github"
        assert filed.tracker_ref == "77"
        assert filed.tracker_url == "https://github.com/myorg/myrepo/issues/77"

    async def test_add_occurrence_comment(self):
        settings = self._settings()
        filer = GitHubIssuesFiler(settings=settings)
        report = _report().model_copy(update={"tracker_ref": "77"})

        with respx.mock(base_url="https://api.github.com") as mock:
            route = mock.post("/repos/myorg/myrepo/issues/77/comments").mock(
                return_value=httpx.Response(201, json={"id": 1})
            )
            await filer.add_occurrence_comment(report, occurrence_count=3)

        assert route.called


@pytest.mark.asyncio
class TestLinearFiler:
    def _settings(self) -> Settings:
        return Settings(linear_api_key=SecretStr("key"), linear_team_id="team-1")

    def test_requires_credentials(self):
        with pytest.raises(LinearFilerError):
            LinearFiler(settings=Settings())

    async def test_files_issue(self):
        settings = self._settings()
        filer = LinearFiler(settings=settings)

        with respx.mock(base_url="https://api.linear.app") as mock:
            mock.post("/graphql").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "data": {
                            "issueCreate": {
                                "success": True,
                                "issue": {
                                    "id": "internal-1",
                                    "identifier": "ENG-88",
                                    "url": "https://linear.app/myorg/issue/ENG-88",
                                },
                            }
                        }
                    },
                )
            )
            filed = await filer.file(_report())

        assert filed.tracker_name == "linear"
        assert filed.tracker_ref == "ENG-88"

    async def test_graphql_errors_raise(self):
        settings = self._settings()
        filer = LinearFiler(settings=settings)

        with respx.mock(base_url="https://api.linear.app") as mock:
            mock.post("/graphql").mock(
                return_value=httpx.Response(200, json={"errors": [{"message": "bad input"}]})
            )
            with pytest.raises(LinearFilerError):
                await filer.file(_report())

    async def test_add_occurrence_comment_raises_known_limitation(self):
        """Linear's comment mutation needs the issue's internal ID, which
        BugReport does not retain - this is a documented limitation, not
        a silent failure."""
        settings = self._settings()
        filer = LinearFiler(settings=settings)
        report = _report().model_copy(update={"tracker_ref": "ENG-88"})

        with pytest.raises(LinearFilerError):
            await filer.add_occurrence_comment(report, occurrence_count=2)
