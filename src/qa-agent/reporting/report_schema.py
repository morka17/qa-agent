"""
Data contracts for a filed bug report, plus the shared interface every
issue-tracker filer (`issue_trackers/jira_filer.py`,
`github_issues_filer.py`, `linear_filer.py`) implements.

A `BugReport` is deliberately tracker-agnostic: `bug_report_writer.py`
builds one from run evidence without knowing (or caring) which tracker
it'll eventually be filed against, and each filer is responsible for
translating this generic shape into its tracker's specific API payload.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from qa_agent.triage.failure_classifier import FailureCategory


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"  # non-actionable findings: flaky tests, selector drift, environment noise


class EvidenceType(str, Enum):
    SCREENSHOT = "screenshot"
    VIDEO = "video"
    TRACE = "trace"
    CONSOLE_LOG = "console_log"
    NETWORK_LOG = "network_log"
    DOM_SNAPSHOT = "dom_snapshot"


class EvidenceAttachment(BaseModel):
    """One piece of proof attached to the report — the thing that makes Sentinel's reports 'evidence-first' rather than a bare assertion of failure."""

    type: EvidenceType
    description: str = Field(..., min_length=1)
    path_or_url: str = Field(..., description="Local filesystem path or, once uploaded, a public/artifact-store URL.")
    is_uploaded: bool = Field(
        default=False, description="True once evidence_bundler.py has pushed this to durable artifact storage."
    )


class ReproStep(BaseModel):
    order: int = Field(..., ge=0)
    description: str = Field(..., min_length=1)


class BugReport(BaseModel):
    """
    The complete, tracker-agnostic representation of a filed (or
    about-to-be-filed) bug. `bug_report_writer.py` produces this;
    `issue_trackers/*` each know how to render it into their own
    tracker's payload.
    """

    id: str = Field(default_factory=_new_id)
    title: str = Field(..., min_length=1, max_length=200)
    severity: Severity
    category: FailureCategory
    summary: str = Field(..., min_length=1, description="One or two sentence plain-language summary.")
    description: str = Field(..., min_length=1, description="Full narrative description, markdown-formatted.")
    expected_behavior: str
    actual_behavior: str
    repro_steps: list[ReproStep] = Field(default_factory=list)
    evidence: list[EvidenceAttachment] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)

    # Provenance / context
    run_id: str
    target_url: str | None = None
    story_id: str | None = None
    test_plan_id: str | None = None
    cluster_id: str | None = Field(
        default=None, description="Foreign key to triage.dedupe_engine.FailureCluster.id, if deduped."
    )
    occurrence_count: int = Field(default=1, ge=1)

    # Filing state — populated once an issue_trackers/* filer succeeds.
    tracker_name: str | None = None
    tracker_ref: str | None = Field(default=None, description="e.g. Jira key, GitHub issue number, Linear identifier.")
    tracker_url: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    filed_at: datetime | None = None

    def as_markdown_body(self) -> str:
        """
        Renders the report body in a markdown shape every tracker
        understands (Jira's ADF converter, GitHub, and Linear all accept
        markdown), so filers can share this instead of each re-formatting.
        """
        lines = [
            f"**Summary:** {self.summary}",
            "",
            f"**Severity:** {self.severity.value} &nbsp; **Category:** {self.category.value}",
            "",
            "## Description",
            self.description,
            "",
            "## Expected Behavior",
            self.expected_behavior,
            "",
            "## Actual Behavior",
            self.actual_behavior,
        ]
        if self.repro_steps:
            lines += ["", "## Steps to Reproduce"]
            lines += [f"{s.order + 1}. {s.description}" for s in sorted(self.repro_steps, key=lambda s: s.order)]
        if self.evidence:
            lines += ["", "## Evidence"]
            lines += [f"- **{e.type.value}**: {e.description} — {e.path_or_url}" for e in self.evidence]
        if self.occurrence_count > 1:
            lines += ["", f"_Observed {self.occurrence_count} time(s) across automated runs._"]
        lines += ["", f"_Filed automatically by Sentinel-QA from run `{self.run_id}`._"]
        return "\n".join(lines)


class IssueTrackerFilingError(Exception):
    """Raised by any filer for non-recoverable API failures (auth, network, malformed response)."""


class IssueTrackerFiler(Protocol):
    """
    The shared interface every tracker filer implements. `agent_loop.py`
    depends only on this Protocol, never on a specific tracker's client,
    so swapping Jira for Linear (or adding a new tracker) never touches
    orchestration code — only `qa_agent.config.settings.IssueTracker`
    and a new module under `issue_trackers/`.
    """

    async def file(self, report: BugReport) -> BugReport:
        """File a new issue and return the report with tracker_ref/tracker_url/filed_at populated."""
        ...

    async def add_occurrence_comment(self, report: BugReport, occurrence_count: int) -> None:
        """Post a 'seen again' comment on an already-filed issue, for deduped repeat occurrences."""
        ...