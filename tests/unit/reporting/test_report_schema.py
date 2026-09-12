"""Unit tests for qa_agent.reporting.report_schema."""

from qa_agent.reporting.report_schema import (
    BugReport,
    EvidenceAttachment,
    EvidenceType,
    ReproStep,
    Severity,
)
from qa_agent.triage.failure_classifier import FailureCategory


def _report(**overrides) -> BugReport:
    defaults = dict(
        title="Checkout fails with server error",
        severity=Severity.HIGH,
        category=FailureCategory.APP_BUG,
        summary="Checkout fails.",
        description="Placing an order triggers a 500.",
        expected_behavior="Order confirmed.",
        actual_behavior="500 error shown.",
        run_id="run-xyz",
    )
    defaults.update(overrides)
    return BugReport(**defaults)


class TestBugReport:
    def test_defaults(self):
        report = _report()
        assert report.occurrence_count == 1
        assert report.tracker_ref is None
        assert report.filed_at is None
        assert report.repro_steps == []
        assert report.evidence == []

    def test_as_markdown_body_includes_core_sections(self):
        report = _report(
            repro_steps=[
                ReproStep(order=0, description="Go to checkout"),
                ReproStep(order=1, description="Place order"),
            ],
            evidence=[
                EvidenceAttachment(type=EvidenceType.SCREENSHOT, description="failure", path_or_url="/tmp/x.png")
            ],
        )
        markdown = report.as_markdown_body()

        assert "## Description" in markdown
        assert "## Expected Behavior" in markdown
        assert "## Actual Behavior" in markdown
        assert "## Steps to Reproduce" in markdown
        assert "1. Go to checkout" in markdown
        assert "2. Place order" in markdown
        assert "## Evidence" in markdown
        assert report.run_id in markdown

    def test_repro_steps_rendered_in_order_regardless_of_list_order(self):
        report = _report(
            repro_steps=[
                ReproStep(order=1, description="Second step"),
                ReproStep(order=0, description="First step"),
            ]
        )
        markdown = report.as_markdown_body()
        assert markdown.index("First step") < markdown.index("Second step")

    def test_occurrence_count_greater_than_one_is_mentioned(self):
        report = _report(occurrence_count=5)
        markdown = report.as_markdown_body()
        assert "5 time(s)" in markdown

    def test_occurrence_count_of_one_is_not_mentioned(self):
        report = _report(occurrence_count=1)
        markdown = report.as_markdown_body()
        assert "time(s)" not in markdown

    def test_evidence_omitted_section_when_empty(self):
        report = _report(evidence=[])
        markdown = report.as_markdown_body()
        assert "## Evidence" not in markdown
