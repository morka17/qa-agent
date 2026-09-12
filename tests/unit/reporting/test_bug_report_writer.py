"""Unit tests for qa_agent.reporting.bug_report_writer."""

import json

import pytest

from qa_agent.ingestion.schemas import Priority, TestIntent
from qa_agent.planning.step_schema import ActionType, Step, TestPlan
from qa_agent.reporting.bug_report_writer import (
    BugReportWriter,
    BugReportWritingError,
    determine_severity,
)
from qa_agent.reporting.report_schema import Severity
from qa_agent.triage.failure_classifier import ClassificationResult, FailureCategory, FailureEvidence
from qa_agent.triage.root_cause_analyzer import RootCauseAnalysis
from qa_agent.verification.assertion_engine import AssertionResult


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


def _intent(priority=Priority.HIGH) -> TestIntent:
    return TestIntent(
        story_id="s1",
        persona="a customer",
        goal="checkout an order",
        priority=priority,
        target_url="https://shop.example.com/checkout",
    )


def _plan(intent: TestIntent) -> TestPlan:
    return TestPlan(
        test_intent_id=intent.id,
        target_url=intent.target_url,
        steps=[
            Step(order=0, action=ActionType.NAVIGATE, value=intent.target_url, description="Go to checkout"),
            Step(
                order=1,
                action=ActionType.CLICK,
                target_description="the 'Place Order' button",
                description="Place the order",
            ),
        ],
    )


def _classification(category=FailureCategory.APP_BUG) -> ClassificationResult:
    return ClassificationResult(category=category, confidence=0.9, reasoning="x", signals=[], scores={})


def _rca() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        summary="Checkout fails with a server error.",
        likely_cause="Backend 500 on order submission.",
        contributing_factors=["5xx response"],
        confidence=0.8,
        suggested_category=FailureCategory.APP_BUG,
        recommended_action="Check backend logs.",
        raw_llm_output="{}",
    )


class TestDetermineSeverity:
    def test_app_bug_uses_priority_directly(self):
        assert determine_severity(FailureCategory.APP_BUG, Priority.CRITICAL) == Severity.CRITICAL
        assert determine_severity(FailureCategory.APP_BUG, Priority.LOW) == Severity.LOW

    def test_selector_drift_is_capped_to_info_even_at_critical_priority(self):
        assert determine_severity(FailureCategory.SELECTOR_DRIFT, Priority.CRITICAL) == Severity.INFO

    def test_flaky_test_is_capped_to_low(self):
        assert determine_severity(FailureCategory.FLAKY_TEST, Priority.CRITICAL) == Severity.LOW

    def test_cap_never_raises_severity_above_priority_derived_base(self):
        # A low-priority story's app_bug should stay LOW, not be inflated.
        assert determine_severity(FailureCategory.APP_BUG, Priority.LOW) == Severity.LOW


@pytest.mark.asyncio
class TestBugReportWriter:
    async def test_writes_complete_report(self):
        intent = _intent()
        plan = _plan(intent)
        evidence = FailureEvidence(
            assertion_results=[
                AssertionResult(
                    assertion_id="a1",
                    passed=False,
                    actual="error page",
                    expected="confirmation page",
                    reason="confirmation not shown",
                )
            ]
        )
        llm_response = json.dumps(
            {
                "title": "Checkout fails with server error when placing an order",
                "expected_behavior": "The order should be confirmed.",
                "actual_behavior": "The backend returns a 500 error.",
                "description": "Placing an order triggers a server-side error.",
            }
        )
        writer = BugReportWriter(llm_client=StubLLM(llm_response))
        report = await writer.write(
            run_id="run-xyz",
            intent=intent,
            plan=plan,
            evidence=evidence,
            classification=_classification(),
            rca=_rca(),
            evidence_attachments=[],
        )

        assert report.title.startswith("Checkout fails")
        assert report.severity == Severity.HIGH  # HIGH priority + app_bug
        assert report.category == FailureCategory.APP_BUG
        assert [s.description for s in report.repro_steps] == ["Go to checkout", "Place the order"]
        assert "sentinel-qa" in report.labels
        assert "category:app_bug" in report.labels

    async def test_missing_required_field_raises(self):
        intent = _intent()
        plan = _plan(intent)
        evidence = FailureEvidence()
        llm_response = json.dumps({"title": "x"})  # missing other required fields
        writer = BugReportWriter(llm_client=StubLLM(llm_response))

        with pytest.raises(BugReportWritingError):
            await writer.write(
                run_id="run-1",
                intent=intent,
                plan=plan,
                evidence=evidence,
                classification=_classification(),
                rca=_rca(),
                evidence_attachments=[],
            )

    async def test_malformed_json_raises(self):
        intent = _intent()
        plan = _plan(intent)
        writer = BugReportWriter(llm_client=StubLLM("not json"))

        with pytest.raises(BugReportWritingError):
            await writer.write(
                run_id="run-1",
                intent=intent,
                plan=plan,
                evidence=FailureEvidence(),
                classification=_classification(),
                rca=_rca(),
                evidence_attachments=[],
            )
