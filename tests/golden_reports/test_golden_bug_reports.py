"""
Golden-file regression test: regenerates a BugReport from fixed,
deterministic inputs (including a fixed stub LLM response, not a live
model call) and asserts the output matches the checked-in golden files
byte-for-byte. See README.md in this directory for how to update a
golden file when a deliberate behavior change requires it.
"""

import json
from pathlib import Path

import pytest

from qa_agent.ingestion.schemas import Priority, TestIntent
from qa_agent.planning.step_schema import ActionType, Assertion, AssertionType, Step, TestPlan
from qa_agent.reporting.bug_report_writer import BugReportWriter
from qa_agent.triage.failure_classifier import ClassificationResult, FailureCategory, FailureEvidence
from qa_agent.triage.root_cause_analyzer import RootCauseAnalysis
from qa_agent.verification.assertion_engine import AssertionResult

GOLDEN_DIR = Path(__file__).parent


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


def _cart_total_bug_report_response() -> str:
    return json.dumps(
        {
            "title": "Cart total does not update after adding an item to the cart",
            "expected_behavior": "After adding an item to the cart, the cart total should display the item's price.",
            "actual_behavior": "The cart total remains blank after the item is added, even though the item "
            "count increments correctly.",
            "description": "Adding an item to the cart correctly increments the item count but never updates "
            "the displayed cart total, leaving shoppers unable to see how much they will be charged.",
        }
    )


def _cart_total_bug_intent() -> TestIntent:
    return TestIntent(
        id="intent-golden-001",
        story_id="story-golden-001",
        persona="a shopper",
        goal="add an item to the cart and see the updated total",
        rationale="so they know how much they'll be charged",
        priority=Priority.HIGH,
        target_url="http://localhost:8123",
    )


def _cart_total_bug_plan(intent: TestIntent) -> TestPlan:
    return TestPlan(
        id="plan-golden-001",
        test_intent_id=intent.id,
        target_url=intent.target_url,
        steps=[
            Step(
                id="step-0",
                order=0,
                action=ActionType.NAVIGATE,
                value="http://localhost:8123",
                description="Go to the product page",
            ),
            Step(
                id="step-1",
                order=1,
                action=ActionType.CLICK,
                target_description="the 'Add to Cart' button",
                description="Add the item to the cart",
            ),
        ],
        assertions=[
            Assertion(
                id="assertion-0",
                after_step_id="step-1",
                type=AssertionType.TEXT_CONTAINS,
                target_description="the cart total",
                expected="$79.99",
                description="Cart total reflects the item price",
            )
        ],
    )


def _cart_total_bug_evidence() -> FailureEvidence:
    return FailureEvidence(
        assertion_results=[
            AssertionResult(
                assertion_id="assertion-0",
                passed=False,
                actual="",
                expected="$79.99",
                reason="substring not found in element text",
            )
        ]
    )


def _cart_total_bug_classification() -> ClassificationResult:
    return ClassificationResult(
        category=FailureCategory.APP_BUG,
        confidence=1.0,
        reasoning="Rule-based classification: 1 assertion(s) failed with no execution or resolution errors",
        signals=["1 assertion(s) failed with no execution or resolution errors"],
        scores={FailureCategory.APP_BUG: 1.0},
    )


def _cart_total_bug_rca() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        summary="The cart total does not update when an item is added to the cart.",
        likely_cause="The add-to-cart handler updates the item count but never recalculates or renders "
        "the cart total.",
        contributing_factors=["Cart total field remained empty after Add to Cart was clicked"],
        confidence=0.9,
        suggested_category=FailureCategory.APP_BUG,
        recommended_action="Check the add-to-cart click handler for the missing cart-total update.",
        raw_llm_output="{}",
    )


@pytest.mark.asyncio
class TestCartTotalBugGolden:
    async def _generate_report(self):
        intent = _cart_total_bug_intent()
        plan = _cart_total_bug_plan(intent)
        writer = BugReportWriter(llm_client=StubLLM(_cart_total_bug_report_response()))
        return await writer.write(
            run_id="run-golden-001",
            intent=intent,
            plan=plan,
            evidence=_cart_total_bug_evidence(),
            classification=_cart_total_bug_classification(),
            rca=_cart_total_bug_rca(),
            evidence_attachments=[],
        )

    async def test_markdown_matches_golden_file(self):
        report = await self._generate_report()
        expected = (GOLDEN_DIR / "cart_total_bug.md").read_text()
        assert report.as_markdown_body() + "\n" == expected

    async def test_structural_fields_match_golden_file(self):
        report = await self._generate_report()
        expected = json.loads((GOLDEN_DIR / "cart_total_bug.json").read_text())

        actual = {
            "title": report.title,
            "severity": report.severity.value,
            "category": report.category.value,
            "summary": report.summary,
            "expected_behavior": report.expected_behavior,
            "actual_behavior": report.actual_behavior,
            "repro_steps": [{"order": s.order, "description": s.description} for s in report.repro_steps],
            "labels": report.labels,
            "occurrence_count": report.occurrence_count,
        }
        assert actual == expected

    async def test_severity_is_high_for_high_priority_app_bug(self):
        """Documents the specific regression this golden file protects
        against: HIGH-priority + app_bug must always yield HIGH severity."""
        report = await self._generate_report()
        assert report.severity.value == "high"
