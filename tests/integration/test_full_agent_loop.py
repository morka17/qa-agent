"""
Integration test for the complete pipeline: UserStory -> ingestion ->
planning -> execution -> verification -> triage -> reporting -> issue
filing. Every LLM call and the browser page are faked, but every real
`qa_agent` module in the chain participates unmodified - this is the
test that proves the modules actually compose, not just that each one
works in isolation (that's what tests/unit/ covers).

Run alone with:
    pytest tests/integration/test_full_agent_loop.py -v
"""

import json
from contextlib import asynccontextmanager

import pytest

from qa_agent.config.settings import get_settings
from qa_agent.ingestion.schemas import Priority, StorySource, UserStory
from qa_agent.ingestion.story_parser import StoryParser
from qa_agent.orchestration.agent_loop import AgentLoop
from qa_agent.orchestration.state_machine import RunState
from qa_agent.perception.element_resolver import ElementResolver
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache
from qa_agent.planning.test_planner import TestPlanner
from qa_agent.reporting.bug_report_writer import BugReportWriter
from qa_agent.reporting.report_schema import BugReport
from qa_agent.triage.root_cause_analyzer import RootCauseAnalyzer


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


DOM_RAW = {
    "url": "https://shop.example.com/checkout",
    "title": "Checkout",
    "elements": [
        {
            "index": 0,
            "tag": "button",
            "role": "button",
            "accessible_name": "Place Order",
            "text": "Place Order",
            "value": None,
            "attributes": {},
            "dom_path": "#place-order",
            "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 30},
            "is_visible": True,
            "is_enabled": True,
        },
        {
            "index": 1,
            "tag": "span",
            "role": None,
            "accessible_name": "order status message",
            "text": "Something went wrong",
            "value": None,
            "attributes": {},
            "dom_path": "#order-status",
            "bounding_box": {"x": 0, "y": 40, "width": 200, "height": 20},
            "is_visible": True,
            "is_enabled": True,
        },
    ],
}


class FakeLocator:
    def __init__(self, selector):
        self.selector = selector

    async def click(self, **kwargs):
        pass

    async def text_content(self):
        return "Something went wrong" if "order-status" in self.selector else None

    async def is_visible(self):
        return True

    async def count(self):
        return 1


class FakePage:
    video = None

    def __init__(self):
        self.url = DOM_RAW["url"]

    async def title(self):
        return DOM_RAW["title"]

    async def evaluate(self, script):
        return DOM_RAW

    async def screenshot(self, **kwargs):
        return b"fakepng"

    async def goto(self, url, **kwargs):
        pass

    async def go_back(self, **kwargs):
        pass

    async def reload(self, **kwargs):
        pass

    def locator(self, selector):
        return FakeLocator(selector)

    def on(self, event, handler):
        pass

    def remove_listener(self, event, handler):
        pass

    async def route(self, pattern, handler):
        pass

    async def unroute(self, pattern):
        pass


class FakeTracing:
    async def start(self, **kwargs):
        pass

    async def stop(self, path=None):
        with open(path, "wb") as f:
            f.write(b"trace")


class FakeContext:
    tracing = FakeTracing()


class FakeBrowserManager:
    is_running = False

    @asynccontextmanager
    async def new_context(self, record_video_dir=None, **kwargs):
        yield FakeContext()

    async def new_page(self, context):
        return FakePage()


class StubIssueTracker:
    def __init__(self):
        self.filed: list[BugReport] = []

    async def file(self, report: BugReport) -> BugReport:
        self.filed.append(report)
        return report.model_copy(
            update={"tracker_name": "stub", "tracker_ref": "STUB-1", "tracker_url": "https://stub/STUB-1"}
        )

    async def add_occurrence_comment(self, report, occurrence_count):
        pass


@pytest.fixture(autouse=True)
def _allow_irreversible_submit(monkeypatch):
    """This scenario's plan includes an irreversible_submit step (placing
    an order) - allowlist it so the test exercises the failure/reporting
    path rather than the approval-gate path, which is already covered by
    tests/unit/planning/test_plan_validator.py."""
    monkeypatch.setenv("DESTRUCTIVE_ACTION_ALLOWLIST", '["irreversible_submit"]')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def story() -> UserStory:
    return UserStory(
        source=StorySource.MANUAL,
        title="Checkout confirmation",
        narrative="As a customer, I want to place an order, so that I receive an order confirmation.",
        priority=Priority.HIGH,
        target_url="https://shop.example.com/checkout",
    )


@pytest.fixture
def agent_loop() -> tuple[AgentLoop, StubIssueTracker]:
    plan_response = json.dumps(
        {
            "preconditions": [],
            "steps": [
                {
                    "action": "navigate",
                    "target_description": None,
                    "value": "https://shop.example.com/checkout",
                    "description": "Go to checkout",
                    "destructive_category": "none",
                },
                {
                    "action": "click",
                    "target_description": "the 'Place Order' button",
                    "value": None,
                    "description": "Place the order",
                    "destructive_category": "irreversible_submit",
                },
            ],
            "assertions": [
                {
                    "after_step_index": 1,
                    "type": "text_equals",
                    "target_description": "the order status message",
                    "expected": "Order Confirmed",
                    "description": "Order confirmation is shown",
                }
            ],
        }
    )
    rca_response = json.dumps(
        {
            "summary": "Placing an order does not show a confirmation message.",
            "likely_cause": "The order status message never updates to a confirmation state.",
            "contributing_factors": ['Status message text remained "Something went wrong"'],
            "confidence": 0.8,
            "suggested_category": "app_bug",
            "recommended_action": "Check the order submission handler for a swallowed failure.",
        }
    )
    report_response = json.dumps(
        {
            "title": "Order confirmation message not shown after placing an order",
            "expected_behavior": "After placing an order, a confirmation message should be shown.",
            "actual_behavior": "The status message shows an error instead of a confirmation.",
            "description": "Placing an order via checkout does not result in a confirmation message.",
        }
    )

    issue_tracker = StubIssueTracker()
    loop = AgentLoop(
        browser_manager=FakeBrowserManager(),
        story_parser=StoryParser(),  # canonical narrative - no LLM needed
        test_planner=TestPlanner(llm_client=StubLLM(plan_response)),
        element_resolver=ElementResolver(vision_client=None, cache=SelectorCache()),
        bug_report_writer=BugReportWriter(llm_client=StubLLM(report_response)),
        issue_tracker=issue_tracker,
        rca_analyzer=RootCauseAnalyzer(llm_client=StubLLM(rca_response)),
    )
    return loop, issue_tracker


@pytest.mark.asyncio
class TestFullAgentLoop:
    async def test_run_reaches_failed_state_with_a_filed_bug(self, agent_loop, story):
        loop, issue_tracker = agent_loop
        result = await loop.run(story)

        assert result.final_state == RunState.FAILED
        assert result.error is None
        assert all(r.success for r in result.execution_results)  # actions succeeded; the *assertion* failed

    async def test_bug_report_has_correct_content(self, agent_loop, story):
        loop, issue_tracker = agent_loop
        result = await loop.run(story)

        assert result.bug_report is not None
        assert "confirmation" in result.bug_report.title.lower()
        assert result.bug_report.severity.value == "high"  # HIGH priority + app_bug -> HIGH
        assert result.bug_report.category.value == "app_bug"
        assert len(result.bug_report.evidence) >= 1  # at least the trace

    async def test_bug_is_filed_against_the_issue_tracker(self, agent_loop, story):
        loop, issue_tracker = agent_loop
        result = await loop.run(story)

        assert len(issue_tracker.filed) == 1
        assert result.bug_report.tracker_ref == "STUB-1"

    async def test_artifacts_are_produced(self, agent_loop, story):
        loop, _ = agent_loop
        result = await loop.run(story)

        assert result.artifacts is not None
        assert result.artifacts.trace_path is not None
        assert result.artifacts.trace_path.exists()
