"""Unit tests for qa_agent.execution.action_executor."""

import pytest
from playwright.async_api import Error as PlaywrightError

from qa_agent.execution.action_executor import ActionExecutor
from qa_agent.execution.retry_policy import RetryPolicy
from qa_agent.perception.dom_snapshot import BoundingBox, DomElementNode, DomSnapshot
from qa_agent.perception.element_resolver import ElementResolver
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache
from qa_agent.planning.step_schema import ActionType, Step


def _snapshot() -> DomSnapshot:
    el = DomElementNode(
        index=0,
        tag="button",
        role="button",
        accessible_name="Add to Cart",
        text="Add to Cart",
        value=None,
        attributes={},
        dom_path="#add-to-cart",
        bounding_box=BoundingBox(0, 0, 10, 10),
        is_visible=True,
        is_enabled=True,
    )
    return DomSnapshot(url="https://shop.example.com/product/42", title="Product", elements=[el])


class FakeLocator:
    def __init__(self, selector, fail_first_n=0):
        self.selector = selector
        self._fail_first_n = fail_first_n
        self.click_calls = 0

    async def click(self, **kwargs):
        self.click_calls += 1
        if self.click_calls <= self._fail_first_n:
            raise PlaywrightError("Element is not attached to the DOM")


class FakePage:
    def __init__(self, snapshot: DomSnapshot, fail_first_n: int = 0):
        self._snapshot = snapshot
        self.url = snapshot.url
        self._fail_first_n = fail_first_n
        self.evaluate_calls = 0
        self._shared_locator: FakeLocator | None = None

    async def title(self):
        return self._snapshot.title

    async def evaluate(self, script):
        self.evaluate_calls += 1
        return {
            "url": self._snapshot.url,
            "title": self._snapshot.title,
            "elements": [
                {
                    "index": e.index,
                    "tag": e.tag,
                    "role": e.role,
                    "accessible_name": e.accessible_name,
                    "text": e.text,
                    "value": e.value,
                    "attributes": e.attributes,
                    "dom_path": e.dom_path,
                    "bounding_box": {"x": 0, "y": 0, "width": 10, "height": 10},
                    "is_visible": e.is_visible,
                    "is_enabled": e.is_enabled,
                }
                for e in self._snapshot.elements
            ],
        }

    async def screenshot(self):
        raise AssertionError("vision should not be needed for these tests")

    async def goto(self, url, **kwargs):
        pass

    async def go_back(self, **kwargs):
        pass

    async def reload(self, **kwargs):
        pass

    def locator(self, selector):
        # Share one locator instance across attempts so its failure
        # counter persists between self-healing retries.
        if self._shared_locator is None:
            self._shared_locator = FakeLocator(selector, fail_first_n=self._fail_first_n)
        return self._shared_locator


@pytest.mark.asyncio
class TestActionExecutor:
    async def test_click_succeeds_immediately(self):
        snapshot = _snapshot()
        page = FakePage(snapshot)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        executor = ActionExecutor(
            resolver=resolver, retry_policy=RetryPolicy(max_attempts=3, base_delay_ms=1, max_delay_ms=2)
        )
        step = Step(
            order=0, action=ActionType.CLICK, target_description="the 'Add to Cart' button", description="click"
        )

        result = await executor.execute_step(page, step, snapshot)

        assert result.success
        assert result.resolved_element is not None
        assert page.evaluate_calls == 0  # no self-healing needed

    async def test_self_healing_recovers_after_transient_failures(self):
        snapshot = _snapshot()
        page = FakePage(snapshot, fail_first_n=2)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        executor = ActionExecutor(
            resolver=resolver, retry_policy=RetryPolicy(max_attempts=4, base_delay_ms=1, max_delay_ms=2)
        )
        step = Step(
            order=0, action=ActionType.CLICK, target_description="the 'Add to Cart' button", description="click"
        )

        result = await executor.execute_step(page, step, snapshot)

        assert result.success
        assert page.evaluate_calls == 2  # one re-snapshot per failed attempt

    async def test_exhausted_retries_produce_failed_result_not_a_crash(self):
        snapshot = _snapshot()
        page = FakePage(snapshot, fail_first_n=99)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        executor = ActionExecutor(
            resolver=resolver, retry_policy=RetryPolicy(max_attempts=2, base_delay_ms=1, max_delay_ms=2)
        )
        step = Step(
            order=0, action=ActionType.CLICK, target_description="the 'Add to Cart' button", description="click"
        )

        result = await executor.execute_step(page, step, snapshot)

        assert not result.success
        assert result.error is not None

    async def test_navigate_action_does_not_resolve_a_target(self):
        snapshot = _snapshot()
        page = FakePage(snapshot)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        executor = ActionExecutor(resolver=resolver)
        step = Step(order=0, action=ActionType.NAVIGATE, value="https://shop.example.com", description="go")

        result = await executor.execute_step(page, step, snapshot)

        assert result.success
        assert result.resolved_element is None
