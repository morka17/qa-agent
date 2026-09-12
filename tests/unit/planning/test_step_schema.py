"""Unit tests for qa_agent.planning.step_schema."""

import pytest
from pydantic import ValidationError

from qa_agent.planning.step_schema import (
    ActionType,
    Assertion,
    AssertionType,
    DestructiveCategory,
    Precondition,
    Step,
    TestPlan,
)


class TestStep:
    def test_target_bearing_action_requires_target_description(self):
        with pytest.raises(ValidationError):
            Step(order=0, action=ActionType.CLICK, description="click something")

    def test_navigate_does_not_require_target_description(self):
        step = Step(order=0, action=ActionType.NAVIGATE, value="https://x.com", description="go")
        assert step.target_description is None

    def test_fill_requires_value(self):
        with pytest.raises(ValidationError):
            Step(order=0, action=ActionType.FILL, target_description="the search box", description="fill")

    def test_is_destructive_property(self):
        safe = Step(order=0, action=ActionType.CLICK, target_description="a button", description="d")
        risky = Step(
            order=0,
            action=ActionType.CLICK,
            target_description="delete button",
            description="d",
            destructive_category=DestructiveCategory.DELETE,
        )
        assert not safe.is_destructive
        assert risky.is_destructive


class TestAssertion:
    def test_element_visible_requires_target(self):
        with pytest.raises(ValidationError):
            Assertion(after_step_id="s1", type=AssertionType.ELEMENT_VISIBLE, description="d")

    def test_text_equals_requires_expected(self):
        with pytest.raises(ValidationError):
            Assertion(
                after_step_id="s1",
                type=AssertionType.TEXT_EQUALS,
                target_description="x",
                description="d",
            )

    def test_url_contains_does_not_require_target(self):
        assertion = Assertion(
            after_step_id="s1", type=AssertionType.URL_CONTAINS, expected="/cart", description="on cart"
        )
        assert assertion.target_description is None


class TestTestPlan:
    def _make_step(self, order: int) -> Step:
        return Step(order=order, action=ActionType.NAVIGATE, value="https://x.com", description="go")

    def test_requires_at_least_one_step(self):
        with pytest.raises(ValidationError):
            TestPlan(test_intent_id="i1", steps=[])

    def test_steps_in_order(self):
        s0, s1 = self._make_step(0), self._make_step(1)
        plan = TestPlan(test_intent_id="i1", steps=[s1, s0])
        assert plan.steps_in_order() == [s0, s1]

    def test_step_by_id(self):
        s0 = self._make_step(0)
        plan = TestPlan(test_intent_id="i1", steps=[s0])
        assert plan.step_by_id(s0.id) is s0
        assert plan.step_by_id("missing") is None

    def test_assertions_after(self):
        s0 = self._make_step(0)
        a1 = Assertion(after_step_id=s0.id, type=AssertionType.URL_CONTAINS, expected="x", description="d")
        a2 = Assertion(after_step_id="other", type=AssertionType.URL_CONTAINS, expected="y", description="d")
        plan = TestPlan(test_intent_id="i1", steps=[s0], assertions=[a1, a2])
        assert plan.assertions_after(s0.id) == [a1]

    def test_has_destructive_steps(self):
        safe = self._make_step(0)
        risky = Step(
            order=1,
            action=ActionType.CLICK,
            target_description="delete",
            description="d",
            destructive_category=DestructiveCategory.DELETE,
        )
        plan = TestPlan(test_intent_id="i1", steps=[safe, risky])
        assert plan.has_destructive_steps


class TestPrecondition:
    def test_defaults(self):
        pc = Precondition(description="user is logged in")
        assert pc.setup_step_id is None
        assert pc.is_environmental is False
