"""Unit tests for qa_agent.planning.plan_validator."""

from qa_agent.config.settings import Settings
from qa_agent.planning.plan_validator import Severity, validate_plan
from qa_agent.planning.step_schema import (
    ActionType,
    Assertion,
    AssertionType,
    DestructiveCategory,
    Precondition,
    Step,
    TestPlan,
)


def _navigate_step(order: int) -> Step:
    return Step(order=order, action=ActionType.NAVIGATE, value="https://x.com", description="go")


class TestStructuralValidation:
    def test_valid_plan_has_no_errors(self):
        step = _navigate_step(0)
        assertion = Assertion(
            after_step_id=step.id, type=AssertionType.URL_CONTAINS, expected="x.com", description="on site"
        )
        plan = TestPlan(test_intent_id="i1", steps=[step], assertions=[assertion])
        result = validate_plan(plan)
        assert result.is_valid
        assert result.errors == []

    def test_duplicate_step_order_is_an_error(self):
        s0 = Step(order=0, action=ActionType.NAVIGATE, value="https://x.com", description="a")
        s1 = Step(order=0, action=ActionType.NAVIGATE, value="https://y.com", description="b")
        plan = TestPlan(test_intent_id="i1", steps=[s0, s1])
        result = validate_plan(plan)
        assert not result.is_valid
        assert any(i.code == "DUPLICATE_STEP_ORDER" for i in result.errors)

    def test_non_contiguous_step_order_is_an_error(self):
        s0 = Step(order=0, action=ActionType.NAVIGATE, value="https://x.com", description="a")
        s2 = Step(order=2, action=ActionType.NAVIGATE, value="https://y.com", description="b")
        plan = TestPlan(test_intent_id="i1", steps=[s0, s2])
        result = validate_plan(plan)
        assert any(i.code == "NON_CONTIGUOUS_STEP_ORDER" for i in result.errors)

    def test_dangling_assertion_reference_is_an_error(self):
        step = _navigate_step(0)
        assertion = Assertion(
            after_step_id="does-not-exist", type=AssertionType.URL_CONTAINS, expected="x", description="d"
        )
        plan = TestPlan(test_intent_id="i1", steps=[step], assertions=[assertion])
        result = validate_plan(plan)
        assert any(i.code == "DANGLING_ASSERTION_REFERENCE" for i in result.errors)

    def test_dangling_precondition_reference_is_an_error(self):
        step = _navigate_step(0)
        precondition = Precondition(description="x", setup_step_id="does-not-exist")
        plan = TestPlan(test_intent_id="i1", steps=[step], preconditions=[precondition])
        result = validate_plan(plan)
        assert any(i.code == "DANGLING_PRECONDITION_REFERENCE" for i in result.errors)

    def test_no_assertions_is_a_warning_not_an_error(self):
        plan = TestPlan(test_intent_id="i1", steps=[_navigate_step(0)])
        result = validate_plan(plan)
        assert result.is_valid  # warnings don't block execution
        assert any(i.code == "NO_ASSERTIONS" for i in result.warnings)

    def test_unusually_long_plan_is_a_warning(self):
        steps = [
            Step(order=i, action=ActionType.NAVIGATE, value="https://x.com", description="go")
            for i in range(61)
        ]
        plan = TestPlan(test_intent_id="i1", steps=steps)
        result = validate_plan(plan)
        assert any(i.code == "PLAN_UNUSUALLY_LONG" for i in result.warnings)


class TestGuardrails:
    def test_destructive_step_in_allowlist_produces_no_warning(self):
        step = Step(
            order=0,
            action=ActionType.CLICK,
            target_description="delete account button",
            description="delete",
            destructive_category=DestructiveCategory.DELETE,
        )
        plan = TestPlan(test_intent_id="i1", steps=[step])
        settings = Settings(destructive_action_allowlist=["delete"])
        result = validate_plan(plan, settings)
        assert not result.requires_human_approval
        assert not any(i.code.startswith("DESTRUCTIVE_ACTION") for i in result.issues)

    def test_destructive_step_not_allowlisted_requires_approval_by_default(self):
        step = Step(
            order=0,
            action=ActionType.CLICK,
            target_description="delete account button",
            description="delete",
            destructive_category=DestructiveCategory.DELETE,
        )
        plan = TestPlan(test_intent_id="i1", steps=[step])
        settings = Settings(destructive_action_allowlist=[], require_human_approval_for_destructive_actions=True)
        result = validate_plan(plan, settings)
        assert result.is_valid  # approval-required is a warning, not a blocking error
        assert result.requires_human_approval
        assert any(i.code == "DESTRUCTIVE_ACTION_NEEDS_APPROVAL" for i in result.warnings)

    def test_destructive_step_ungated_when_approval_disabled(self):
        step = Step(
            order=0,
            action=ActionType.CLICK,
            target_description="delete account button",
            description="delete",
            destructive_category=DestructiveCategory.DELETE,
        )
        plan = TestPlan(test_intent_id="i1", steps=[step])
        settings = Settings(
            destructive_action_allowlist=[], require_human_approval_for_destructive_actions=False
        )
        result = validate_plan(plan, settings)
        assert not result.requires_human_approval
        assert any(i.code == "DESTRUCTIVE_ACTION_UNGATED" for i in result.warnings)

    def test_severity_classification(self):
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
