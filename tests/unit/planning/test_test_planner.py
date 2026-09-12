"""Unit tests for qa_agent.planning.test_planner."""

import json

import pytest

from qa_agent.ingestion.schemas import TestIntent
from qa_agent.planning.test_planner import ApprovalRequiredError, TestPlanner, TestPlanningError


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


def _intent() -> TestIntent:
    return TestIntent(
        story_id="s1",
        persona="a logged-in customer",
        goal="add an item to the cart",
        rationale="so they can check out",
        preconditions=["the customer is logged in"],
        success_criteria=["the cart shows 1 item"],
        target_url="https://shop.example.com/product/42",
    )


@pytest.mark.asyncio
class TestTestPlanner:
    async def test_generates_valid_plan_from_llm_output(self):
        response = json.dumps(
            {
                "preconditions": [{"description": "the customer is logged in", "is_environmental": True}],
                "steps": [
                    {
                        "action": "navigate",
                        "target_description": None,
                        "value": "https://shop.example.com/product/42",
                        "description": "Go to product page",
                        "destructive_category": "none",
                    },
                    {
                        "action": "click",
                        "target_description": "the 'Add to Cart' button",
                        "value": None,
                        "description": "Add item to cart",
                        "destructive_category": "none",
                    },
                ],
                "assertions": [
                    {
                        "after_step_index": 1,
                        "type": "text_equals",
                        "target_description": "the cart item count",
                        "expected": "1",
                        "description": "Cart shows 1 item",
                    }
                ],
            }
        )
        planner = TestPlanner(llm_client=StubLLM(response))
        plan = await planner.plan(_intent())

        assert len(plan.steps) == 2
        assert len(plan.assertions) == 1
        assert [s.order for s in plan.steps_in_order()] == [0, 1]
        assert plan.preconditions[0].is_environmental is True

    async def test_destructive_step_requiring_approval_raises(self):
        response = json.dumps(
            {
                "preconditions": [],
                "steps": [
                    {
                        "action": "click",
                        "target_description": "the 'Delete Account' button",
                        "value": None,
                        "description": "Delete the account",
                        "destructive_category": "delete",
                    }
                ],
                "assertions": [],
            }
        )
        planner = TestPlanner(llm_client=StubLLM(response))
        with pytest.raises(ApprovalRequiredError) as exc_info:
            await planner.plan(_intent())
        assert exc_info.value.plan is not None
        assert exc_info.value.plan.has_destructive_steps

    async def test_invalid_assertion_index_raises_planning_error(self):
        response = json.dumps(
            {
                "preconditions": [],
                "steps": [
                    {
                        "action": "navigate",
                        "target_description": None,
                        "value": "https://x.com",
                        "description": "go",
                        "destructive_category": "none",
                    }
                ],
                "assertions": [
                    {
                        "after_step_index": 5,  # out of range for a 1-step plan
                        "type": "url_contains",
                        "target_description": None,
                        "expected": "x",
                        "description": "d",
                    }
                ],
            }
        )
        planner = TestPlanner(llm_client=StubLLM(response))
        with pytest.raises(TestPlanningError):
            await planner.plan(_intent())

    async def test_malformed_json_raises_planning_error(self):
        planner = TestPlanner(llm_client=StubLLM("not valid json"))
        with pytest.raises(TestPlanningError):
            await planner.plan(_intent())

    async def test_unrecognized_action_raises_planning_error(self):
        response = json.dumps(
            {
                "preconditions": [],
                "steps": [
                    {
                        "action": "teleport",  # not a real ActionType
                        "target_description": None,
                        "value": None,
                        "description": "go",
                        "destructive_category": "none",
                    }
                ],
                "assertions": [],
            }
        )
        planner = TestPlanner(llm_client=StubLLM(response))
        with pytest.raises(TestPlanningError):
            await planner.plan(_intent())
