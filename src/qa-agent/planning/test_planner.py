"""
Generates an executable TestPlan (ordered steps + assertions) from a
TestIntent.

Like `ingestion/story_parser.py`, the LLM dependency is expressed as a
local `Protocol` rather than a concrete import, so this module has no
hard dependency on which provider is wired up in `qa_agent.llm`.

The planner's output is *always* run through `plan_validator.py` before
being returned — a plan that fails structural validation is a planner
bug, not something callers should have to remember to check for
themselves. A plan that requires human approval (destructive actions) is
still returned, with `TestPlanner.plan()` raising `ApprovalRequiredError`
so the orchestration layer can route it to an approval queue instead of
silently executing.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.ingestion.schemas import TestIntent
from qa_agent.planning.plan_validator import (
    PlanValidationError,
    ValidationResult,
    validate_plan,
)
from qa_agent.planning.step_schema import (
    ActionType,
    Assertion,
    AssertionType,
    DestructiveCategory,
    Precondition,
    Step,
    TestPlan,
)

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a senior QA engineer converting a test intent into a
precise, ordered browser test plan. Return ONLY a JSON object (no markdown, no
prose) with this exact shape:

{
  "preconditions": [
    {"description": "<state that must hold before starting>", "is_environmental": <true|false>}
  ],
  "steps": [
    {
      "action": "<one of: navigate, click, fill, select_option, check, uncheck, hover, "
                "scroll_to, press_key, upload_file, wait_for, go_back, reload>",
      "target_description": "<natural-language description of the element, or null for "
                             "navigate/go_back/reload>",
      "value": "<input value if the action needs one, else null>",
      "description": "<human-readable summary of this step>",
      "destructive_category": "<one of: delete, payment, irreversible_submit, "
                               "account_modification, data_export, none>"
    }
  ],
  "assertions": [
    {
      "after_step_index": <integer index into the steps array above, 0-indexed>,
      "type": "<one of: element_visible, element_not_visible, text_equals, text_contains, "
              "url_equals, url_contains, element_count, no_console_errors, custom>",
      "target_description": "<element description, or null if not applicable>",
      "expected": "<expected value/text/url/count, or null if not applicable>",
      "description": "<human-readable summary of what this checks>"
    }
  ]
}

Rules:
- Never use a CSS selector, XPath, or ID in target_description — describe the
  element the way a human tester would ("the 'Submit Order' button in the
  checkout form").
- The first step should be a "navigate" action to the target URL unless a
  precondition already establishes the starting page.
- Every success_criteria item from the test intent must be covered by at
  least one assertion.
- Mark destructive_category honestly. Actions that delete data, charge
  payment, or submit an irreversible action (e.g. "Place Order", "Delete
  Account") must be flagged, even if the story's own goal is to test that
  exact action.
- Keep the plan as short as possible while still covering every
  precondition, step, and success criterion implied by the intent."""


class LLMClient(Protocol):
    """Same interface as `ingestion.story_parser.LLMClient` — see that module for context."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class TestPlanningError(Exception):
    """Raised when the LLM output cannot be parsed into a valid TestPlan shape."""


class ApprovalRequiredError(Exception):
    """
    Raised when a generated plan is structurally valid but contains
    destructive actions requiring human sign-off before it may execute.
    Callers should catch this, surface `plan` and `validation` to an
    approval workflow, and re-submit for execution once approved rather
    than treating this as a hard failure.
    """

    def __init__(self, plan: TestPlan, validation: ValidationResult) -> None:
        self.plan = plan
        self.validation = validation
        super().__init__(
            f"Test plan {plan.id} requires human approval before execution "
            f"({len(validation.warnings)} guardrail warning(s))."
        )


def _parse_llm_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TestPlanningError(f"LLM response was not valid JSON: {raw!r}") from exc


def _build_user_prompt(intent: TestIntent) -> str:
    precondition_lines = [f"- {p}" for p in intent.preconditions] or ["- (none stated)"]
    success_lines = [f"- {c}" for c in intent.success_criteria] or ["- (none stated)"]

    lines = [
        f"Persona: {intent.persona}",
        f"Goal: {intent.goal}",
        f"Rationale: {intent.rationale or 'N/A'}",
        f"Target URL: {intent.target_url or 'N/A'}",
        "",
        "Preconditions (from intent):",
        *precondition_lines,
        "",
        "Success criteria (from intent):",
        *success_lines,
    ]
    if intent.acceptance_criteria:
        lines += [
            "",
            "Acceptance criteria (Gherkin):",
            *(f"- {c.as_gherkin_line()}" for c in intent.acceptance_criteria),
        ]
    return "\n".join(lines)


def _build_plan_from_llm_output(intent: TestIntent, raw_plan: dict) -> TestPlan:
    preconditions: list[Precondition] = []
    for pc in raw_plan.get("preconditions", []):
        preconditions.append(
            Precondition(
                description=pc["description"],
                is_environmental=bool(pc.get("is_environmental", False)),
            )
        )

    steps: list[Step] = []
    for order, raw_step in enumerate(raw_plan.get("steps", [])):
        steps.append(
            Step(
                order=order,
                action=ActionType(raw_step["action"]),
                target_description=raw_step.get("target_description"),
                value=raw_step.get("value"),
                description=raw_step["description"],
                destructive_category=DestructiveCategory(
                    raw_step.get("destructive_category", "none")
                ),
            )
        )

    assertions: list[Assertion] = []
    for raw_assertion in raw_plan.get("assertions", []):
        step_index = raw_assertion["after_step_index"]
        if not (0 <= step_index < len(steps)):
            raise TestPlanningError(
                f"Assertion references after_step_index={step_index}, "
                f"but the plan only has {len(steps)} step(s)."
            )
        assertions.append(
            Assertion(
                after_step_id=steps[step_index].id,
                type=AssertionType(raw_assertion["type"]),
                target_description=raw_assertion.get("target_description"),
                expected=raw_assertion.get("expected"),
                description=raw_assertion["description"],
            )
        )

    return TestPlan(
        test_intent_id=intent.id,
        target_url=intent.target_url,
        preconditions=preconditions,
        steps=steps,
        assertions=assertions,
        metadata={"persona": intent.persona, "goal": intent.goal},
    )


class TestPlanner:
    """
    Converts a TestIntent into a validated TestPlan.

    Example:
        planner = TestPlanner(llm_client=my_llm_client)
        plan = await planner.plan(intent)
    """

    def __init__(self, llm_client: LLMClient, settings: Settings | None = None) -> None:
        self._llm_client = llm_client
        self._settings = settings or get_settings()

    async def plan(self, intent: TestIntent) -> TestPlan:
        """
        Generate and validate a TestPlan for the given intent.

        Raises:
            TestPlanningError: the LLM output could not be parsed into a
                well-formed plan at all.
            PlanValidationError: the plan parsed, but failed structural
                validation (dangling references, duplicate step orders, ...).
            ApprovalRequiredError: the plan is structurally valid but
                contains destructive actions requiring human sign-off.
        """
        raw = await self._llm_client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(intent),
        )
        raw_plan = _parse_llm_json(raw)

        try:
            plan = _build_plan_from_llm_output(intent, raw_plan)
        except (KeyError, ValueError) as exc:
            raise TestPlanningError(f"Malformed plan JSON from LLM: {exc}") from exc

        validation = validate_plan(plan, self._settings)
        if not validation.is_valid:
            logger.error(
                "Generated plan %s failed structural validation: %s",
                plan.id,
                [i.message for i in validation.errors],
            )
            raise PlanValidationError(validation)

        if validation.requires_human_approval:
            logger.warning(
                "Generated plan %s requires human approval before execution.", plan.id
            )
            raise ApprovalRequiredError(plan, validation)

        if validation.warnings:
            logger.warning(
                "Generated plan %s has non-blocking warnings: %s",
                plan.id,
                [i.message for i in validation.warnings],
            )

        logger.info(
            "Generated valid plan %s for intent %s (%d step(s), %d assertion(s)).",
            plan.id,
            intent.id,
            len(plan.steps),
            len(plan.assertions),
        )
        return plan