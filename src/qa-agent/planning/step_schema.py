"""
Data contracts for the planning layer's output: an ordered, executable
step graph derived from a TestIntent.

A TestPlan is deliberately *not* tied to any DOM selector — steps
describe actions in terms a human tester would use ("click the 'Add to
Cart' button"), and it is the perception layer's job (element_resolver.py)
to resolve that description to an actual element at execution time. This
separation is what lets plans survive UI refactors: the plan says *what*
to do, perception figures out *how* on the current DOM.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActionType(str, Enum):
    """Every action the execution layer's ActionExecutor knows how to perform."""

    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT_OPTION = "select_option"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"
    SCROLL_TO = "scroll_to"
    PRESS_KEY = "press_key"
    UPLOAD_FILE = "upload_file"
    WAIT_FOR = "wait_for"
    GO_BACK = "go_back"
    RELOAD = "reload"

    @property
    def requires_value(self) -> bool:
        return self in {
            ActionType.NAVIGATE,
            ActionType.FILL,
            ActionType.SELECT_OPTION,
            ActionType.PRESS_KEY,
            ActionType.UPLOAD_FILE,
        }

    @property
    def requires_target(self) -> bool:
        """Actions that operate on a specific element, as opposed to the page as a whole."""
        return self not in {ActionType.NAVIGATE, ActionType.GO_BACK, ActionType.RELOAD}


class DestructiveCategory(str, Enum):
    """
    Classes of action the guardrails layer treats as irreversible or
    high-consequence. A step tagged with one of these must match the
    `destructive_action_allowlist` in settings or be routed to human
    approval before execution — see `plan_validator.py`.
    """

    DELETE = "delete"
    PAYMENT = "payment"
    IRREVERSIBLE_SUBMIT = "irreversible_submit"
    ACCOUNT_MODIFICATION = "account_modification"
    DATA_EXPORT = "data_export"
    NONE = "none"


class Precondition(BaseModel):
    """
    State that must hold before the plan begins executing. Some
    preconditions are satisfied by explicit setup steps prepended to the
    plan (`setup_step_id` set); others describe environment assumptions
    the runner must verify or arrange out-of-band (e.g. "a seeded test
    account exists") and are left unset.
    """

    id: str = Field(default_factory=_new_id)
    description: str = Field(..., min_length=1)
    setup_step_id: str | None = Field(
        default=None,
        description="ID of the Step in this plan that establishes this precondition, if any.",
    )
    is_environmental: bool = Field(
        default=False,
        description="True if this must be true of the test environment itself "
        "(e.g. seeded data) rather than something the agent can set up by acting in-browser.",
    )


class Step(BaseModel):
    """
    A single executable action within a test plan.

    `target_description` is a natural-language description of the
    element to act on (e.g. "the 'Add to Cart' button in the product
    summary card") — never a CSS/XPath selector. The perception layer
    resolves it against the live DOM at execution time.
    """

    id: str = Field(default_factory=_new_id)
    order: int = Field(..., ge=0, description="0-indexed execution position within the plan.")
    action: ActionType
    target_description: str | None = Field(
        default=None,
        description="NL description of the element to act on. Required unless "
        "the action operates on the page as a whole (navigate/go_back/reload).",
    )
    value: str | None = Field(
        default=None,
        description="Input value for actions that need one (fill text, URL to "
        "navigate to, option to select, key to press, file path to upload).",
    )
    description: str = Field(
        ..., min_length=1, description="Human-readable summary of this step, for reports/logs."
    )
    destructive_category: DestructiveCategory = DestructiveCategory.NONE
    timeout_ms: int | None = Field(
        default=None, description="Override for this step's action timeout, if the default is insufficient."
    )
    depends_on_precondition_ids: list[str] = Field(default_factory=list)

    @property
    def is_destructive(self) -> bool:
        return self.destructive_category != DestructiveCategory.NONE

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "Step":
        if self.action.requires_target and not self.target_description:
            raise ValueError(
                f"action {self.action.value!r} requires a target_description."
            )
        if self.action.requires_value and not self.value:
            raise ValueError(f"action {self.action.value!r} requires a value.")
        return self


class AssertionType(str, Enum):
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_NOT_VISIBLE = "element_not_visible"
    TEXT_EQUALS = "text_equals"
    TEXT_CONTAINS = "text_contains"
    URL_EQUALS = "url_equals"
    URL_CONTAINS = "url_contains"
    ELEMENT_COUNT = "element_count"
    NO_CONSOLE_ERRORS = "no_console_errors"
    CUSTOM = "custom"  # falls back to LLM-judged evaluation against `expected`

    @property
    def requires_target(self) -> bool:
        return self in {
            AssertionType.ELEMENT_VISIBLE,
            AssertionType.ELEMENT_NOT_VISIBLE,
            AssertionType.TEXT_EQUALS,
            AssertionType.TEXT_CONTAINS,
            AssertionType.ELEMENT_COUNT,
        }

    @property
    def requires_expected(self) -> bool:
        return self in {
            AssertionType.TEXT_EQUALS,
            AssertionType.TEXT_CONTAINS,
            AssertionType.URL_EQUALS,
            AssertionType.URL_CONTAINS,
            AssertionType.ELEMENT_COUNT,
            AssertionType.CUSTOM,
        }


class Assertion(BaseModel):
    """
    A single checkable condition evaluated after a specific step executes.
    This is the plan's oracle: what "correct" looks like, checked by
    `verification/assertion_engine.py` at runtime.
    """

    id: str = Field(default_factory=_new_id)
    after_step_id: str = Field(
        ..., description="ID of the Step after which this assertion is evaluated."
    )
    type: AssertionType
    target_description: str | None = Field(
        default=None, description="NL description of the element the assertion checks, if applicable."
    )
    expected: str | None = Field(
        default=None, description="Expected value/text/count/URL, if applicable to the assertion type."
    )
    description: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_assertion_requirements(self) -> "Assertion":
        if self.type.requires_target and not self.target_description:
            raise ValueError(f"assertion type {self.type.value!r} requires a target_description.")
        if self.type.requires_expected and not self.expected:
            raise ValueError(f"assertion type {self.type.value!r} requires an expected value.")
        return self


class TestPlan(BaseModel):
    """
    The full, ordered, executable representation of a TestIntent: setup
    preconditions, a sequence of steps, and the assertions that determine
    pass/fail. This is what `orchestration/agent_loop.py` consumes.
    """

    id: str = Field(default_factory=_new_id)
    test_intent_id: str = Field(..., description="Foreign key back to the originating TestIntent.id")
    target_url: str | None = None
    preconditions: list[Precondition] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("steps")
    @classmethod
    def _steps_not_empty(cls, v: list[Step]) -> list[Step]:
        if not v:
            raise ValueError("a TestPlan must contain at least one step.")
        return v

    def steps_in_order(self) -> list[Step]:
        return sorted(self.steps, key=lambda s: s.order)

    def step_by_id(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def assertions_after(self, step_id: str) -> list[Assertion]:
        return [a for a in self.assertions if a.after_step_id == step_id]

    @property
    def has_destructive_steps(self) -> bool:
        return any(step.is_destructive for step in self.steps)