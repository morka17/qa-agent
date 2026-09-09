"""
Static validation of a TestPlan before it is ever handed to the execution
layer.

This module catches two distinct classes of problem:

  1. **Structural integrity** — malformed step ordering, assertions or
     preconditions that reference steps which don't exist, duplicate step
     orders, plans that are suspiciously long, etc. These are bugs in the
     planner (or in the LLM's output) and should never reach a browser.

  2. **Safety / guardrails** — destructive actions (delete, payment,
     irreversible submit, ...) that aren't explicitly allowlisted require
     human approval before execution, per
     `settings.require_human_approval_for_destructive_actions`.

Validation never mutates the plan — it only reports. Callers decide
whether a plan with warnings-but-no-errors is safe to run, and whether an
`ApprovalRequired` result blocks automatic execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from qa_agent.config.settings import Settings, get_settings
from qa_agent.planning.step_schema import Step, TestPlan

# Sanity ceiling: a plan this long almost certainly indicates a planning
# failure (e.g. the LLM looping or over-decomposing a simple flow) rather
# than a genuinely long legitimate test.
_MAX_REASONABLE_STEPS = 60


class Severity(str, Enum):
    ERROR = "error"  # plan must not execute
    WARNING = "warning"  # plan may execute, but the issue should be surfaced


@dataclass
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    step_id: str | None = None


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)
    requires_human_approval: bool = False

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """A plan is executable if it has no ERROR-level issues (WARNINGs are non-blocking)."""
        return len(self.errors) == 0

    def add(self, severity: Severity, code: str, message: str, step_id: str | None = None) -> None:
        self.issues.append(ValidationIssue(severity=severity, code=code, message=message, step_id=step_id))


class PlanValidationError(Exception):
    """Raised by `validate_plan_or_raise` when a plan has blocking errors."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        summary = "; ".join(f"[{i.code}] {i.message}" for i in result.errors)
        super().__init__(f"Test plan failed validation: {summary}")


def _validate_step_ordering(plan: TestPlan, result: ValidationResult) -> None:
    orders = [step.order for step in plan.steps]

    if len(orders) != len(set(orders)):
        result.add(
            Severity.ERROR,
            "DUPLICATE_STEP_ORDER",
            f"Step 'order' values must be unique; found duplicates in {sorted(orders)}.",
        )

    expected = list(range(len(plan.steps)))
    if sorted(orders) != expected:
        result.add(
            Severity.ERROR,
            "NON_CONTIGUOUS_STEP_ORDER",
            f"Step orders must be a contiguous 0-indexed sequence; got {sorted(orders)}, "
            f"expected {expected}.",
        )


def _validate_reference_integrity(plan: TestPlan, result: ValidationResult) -> None:
    step_ids = {step.id for step in plan.steps}

    for assertion in plan.assertions:
        if assertion.after_step_id not in step_ids:
            result.add(
                Severity.ERROR,
                "DANGLING_ASSERTION_REFERENCE",
                f"Assertion {assertion.id} references after_step_id "
                f"{assertion.after_step_id!r}, which is not a step in this plan.",
                step_id=assertion.after_step_id,
            )

    for precondition in plan.preconditions:
        if precondition.setup_step_id and precondition.setup_step_id not in step_ids:
            result.add(
                Severity.ERROR,
                "DANGLING_PRECONDITION_REFERENCE",
                f"Precondition {precondition.id} references setup_step_id "
                f"{precondition.setup_step_id!r}, which is not a step in this plan.",
                step_id=precondition.setup_step_id,
            )

    for step in plan.steps:
        for precondition_id in step.depends_on_precondition_ids:
            if precondition_id not in {p.id for p in plan.preconditions}:
                result.add(
                    Severity.ERROR,
                    "DANGLING_STEP_PRECONDITION_REFERENCE",
                    f"Step {step.id} depends on precondition {precondition_id!r}, "
                    "which is not declared in this plan.",
                    step_id=step.id,
                )


def _validate_plan_size(plan: TestPlan, result: ValidationResult) -> None:
    if len(plan.steps) > _MAX_REASONABLE_STEPS:
        result.add(
            Severity.WARNING,
            "PLAN_UNUSUALLY_LONG",
            f"Plan has {len(plan.steps)} steps, exceeding the sanity threshold of "
            f"{_MAX_REASONABLE_STEPS}. Consider whether this indicates a planning failure.",
        )


def _validate_assertions_present(plan: TestPlan, result: ValidationResult) -> None:
    if not plan.assertions:
        result.add(
            Severity.WARNING,
            "NO_ASSERTIONS",
            "Plan has no assertions — the agent will execute the flow but cannot "
            "verify success or detect a bug.",
        )


def _enforce_destructive_action_guardrails(
    plan: TestPlan, settings: Settings, result: ValidationResult
) -> None:
    allowlist = set(settings.destructive_action_allowlist)

    destructive_steps: list[Step] = [s for s in plan.steps if s.is_destructive]
    for step in destructive_steps:
        category = step.destructive_category.value
        if category in allowlist:
            continue

        if settings.require_human_approval_for_destructive_actions:
            result.requires_human_approval = True
            result.add(
                Severity.WARNING,
                "DESTRUCTIVE_ACTION_NEEDS_APPROVAL",
                f"Step {step.id} ({step.description!r}) is categorized as "
                f"{category!r} and is not in the destructive_action_allowlist. "
                "Human approval is required before this plan can execute.",
                step_id=step.id,
            )
        else:
            # Approval is not required by configuration, but this is still
            # worth a loud warning — silent destructive execution is the
            # single worst failure mode for an autonomous agent.
            result.add(
                Severity.WARNING,
                "DESTRUCTIVE_ACTION_UNGATED",
                f"Step {step.id} ({step.description!r}) performs a "
                f"{category!r} action without human-approval gating enabled. "
                "Verify this is intentional.",
                step_id=step.id,
            )


def validate_plan(plan: TestPlan, settings: Settings | None = None) -> ValidationResult:
    """
    Run every structural and safety check against `plan` and return the
    aggregated result. Never raises — inspect `result.is_valid` and
    `result.requires_human_approval` to decide how to proceed.
    """
    settings = settings or get_settings()
    result = ValidationResult()

    _validate_step_ordering(plan, result)
    _validate_reference_integrity(plan, result)
    _validate_plan_size(plan, result)
    _validate_assertions_present(plan, result)
    _enforce_destructive_action_guardrails(plan, settings, result)

    return result


def validate_plan_or_raise(plan: TestPlan, settings: Settings | None = None) -> ValidationResult:
    """
    Convenience wrapper for call sites (e.g. `test_planner.py`) that want
    a hard failure on structural errors rather than inspecting the result
    themselves. Human-approval-required plans do NOT raise — that's a
    valid, non-error outcome the orchestration layer must handle
    explicitly (route to an approval queue), not an exception.
    """
    result = validate_plan(plan, settings)
    if not result.is_valid:
        raise PlanValidationError(result)
    return result