"""
Evaluates a plan's `Assertion` objects (from `planning/step_schema.py`)
against the live page after a step executes — the oracle that determines
pass/fail, and therefore what `triage/failure_classifier.py` has to work
with when something doesn't match.

Element-bearing assertion types reuse `ElementResolver` (the same
semantic resolution used for actions) so an assertion's
`target_description` is exactly as selector-independent as a step's —
"the cart total" is resolved the same way whether it's being clicked or
being checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.perception.dom_snapshot import DomSnapshot
from qa_agent.perception.element_resolver import ElementResolutionError, ElementResolver
from qa_agent.planning.step_schema import Assertion, AssertionType

logger = get_logger(__name__)


class Locator(Protocol):
    async def text_content(self) -> str | None: ...
    async def is_visible(self) -> bool: ...
    async def count(self) -> int: ...


class VerifiablePage(Protocol):
    @property
    def url(self) -> str: ...
    def locator(self, selector: str) -> Locator: ...


class CustomAssertionJudge(Protocol):
    """
    Hook for `AssertionType.CUSTOM`: assertions whose pass/fail condition
    can't be expressed as a simple structural check and instead needs an
    LLM to judge (e.g. "the confirmation message is friendly and
    mentions the order number"). Kept optional — `AssertionEngine` works
    fully for every other assertion type without one.
    """

    async def judge(self, description: str, expected: str | None, observed_context: str) -> bool:
        ...


@dataclass(frozen=True)
class AssertionResult:
    assertion_id: str
    passed: bool
    actual: str | None
    expected: str | None
    reason: str


class AssertionEvaluationError(Exception):
    """
    Raised when an assertion cannot be *evaluated* at all (e.g. its
    target can't be resolved on the page) — distinct from the assertion
    resolving fine and simply failing (`AssertionResult.passed = False`),
    which is a normal, expected outcome, not an error.
    """


class AssertionEngine:
    """
    Example:
        engine = AssertionEngine(resolver=my_resolver)
        result = await engine.evaluate(page, snapshot, assertion)
    """

    def __init__(
        self,
        resolver: ElementResolver,
        custom_judge: CustomAssertionJudge | None = None,
    ) -> None:
        self._resolver = resolver
        self._custom_judge = custom_judge

    async def evaluate_all(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertions: list[Assertion]
    ) -> list[AssertionResult]:
        results = []
        for assertion in assertions:
            results.append(await self.evaluate(page, snapshot, assertion))
        return results

    async def evaluate(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        try:
            handler = self._HANDLERS[assertion.type]
            return await handler(self, page, snapshot, assertion)
        except ElementResolutionError as exc:
            # A resolution failure for an assertion's target is itself
            # meaningful signal (the expected element genuinely isn't
            # there), so it's reported as a failed assertion, not raised
            # as a system error — that's what lets triage distinguish
            # "the button never appeared" (a real bug) from "the plan
            # itself is broken" (a validation-layer concern).
            return AssertionResult(
                assertion_id=assertion.id,
                passed=False,
                actual=None,
                expected=assertion.expected,
                reason=f"target could not be resolved: {exc}",
            )

    async def _eval_element_visible(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        resolved = await self._resolver.resolve(page, snapshot, assertion.target_description)  # type: ignore[arg-type]
        visible = await page.locator(resolved.selector).is_visible()
        return AssertionResult(
            assertion_id=assertion.id,
            passed=visible,
            actual=str(visible),
            expected="True",
            reason="element is visible" if visible else "element resolved but is not visible",
        )

    async def _eval_element_not_visible(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        try:
            resolved = await self._resolver.resolve(page, snapshot, assertion.target_description)  # type: ignore[arg-type]
        except ElementResolutionError:
            # Element genuinely not present at all satisfies "not visible".
            return AssertionResult(
                assertion_id=assertion.id,
                passed=True,
                actual="not present",
                expected="False",
                reason="element does not exist in the current DOM",
            )
        visible = await page.locator(resolved.selector).is_visible()
        return AssertionResult(
            assertion_id=assertion.id,
            passed=not visible,
            actual=str(visible),
            expected="False",
            reason="element exists but is hidden" if not visible else "element is visible",
        )

    async def _eval_text_equals(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        resolved = await self._resolver.resolve(page, snapshot, assertion.target_description)  # type: ignore[arg-type]
        actual = (await page.locator(resolved.selector).text_content() or "").strip()
        expected = (assertion.expected or "").strip()
        passed = actual == expected
        return AssertionResult(
            assertion_id=assertion.id,
            passed=passed,
            actual=actual,
            expected=expected,
            reason="exact match" if passed else "text did not match exactly",
        )

    async def _eval_text_contains(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        resolved = await self._resolver.resolve(page, snapshot, assertion.target_description)  # type: ignore[arg-type]
        actual = (await page.locator(resolved.selector).text_content() or "").strip()
        expected = (assertion.expected or "").strip()
        passed = expected.lower() in actual.lower()
        return AssertionResult(
            assertion_id=assertion.id,
            passed=passed,
            actual=actual,
            expected=expected,
            reason="substring found" if passed else "substring not found in element text",
        )

    async def _eval_url_equals(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        expected = (assertion.expected or "").strip()
        actual = page.url
        passed = actual.rstrip("/") == expected.rstrip("/")
        return AssertionResult(
            assertion_id=assertion.id,
            passed=passed,
            actual=actual,
            expected=expected,
            reason="URL matches exactly" if passed else "URL does not match",
        )

    async def _eval_url_contains(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        expected = (assertion.expected or "").strip()
        actual = page.url
        passed = expected in actual
        return AssertionResult(
            assertion_id=assertion.id,
            passed=passed,
            actual=actual,
            expected=expected,
            reason="substring found in URL" if passed else "substring not found in URL",
        )

    async def _eval_element_count(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        resolved = await self._resolver.resolve(page, snapshot, assertion.target_description)  # type: ignore[arg-type]
        # `resolved.selector` identifies one specific element; for a
        # count assertion we query all elements sharing its dom_path's
        # tag-level pattern is out of scope here, so count is evaluated
        # against the resolved selector directly (typically a container).
        actual_count = await page.locator(resolved.selector).count()
        try:
            expected_count = int(assertion.expected)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise AssertionEvaluationError(
                f"Assertion {assertion.id} has non-integer expected count: {assertion.expected!r}"
            )
        passed = actual_count == expected_count
        return AssertionResult(
            assertion_id=assertion.id,
            passed=passed,
            actual=str(actual_count),
            expected=str(expected_count),
            reason="count matches" if passed else "count does not match",
        )

    async def _eval_no_console_errors(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        # This assertion type's actual data comes from
        # `console_error_monitor.py`, not the page itself — the engine
        # can't evaluate it in isolation. Callers (the orchestration
        # loop) should route NO_CONSOLE_ERRORS assertions through
        # `ConsoleErrorMonitor.to_assertion_result()` instead of calling
        # `evaluate()` for them directly.
        raise AssertionEvaluationError(
            f"Assertion {assertion.id} is type NO_CONSOLE_ERRORS, which requires "
            "a ConsoleErrorMonitor and cannot be evaluated by AssertionEngine alone."
        )

    async def _eval_custom(
        self, page: VerifiablePage, snapshot: DomSnapshot, assertion: Assertion
    ) -> AssertionResult:
        if self._custom_judge is None:
            raise AssertionEvaluationError(
                f"Assertion {assertion.id} is type CUSTOM but no CustomAssertionJudge "
                "was configured on this AssertionEngine."
            )
        context = f"Current URL: {page.url}\nPage title: {snapshot.title}"
        passed = await self._custom_judge.judge(assertion.description, assertion.expected, context)
        return AssertionResult(
            assertion_id=assertion.id,
            passed=passed,
            actual="(LLM-judged)",
            expected=assertion.expected,
            reason="custom judge approved" if passed else "custom judge rejected",
        )

    _HANDLERS: dict = {}  # populated below the class body


AssertionEngine._HANDLERS = {
    AssertionType.ELEMENT_VISIBLE: AssertionEngine._eval_element_visible,
    AssertionType.ELEMENT_NOT_VISIBLE: AssertionEngine._eval_element_not_visible,
    AssertionType.TEXT_EQUALS: AssertionEngine._eval_text_equals,
    AssertionType.TEXT_CONTAINS: AssertionEngine._eval_text_contains,
    AssertionType.URL_EQUALS: AssertionEngine._eval_url_equals,
    AssertionType.URL_CONTAINS: AssertionEngine._eval_url_contains,
    AssertionType.ELEMENT_COUNT: AssertionEngine._eval_element_count,
    AssertionType.NO_CONSOLE_ERRORS: AssertionEngine._eval_no_console_errors,
    AssertionType.CUSTOM: AssertionEngine._eval_custom,
}