"""
Executes a single `Step` (from `planning/step_schema.py`) against a live
page: resolves its `target_description` via `ElementResolver`, then
performs the corresponding Playwright action.

Every target-bearing action is wrapped in `retry_policy.retry_async` with
a self-healing `on_retry` hook: if an action fails (stale element,
timing, a re-render since resolution), the hook invalidates that
description's cache entry, re-snapshots the live DOM, and re-resolves
before the next attempt — so a retry acts on a freshly located element
rather than blindly repeating a call against a selector that may no
longer point at the right thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.execution.retry_policy import RetryPolicy, retry_async
from qa_agent.perception.dom_snapshot import DomSnapshot, DomSnapshotExtractor, Page as SnapshotPage
from qa_agent.perception.element_resolver import (
    ElementResolutionError,
    ElementResolver,
    ResolvedElement,
    ScreenshotProvider,
)
from qa_agent.planning.step_schema import ActionType, Step

logger = get_logger(__name__)

# Playwright exceptions worth retrying: timeouts (element not ready yet /
# navigation still settling) and its generic "Error" class, which covers
# things like "element is not attached to the DOM" after a re-render.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    PlaywrightTimeoutError,
    PlaywrightError,
)

_ACTION_RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay_ms=300, max_delay_ms=3_000)


class Locator(Protocol):
    """The Playwright `Locator` surface `ActionExecutor` depends on."""

    async def click(self, **kwargs: object) -> None: ...
    async def fill(self, value: str, **kwargs: object) -> None: ...
    async def select_option(self, value: str, **kwargs: object) -> object: ...
    async def check(self, **kwargs: object) -> None: ...
    async def uncheck(self, **kwargs: object) -> None: ...
    async def hover(self, **kwargs: object) -> None: ...
    async def scroll_into_view_if_needed(self, **kwargs: object) -> None: ...
    async def press(self, key: str, **kwargs: object) -> None: ...
    async def set_input_files(self, path: str, **kwargs: object) -> None: ...
    async def wait_for(self, **kwargs: object) -> None: ...


class ExecutablePage(SnapshotPage, ScreenshotProvider, Protocol):
    """
    The full page surface this module needs: DOM-snapshotting (from
    `dom_snapshot.Page`), screenshotting (from `element_resolver.ScreenshotProvider`,
    needed transitively by the resolver's vision fallback), plus
    navigation and locator access.
    """

    async def goto(self, url: str, **kwargs: object) -> object: ...
    async def go_back(self, **kwargs: object) -> object: ...
    async def reload(self, **kwargs: object) -> object: ...

    def locator(self, selector: str) -> Locator: ...


@dataclass(frozen=True)
class ExecutionResult:
    step_id: str
    success: bool
    resolved_element: ResolvedElement | None
    error: str | None
    duration_ms: float


class ActionExecutionError(Exception):
    """Raised when a step fails after exhausting retries (and any self-healing attempts)."""


class ActionExecutor:
    """
    Example:
        executor = ActionExecutor(resolver=my_resolver)
        result = await executor.execute_step(page, step, snapshot)
    """

    def __init__(
        self,
        resolver: ElementResolver,
        snapshot_extractor: DomSnapshotExtractor | None = None,
        retry_policy: RetryPolicy = _ACTION_RETRY_POLICY,
        settings: Settings | None = None,
    ) -> None:
        self._resolver = resolver
        self._snapshot_extractor = snapshot_extractor or DomSnapshotExtractor()
        self._retry_policy = retry_policy
        self._settings = settings or get_settings()

    async def execute_step(
        self, page: ExecutablePage, step: Step, snapshot: DomSnapshot
    ) -> ExecutionResult:
        start = monotonic()
        try:
            resolved = await self._dispatch(page, step, snapshot)
            duration_ms = (monotonic() - start) * 1000
            logger.info(
                "Step %s (%s) succeeded in %.0fms.", step.id, step.action.value, duration_ms
            )
            return ExecutionResult(
                step_id=step.id,
                success=True,
                resolved_element=resolved,
                error=None,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 - convert to a domain result, not a bare crash
            duration_ms = (monotonic() - start) * 1000
            logger.error(
                "Step %s (%s) failed after %.0fms: %s", step.id, step.action.value, duration_ms, exc
            )
            return ExecutionResult(
                step_id=step.id,
                success=False,
                resolved_element=None,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def _dispatch(
        self, page: ExecutablePage, step: Step, snapshot: DomSnapshot
    ) -> ResolvedElement | None:
        if step.action == ActionType.NAVIGATE:
            await page.goto(step.value, timeout=self._settings.playwright_navigation_timeout_ms)
            return None
        if step.action == ActionType.GO_BACK:
            await page.go_back(timeout=self._settings.playwright_navigation_timeout_ms)
            return None
        if step.action == ActionType.RELOAD:
            await page.reload(timeout=self._settings.playwright_navigation_timeout_ms)
            return None

        # Every remaining action targets a specific element and goes
        # through the resolve-act-retry-with-self-healing path.
        assert step.target_description is not None  # enforced by Step's own validation
        return await self._act_on_target(page, step, snapshot)

    async def _act_on_target(
        self, page: ExecutablePage, step: Step, initial_snapshot: DomSnapshot
    ) -> ResolvedElement:
        # A mutable holder so the retry hook can swap in a freshly
        # captured snapshot for the next attempt, while the retried
        # closure below always reads the *current* value.
        snapshot_holder: list[DomSnapshot] = [initial_snapshot]
        last_resolved: list[ResolvedElement | None] = [None]

        async def attempt() -> ResolvedElement:
            resolved = await self._resolver.resolve(
                page, snapshot_holder[0], step.target_description  # type: ignore[arg-type]
            )
            last_resolved[0] = resolved
            locator = page.locator(resolved.selector)
            await self._perform(locator, step)
            return resolved

        async def on_retry(attempt_number: int, exc: Exception) -> None:
            logger.info(
                "Self-healing step %s after attempt %d failure (%s): "
                "invalidating cache and re-resolving against a fresh snapshot.",
                step.id,
                attempt_number,
                type(exc).__name__,
            )
            cache = self._resolver.cache
            await cache.invalidate(snapshot_holder[0].page_signature, step.target_description)
            snapshot_holder[0] = await self._snapshot_extractor.capture(page)

        try:
            return await retry_async(
                attempt,
                policy=self._retry_policy,
                retryable_exceptions=_RETRYABLE_EXCEPTIONS + (ElementResolutionError,),
                on_retry=on_retry,
            )
        except Exception as exc:
            raise ActionExecutionError(
                f"Step {step.id} ({step.action.value} on {step.target_description!r}) "
                f"failed: {exc}"
            ) from exc

    @staticmethod
    async def _perform(locator: Locator, step: Step) -> None:
        action = step.action
        timeout_kwargs = {"timeout": step.timeout_ms} if step.timeout_ms else {}

        if action == ActionType.CLICK:
            await locator.click(**timeout_kwargs)
        elif action == ActionType.FILL:
            await locator.fill(step.value or "", **timeout_kwargs)
        elif action == ActionType.SELECT_OPTION:
            await locator.select_option(step.value or "", **timeout_kwargs)
        elif action == ActionType.CHECK:
            await locator.check(**timeout_kwargs)
        elif action == ActionType.UNCHECK:
            await locator.uncheck(**timeout_kwargs)
        elif action == ActionType.HOVER:
            await locator.hover(**timeout_kwargs)
        elif action == ActionType.SCROLL_TO:
            await locator.scroll_into_view_if_needed(**timeout_kwargs)
        elif action == ActionType.PRESS_KEY:
            await locator.press(step.value or "", **timeout_kwargs)
        elif action == ActionType.UPLOAD_FILE:
            await locator.set_input_files(step.value or "", **timeout_kwargs)
        elif action == ActionType.WAIT_FOR:
            await locator.wait_for(**timeout_kwargs)
        else:
            raise ActionExecutionError(f"Unhandled target-bearing action: {action.value!r}")