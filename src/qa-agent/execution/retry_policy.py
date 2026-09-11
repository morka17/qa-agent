"""
Generic retry-with-backoff machinery, plus the specific "self-healing"
hook pattern `action_executor.py` uses: on a retryable failure, re-run a
caller-supplied recovery step (typically re-resolving the target element
against a fresh DOM snapshot) before the next attempt, rather than
blindly retrying the exact same action against a selector that may no
longer be valid.

This module has no Playwright dependency — it operates purely on
awaitables and exception types, so it's reusable anywhere in the codebase
that needs bounded retries (execution actions, LLM calls, connector
requests), not just browser actions.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

OnRetryHook = Callable[[int, Exception], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    """
    Exponential backoff with jitter.

    Attributes:
        max_attempts: Total attempts including the first (non-retry) try.
        base_delay_ms: Delay before the first retry.
        max_delay_ms: Ceiling on any single delay, regardless of backoff growth.
        backoff_multiplier: Growth factor applied to the delay after each attempt.
        jitter_ratio: Randomizes each delay by +/- this fraction, to avoid
            thundering-herd retries when many steps fail at once (e.g. a
            target app briefly returning 503s).
    """

    max_attempts: int = 3
    base_delay_ms: int = 250
    max_delay_ms: int = 4_000
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.2

    def delay_seconds(self, attempt: int) -> float:
        """`attempt` is 1-indexed: the delay taken *before* attempt number `attempt`."""
        raw_ms = self.base_delay_ms * (self.backoff_multiplier ** (attempt - 1))
        capped_ms = min(raw_ms, self.max_delay_ms)
        jitter = capped_ms * self.jitter_ratio
        jittered_ms = capped_ms + random.uniform(-jitter, jitter)
        return max(jittered_ms, 0) / 1000


# A conservative default suitable for flaky-network / transient-DOM-timing
# situations. Callers dealing with a known-unreliable target app can pass
# a more patient policy explicitly.
DEFAULT_POLICY = RetryPolicy()


class RetryExhaustedError(Exception):
    """
    Raised when every attempt failed. Wraps the *last* exception seen so
    callers get the most relevant failure, while `attempts` records how
    many tries were actually made for observability/reporting.
    """

    def __init__(self, attempts: int, last_exception: Exception) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempt(s); last error: "
            f"{type(last_exception).__name__}: {last_exception}"
        )


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: OnRetryHook | None = None,
) -> T:
    """
    Call `fn()` up to `policy.max_attempts` times, retrying only on
    exceptions matching `retryable_exceptions`. Any other exception
    propagates immediately without retrying.

    `on_retry(attempt, exception)` — if provided — runs after a failed
    attempt and before the backoff sleep for the *next* attempt. This is
    the self-healing seam: `action_executor.py` passes a hook here that
    re-resolves the target element (invalidating any stale cached
    selector) so the next attempt acts against a freshly located element
    rather than repeating the exact same doomed call.

    Raises:
        RetryExhaustedError: every attempt failed with a retryable exception.
        Exception: immediately, if `fn()` raises something not in `retryable_exceptions`.
    """
    last_exception: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except retryable_exceptions as exc:  # type: ignore[misc]
            last_exception = exc
            is_last_attempt = attempt == policy.max_attempts
            logger.warning(
                "Attempt %d/%d failed: %s: %s%s",
                attempt,
                policy.max_attempts,
                type(exc).__name__,
                exc,
                " (giving up)" if is_last_attempt else " (will retry)",
            )

            if is_last_attempt:
                break

            if on_retry is not None:
                await on_retry(attempt, exc)

            await asyncio.sleep(policy.delay_seconds(attempt))

    assert last_exception is not None  # loop always runs at least once
    raise RetryExhaustedError(attempts=policy.max_attempts, last_exception=last_exception)