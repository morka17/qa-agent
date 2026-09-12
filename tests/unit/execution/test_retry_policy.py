"""Unit tests for qa_agent.execution.retry_policy."""

import pytest

from qa_agent.execution.retry_policy import RetryExhaustedError, RetryPolicy, retry_async


@pytest.mark.asyncio
class TestRetryAsync:
    async def test_succeeds_on_first_attempt_without_retry(self):
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            return "ok"

        result = await retry_async(fn, policy=RetryPolicy(max_attempts=3))
        assert result == "ok"
        assert calls["n"] == 1

    async def test_succeeds_after_transient_failures(self):
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return "ok"

        result = await retry_async(
            fn,
            policy=RetryPolicy(max_attempts=5, base_delay_ms=1, max_delay_ms=2),
            retryable_exceptions=(ValueError,),
        )
        assert result == "ok"
        assert calls["n"] == 3

    async def test_raises_retry_exhausted_after_max_attempts(self):
        async def always_fails():
            raise RuntimeError("nope")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await retry_async(
                always_fails,
                policy=RetryPolicy(max_attempts=3, base_delay_ms=1, max_delay_ms=2),
                retryable_exceptions=(RuntimeError,),
            )
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, RuntimeError)

    async def test_non_retryable_exception_propagates_immediately(self):
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            raise KeyError("boom")

        with pytest.raises(KeyError):
            await retry_async(fn, policy=RetryPolicy(max_attempts=5), retryable_exceptions=(ValueError,))
        assert calls["n"] == 1  # never retried

    async def test_on_retry_hook_called_between_attempts(self):
        retries_seen = []

        async def fn():
            if len(retries_seen) < 2:
                raise ValueError("transient")
            return "ok"

        async def on_retry(attempt, exc):
            retries_seen.append(attempt)

        await retry_async(
            fn,
            policy=RetryPolicy(max_attempts=5, base_delay_ms=1, max_delay_ms=2),
            retryable_exceptions=(ValueError,),
            on_retry=on_retry,
        )
        assert retries_seen == [1, 2]


class TestRetryPolicyDelay:
    def test_delay_seconds_respects_max_delay_ceiling(self):
        policy = RetryPolicy(base_delay_ms=1000, max_delay_ms=2000, backoff_multiplier=10, jitter_ratio=0)
        # Without a ceiling, attempt 3 would be 1000 * 10^2 = 100,000ms
        assert policy.delay_seconds(3) <= 2.0

    def test_delay_seconds_never_negative(self):
        policy = RetryPolicy(base_delay_ms=100, jitter_ratio=2.0)  # extreme jitter
        for attempt in range(1, 5):
            assert policy.delay_seconds(attempt) >= 0
