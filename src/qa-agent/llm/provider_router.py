"""
The single point every other module's `LLMClient` Protocol is actually
satisfied by: routes `complete()` calls to whichever backend
(Anthropic/OpenAI/local) is configured, applies PII redaction to outgoing
prompts, records cost via `cost_tracker.py`, and retries transient
failures via the same `execution/retry_policy.py` used for browser
actions.

Provider SDKs are imported lazily inside each backend's `__init__`, so
importing this module — or `qa_agent` generally — never requires every
possible LLM SDK to be installed, only whichever one `settings.llm_provider`
actually selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import LLMProvider, Settings, get_settings
from qa_agent.execution.retry_policy import RetryPolicy, retry_async
from qa_agent.llm.cost_tracker import CostTracker
from qa_agent.llm.guardrails import redact_pii

logger = get_logger(__name__)

_LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay_ms=500, max_delay_ms=8_000)


@dataclass(frozen=True)
class CompletionUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    usage: CompletionUsage


class ProviderBackendError(Exception):
    """Raised for non-retryable backend failures (bad request, auth, unsupported operation)."""


class ProviderBackend(Protocol):
    """
    What `LLMRouter` needs from a concrete provider integration.
    `complete_with_usage` (not just `complete`) so `LLMRouter` can feed
    real token counts to `cost_tracker.py` instead of estimating them.
    """

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion: ...

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        """Vision-grounding support. Backends without vision support raise ProviderBackendError."""
        ...


class AnthropicBackend:
    """Wraps the Anthropic Messages API. Lazily imports `anthropic` so it's only required when actually selected."""

    def __init__(self, settings: Settings) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ProviderBackendError(
                "AnthropicBackend requires the 'anthropic' package to be installed."
            ) from exc
        self._settings = settings
        self._client = AsyncAnthropic(
            api_key=settings.llm_api_key.get_secret_value(), timeout=settings.llm_request_timeout_s
        )

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion:
        response = await self._client.messages.create(
            model=self._settings.llm_model,
            max_tokens=self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return Completion(
            text=text,
            usage=CompletionUsage(
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
            ),
        )

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        import base64
        import json as _json

        prompt = (
            f"The image has {mark_count} numbered marks (0-{mark_count - 1}) drawn around "
            f"candidate elements. Which mark best matches: {description!r}? "
            'Return ONLY JSON: {"mark_index": <int or null if none match>}'
        )
        response = await self._client.messages.create(
            model=self._settings.llm_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        try:
            parsed = _json.loads(text.strip().strip("`"))
            return parsed.get("mark_index")
        except (ValueError, AttributeError) as exc:
            raise ProviderBackendError(f"Could not parse vision response: {text!r}") from exc


class OpenAIBackend:
    """Wraps the OpenAI Chat Completions API. Lazily imports `openai`."""

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderBackendError(
                "OpenAIBackend requires the 'openai' package to be installed."
            ) from exc
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(), timeout=settings.llm_request_timeout_s
        )

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion:
        response = await self._client.chat.completions.create(
            model=self._settings.llm_model,
            max_tokens=self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return Completion(
            text=response.choices[0].message.content or "",
            usage=CompletionUsage(
                input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens
            ),
        )

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        raise ProviderBackendError(
            "OpenAIBackend does not implement vision-grounded element location in this "
            "build. Configure an Anthropic-backed vision client for visual_grounding.py, "
            "or extend this method with an OpenAI vision-capable model call."
        )


class LocalBackend:
    """
    Wraps an OpenAI-compatible local inference server (e.g. Ollama, vLLM,
    LM Studio) via plain HTTP, so no vendor SDK is required at all for
    fully local/offline operation.
    """

    def __init__(self, settings: Settings, base_url: str = "http://localhost:11434/v1") -> None:
        self._settings = settings
        self._base_url = base_url

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion:
        import httpx

        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._settings.llm_request_timeout_s) as client:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self._settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self._settings.llm_temperature,
                },
            )
            if response.status_code != 200:
                raise ProviderBackendError(
                    f"Local LLM server returned {response.status_code}: {response.text[:300]}"
                )
            data = response.json()

        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            usage=CompletionUsage(
                input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0)
            ),
        )

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        raise ProviderBackendError("LocalBackend does not support vision-grounded element location.")


def _build_backend(settings: Settings) -> ProviderBackend:
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicBackend(settings)
    if settings.llm_provider == LLMProvider.OPENAI:
        return OpenAIBackend(settings)
    return LocalBackend(settings)


class LLMRouter:
    """
    Satisfies every `LLMClient` Protocol used across the codebase
    (`ingestion.story_parser.LLMClient`, `planning.test_planner.LLMClient`,
    `triage.failure_classifier.LLMClient`, etc. — they're all structurally
    identical `async def complete(system_prompt, user_prompt) -> str`) as
    well as `perception.selector_strategies.visual_grounding.VisionLLMClient`
    (via `locate_marked_element`).

    Example:
        router = LLMRouter(cost_tracker=my_cost_tracker)
        title_json = await router.complete(system_prompt, user_prompt, purpose="bug_title", run_id=run_id)
    """

    def __init__(
        self,
        backend: ProviderBackend | None = None,
        settings: Settings | None = None,
        cost_tracker: CostTracker | None = None,
        redact_outgoing_pii: bool = True,
        retry_policy: RetryPolicy = _LLM_RETRY_POLICY,
    ) -> None:
        self._settings = settings or get_settings()
        self._backend = backend or _build_backend(self._settings)
        self._cost_tracker = cost_tracker
        self._redact_pii = redact_outgoing_pii
        self._retry_policy = retry_policy

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "unspecified",
        run_id: str | None = None,
    ) -> str:
        if self._redact_pii:
            redacted = redact_pii(user_prompt)
            user_prompt = redacted.redacted_text

        async def attempt() -> Completion:
            return await self._backend.complete_with_usage(system_prompt, user_prompt)

        completion = await retry_async(
            attempt,
            policy=self._retry_policy,
            retryable_exceptions=(ProviderBackendError, ConnectionError, TimeoutError),
        )

        if self._cost_tracker is not None:
            self._cost_tracker.record(
                provider=self._settings.llm_provider.value,
                model=self._settings.llm_model,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                purpose=purpose,
                run_id=run_id,
            )

        return completion.text

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        return await self._backend.locate_marked_element(image_bytes, description, mark_count)"""
The single point every other module's `LLMClient` Protocol is actually
satisfied by: routes `complete()` calls to whichever backend
(Anthropic/OpenAI/local) is configured, applies PII redaction to outgoing
prompts, records cost via `cost_tracker.py`, and retries transient
failures via the same `execution/retry_policy.py` used for browser
actions.

Provider SDKs are imported lazily inside each backend's `__init__`, so
importing this module — or `qa_agent` generally — never requires every
possible LLM SDK to be installed, only whichever one `settings.llm_provider`
actually selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import LLMProvider, Settings, get_settings
from qa_agent.execution.retry_policy import RetryPolicy, retry_async
from qa_agent.llm.cost_tracker import CostTracker
from qa_agent.llm.guardrails import redact_pii

logger = get_logger(__name__)

_LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay_ms=500, max_delay_ms=8_000)


@dataclass(frozen=True)
class CompletionUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    usage: CompletionUsage


class ProviderBackendError(Exception):
    """Raised for non-retryable backend failures (bad request, auth, unsupported operation)."""


class ProviderBackend(Protocol):
    """
    What `LLMRouter` needs from a concrete provider integration.
    `complete_with_usage` (not just `complete`) so `LLMRouter` can feed
    real token counts to `cost_tracker.py` instead of estimating them.
    """

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion: ...

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        """Vision-grounding support. Backends without vision support raise ProviderBackendError."""
        ...


class AnthropicBackend:
    """Wraps the Anthropic Messages API. Lazily imports `anthropic` so it's only required when actually selected."""

    def __init__(self, settings: Settings) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ProviderBackendError(
                "AnthropicBackend requires the 'anthropic' package to be installed."
            ) from exc
        self._settings = settings
        self._client = AsyncAnthropic(
            api_key=settings.llm_api_key.get_secret_value(), timeout=settings.llm_request_timeout_s
        )

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion:
        response = await self._client.messages.create(
            model=self._settings.llm_model,
            max_tokens=self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return Completion(
            text=text,
            usage=CompletionUsage(
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
            ),
        )

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        import base64
        import json as _json

        prompt = (
            f"The image has {mark_count} numbered marks (0-{mark_count - 1}) drawn around "
            f"candidate elements. Which mark best matches: {description!r}? "
            'Return ONLY JSON: {"mark_index": <int or null if none match>}'
        )
        response = await self._client.messages.create(
            model=self._settings.llm_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        try:
            parsed = _json.loads(text.strip().strip("`"))
            return parsed.get("mark_index")
        except (ValueError, AttributeError) as exc:
            raise ProviderBackendError(f"Could not parse vision response: {text!r}") from exc


class OpenAIBackend:
    """Wraps the OpenAI Chat Completions API. Lazily imports `openai`."""

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderBackendError(
                "OpenAIBackend requires the 'openai' package to be installed."
            ) from exc
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(), timeout=settings.llm_request_timeout_s
        )

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion:
        response = await self._client.chat.completions.create(
            model=self._settings.llm_model,
            max_tokens=self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return Completion(
            text=response.choices[0].message.content or "",
            usage=CompletionUsage(
                input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens
            ),
        )

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        raise ProviderBackendError(
            "OpenAIBackend does not implement vision-grounded element location in this "
            "build. Configure an Anthropic-backed vision client for visual_grounding.py, "
            "or extend this method with an OpenAI vision-capable model call."
        )


class LocalBackend:
    """
    Wraps an OpenAI-compatible local inference server (e.g. Ollama, vLLM,
    LM Studio) via plain HTTP, so no vendor SDK is required at all for
    fully local/offline operation.
    """

    def __init__(self, settings: Settings, base_url: str = "http://localhost:11434/v1") -> None:
        self._settings = settings
        self._base_url = base_url

    async def complete_with_usage(self, system_prompt: str, user_prompt: str) -> Completion:
        import httpx

        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._settings.llm_request_timeout_s) as client:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self._settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self._settings.llm_temperature,
                },
            )
            if response.status_code != 200:
                raise ProviderBackendError(
                    f"Local LLM server returned {response.status_code}: {response.text[:300]}"
                )
            data = response.json()

        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            usage=CompletionUsage(
                input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0)
            ),
        )

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        raise ProviderBackendError("LocalBackend does not support vision-grounded element location.")


def _build_backend(settings: Settings) -> ProviderBackend:
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicBackend(settings)
    if settings.llm_provider == LLMProvider.OPENAI:
        return OpenAIBackend(settings)
    return LocalBackend(settings)


class LLMRouter:
    """
    Satisfies every `LLMClient` Protocol used across the codebase
    (`ingestion.story_parser.LLMClient`, `planning.test_planner.LLMClient`,
    `triage.failure_classifier.LLMClient`, etc. — they're all structurally
    identical `async def complete(system_prompt, user_prompt) -> str`) as
    well as `perception.selector_strategies.visual_grounding.VisionLLMClient`
    (via `locate_marked_element`).

    Example:
        router = LLMRouter(cost_tracker=my_cost_tracker)
        title_json = await router.complete(system_prompt, user_prompt, purpose="bug_title", run_id=run_id)
    """

    def __init__(
        self,
        backend: ProviderBackend | None = None,
        settings: Settings | None = None,
        cost_tracker: CostTracker | None = None,
        redact_outgoing_pii: bool = True,
        retry_policy: RetryPolicy = _LLM_RETRY_POLICY,
    ) -> None:
        self._settings = settings or get_settings()
        self._backend = backend or _build_backend(self._settings)
        self._cost_tracker = cost_tracker
        self._redact_pii = redact_outgoing_pii
        self._retry_policy = retry_policy

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "unspecified",
        run_id: str | None = None,
    ) -> str:
        if self._redact_pii:
            redacted = redact_pii(user_prompt)
            user_prompt = redacted.redacted_text

        async def attempt() -> Completion:
            return await self._backend.complete_with_usage(system_prompt, user_prompt)

        completion = await retry_async(
            attempt,
            policy=self._retry_policy,
            retryable_exceptions=(ProviderBackendError, ConnectionError, TimeoutError),
        )

        if self._cost_tracker is not None:
            self._cost_tracker.record(
                provider=self._settings.llm_provider.value,
                model=self._settings.llm_model,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                purpose=purpose,
                run_id=run_id,
            )

        return completion.text

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        return await self._backend.locate_marked_element(image_bytes, description, mark_count)