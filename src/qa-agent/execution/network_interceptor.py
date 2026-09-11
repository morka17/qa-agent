"""
Intercepts network traffic during a test run: records every request/
response matching configured patterns (for evidence bundling and
verification's assertion oracle), and optionally mocks responses so a
test can exercise a flow without depending on a real backend or
third-party service being reachable/stable.

Wraps Playwright's `page.route()`, but is expressed against a local
`RoutablePage`/`Route`/`Request` Protocol so the recording/matching logic
is unit-testable without a real browser — the same pattern used
throughout `execution/` and `perception/`.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)


class Request(Protocol):
    @property
    def url(self) -> str: ...
    @property
    def method(self) -> str: ...
    def post_data(self) -> str | None: ...
    @property
    def headers(self) -> dict[str, str]: ...


class Response(Protocol):
    @property
    def status(self) -> int: ...
    async def text(self) -> str: ...
    @property
    def headers(self) -> dict[str, str]: ...


class Route(Protocol):
    @property
    def request(self) -> Request: ...
    async def fulfill(self, **kwargs: object) -> None: ...
    async def continue_(self, **kwargs: object) -> None: ...
    async def abort(self, error_code: str = "failed") -> None: ...


class RoutablePage(Protocol):
    async def route(self, url_pattern: str, handler: Callable[[Route], Awaitable[None]]) -> None: ...
    async def unroute(self, url_pattern: str) -> None: ...
    def on(self, event: str, handler: Callable[..., None]) -> None: ...


@dataclass(frozen=True)
class MockRule:
    """
    A request-matching rule: any request whose URL matches `url_pattern`
    (a glob, e.g. `**/api/checkout`) and, if given, whose method matches
    `method`, is fulfilled with the configured mock response instead of
    reaching the real network.
    """

    url_pattern: str
    status: int = 200
    body: Any = None  # dict/list -> JSON-encoded; str -> sent as-is
    content_type: str = "application/json"
    method: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def matches(self, request: Request) -> bool:
        if self.method and request.method.upper() != self.method.upper():
            return False
        return fnmatch.fnmatch(request.url, self.url_pattern)

    def render_body(self) -> str:
        if isinstance(self.body, str) or self.body is None:
            return self.body or ""
        return json.dumps(self.body)


@dataclass(frozen=True)
class RecordedExchange:
    """One observed request/response pair, kept for evidence bundling and assertions."""

    url: str
    method: str
    request_body: str | None
    status: int | None
    response_body: str | None
    was_mocked: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NetworkInterceptor:
    """
    Example:
        interceptor = NetworkInterceptor()
        interceptor.mock("**/api/payment", status=200, body={"status": "approved"})
        await interceptor.attach(page)
        ...
        failed_requests = interceptor.exchanges_matching(lambda e: e.status and e.status >= 400)
    """

    def __init__(self, observe_pattern: str = "**") -> None:
        self._mock_rules: list[MockRule] = []
        self._exchanges: list[RecordedExchange] = []
        self._observe_pattern = observe_pattern
        self._attached_page: RoutablePage | None = None

    def mock(
        self,
        url_pattern: str,
        status: int = 200,
        body: Any = None,
        content_type: str = "application/json",
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Register a mock rule. Later-registered rules take precedence over earlier, broader ones."""
        self._mock_rules.append(
            MockRule(
                url_pattern=url_pattern,
                status=status,
                body=body,
                content_type=content_type,
                method=method,
                headers=headers or {},
            )
        )

    def clear_mocks(self) -> None:
        self._mock_rules.clear()

    @property
    def exchanges(self) -> list[RecordedExchange]:
        return list(self._exchanges)

    def exchanges_matching(
        self, predicate: Callable[[RecordedExchange], bool]
    ) -> list[RecordedExchange]:
        return [e for e in self._exchanges if predicate(e)]

    def failed_exchanges(self, status_threshold: int = 400) -> list[RecordedExchange]:
        """Convenience accessor for verification's network-error checks."""
        return self.exchanges_matching(
            lambda e: e.status is not None and e.status >= status_threshold
        )

    async def attach(self, page: RoutablePage) -> None:
        """Start intercepting/observing traffic on `page` for the observe pattern."""
        self._attached_page = page
        await page.route(self._observe_pattern, self._handle_route)
        logger.debug("NetworkInterceptor attached with %d mock rule(s).", len(self._mock_rules))

    async def detach(self) -> None:
        if self._attached_page is not None:
            await self._attached_page.unroute(self._observe_pattern)
            self._attached_page = None

    def _matching_rule(self, request: Request) -> MockRule | None:
        for rule in reversed(self._mock_rules):
            if rule.matches(request):
                return rule
        return None

    async def _handle_route(self, route: Route) -> None:
        request = route.request
        rule = self._matching_rule(request)

        if rule is not None:
            await route.fulfill(
                status=rule.status,
                content_type=rule.content_type,
                body=rule.render_body(),
                headers=rule.headers or None,
            )
            self._exchanges.append(
                RecordedExchange(
                    url=request.url,
                    method=request.method,
                    request_body=request.post_data(),
                    status=rule.status,
                    response_body=rule.render_body(),
                    was_mocked=True,
                )
            )
            logger.debug("Mocked %s %s -> %d", request.method, request.url, rule.status)
            return

        # Not mocked: let it hit the real network, then record the
        # real outcome for evidence/assertions. `continue_()` doesn't
        # give us the response directly, so this relies on the caller
        # also having registered a `page.on("response", ...)` listener
        # via `observe_response` below for real (non-mocked) traffic.
        await route.continue_()

    def observe_response_handler(self) -> Callable[[Response], Awaitable[None]]:
        """
        Returns an async handler suitable for `page.on("response", handler)`,
        recording real (non-mocked) responses. Wiring this up is the
        caller's responsibility (typically done alongside `attach()`)
        since Playwright's `route`/`on("response")` are separate hooks.
        """

        async def _handler(response: Response) -> None:
            try:
                body = await response.text()
            except Exception:  # noqa: BLE001 - binary/streamed responses aren't text-readable
                body = None
            # Real responses don't carry the originating request's method
            # via this handler in all Playwright versions uniformly, so
            # this intentionally records what's reliably available.
            self._exchanges.append(
                RecordedExchange(
                    url=getattr(response, "url", "unknown"),
                    method="unknown",
                    request_body=None,
                    status=response.status,
                    response_body=body,
                    was_mocked=False,
                )
            )

        return _handler


def url_matches_any(url: str, patterns: list[str]) -> bool:
    """Small helper reused by verification's network-error checks. Each pattern is tried as a glob first, then as a regex if it isn't valid glob-only syntax."""
    for pattern in patterns:
        if fnmatch.fnmatch(url, pattern):
            return True
        try:
            if re.search(pattern, url):
                return True
        except re.error:
            continue  # pattern was glob-only syntax (e.g. "**/api/*"), not valid regex - already checked above
    return False