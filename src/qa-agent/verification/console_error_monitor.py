"""
Listens to a page's `console` and `pageerror` events for the lifetime of
a run and classifies what it hears. This is what backs the
`NO_CONSOLE_ERRORS` assertion type (which `assertion_engine.py`
deliberately doesn't handle itself — see that module's `_eval_no_console_errors`)
and is independently useful evidence for `triage/failure_classifier.py`
even when no explicit assertion asked for it: an uncaught exception in
the console at the moment a step failed is often the actual root cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.planning.step_schema import Assertion

logger = get_logger(__name__)


class ConsoleMessageLevel(str, Enum):
    LOG = "log"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ConsoleMessage(Protocol):
    """The Playwright `ConsoleMessage` surface this module depends on."""

    @property
    def type(self) -> str: ...
    @property
    def text(self) -> str: ...
    @property
    def location(self) -> dict[str, object]: ...


class PageError(Protocol):
    """The Playwright `page.on("pageerror", ...)` payload — typically an Error-like object."""

    def __str__(self) -> str: ...


class MonitorablePage(Protocol):
    def on(self, event: str, handler: Callable[..., None]) -> None: ...
    def remove_listener(self, event: str, handler: Callable[..., None]) -> None: ...


@dataclass(frozen=True)
class RecordedConsoleEntry:
    level: ConsoleMessageLevel
    text: str
    source: str  # "console" or "pageerror"
    url: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Noise every real-world app tends to emit that is not, itself, evidence
# of a bug: browser extension chatter, expected dev-mode warnings, known
# benign third-party SDK logging. Kept intentionally small and
# conservative — broad suppression here would hide real bugs, which
# defeats the entire point of this module.
_DEFAULT_IGNORE_PATTERNS = (
    r"Download the React DevTools",
    r"\[HMR\]",
    r"chrome-extension://",
)


class ConsoleErrorMonitor:
    """
    Example:
        monitor = ConsoleErrorMonitor()
        monitor.attach(page)
        ... run the test ...
        if monitor.has_errors():
            print(monitor.summary())
        monitor.detach(page)
    """

    def __init__(self, ignore_patterns: list[str] | None = None) -> None:
        self._patterns = [
            re.compile(p) for p in (ignore_patterns or list(_DEFAULT_IGNORE_PATTERNS))
        ]
        self._entries: list[RecordedConsoleEntry] = []
        self._console_handler: Callable[[ConsoleMessage], None] | None = None
        self._pageerror_handler: Callable[[PageError], None] | None = None

    @property
    def entries(self) -> list[RecordedConsoleEntry]:
        return list(self._entries)

    def _is_ignored(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._patterns)

    def _record(self, level: ConsoleMessageLevel, text: str, source: str, url: str | None) -> None:
        if self._is_ignored(text):
            logger.debug("Ignored console entry matching a suppression pattern: %.80s", text)
            return
        self._entries.append(
            RecordedConsoleEntry(level=level, text=text, source=source, url=url)
        )
        if level == ConsoleMessageLevel.ERROR:
            logger.warning("Console error captured: %.200s", text)

    def attach(self, page: MonitorablePage) -> None:
        def on_console(message: ConsoleMessage) -> None:
            try:
                level = ConsoleMessageLevel(message.type)
            except ValueError:
                level = ConsoleMessageLevel.LOG
            location = message.location or {}
            self._record(level, message.text, source="console", url=location.get("url"))  # type: ignore[arg-type]

        def on_pageerror(error: PageError) -> None:
            self._record(ConsoleMessageLevel.ERROR, str(error), source="pageerror", url=None)

        self._console_handler = on_console
        self._pageerror_handler = on_pageerror
        page.on("console", on_console)
        page.on("pageerror", on_pageerror)
        logger.debug("ConsoleErrorMonitor attached.")

    def detach(self, page: MonitorablePage) -> None:
        if self._console_handler is not None:
            page.remove_listener("console", self._console_handler)
            self._console_handler = None
        if self._pageerror_handler is not None:
            page.remove_listener("pageerror", self._pageerror_handler)
            self._pageerror_handler = None

    def errors(self) -> list[RecordedConsoleEntry]:
        return [e for e in self._entries if e.level == ConsoleMessageLevel.ERROR]

    def has_errors(self) -> bool:
        return len(self.errors()) > 0

    def clear(self) -> None:
        """
        Reset captured entries without detaching listeners — useful when
        a monitor is reused across steps and callers want a per-step
        error count rather than a whole-run cumulative one.
        """
        self._entries.clear()

    def summary(self) -> str:
        error_entries = self.errors()
        if not error_entries:
            return "No console errors captured."
        lines = [f"{len(error_entries)} console error(s):"]
        lines += [f"  [{e.source}] {e.text[:200]}" for e in error_entries]
        return "\n".join(lines)

    def to_assertion_result(self, assertion: Assertion) -> "AssertionResultLike":
        """
        Builds the result `assertion_engine.py` couldn't produce itself
        for a `NO_CONSOLE_ERRORS` assertion, using a plain, dependency-free
        local namedtuple-like shape so this module doesn't need to import
        `assertion_engine` (which would create a cycle, since that module
        already documents this monitor as its companion for that type).
        """
        passed = not self.has_errors()
        return AssertionResultLike(
            assertion_id=assertion.id,
            passed=passed,
            actual=str(len(self.errors())),
            expected="0",
            reason="no console errors" if passed else self.summary(),
        )


@dataclass(frozen=True)
class AssertionResultLike:
    """
    Structurally identical to `assertion_engine.AssertionResult` — kept
    as a separate definition (rather than importing it) purely to avoid
    a circular import between these two verification modules. Callers
    that need the real `AssertionResult` type can construct one from
    this object's fields directly.
    """

    assertion_id: str
    passed: bool
    actual: str | None
    expected: str | None
    reason: str