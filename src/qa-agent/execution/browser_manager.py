"""
Owns the Playwright process, browser, and context lifecycle.

A single `BrowserManager` is shared across an agent worker process: one
Playwright browser instance is expensive to start and is reused across
many test runs, while individual `BrowserContext`s (cheap, isolated —
separate cookies/storage/cache) are created and torn down per run so
tests never leak state into each other.

Concurrency is bounded by `settings.max_concurrent_browser_contexts` via
a semaphore, so a burst of queued runs can't exhaust host memory by
spinning up unbounded contexts.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import TracebackType
from typing import AsyncIterator

from playwright.async_api import (
    Browser,
    BrowserContext,
    BrowserType,
    Page,
    Playwright,
    async_playwright,
)

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings

logger = get_logger(__name__)


class BrowserManagerError(Exception):
    """Raised for lifecycle errors: launching before start(), double-start, etc."""


class BrowserManager:
    """
    Example:
        manager = BrowserManager()
        await manager.start()
        async with manager.new_context() as context:
            page = await manager.new_page(context)
            await page.goto("https://example.com")
        await manager.stop()

    Or, as an async context manager for the whole manager lifecycle:
        async with BrowserManager() as manager:
            async with manager.new_context() as context:
                ...
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context_semaphore = asyncio.Semaphore(
            self._settings.max_concurrent_browser_contexts
        )
        self._active_context_count = 0

    async def start(self) -> None:
        if self._browser is not None:
            raise BrowserManagerError("BrowserManager.start() called more than once.")

        self._playwright = await async_playwright().start()
        browser_type: BrowserType = getattr(
            self._playwright, self._settings.playwright_browser
        )
        self._browser = await browser_type.launch(headless=self._settings.playwright_headless)
        logger.info(
            "Launched %s browser (headless=%s, max_concurrent_contexts=%d).",
            self._settings.playwright_browser,
            self._settings.playwright_headless,
            self._settings.max_concurrent_browser_contexts,
        )

    async def stop(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser manager stopped.")

    @property
    def is_running(self) -> bool:
        return self._browser is not None

    @asynccontextmanager
    async def new_context(
        self,
        record_video_dir: str | None = None,
        **context_kwargs: object,
    ) -> AsyncIterator[BrowserContext]:
        """
        Acquire a fresh, isolated `BrowserContext`, bounded by the
        concurrency semaphore. Always use this as an `async with` block —
        it guarantees the context is closed (and the semaphore slot
        released) even if the test run raises.
        """
        if self._browser is None:
            raise BrowserManagerError("Cannot create a context before start() has been called.")

        async with self._context_semaphore:
            self._active_context_count += 1
            logger.debug(
                "Opening browser context (%d/%d active).",
                self._active_context_count,
                self._settings.max_concurrent_browser_contexts,
            )
            context = await self._browser.new_context(
                record_video_dir=record_video_dir,
                **context_kwargs,  # type: ignore[arg-type]
            )
            context.set_default_timeout(self._settings.playwright_default_timeout_ms)
            context.set_default_navigation_timeout(
                self._settings.playwright_navigation_timeout_ms
            )
            try:
                yield context
            finally:
                await context.close()
                self._active_context_count -= 1
                logger.debug(
                    "Closed browser context (%d/%d active).",
                    self._active_context_count,
                    self._settings.max_concurrent_browser_contexts,
                )

    async def new_page(self, context: BrowserContext) -> Page:
        """Convenience wrapper: most contexts in Sentinel only ever need a single page."""
        return await context.new_page()

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()