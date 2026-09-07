"""
Structured logging setup for Sentinel-QA.

Emits JSON logs in non-local environments (for ingestion by log
aggregators / OpenTelemetry collectors) and human-readable colored logs
locally. Every log line automatically carries a `run_id` when one is bound
via `bind_run_context`, which makes it trivial to grep a single agent
run's full log trail out of a shared stream.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import sys
from datetime import datetime, timezone
from typing import Any

from qa_agent.config.settings import Environment, get_settings

# Context var so any log call anywhere in the async call stack automatically
# tags its run_id/story_id without threading them through every function.
_run_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "run_id", default=None
)
_story_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "story_id", default=None
)


def bind_run_context(run_id: str | None = None, story_id: str | None = None) -> None:
    """Bind the current run/story id so subsequent log records include it."""
    if run_id is not None:
        _run_id_ctx.set(run_id)
    if story_id is not None:
        _story_id_ctx.set(story_id)


def clear_run_context() -> None:
    _run_id_ctx.set(None)
    _story_id_ctx.set(None)


class ContextFilter(logging.Filter):
    """Injects run_id/story_id from contextvars onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id_ctx.get()
        record.story_id = _story_id_ctx.get()
        return True


class JSONFormatter(logging.Formatter):
    """Renders log records as single-line JSON for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        run_id = getattr(record, "run_id", None)
        story_id = getattr(record, "story_id", None)
        if run_id:
            payload["run_id"] = run_id
        if story_id:
            payload["story_id"] = story_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Compact, colorized-friendly formatter for local development."""

    _BASE_FMT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._BASE_FMT, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        run_id = getattr(record, "run_id", None)
        story_id = getattr(record, "story_id", None)
        base = super().format(record)
        tags = []
        if run_id:
            tags.append(f"run={run_id}")
        if story_id:
            tags.append(f"story={story_id}")
        if tags:
            base = f"{base}  [{' '.join(tags)}]"
        return base


def configure_logging() -> None:
    """
    Configure the root logger for the process.

    Call this once at process startup (CLI entrypoint, API app startup,
    worker startup) — never inside library code, to avoid double
    configuration when qa_agent modules are imported by a host application.
    """
    settings = get_settings()
    use_json = settings.environment != Environment.LOCAL

    formatter_key = "json" if use_json else "human"

    logging_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context": {"()": ContextFilter},
        },
        "formatters": {
            "json": {"()": JSONFormatter},
            "human": {"()": HumanFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": formatter_key,
                "filters": ["context"],
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console"],
        },
        "loggers": {
            # Quiet down noisy third-party libraries by default.
            "playwright": {"level": "WARNING", "propagate": True},
            "httpx": {"level": "WARNING", "propagate": True},
            "httpcore": {"level": "WARNING", "propagate": True},
            "asyncio": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(logging_config)
    logging.getLogger(__name__).debug(
        "Logging configured (format=%s, level=%s, env=%s)",
        formatter_key,
        settings.log_level,
        settings.environment.value,
    )


def get_logger(name: str) -> logging.Logger:
    """Thin convenience wrapper so call sites don't import stdlib logging directly."""
    return logging.getLogger(name)