#!/usr/bin/env python3
"""
Sentinel-QA command-line entrypoint.

Installed as the `qa_agent` console script (see `pyproject.toml`'s
`[tool.poetry.scripts]`, which points at the `app` object below), so
after `poetry install` this is invoked as:

    qa_agent run --story "As a user, I can add an item to my cart." --target-url https://staging.example.com
    qa_agent run --stories ./backlog.csv
    qa_agent serve --port 8000
    qa_agent worker

Every command wires the same collaborators `api/dependencies.py` wires
for the HTTP layer (`build_agent_loop` below is the CLI's equivalent of
`api.dependencies.get_agent_loop`) so behavior never drifts between
"run it from the CLI" and "trigger it through the API" — both ultimately
call the same `AgentLoop.run()`.
"""

from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from typing import Optional

import typer

from qa_agent.config.logging_config import bind_run_context, clear_run_context, configure_logging, get_logger
from qa_agent.config.settings import IssueTracker, Settings, get_settings
from qa_agent.execution.browser_manager import BrowserManager
from qa_agent.ingestion.connectors.csv_connector import CSVConnector
from qa_agent.ingestion.schemas import Priority, StorySource, UserStory
from qa_agent.ingestion.story_parser import StoryParser
from qa_agent.llm.cost_tracker import CostTracker
from qa_agent.llm.provider_router import LLMRouter
from qa_agent.orchestration.agent_loop import AgentLoop, RunResult
from qa_agent.perception.element_resolver import ElementResolver
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache
from qa_agent.planning.test_planner import TestPlanner
from qa_agent.reporting.bug_report_writer import BugReportWriter
from qa_agent.reporting.report_schema import IssueTrackerFiler
from qa_agent.triage.dedupe_engine import DedupeEngine
from qa_agent.triage.failure_classifier import FailureClassifier
from qa_agent.triage.root_cause_analyzer import RootCauseAnalyzer

logger = get_logger(__name__)

app = typer.Typer(
    name="qa_agent",
    help="Sentinel-QA — an autonomous QA agent that reads user stories, drives a "
    "real browser, and files evidence-backed bug reports.",
    add_completion=False,
    no_args_is_help=True,
)


# --------------------------------------------------------------------------
# Collaborator wiring
# --------------------------------------------------------------------------
# Mirrors api/dependencies.py's construction, but as a plain function
# rather than FastAPI Depends() providers, since the CLI has no request
# scope to cache collaborators against. Kept in one place so `run` and
# `worker` — and the API — never wire the pipeline differently.


def _build_issue_tracker(settings: Settings) -> IssueTrackerFiler | None:
    if settings.issue_tracker == IssueTracker.JIRA:
        from qa_agent.reporting.issue_trackers.jira_filer import JiraFiler

        return JiraFiler(settings=settings)
    if settings.issue_tracker == IssueTracker.LINEAR:
        from qa_agent.reporting.issue_trackers.linear_filer import LinearFiler

        return LinearFiler(settings=settings)
    if settings.issue_tracker == IssueTracker.GITHUB:
        from qa_agent.reporting.issue_trackers.github_issues_filer import GitHubIssuesFiler

        return GitHubIssuesFiler(settings=settings)
    return None


def build_agent_loop(settings: Settings, browser_manager: BrowserManager) -> AgentLoop:
    """
    Constructs one fully-wired `AgentLoop`. `browser_manager` is passed
    in (rather than built here) because its lifecycle — `start()`/`stop()`
    — spans multiple calls to `agent_loop.run()` in both the `run` and
    `worker` commands, so the caller owns starting and stopping it.
    """
    cost_tracker = CostTracker()
    llm_router = LLMRouter(cost_tracker=cost_tracker, settings=settings)

    return AgentLoop(
        browser_manager=browser_manager,
        story_parser=StoryParser(llm_client=llm_router),
        test_planner=TestPlanner(llm_client=llm_router, settings=settings),
        element_resolver=ElementResolver(vision_client=llm_router, cache=SelectorCache(), settings=settings),
        bug_report_writer=BugReportWriter(llm_client=llm_router),
        issue_tracker=_build_issue_tracker(settings),
        classifier=FailureClassifier(llm_client=llm_router),
        rca_analyzer=RootCauseAnalyzer(llm_client=llm_router),
        dedupe_engine=DedupeEngine(),
    )


# --------------------------------------------------------------------------
# `run` — ad hoc / CSV-backlog execution
# --------------------------------------------------------------------------


def _load_stories(
    story: Optional[str],
    stories_path: Optional[Path],
    target_url: Optional[str],
    priority: Priority,
    title: Optional[str],
) -> list[UserStory]:
    if bool(story) == bool(stories_path):
        raise typer.BadParameter("Pass exactly one of --story or --stories.")

    if stories_path is not None:
        return CSVConnector().read_stories(stories_path, target_url=target_url)

    assert story is not None  # guaranteed by the exactly-one check above
    return [
        UserStory(
            source=StorySource.MANUAL,
            title=title or (story[:80] + ("…" if len(story) > 80 else "")),
            narrative=story,
            priority=priority,
            target_url=target_url,
        )
    ]


def _print_result(result: RunResult) -> None:
    typer.echo(f"[{result.final_state.value.upper()}] run={result.run_id}")
    if result.error:
        typer.echo(f"    error: {result.error}")
    if result.bug_report is not None:
        typer.echo(f"    bug filed: {result.bug_report.title!r} (severity={result.bug_report.severity.value})")
        if result.bug_report.tracker_url:
            typer.echo(f"    tracker: {result.bug_report.tracker_url}")


async def _run_all(stories: list[UserStory], settings: Settings) -> list[RunResult]:
    browser_manager = BrowserManager(settings=settings)
    await browser_manager.start()
    agent_loop = build_agent_loop(settings, browser_manager)

    results: list[RunResult] = []
    try:
        for story in stories:
            bind_run_context(story_id=story.id)
            typer.echo(f"Running story {story.id[:8]}: {story.title!r}")
            result = await agent_loop.run(story)
            results.append(result)
            _print_result(result)
            clear_run_context()
    finally:
        await browser_manager.stop()

    return results


@app.command()
def run(
    story: Optional[str] = typer.Option(
        None, "--story", help="A single user story narrative, e.g. 'As a user, I can add an item to my cart.'"
    ),
    stories: Optional[Path] = typer.Option(
        None,
        "--stories",
        exists=True,
        dir_okay=False,
        help="Path to a CSV backlog file (see ingestion/connectors/csv_connector.py for the expected columns).",
    ),
    target_url: Optional[str] = typer.Option(
        None,
        "--target-url",
        help="Base URL of the app under test. Required with --story; optional with --stories "
        "(each row may set its own target_url column instead).",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", help="Title for the ad-hoc story (--story only). Defaults to a truncated narrative."
    ),
    priority: Priority = typer.Option(Priority.MEDIUM, "--priority", help="Priority for the ad-hoc story (--story only)."),
) -> None:
    """Run one story ad hoc, or a CSV backlog of stories, against a target app."""
    configure_logging()
    settings = get_settings()

    if story and not target_url:
        raise typer.BadParameter("--target-url is required when using --story.")

    story_list = _load_stories(story, stories, target_url, priority, title)
    if not story_list:
        typer.echo("No valid stories to run (check the CSV for malformed rows).")
        raise typer.Exit(code=1)

    results = asyncio.run(_run_all(story_list, settings))

    passed = sum(1 for r in results if r.passed)
    typer.echo("")
    typer.echo(f"{passed}/{len(results)} run(s) passed.")
    if passed < len(results):
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------
# `serve` — control-plane API
# --------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host for the control-plane API."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Enable auto-reload for local development."),
) -> None:
    """Start the FastAPI control-plane API (`qa_agent.api.app:app`)."""
    import uvicorn

    configure_logging()
    uvicorn.run("qa_agent.api.app:app", host=host, port=port, reload=reload)


# --------------------------------------------------------------------------
# `worker` — standalone Redis-backed job consumer
# --------------------------------------------------------------------------
# A lightweight alternative to Celery/Temporal for teams that don't want
# to operate either: jobs are JSON payloads pushed onto a Redis list,
# matching the shape api/routers/runs.py enqueues
# ({title, narrative, priority, target_url}). Deployments already using
# Celery/Temporal should run those frameworks' native worker CLIs against
# the task bodies described in orchestration/task_queue.py's
# CeleryTaskQueue/TemporalTaskQueue docstrings instead of this command.

REDIS_JOB_KEY = "sentinel-qa:jobs:run_story"


async def _process_job_payload(agent_loop: AgentLoop, raw_payload: bytes | str) -> RunResult:
    payload = json.loads(raw_payload)
    story = UserStory(
        source=StorySource.API,
        title=payload["title"],
        narrative=payload["narrative"],
        priority=Priority(payload.get("priority", Priority.MEDIUM.value)),
        target_url=payload.get("target_url"),
    )
    return await agent_loop.run(story)


async def worker_loop(
    redis_client,
    agent_loop: AgentLoop,
    queue_key: str = REDIS_JOB_KEY,
    poll_timeout_s: int = 5,
    max_iterations: Optional[int] = None,
) -> None:
    """
    Blocks on `queue_key` and processes one job at a time.

    `max_iterations` bounds the loop — used by tests; production callers
    omit it (the default, `None`) to run until the process receives
    SIGINT/SIGTERM. A single malformed or failing job is logged and
    skipped rather than crashing the worker process.
    """
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        popped = await redis_client.blpop(queue_key, timeout=poll_timeout_s)
        iterations += 1
        if popped is None:
            continue  # BLPOP timed out - loop again so Ctrl+C/SIGTERM can interrupt promptly

        _, raw_payload = popped
        try:
            result = await _process_job_payload(agent_loop, raw_payload)
            logger.info("Processed job -> run %s (%s).", result.run_id, result.final_state.value)
        except Exception as exc:  # noqa: BLE001 - one bad job must never take down the worker process
            logger.error("Job processing failed: %s", exc, exc_info=True)


@app.command()
def worker(
    queue_key: str = typer.Option(REDIS_JOB_KEY, help="Redis list key to BLPOP job payloads from."),
    poll_timeout: int = typer.Option(5, help="Seconds to block on each BLPOP before looping again."),
) -> None:
    """Start a standalone worker that consumes job payloads from Redis and runs them through the AgentLoop."""
    import redis.asyncio as redis

    configure_logging()
    settings = get_settings()
    redis_client = redis.from_url(settings.redis_url)

    async def _main() -> None:
        browser_manager = BrowserManager(settings=settings)
        await browser_manager.start()
        agent_loop = build_agent_loop(settings, browser_manager)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass  # not available on every platform - Ctrl+C still raises KeyboardInterrupt there

        typer.echo(f"Worker listening on Redis key {queue_key!r} (Ctrl+C to stop)...")
        worker_task = asyncio.create_task(worker_loop(redis_client, agent_loop, queue_key, poll_timeout))
        await stop_event.wait()
        worker_task.cancel()

        await browser_manager.stop()
        await redis_client.aclose()

    asyncio.run(_main())


# --------------------------------------------------------------------------
# `version`
# --------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed Sentinel-QA version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        typer.echo(pkg_version("sentinel-qa"))
    except PackageNotFoundError:
        typer.echo("sentinel-qa (version unknown — not installed as a package)")


if __name__ == "__main__":
    app()