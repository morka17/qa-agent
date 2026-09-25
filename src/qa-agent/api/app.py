"""
FastAPI control-plane application entrypoint. Wires the HTTP layer
(`routers/*`) to the actual pipeline (`orchestration.agent_loop.AgentLoop`)
by registering the `"run_story"` job handler on the task queue at
startup — this is the one place in the whole codebase where an inbound
API request's payload turns into a real, executed `AgentLoop.run()` call.

Run with:
    uvicorn qa_agent.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI

from qa_agent.api.dependencies import (
    get_agent_loop,
    get_browser_manager,
    get_dedupe_engine,
    get_llm_router,
    get_task_queue,
    _get_engine,
    _get_sessionmaker,
)
from qa_agent.api.routers import bugs, runs, stories, webhooks
from qa_agent.config.logging_config import configure_logging, get_logger
from qa_agent.config.settings import get_settings
from qa_agent.ingestion.schemas import Priority, StorySource, UserStory
from qa_agent.orchestration.agent_loop import AgentLoop
from qa_agent.storage.models import ArtifactORM, BugORM, RunORM, StepORM, create_all_tables

logger = get_logger(__name__)


async def _run_story_job_handler(payload: dict[str, Any]) -> Any:
    """
    The actual work behind every `POST /runs` and `POST /webhooks/ci`
    call: builds a `UserStory` from the enqueued payload, runs it through
    the `AgentLoop`, and persists the result to the database before
    returning it (so `GET /runs/{job_id}/status` and `GET /runs/{run_id}`
    both see a consistent picture).
    """
    story = UserStory(
        source=StorySource.API,
        title=payload["title"],
        narrative=payload["narrative"],
        priority=Priority(payload.get("priority", Priority.MEDIUM.value)),
        target_url=payload.get("target_url"),
    )

    agent_loop: AgentLoop = _runtime_state["agent_loop"]
    result = await agent_loop.run(story)
    await _persist_run_result(story, result, sessionmaker=_runtime_state["sessionmaker"])
    return result


async def _persist_run_result(story: UserStory, result: Any, sessionmaker: Any) -> None:
    """
    `sessionmaker` is passed explicitly (rather than resolved from the
    global cached `_get_sessionmaker()` internally) so this function —
    and the job handler that calls it — can be exercised against a test
    database without needing the real `settings.database_url` to be
    reachable. `lifespan()` populates `_runtime_state["sessionmaker"]`
    with the real one for actual deployments; tests populate it with
    their own before invoking the handler.
    """
    async with sessionmaker() as session:
        run_row = RunORM(
            id=result.run_id,
            story_id=story.id,
            test_plan_id=result.plan.id if result.plan else None,
            target_url=story.target_url,
            state=result.final_state.value,
            error=result.error,
            finished_at=None,
        )
        session.add(run_row)

        # execution_results is produced by AgentLoop._execute_plan() by
        # iterating plan.steps_in_order() and appending exactly one
        # result per step (halting early on the first failure), so the
        # two sequences are index-aligned — zip recovers the action/
        # target_description that ExecutionResult itself doesn't carry.
        plan_steps = result.plan.steps_in_order() if result.plan else []
        for order, exec_result in enumerate(result.execution_results):
            matching_step = plan_steps[order] if order < len(plan_steps) else None
            resolved = exec_result.resolved_element
            session.add(
                StepORM(
                    run_id=run_row.id,
                    step_order=order,
                    action=matching_step.action.value if matching_step else "unknown",
                    target_description=matching_step.target_description if matching_step else None,
                    description=matching_step.description if matching_step else exec_result.step_id,
                    success=exec_result.success,
                    error=exec_result.error,
                    duration_ms=exec_result.duration_ms,
                    resolution_strategy=resolved.strategy.value if resolved else None,
                    resolution_confidence=resolved.confidence if resolved else None,
                )
            )

        if result.bug_report is not None:
            report = result.bug_report
            session.add(
                BugORM(
                    id=report.id,
                    run_id=run_row.id,
                    title=report.title,
                    severity=report.severity.value,
                    category=report.category.value,
                    summary=report.summary,
                    description=report.description,
                    expected_behavior=report.expected_behavior,
                    actual_behavior=report.actual_behavior,
                    labels=report.labels,
                    cluster_id=report.cluster_id,
                    occurrence_count=report.occurrence_count,
                    tracker_name=report.tracker_name,
                    tracker_ref=report.tracker_ref,
                    tracker_url=report.tracker_url,
                    filed_at=report.filed_at,
                )
            )
            for evidence in report.evidence:
                session.add(
                    ArtifactORM(
                        run_id=run_row.id,
                        type=evidence.type.value,
                        description=evidence.description,
                        path_or_url=evidence.path_or_url,
                        is_uploaded=evidence.is_uploaded,
                    )
                )

        await session.commit()
    logger.info("Persisted run %s (state=%s) to the database.", result.run_id, result.final_state.value)


# Process-wide state the job handler needs but can't receive via
# `Depends(...)`, since it runs outside any single HTTP request's scope.
# Populated once by `lifespan()` at real startup; a test can populate it
# directly (see api tests) to exercise `_run_story_job_handler` against
# fakes without going through `lifespan()` at all.
_runtime_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    logger.info("Starting Sentinel-QA control plane (environment=%s).", settings.environment.value)

    engine = _get_engine()
    await create_all_tables(engine)  # dev convenience; production applies storage/migrations/ via Alembic instead

    browser_manager = get_browser_manager()
    await browser_manager.start()

    task_queue = get_task_queue()
    task_queue.register_handler("run_story", _run_story_job_handler)  # type: ignore[union-attr]

    # The job handler runs outside any single HTTP request's
    # dependency-injection scope, so its AgentLoop is built once here
    # directly from the same cached collaborators `get_agent_loop`
    # would otherwise receive via `Depends(...)` inside a request.
    llm_router = get_llm_router()
    _runtime_state["agent_loop"] = get_agent_loop(
        llm_router=llm_router,
        browser_manager=browser_manager,
        dedupe_engine=get_dedupe_engine(),
    )
    _runtime_state["sessionmaker"] = _get_sessionmaker()

    yield

    logger.info("Shutting down Sentinel-QA control plane.")
    await browser_manager.stop()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentinel-QA Control Plane",
        description="Submit stories, trigger autonomous QA runs, and review filed bugs.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(stories.router)
    app.include_router(runs.router)
    app.include_router(bugs.router)
    app.include_router(webhooks.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()