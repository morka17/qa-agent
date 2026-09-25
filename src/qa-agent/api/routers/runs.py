"""
Run lifecycle endpoints: trigger a run for a story, poll its status, and
list historical runs. Triggering enqueues onto the configured
`TaskQueue` (`api/dependencies.py::get_task_queue`) rather than running
inline in the request handler — an agent run can take minutes (browser
automation + several LLM calls), far too long for a synchronous HTTP
request.

The `"run_story"` job handler that actually executes an `AgentLoop` run
and persists its result is registered against the task queue at
app-startup (`api/app.py`), not here — this router only enqueues and
polls, consistent with `TaskQueue` being the single seam between "an API
call happened" and "an agent run actually executes" (see
`orchestration/task_queue.py`).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from qa_agent.api.dependencies import get_db_session, get_task_queue
from qa_agent.ingestion.schemas import Priority
from qa_agent.orchestration.task_queue import JobStatus, TaskQueue
from qa_agent.storage.models import RunORM

router = APIRouter(prefix="/runs", tags=["runs"])


class RunTriggerRequest(BaseModel):
    title: str = Field(..., min_length=1)
    narrative: str = Field(..., min_length=1)
    priority: Priority = Priority.MEDIUM
    target_url: str | None = None


class RunTriggerResponse(BaseModel):
    job_id: str
    status: JobStatus


class RunStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    run_id: str | None = None
    final_state: str | None = None
    bug_report_id: str | None = None
    error: str | None = None


class RunSummary(BaseModel):
    id: str
    story_id: str | None
    target_url: str | None
    state: str
    created_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_orm_row(cls, row: RunORM) -> "RunSummary":
        return cls(
            id=row.id,
            story_id=row.story_id,
            target_url=row.target_url,
            state=row.state,
            created_at=row.created_at,
            finished_at=row.finished_at,
        )


@router.post("", response_model=RunTriggerResponse, status_code=202)
async def trigger_run(
    request: RunTriggerRequest, task_queue: TaskQueue = Depends(get_task_queue)
) -> RunTriggerResponse:
    """Enqueues a run and returns immediately with a job_id to poll via `GET /runs/{job_id}/status`."""
    job_id = await task_queue.enqueue(
        "run_story",
        {
            "title": request.title,
            "narrative": request.narrative,
            "priority": request.priority.value,
            "target_url": request.target_url,
        },
    )
    return RunTriggerResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get("/{job_id}/status", response_model=RunStatusResponse)
async def get_run_status(
    job_id: str, task_queue: TaskQueue = Depends(get_task_queue)
) -> RunStatusResponse:
    record = await task_queue.get_status(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No job found with id {job_id!r}.")

    response = RunStatusResponse(job_id=job_id, status=record.status, error=record.error)
    if record.status == JobStatus.SUCCEEDED and record.result is not None:
        result = record.result  # the orchestration.agent_loop.RunResult the job handler returned
        response.run_id = result.run_id
        response.final_state = result.final_state.value
        if result.bug_report is not None:
            response.bug_report_id = result.bug_report.id
    return response


@router.get("", response_model=list[RunSummary])
async def list_runs(
    state: str | None = Query(default=None, description="Filter by run state, e.g. 'failed'."),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> list[RunSummary]:
    stmt = select(RunORM).order_by(RunORM.created_at.desc()).offset(offset).limit(limit)
    if state is not None:
        stmt = stmt.where(RunORM.state == state)
    result = await session.execute(stmt)
    return [RunSummary.from_orm_row(row) for row in result.scalars().all()]


@router.get("/{run_id}", response_model=RunSummary)
async def get_run(run_id: str, session: AsyncSession = Depends(get_db_session)) -> RunSummary:
    result = await session.execute(select(RunORM).where(RunORM.id == run_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No run found with id {run_id!r}.")
    return RunSummary.from_orm_row(row)