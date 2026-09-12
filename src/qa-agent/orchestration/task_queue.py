"""
Abstracts "run this job somewhere, eventually, possibly on another
process" behind one small interface, so `agent_loop.py` and
`run_scheduler.py` don't need to know or care whether jobs are actually
executed by Celery, RQ, Temporal, or (for local dev/tests) an in-process
asyncio queue.

Only one concrete implementation (`InMemoryTaskQueue`) is fully wired up
here, since it has no external dependency — the Celery/Temporal adapters
are thin, deliberately minimal wrappers that import their client
libraries lazily (only when actually instantiated) so `qa_agent` doesn't
hard-require every possible queue backend just to be imported.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None


class TaskQueue(Protocol):
    """
    The interface `agent_loop.py`/`run_scheduler.py` depend on. `payload`
    is whatever the job handler needs (typically a story ID or a
    pre-built `TestIntent`) — the queue itself doesn't interpret it, it
    just passes it through to whatever handler was registered for
    `job_type`.
    """

    async def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        """Enqueue a job and return its job_id immediately (does not wait for completion)."""
        ...

    async def get_status(self, job_id: str) -> JobRecord | None: ...


class TaskQueueError(Exception):
    """Raised for backend-specific submission failures."""


JobHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class InMemoryTaskQueue:
    """
    Process-local queue backed by `asyncio`. Suitable for local dev,
    tests, and single-process deployments; not shared across worker
    processes — use `CeleryTaskQueue` or `TemporalTaskQueue` for anything
    that needs to survive a process restart or scale beyond one worker.

    Example:
        queue = InMemoryTaskQueue()
        queue.register_handler("run_story", my_agent_loop.run_from_payload)
        job_id = await queue.enqueue("run_story", {"story_id": "abc"})
        status = await queue.get_status(job_id)
    """

    def __init__(self, max_concurrent_jobs: int = 5) -> None:
        self._handlers: dict[str, JobHandler] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._tasks: set[asyncio.Task] = set()

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    async def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        if job_type not in self._handlers:
            raise TaskQueueError(f"No handler registered for job_type {job_type!r}.")

        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobRecord(
            job_id=job_id, status=JobStatus.QUEUED, queued_at=datetime.now(timezone.utc)
        )
        task = asyncio.create_task(self._run_job(job_id, job_type, payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        logger.info("Enqueued job %s (type=%s).", job_id, job_type)
        return job_id

    async def _run_job(self, job_id: str, job_type: str, payload: dict[str, Any]) -> None:
        async with self._semaphore:
            record = self._jobs[job_id]
            record.status = JobStatus.RUNNING
            record.started_at = datetime.now(timezone.utc)
            try:
                handler = self._handlers[job_type]
                result = await handler(payload)
                record.result = result
                record.status = JobStatus.SUCCEEDED
                logger.info("Job %s (type=%s) succeeded.", job_id, job_type)
            except Exception as exc:  # noqa: BLE001 - job failures are data (JobRecord.error), not process crashes
                record.error = str(exc)
                record.status = JobStatus.FAILED
                logger.error("Job %s (type=%s) failed: %s", job_id, job_type, exc)
            finally:
                record.finished_at = datetime.now(timezone.utc)

    async def get_status(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def wait_for(self, job_id: str, poll_interval_s: float = 0.05) -> JobRecord:
        """Convenience for tests/CLI usage: block until a job reaches a terminal status."""
        while True:
            record = self._jobs.get(job_id)
            if record is None:
                raise TaskQueueError(f"Unknown job_id {job_id!r}.")
            if record.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return record
            await asyncio.sleep(poll_interval_s)


class CeleryTaskQueue:
    """
    Adapter over a Celery app. Imports `celery` lazily in `__init__` so
    that merely importing `qa_agent.orchestration.task_queue` doesn't
    require Celery to be installed for deployments that use a different
    backend.

    `job_type` is used as the registered Celery task's name
    (`celery_app.task(name=job_type)`); the caller is responsible for
    having registered that task on the Celery app before jobs of that
    type are enqueued — this adapter only submits and polls, it doesn't
    define task bodies itself (Celery tasks aren't natively async, so the
    handler registration story is intentionally left to the caller's
    Celery app setup rather than mirrored here).
    """

    def __init__(self, celery_app: Any) -> None:
        try:
            from celery.result import AsyncResult  # noqa: F401
        except ImportError as exc:
            raise TaskQueueError(
                "CeleryTaskQueue requires the 'celery' package to be installed."
            ) from exc
        self._celery_app = celery_app

    async def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        # Celery's `.delay()`/`.apply_async()` are synchronous calls that
        # perform network I/O (publishing to the broker); run in a thread
        # so this coroutine doesn't block the event loop.
        async_result = await asyncio.to_thread(
            self._celery_app.send_task, job_type, kwargs={"payload": payload}
        )
        logger.info("Enqueued Celery job %s (type=%s).", async_result.id, job_type)
        return async_result.id

    async def get_status(self, job_id: str) -> JobRecord | None:
        from celery.result import AsyncResult

        result = AsyncResult(job_id, app=self._celery_app)
        status_map = {
            "PENDING": JobStatus.QUEUED,
            "STARTED": JobStatus.RUNNING,
            "SUCCESS": JobStatus.SUCCEEDED,
            "FAILURE": JobStatus.FAILED,
        }
        status = status_map.get(result.state, JobStatus.QUEUED)
        return JobRecord(
            job_id=job_id,
            status=status,
            queued_at=datetime.now(timezone.utc),  # Celery doesn't expose original queue time via AsyncResult
            result=result.result if status == JobStatus.SUCCEEDED else None,
            error=str(result.result) if status == JobStatus.FAILED else None,
        )


class TemporalTaskQueue:
    """
    Adapter over a Temporal client/workflow. Like `CeleryTaskQueue`,
    imports `temporalio` lazily. `job_type` maps to a registered Temporal
    workflow name; the workflow itself (e.g. `AgentLoopWorkflow`) is
    defined and registered with the Temporal worker separately — this
    adapter only starts and polls executions.
    """

    def __init__(self, temporal_client: Any, task_queue_name: str) -> None:
        try:
            import temporalio.client  # noqa: F401
        except ImportError as exc:
            raise TaskQueueError(
                "TemporalTaskQueue requires the 'temporalio' package to be installed."
            ) from exc
        self._client = temporal_client
        self._task_queue_name = task_queue_name

    async def enqueue(self, job_type: str, payload: dict[str, Any]) -> str:
        workflow_id = str(uuid.uuid4())
        await self._client.start_workflow(
            job_type,
            payload,
            id=workflow_id,
            task_queue=self._task_queue_name,
        )
        logger.info("Started Temporal workflow %s (type=%s).", workflow_id, job_type)
        return workflow_id

    async def get_status(self, job_id: str) -> JobRecord | None:
        handle = self._client.get_workflow_handle(job_id)
        description = await handle.describe()
        status_map = {
            "RUNNING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.SUCCEEDED,
            "FAILED": JobStatus.FAILED,
        }
        status = status_map.get(description.status.name, JobStatus.QUEUED)
        result = None
        error = None
        if status == JobStatus.SUCCEEDED:
            result = await handle.result()
        elif status == JobStatus.FAILED:
            error = "Temporal workflow failed; see Temporal UI for details."
        return JobRecord(
            job_id=job_id,
            status=status,
            queued_at=description.start_time,
            result=result,
            error=error,
        )