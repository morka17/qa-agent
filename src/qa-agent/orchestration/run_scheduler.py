"""
Triggers agent runs on a schedule (nightly regression sweeps, periodic
smoke tests) or on demand (a CI webhook firing after a deploy). Every
trigger path — cron, interval, or ad hoc — funnels through the same
`enqueue()` call into a `TaskQueue`, so `agent_loop.py` never needs to
know whether a given run was scheduled or manually triggered.

Cron expressions are supported via the optional `croniter` package,
imported lazily so a deployment that only uses interval/CI-triggered
schedules doesn't need to install it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from qa_agent.config.logging_config import get_logger
from qa_agent.orchestration.task_queue import TaskQueue

logger = get_logger(__name__)


class ScheduleType(str, Enum):
    INTERVAL = "interval"
    CRON = "cron"


@dataclass
class ScheduleSpec:
    """
    One recurring schedule. Exactly one of `interval_seconds` /
    `cron_expression` should be set, matching `schedule_type`.
    """

    id: str
    job_type: str
    payload: dict[str, Any]
    schedule_type: ScheduleType
    interval_seconds: int | None = None
    cron_expression: str | None = None  # standard 5-field cron, e.g. "0 2 * * *"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.schedule_type == ScheduleType.INTERVAL and self.interval_seconds is None:
            raise ValueError(f"Schedule {self.id!r}: interval schedules require interval_seconds.")
        if self.schedule_type == ScheduleType.INTERVAL and self.interval_seconds < 0:
            raise ValueError(f"Schedule {self.id!r}: interval_seconds must be >= 0.")
        if self.schedule_type == ScheduleType.CRON and not self.cron_expression:
            raise ValueError(f"Schedule {self.id!r}: cron schedules require cron_expression.")


class RunSchedulerError(Exception):
    pass


def _next_cron_fire_time(cron_expression: str, after: datetime) -> datetime:
    try:
        from croniter import croniter
    except ImportError as exc:
        raise RunSchedulerError(
            "Cron-based schedules require the optional 'croniter' package "
            "(pip install croniter)."
        ) from exc
    return croniter(cron_expression, after).get_next(datetime)


class RunScheduler:
    """
    Example:
        scheduler = RunScheduler(task_queue=queue)
        scheduler.add_schedule(ScheduleSpec(
            id="nightly-regression",
            job_type="run_story_batch",
            payload={"suite": "regression"},
            schedule_type=ScheduleType.CRON,
            cron_expression="0 2 * * *",
        ))
        await scheduler.start()
        ...
        await scheduler.stop()

    Or trigger a run immediately (e.g. from a CI webhook handler), bypassing any schedule:
        await scheduler.trigger_now("run_story", {"story_id": "abc"})
    """

    def __init__(self, task_queue: TaskQueue, poll_interval_s: float = 1.0) -> None:
        self._task_queue = task_queue
        self._poll_interval_s = poll_interval_s
        self._schedules: dict[str, ScheduleSpec] = {}
        self._next_fire: dict[str, datetime] = {}
        self._loop_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    def add_schedule(self, spec: ScheduleSpec) -> None:
        if spec.id in self._schedules:
            raise RunSchedulerError(f"A schedule with id {spec.id!r} is already registered.")
        self._schedules[spec.id] = spec
        self._next_fire[spec.id] = self._compute_next_fire(spec, after=datetime.now(timezone.utc))
        logger.info(
            "Registered schedule %r (%s), next fire at %s.",
            spec.id,
            spec.schedule_type.value,
            self._next_fire[spec.id].isoformat(),
        )

    def remove_schedule(self, schedule_id: str) -> None:
        self._schedules.pop(schedule_id, None)
        self._next_fire.pop(schedule_id, None)

    def set_enabled(self, schedule_id: str, enabled: bool) -> None:
        if schedule_id not in self._schedules:
            raise RunSchedulerError(f"Unknown schedule id {schedule_id!r}.")
        self._schedules[schedule_id].enabled = enabled

    @staticmethod
    def _compute_next_fire(spec: ScheduleSpec, after: datetime) -> datetime:
        if spec.schedule_type == ScheduleType.INTERVAL:
            assert spec.interval_seconds is not None
            return after + timedelta(seconds=spec.interval_seconds)
        assert spec.cron_expression is not None
        return _next_cron_fire_time(spec.cron_expression, after)

    async def trigger_now(self, job_type: str, payload: dict[str, Any]) -> str:
        """
        Enqueue a run immediately, independent of any registered
        schedule. This is the entrypoint a CI webhook handler (e.g. a
        FastAPI route in `api/routers/webhooks.py`) calls after a deploy
        completes.
        """
        job_id = await self._task_queue.enqueue(job_type, payload)
        logger.info("Triggered ad hoc run: job %s (type=%s).", job_id, job_type)
        return job_id

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        for schedule_id, spec in list(self._schedules.items()):
            if not spec.enabled:
                continue
            next_fire = self._next_fire.get(schedule_id)
            if next_fire is not None and now >= next_fire:
                try:
                    await self.trigger_now(spec.job_type, spec.payload)
                except Exception as exc:  # noqa: BLE001 - one bad schedule shouldn't kill the scheduler loop
                    logger.error("Failed to trigger scheduled run %r: %s", schedule_id, exc)
                self._next_fire[schedule_id] = self._compute_next_fire(spec, after=now)
                logger.debug(
                    "Schedule %r next fire recomputed: %s.",
                    schedule_id,
                    self._next_fire[schedule_id].isoformat(),
                )

    async def _run_loop(self) -> None:
        while not self._stopped.is_set():
            await self._tick()
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._poll_interval_s)
            except asyncio.TimeoutError:
                pass  # normal: just means it's time for the next tick

    async def start(self) -> None:
        if self._loop_task is not None:
            raise RunSchedulerError("RunScheduler.start() called more than once.")
        self._stopped.clear()
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("RunScheduler started with %d schedule(s).", len(self._schedules))

    async def stop(self) -> None:
        self._stopped.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None
        logger.info("RunScheduler stopped.")

    def next_fire_times(self) -> dict[str, datetime]:
        return dict(self._next_fire)