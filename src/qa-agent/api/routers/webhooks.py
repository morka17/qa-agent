"""
CI-triggered run entrypoint — the HTTP counterpart of
`orchestration.run_scheduler.RunScheduler.trigger_now`. A CI pipeline
(post-deploy step) or generic webhook sender POSTs here with one or more
stories to run against the just-deployed environment; each becomes its
own enqueued job, same as `POST /runs`.

No webhook-signature verification is implemented here — a production
deployment behind a CI provider (GitHub Actions, GitLab CI, etc.) should
add HMAC signature verification specific to that provider before this
handler is reachable from the public internet. Left out here to keep
this module provider-agnostic; add it as a dependency in `app.py`'s
router inclusion if/when a specific CI provider is targeted.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from qa_agent.api.dependencies import get_task_queue
from qa_agent.ingestion.schemas import Priority
from qa_agent.orchestration.task_queue import TaskQueue

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class CITriggeredStory(BaseModel):
    title: str = Field(..., min_length=1)
    narrative: str = Field(..., min_length=1)
    priority: Priority = Priority.HIGH  # CI-triggered smoke/regression runs default to HIGH
    target_url: str | None = None


class CIWebhookPayload(BaseModel):
    deployment_id: str | None = None
    environment: str | None = None
    stories: list[CITriggeredStory] = Field(..., min_length=1)


class CIWebhookResponse(BaseModel):
    triggered_job_ids: list[str]


@router.post("/ci", response_model=CIWebhookResponse, status_code=202)
async def ci_triggered_run(
    payload: CIWebhookPayload, task_queue: TaskQueue = Depends(get_task_queue)
) -> CIWebhookResponse:
    """Enqueues one run per story in the payload, e.g. a post-deploy smoke-test suite."""
    job_ids = []
    for story in payload.stories:
        job_id = await task_queue.enqueue(
            "run_story",
            {
                "title": story.title,
                "narrative": story.narrative,
                "priority": story.priority.value,
                "target_url": story.target_url,
                "deployment_id": payload.deployment_id,
                "environment": payload.environment,
            },
        )
        job_ids.append(job_id)
    return CIWebhookResponse(triggered_job_ids=job_ids)