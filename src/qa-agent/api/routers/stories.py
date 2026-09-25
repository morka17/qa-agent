"""
Story intake endpoints. Stories themselves aren't persisted to the
database (there's no `StoryORM` — only runs/steps/bugs/artifacts are
durable, per `storage/models.py`); this router validates and normalizes
incoming story submissions into `ingestion.schemas.UserStory`, which the
caller then passes to `POST /runs` to actually execute. Keeping "define
a story" and "run a story" as separate calls lets a UI show the parsed
story back to a user for confirmation before committing to a run.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from qa_agent.ingestion.schemas import Priority, StorySource, UserStory

router = APIRouter(prefix="/stories", tags=["stories"])


class StorySubmission(BaseModel):
    title: str = Field(..., min_length=1)
    narrative: str = Field(..., min_length=1)
    priority: Priority = Priority.MEDIUM
    labels: list[str] = Field(default_factory=list)
    target_url: str | None = None


class StoryResponse(BaseModel):
    id: str
    external_id: str | None
    source: StorySource
    title: str
    narrative: str
    priority: Priority
    labels: list[str]
    target_url: str | None

    @classmethod
    def from_domain(cls, story: UserStory) -> "StoryResponse":
        return cls(
            id=story.id,
            external_id=story.external_id,
            source=story.source,
            title=story.title,
            narrative=story.narrative,
            priority=story.priority,
            labels=story.labels,
            target_url=story.target_url,
        )


@router.post("", response_model=StoryResponse, status_code=201)
async def submit_story(submission: StorySubmission) -> StoryResponse:
    """
    Validates and normalizes a story submission. The returned `id` is
    what `POST /runs` expects as `story_id` — the caller is responsible
    for holding onto the full story payload (or re-submitting it) since
    it isn't persisted server-side by this endpoint alone.
    """
    try:
        story = UserStory(
            source=StorySource.API,
            title=submission.title,
            narrative=submission.narrative,
            priority=submission.priority,
            labels=submission.labels,
            target_url=submission.target_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return StoryResponse.from_domain(story)