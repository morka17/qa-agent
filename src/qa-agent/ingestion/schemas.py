"""
Core data contracts for the ingestion layer.

Every connector (Jira, Linear, CSV, ...) and every downstream stage
(planning, execution, reporting) speaks these schemas — they are the
stable interface that decouples "where a story came from" from
"what the agent does with it".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StorySource(str, Enum):
    JIRA = "jira"
    LINEAR = "linear"
    CSV = "csv"
    MANUAL = "manual"
    API = "api"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GherkinKeyword(str, Enum):
    GIVEN = "given"
    WHEN = "when"
    THEN = "then"
    AND = "and"
    BUT = "but"


class AcceptanceCriterion(BaseModel):
    """
    A single structured Given/When/Then clause extracted from a story's
    acceptance criteria. A story typically has several of these, forming
    one or more full scenarios.
    """

    id: str = Field(default_factory=_new_id)
    scenario_name: str | None = Field(
        default=None, description="Optional scenario grouping, e.g. 'Successful checkout'."
    )
    keyword: GherkinKeyword
    text: str = Field(..., min_length=1, description="The clause text, without the leading keyword.")
    order: int = Field(..., ge=0, description="Position within its scenario, 0-indexed.")

    def as_gherkin_line(self) -> str:
        return f"{self.keyword.value.capitalize()} {self.text}"


class UserStory(BaseModel):
    """
    A normalized user story, regardless of which system it originated from.
    This is the single object every ingestion connector must produce.
    """

    id: str = Field(default_factory=_new_id)
    external_id: str | None = Field(
        default=None, description="Source-system ID, e.g. Jira key 'PROJ-123' or Linear issue ID."
    )
    source: StorySource
    title: str = Field(..., min_length=1)
    narrative: str = Field(
        ..., description="Raw 'As a ... I want ... so that ...' text or free-form description."
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    labels: list[str] = Field(default_factory=list)
    target_url: str | None = Field(
        default=None, description="Base URL of the app under test this story applies to."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Source-specific extra fields preserved for traceability."
    )
    ingested_at: datetime = Field(default_factory=_utcnow)

    @field_validator("narrative")
    @classmethod
    def _narrative_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative must not be blank")
        return v.strip()


class TestIntentKind(str, Enum):
    FUNCTIONAL = "functional"
    REGRESSION = "regression"
    SMOKE = "smoke"
    NEGATIVE = "negative"  # intentionally exercising error/edge paths


class TestIntent(BaseModel):
    """
    The structured, agent-actionable distillation of a UserStory: what
    persona is acting, what goal they're pursuing, and what conditions
    define success/failure. This is the direct input to the planning
    stage's step-graph generator — it does not itself contain execution
    steps, only intent.
    """

    id: str = Field(default_factory=_new_id)
    story_id: str = Field(..., description="Foreign key back to the originating UserStory.id")
    kind: TestIntentKind = TestIntentKind.FUNCTIONAL
    persona: str = Field(..., description="Who is acting, e.g. 'a logged-in customer'.")
    goal: str = Field(..., description="What the persona is trying to accomplish.")
    rationale: str | None = Field(
        default=None, description="The 'so that ...' clause — why the goal matters."
    )
    preconditions: list[str] = Field(
        default_factory=list, description="State that must hold before the test begins."
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Observable, checkable conditions that define a passing outcome.",
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    target_url: str | None = None
    priority: Priority = Priority.MEDIUM
    created_at: datetime = Field(default_factory=_utcnow)

    def has_explicit_criteria(self) -> bool:
        return len(self.acceptance_criteria) > 0