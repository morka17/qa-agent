"""
Read endpoints over filed bugs (`storage.models.BugORM`). Bugs are
written by the `"run_story"` job handler (`api/app.py`) as part of
persisting a completed run, not by this router — this router is
read-only, matching the fact that bug *creation* only ever happens as a
byproduct of an agent run, never as a direct API call.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from qa_agent.api.dependencies import get_db_session
from qa_agent.storage.models import BugORM

router = APIRouter(prefix="/bugs", tags=["bugs"])


class BugSummary(BaseModel):
    id: str
    run_id: str
    title: str
    severity: str
    category: str
    occurrence_count: int
    tracker_name: str | None
    tracker_ref: str | None
    tracker_url: str | None
    created_at: datetime

    @classmethod
    def from_orm_row(cls, row: BugORM) -> "BugSummary":
        return cls(
            id=row.id,
            run_id=row.run_id,
            title=row.title,
            severity=row.severity,
            category=row.category,
            occurrence_count=row.occurrence_count,
            tracker_name=row.tracker_name,
            tracker_ref=row.tracker_ref,
            tracker_url=row.tracker_url,
            created_at=row.created_at,
        )


class BugDetail(BugSummary):
    summary: str
    description: str
    expected_behavior: str
    actual_behavior: str
    labels: list[str]
    cluster_id: str | None
    filed_at: datetime | None

    @classmethod
    def from_orm_row(cls, row: BugORM) -> "BugDetail":
        return cls(
            id=row.id,
            run_id=row.run_id,
            title=row.title,
            severity=row.severity,
            category=row.category,
            occurrence_count=row.occurrence_count,
            tracker_name=row.tracker_name,
            tracker_ref=row.tracker_ref,
            tracker_url=row.tracker_url,
            created_at=row.created_at,
            summary=row.summary,
            description=row.description,
            expected_behavior=row.expected_behavior,
            actual_behavior=row.actual_behavior,
            labels=row.labels,
            cluster_id=row.cluster_id,
            filed_at=row.filed_at,
        )


@router.get("", response_model=list[BugSummary])
async def list_bugs(
    severity: str | None = Query(default=None),
    category: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> list[BugSummary]:
    stmt = select(BugORM).order_by(BugORM.created_at.desc()).offset(offset).limit(limit)
    if severity is not None:
        stmt = stmt.where(BugORM.severity == severity)
    if category is not None:
        stmt = stmt.where(BugORM.category == category)
    result = await session.execute(stmt)
    return [BugSummary.from_orm_row(row) for row in result.scalars().all()]


@router.get("/{bug_id}", response_model=BugDetail)
async def get_bug(bug_id: str, session: AsyncSession = Depends(get_db_session)) -> BugDetail:
    result = await session.execute(select(BugORM).where(BugORM.id == bug_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No bug found with id {bug_id!r}.")
    return BugDetail.from_orm_row(row)