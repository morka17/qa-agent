"""
SQLAlchemy 2.0 async ORM models backing the control-plane database:
one row per run, one row per executed step within a run, one row per
filed bug, and one row per evidence artifact. This is the persistent
record `api/routers/runs.py` and `api/routers/bugs.py` query — the
in-memory dataclasses used elsewhere (`orchestration.agent_loop.RunResult`,
`reporting.report_schema.BugReport`, etc.) are the *working* shapes each
module operates on; these models are the *durable* shapes a finished run
gets persisted into.

Mapped with SQLAlchemy's modern `Mapped`/`mapped_column` declarative
style (2.0+) rather than the legacy `Column(...)` style, and async
throughout via `AsyncAttrs` + an async engine/session — see
`get_engine`/`get_sessionmaker` below, which read the connection URL from
`config.settings.Settings.database_url` (an `asyncpg` DSN in production;
tests can point this at `sqlite+aiosqlite:///:memory:` instead).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from qa_agent.config.settings import Settings, get_settings


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(AsyncAttrs, DeclarativeBase):
    """Shared declarative base for every Sentinel-QA table. `storage/migrations/env.py` targets this metadata."""


class RunORM(Base):
    """
    One row per agent run — the durable counterpart of
    `orchestration.agent_loop.RunResult` and
    `orchestration.state_machine.RunStateMachine`. `state` mirrors
    `RunState.value`; kept as a plain string column (not a DB enum) so
    adding a new `RunState` never requires a migration.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    story_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    test_plan_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    target_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list["StepORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="StepORM.step_order"
    )
    bugs: Mapped[list["BugORM"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    artifacts: Mapped[list["ArtifactORM"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_runs_state_created_at", "state", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"<RunORM id={self.id!r} state={self.state!r}>"


class StepORM(Base):
    """One executed step within a run — the durable counterpart of `planning.step_schema.Step` + `execution.action_executor.ExecutionResult` merged into a single row."""

    __tablename__ = "steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    target_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_strategy: Mapped[str | None] = mapped_column(
        String(32), nullable=True, doc="perception.element_resolver.ResolutionStrategy value, if this step resolved a target element."
    )
    resolution_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["RunORM"] = relationship(back_populates="steps")

    __table_args__ = (Index("ix_steps_run_order", "run_id", "step_order", unique=True),)


class BugORM(Base):
    """Durable counterpart of `reporting.report_schema.BugReport`."""

    __tablename__ = "bugs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    expected_behavior: Mapped[str] = mapped_column(Text, nullable=False)
    actual_behavior: Mapped[str] = mapped_column(Text, nullable=False)
    labels: Mapped[list[str]] = mapped_column(JSON, default=list)
    cluster_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    tracker_name: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tracker_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tracker_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    filed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped["RunORM"] = relationship(back_populates="bugs")


class ArtifactORM(Base):
    """Durable counterpart of `reporting.report_schema.EvidenceAttachment` / `execution.recorder.RunArtifacts`."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    path_or_url: Mapped[str] = mapped_column(Text, nullable=False)
    is_uploaded: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped["RunORM"] = relationship(back_populates="artifacts")


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    return create_async_engine(settings.database_url, echo=settings.database_echo_sql, pool_size=settings.database_pool_size)


def get_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_all_tables(engine: AsyncEngine) -> None:
    """
    Convenience for local dev/tests: creates every table directly from
    the ORM metadata, bypassing Alembic. Production deployments should
    use `storage/migrations/` (`alembic upgrade head`) instead — this
    function exists so a fresh SQLite/Postgres instance can be stood up
    in one call for a test fixture without an Alembic dependency.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)