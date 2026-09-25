"""
FastAPI dependency providers: the wiring layer between the control-plane
API and every collaborator (DB session, task queue, agent loop) it
needs. Kept as `Depends()`-yielding functions rather than global
singletons so tests can override any of them via
`app.dependency_overrides[...]` without needing a real database, real
browser, or real LLM provider — see `api/app.py`'s test suite for the
pattern this enables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from qa_agent.config.settings import Settings, get_settings
from qa_agent.execution.browser_manager import BrowserManager
from qa_agent.ingestion.story_parser import StoryParser
from qa_agent.llm.cost_tracker import CostTracker
from qa_agent.llm.provider_router import LLMRouter
from qa_agent.orchestration.agent_loop import AgentLoop
from qa_agent.orchestration.task_queue import InMemoryTaskQueue, TaskQueue
from qa_agent.perception.element_resolver import ElementResolver
from qa_agent.planning.test_planner import TestPlanner
from qa_agent.reporting.bug_report_writer import BugReportWriter
from qa_agent.storage.models import get_engine, get_sessionmaker
from qa_agent.triage.dedupe_engine import DedupeEngine
from qa_agent.triage.failure_classifier import FailureClassifier
from qa_agent.triage.root_cause_analyzer import RootCauseAnalyzer


def get_settings_dep() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _get_engine():
    return get_engine()


@lru_cache(maxsize=1)
def _get_sessionmaker():
    return get_sessionmaker(_get_engine())


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yields one session per request, always closed afterward regardless of whether the request raised."""
    sessionmaker = _get_sessionmaker()
    async with sessionmaker() as session:
        yield session


@lru_cache(maxsize=1)
def get_cost_tracker() -> CostTracker:
    """No per-run budget cap by default; Settings does not currently expose one — pass `budget_usd_per_run` here directly if a deployment wants one enforced."""
    return CostTracker(budget_usd_per_run=None)


@lru_cache(maxsize=1)
def get_llm_router() -> LLMRouter:
    return LLMRouter(cost_tracker=get_cost_tracker())


@lru_cache(maxsize=1)
def get_task_queue() -> TaskQueue:
    """
    Default is the in-process queue; a deployment that wants Celery or
    Temporal overrides this dependency at app-startup (see `app.py`)
    rather than changing this function, since the choice is
    infrastructure-specific, not application logic.
    """
    return InMemoryTaskQueue(max_concurrent_jobs=5)


@lru_cache(maxsize=1)
def get_browser_manager() -> BrowserManager:
    return BrowserManager()


@lru_cache(maxsize=1)
def get_dedupe_engine() -> DedupeEngine:
    return DedupeEngine()


def get_agent_loop(
    llm_router: LLMRouter = Depends(get_llm_router),
    browser_manager: BrowserManager = Depends(get_browser_manager),
    dedupe_engine: DedupeEngine = Depends(get_dedupe_engine),
) -> AgentLoop:
    """
    Builds one `AgentLoop` per request from shared, cached collaborators.
    The loop object itself is cheap to construct (it just holds
    references), so it is not itself cached — only its expensive
    collaborators (browser manager, LLM router) are.
    """
    return AgentLoop(
        browser_manager=browser_manager,
        story_parser=StoryParser(llm_client=llm_router),
        test_planner=TestPlanner(llm_client=llm_router),
        element_resolver=ElementResolver(vision_client=llm_router),
        bug_report_writer=BugReportWriter(llm_client=llm_router),
        issue_tracker=None,  
        classifier=FailureClassifier(llm_client=llm_router),
        rca_analyzer=RootCauseAnalyzer(llm_client=llm_router),
        dedupe_engine=dedupe_engine,
    )