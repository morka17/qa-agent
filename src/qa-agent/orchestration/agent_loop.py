"""
The top-level orchestrator: takes a `UserStory`, drives it through every
stage of the pipeline — ingestion, planning, execution, verification,
triage, and reporting — and returns a `RunResult`. Every other module in
`qa_agent` is a collaborator this class composes; nothing here does its
own DOM work, its own LLM calls, or its own HTTP requests — it delegates
to the module that owns that concern and is responsible only for
sequencing, state tracking, and error boundaries between stages.

The state machine (`state_machine.py`) is the source of truth for where
a run is; every stage transition happens explicitly so a run can never
silently skip a phase or get stuck.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from qa_agent.config.logging_config import bind_run_context, clear_run_context, get_logger
from qa_agent.execution.action_executor import ActionExecutor, ExecutionResult
from qa_agent.execution.browser_manager import BrowserManager
from qa_agent.execution.network_interceptor import NetworkInterceptor
from qa_agent.execution.recorder import Recorder, RunArtifacts
from qa_agent.ingestion.schemas import UserStory
from qa_agent.ingestion.story_parser import StoryParser
from qa_agent.perception.dom_snapshot import DomSnapshotExtractor
from qa_agent.perception.element_resolver import ElementResolver
from qa_agent.planning.plan_validator import PlanValidationError
from qa_agent.planning.step_schema import TestPlan
from qa_agent.planning.test_planner import ApprovalRequiredError, TestPlanner
from qa_agent.reporting.bug_report_writer import BugReportWriter
from qa_agent.reporting.evidence_bundler import EvidenceBundler
from qa_agent.reporting.report_schema import BugReport, IssueTrackerFiler
from qa_agent.orchestration.state_machine import RunState, RunStateMachine
from qa_agent.triage.dedupe_engine import DedupeEngine, build_fingerprint
from qa_agent.triage.failure_classifier import FailureClassifier, FailureEvidence
from qa_agent.triage.root_cause_analyzer import RootCauseAnalyzer
from qa_agent.verification.assertion_engine import AssertionEngine
from qa_agent.verification.console_error_monitor import ConsoleErrorMonitor

logger = get_logger(__name__)


class PageLike(Protocol):
    """The composite page surface this loop's collaborators need — Playwright's real `Page` satisfies this structurally."""

    url: str

    async def title(self) -> str: ...
    async def evaluate(self, expression: str) -> object: ...
    async def screenshot(self, **kwargs: object) -> bytes: ...
    async def goto(self, url: str, **kwargs: object) -> object: ...

    def locator(self, selector: str) -> object: ...
    def on(self, event: str, handler: object) -> None: ...
    def remove_listener(self, event: str, handler: object) -> None: ...


@dataclass
class RunResult:
    run_id: str
    final_state: RunState
    plan: TestPlan | None = None
    execution_results: list[ExecutionResult] = field(default_factory=list)
    bug_report: BugReport | None = None
    artifacts: RunArtifacts | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.final_state == RunState.PASSED


class AgentLoopError(Exception):
    """Raised for unrecoverable errors that don't map to a normal terminal run state."""


class AgentLoop:
    """
    Composes every pipeline stage. All collaborators are injected so the
    loop itself has zero knowledge of which LLM provider, which browser,
    or which issue tracker is actually configured — see
    `docs/architecture.md` for the full dependency wiring used in
    production (`api/app.py`'s startup builds one `AgentLoop` per worker
    from `config.settings`).

    Example:
        loop = AgentLoop(
            browser_manager=browser_manager,
            story_parser=story_parser,
            test_planner=test_planner,
            element_resolver=element_resolver,
            bug_report_writer=bug_report_writer,
            issue_tracker=jira_filer,
        )
        result = await loop.run(story)
    """

    def __init__(
        self,
        browser_manager: BrowserManager,
        story_parser: StoryParser,
        test_planner: TestPlanner,
        element_resolver: ElementResolver,
        bug_report_writer: BugReportWriter,
        issue_tracker: IssueTrackerFiler | None = None,
        classifier: FailureClassifier | None = None,
        rca_analyzer: RootCauseAnalyzer | None = None,
        dedupe_engine: DedupeEngine | None = None,
        evidence_bundler: EvidenceBundler | None = None,
    ) -> None:
        self._browser_manager = browser_manager
        self._story_parser = story_parser
        self._test_planner = test_planner
        self._element_resolver = element_resolver
        self._bug_report_writer = bug_report_writer
        self._issue_tracker = issue_tracker
        self._classifier = classifier or FailureClassifier()
        self._rca_analyzer = rca_analyzer
        self._dedupe_engine = dedupe_engine or DedupeEngine()
        self._evidence_bundler = evidence_bundler or EvidenceBundler()
        self._snapshot_extractor = DomSnapshotExtractor()

    async def run(self, story: UserStory) -> RunResult:
        run_id = str(uuid.uuid4())
        bind_run_context(run_id=run_id, story_id=story.id)
        machine = RunStateMachine(run_id=run_id)
        recorder = Recorder(run_id=run_id)

        try:
            result = await self._run_pipeline(story, run_id, machine, recorder)
            return result
        except Exception as exc:  # noqa: BLE001 - any unhandled stage failure still produces a well-formed RunResult
            logger.error("Run %s failed with an unhandled error: %s", run_id, exc, exc_info=True)
            if not machine.is_terminal:
                machine.transition(RunState.ERROR, reason=str(exc))
            return RunResult(run_id=run_id, final_state=machine.state, error=str(exc))
        finally:
            clear_run_context()

    async def _run_pipeline(
        self, story: UserStory, run_id: str, machine: RunStateMachine, recorder: Recorder
    ) -> RunResult:
        # --- Ingestion ---
        machine.transition(RunState.INGESTING)
        intent = await self._story_parser.parse(story)

        # --- Planning ---
        machine.transition(RunState.PLANNING)
        try:
            plan = await self._test_planner.plan(intent)
        except ApprovalRequiredError as exc:
            machine.transition(RunState.AWAITING_APPROVAL, reason=str(exc))
            logger.warning("Run %s paused for human approval: %s", run_id, exc)
            return RunResult(run_id=run_id, final_state=machine.state, plan=exc.plan)
        except PlanValidationError as exc:
            machine.transition(RunState.ERROR, reason=str(exc))
            return RunResult(run_id=run_id, final_state=machine.state, error=str(exc))

        # --- Execution + Verification (share one browser context/page) ---
        async with self._browser_manager.new_context(
            record_video_dir=str(recorder.run_dir) if self._browser_manager.is_running else None
        ) as context:
            await recorder.start_tracing(context)
            page = await self._browser_manager.new_page(context)

            console_monitor = ConsoleErrorMonitor()
            console_monitor.attach(page)  # type: ignore[arg-type]
            network_interceptor = NetworkInterceptor()
            await network_interceptor.attach(page)  # type: ignore[arg-type]

            action_executor = ActionExecutor(
                resolver=self._element_resolver, snapshot_extractor=self._snapshot_extractor
            )
            assertion_engine = AssertionEngine(resolver=self._element_resolver)

            machine.transition(RunState.EXECUTING)
            execution_results, resolved_elements = await self._execute_plan(
                page, plan, action_executor  # type: ignore[arg-type]
            )

            machine.transition(RunState.VERIFYING)
            snapshot = await self._snapshot_extractor.capture(page)  # type: ignore[arg-type]
            assertion_results = []
            if all(r.success for r in execution_results):
                for step in plan.steps_in_order():
                    for assertion in plan.assertions_after(step.id):
                        assertion_results.append(
                            await assertion_engine.evaluate(page, snapshot, assertion)  # type: ignore[arg-type]
                        )
            else:
                await recorder.capture_screenshot(page, label="execution_failure")  # type: ignore[arg-type]

            if any(not a.passed for a in assertion_results):
                await recorder.capture_screenshot(page, label="assertion_failure")  # type: ignore[arg-type]

            console_monitor.detach(page)  # type: ignore[arg-type]
            await network_interceptor.detach()
            artifacts = await recorder.finalize(context, page)  # type: ignore[arg-type]

        evidence = FailureEvidence(
            assertion_results=assertion_results,
            execution_results=execution_results,
            console_errors=console_monitor.errors(),
            network_failures=network_interceptor.failed_exchanges(),
            resolved_elements=resolved_elements,
        )

        if not evidence.has_any_failure():
            machine.transition(RunState.PASSED)
            logger.info("Run %s passed.", run_id)
            return RunResult(
                run_id=run_id,
                final_state=machine.state,
                plan=plan,
                execution_results=execution_results,
                artifacts=artifacts,
            )

        # --- Triage ---
        machine.transition(RunState.TRIAGING)
        classification = await self._classifier.classify(evidence)

        rca = None
        if self._rca_analyzer is not None:
            rca = await self._rca_analyzer.analyze(evidence, classification)

        cluster = None
        if plan.target_url:
            primary_error = self._primary_error_text(evidence)
            fingerprint = build_fingerprint(
                target_key=plan.target_url,
                category=classification.category,
                step_description=plan.steps_in_order()[0].description if plan.steps else "",
                assertion_description=evidence.failed_assertions[0].reason
                if evidence.failed_assertions
                else None,
                error_text=primary_error,
            )
            cluster, _is_new = await self._dedupe_engine.find_or_create_cluster(fingerprint, run_id)

        # --- Reporting ---
        machine.transition(RunState.REPORTING)
        evidence_attachments = await self._evidence_bundler.bundle(
            artifacts=artifacts,
            console_errors=evidence.console_errors,
            network_failures=evidence.network_failures,
        )

        bug_report = None
        if rca is not None:
            bug_report = await self._bug_report_writer.write(
                run_id=run_id,
                intent=intent,
                plan=plan,
                evidence=evidence,
                classification=classification,
                rca=rca,
                evidence_attachments=evidence_attachments,
                cluster=cluster,
            )

            if self._issue_tracker is not None:
                if cluster is not None and cluster.filed_bug_ref:
                    await self._issue_tracker.add_occurrence_comment(
                        bug_report, occurrence_count=cluster.occurrence_count
                    )
                    bug_report = bug_report.model_copy(
                        update={"tracker_ref": cluster.filed_bug_ref}
                    )
                else:
                    bug_report = await self._issue_tracker.file(bug_report)
                    if cluster is not None and bug_report.tracker_ref:
                        await self._dedupe_engine.mark_filed(
                            plan.target_url or "", cluster.id, bug_report.tracker_ref
                        )

        machine.transition(RunState.FAILED)
        logger.info("Run %s failed (category=%s).", run_id, classification.category.value)
        return RunResult(
            run_id=run_id,
            final_state=machine.state,
            plan=plan,
            execution_results=execution_results,
            bug_report=bug_report,
            artifacts=artifacts,
        )

    async def _execute_plan(
        self, page: PageLike, plan: TestPlan, action_executor: ActionExecutor
    ) -> tuple[list[ExecutionResult], dict[str, object]]:
        results: list[ExecutionResult] = []
        resolved_elements: dict[str, object] = {}

        for step in plan.steps_in_order():
            snapshot = await self._snapshot_extractor.capture(page)  # type: ignore[arg-type]
            result = await action_executor.execute_step(page, step, snapshot)  # type: ignore[arg-type]
            results.append(result)
            if result.resolved_element is not None:
                resolved_elements[step.id] = result.resolved_element
            if not result.success:
                logger.warning(
                    "Step %s failed; halting remaining plan execution.", step.id
                )
                break

        return results, resolved_elements

    @staticmethod
    def _primary_error_text(evidence: FailureEvidence) -> str:
        if evidence.failed_executions:
            return evidence.failed_executions[0].error or "unknown execution error"
        if evidence.failed_assertions:
            a = evidence.failed_assertions[0]
            return f"{a.reason} (expected={a.expected!r}, actual={a.actual!r})"
        return "unknown failure"