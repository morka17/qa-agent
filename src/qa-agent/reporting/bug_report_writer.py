"""
Assembles the final `BugReport` from everything upstream has already
produced: the test plan (for repro steps), the failure evidence and its
classification (`triage/failure_classifier.py`), the root-cause narrative
(`triage/root_cause_analyzer.py`), and the bundled evidence attachments
(`evidence_bundler.py`).

The LLM's job here is narrow and specific: write a clear, human-readable
title and polish the expected/actual behavior description. It is
deliberately NOT asked to re-derive the category, severity, or repro
steps — those come from structured upstream data (the classifier, the
plan, settings-driven severity rules) precisely because that data is
more reliable than an LLM re-inferring it from scratch, consistent with
the project's "evidence-first" and "no invented facts" principles.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.ingestion.schemas import Priority, TestIntent
from qa_agent.planning.step_schema import TestPlan
from qa_agent.reporting.report_schema import (
    BugReport,
    EvidenceAttachment,
    ReproStep,
    Severity,
)
from qa_agent.triage.dedupe_engine import FailureCluster
from qa_agent.triage.failure_classifier import (
    ClassificationResult,
    FailureCategory,
    FailureEvidence,
)
from qa_agent.triage.root_cause_analyzer import RootCauseAnalysis

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You write the title and behavior description for an automated
QA bug report. Return ONLY a JSON object (no markdown, no prose):

{
  "title": "<a specific, scannable bug title under 100 characters>",
  "expected_behavior": "<1-2 sentences: what should have happened>",
  "actual_behavior": "<1-2 sentences: what actually happened>",
  "description": "<a short markdown-formatted narrative, 2-4 sentences, giving a "
                  "human reviewer enough context to understand the bug without "
                  "reading the raw evidence>"
}

Rules:
- The title must be specific and actionable, e.g. "Checkout fails with 500 error
  when cart total exceeds $100" — never generic like "Test failed" or "Bug in checkout".
- Do not invent details not present in the evidence provided.
- If the failure category is flaky_test, selector_drift, or environment_issue,
  reflect that framing honestly in the description rather than overstating it as
  a confirmed application defect."""

# Severity is driven primarily by the story's own stated priority and
# secondarily by the failure category — a confirmed app_bug on a
# critical-priority story is CRITICAL; the same category on a low-priority
# story is only MEDIUM. Categories that aren't confirmed application
# defects (flaky/selector-drift/environment) are capped well below what
# their nominal priority would otherwise suggest, since filing those as
# high-severity tickets trains humans to ignore the tracker.
_PRIORITY_BASE_SEVERITY: dict[Priority, Severity] = {
    Priority.CRITICAL: Severity.CRITICAL,
    Priority.HIGH: Severity.HIGH,
    Priority.MEDIUM: Severity.MEDIUM,
    Priority.LOW: Severity.LOW,
}

_NON_DEFECT_CATEGORY_SEVERITY_CAP: dict[FailureCategory, Severity] = {
    FailureCategory.FLAKY_TEST: Severity.LOW,
    FailureCategory.SELECTOR_DRIFT: Severity.INFO,
    FailureCategory.ENVIRONMENT_ISSUE: Severity.LOW,
    FailureCategory.PLAN_ERROR: Severity.INFO,
    FailureCategory.UNKNOWN: Severity.LOW,
}

_SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


class LLMClient(Protocol):
    """Same interface as its siblings elsewhere in the codebase."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class BugReportWritingError(Exception):
    """Raised when the LLM's title/description output can't be parsed."""


def _parse_llm_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise BugReportWritingError(f"LLM response was not valid JSON: {raw!r}") from exc


def determine_severity(category: FailureCategory, priority: Priority) -> Severity:
    """
    Combines the story's stated priority with the failure category to
    pick a final severity, capping non-confirmed-defect categories so
    they never outrank an actual application bug of equal or lower
    story priority.
    """
    base = _PRIORITY_BASE_SEVERITY.get(priority, Severity.MEDIUM)
    cap = _NON_DEFECT_CATEGORY_SEVERITY_CAP.get(category)
    if cap is None:
        return base  # app_bug: use the story's priority-derived severity as-is
    base_rank = _SEVERITY_ORDER.index(base)
    cap_rank = _SEVERITY_ORDER.index(cap)
    return _SEVERITY_ORDER[min(base_rank, cap_rank)]


def _build_repro_steps(plan: TestPlan) -> list[ReproStep]:
    return [
        ReproStep(order=step.order, description=step.description) for step in plan.steps_in_order()
    ]


def _build_evidence_summary_prompt(
    evidence: FailureEvidence,
    classification: ClassificationResult,
    rca: RootCauseAnalysis,
    intent: TestIntent,
) -> str:
    lines = [
        f"Story goal: {intent.persona} wants to {intent.goal}",
        f"Failure category: {classification.category.value} (confidence={classification.confidence:.2f})",
        f"Root cause summary: {rca.summary}",
        f"Likely cause: {rca.likely_cause}",
        f"Contributing factors: {rca.contributing_factors}",
        "",
        "Failed assertions:",
    ]
    lines += [
        f"  - {a.reason} (expected={a.expected!r}, actual={a.actual!r})"
        for a in evidence.failed_assertions
    ] or ["  (none)"]
    lines += ["", "Failed execution errors:"]
    lines += [f"  - {e.error}" for e in evidence.failed_executions] or ["  (none)"]
    return "\n".join(lines)


class BugReportWriter:
    """
    Example:
        writer = BugReportWriter(llm_client=my_llm_client)
        report = await writer.write(
            run_id=run_id, intent=intent, plan=plan, evidence=evidence,
            classification=classification, rca=rca, evidence_attachments=attachments,
        )
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def write(
        self,
        run_id: str,
        intent: TestIntent,
        plan: TestPlan,
        evidence: FailureEvidence,
        classification: ClassificationResult,
        rca: RootCauseAnalysis,
        evidence_attachments: list[EvidenceAttachment],
        cluster: FailureCluster | None = None,
    ) -> BugReport:
        user_prompt = _build_evidence_summary_prompt(evidence, classification, rca, intent)
        raw = await self._llm_client.complete(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)
        parsed = _parse_llm_json(raw)

        try:
            title = parsed["title"]
            expected_behavior = parsed["expected_behavior"]
            actual_behavior = parsed["actual_behavior"]
            description = parsed["description"]
        except KeyError as exc:
            raise BugReportWritingError(f"LLM output missing required field: {exc}") from exc

        severity = determine_severity(classification.category, intent.priority)

        report = BugReport(
            title=title[:200],
            severity=severity,
            category=classification.category,
            summary=rca.summary,
            description=description,
            expected_behavior=expected_behavior,
            actual_behavior=actual_behavior,
            repro_steps=_build_repro_steps(plan),
            evidence=evidence_attachments,
            labels=self._build_labels(classification.category, severity),
            run_id=run_id,
            target_url=plan.target_url,
            story_id=intent.story_id,
            test_plan_id=plan.id,
            cluster_id=cluster.id if cluster else None,
            occurrence_count=cluster.occurrence_count if cluster else 1,
        )

        logger.info(
            "Wrote bug report %r (severity=%s, category=%s) for run %s.",
            report.title,
            severity.value,
            classification.category.value,
            run_id,
        )
        return report

    @staticmethod
    def _build_labels(category: FailureCategory, severity: Severity) -> list[str]:
        return ["sentinel-qa", f"category:{category.value}", f"severity:{severity.value}"]