"""
Produces a human-readable root-cause narrative for a failed run by
handing the LLM a structured summary of everything gathered about the
failure — failed assertions, execution errors, console/network evidence,
and the (already computed) rule-based classification — rather than
raw trace/video bytes. This keeps the analysis fast and cheap while
still giving the model everything a human triaging the same failure
would look at first.

This is deliberately a *narrative* layer on top of `failure_classifier.py`,
not a replacement for it: `FailureClassifier` decides the category using
fast, explainable rules; `RootCauseAnalyzer` explains *why*, in prose
suitable for direct inclusion in a filed bug report
(`reporting/bug_report_writer.py` consumes `RootCauseAnalysis.summary`
and `.likely_cause` for exactly that purpose). The analyzer may also
suggest a *refined* category when the evidence, laid out in full, points
somewhere different than the rule scores alone did — this refinement is
advisory, not authoritative; callers decide whether to adopt it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.triage.failure_classifier import (
    ClassificationResult,
    FailureCategory,
    FailureEvidence,
)

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a senior QA engineer writing the root-cause analysis
section of a bug report, based on evidence from an automated browser test run.

Return ONLY a JSON object (no markdown, no prose) with this exact shape:
{
  "summary": "<1-2 sentence plain-language summary of what went wrong>",
  "likely_cause": "<your best-supported hypothesis for the underlying cause>",
  "contributing_factors": ["<factor 1>", "<factor 2>", ...],
  "confidence": <float 0.0-1.0>,
  "suggested_category": "<one of: app_bug, flaky_test, selector_drift, environment_issue, plan_error, unknown>",
  "recommended_action": "<one sentence: what a human should do next>"
}

Guidance:
- Ground every claim in the evidence provided — do not invent details
  (specific error codes, line numbers, causes) that aren't supported by
  what's given.
- "likely_cause" should be specific enough to be actionable (e.g. "the
  checkout API returned a 500 when the cart total exceeded $100" rather
  than "something went wrong with checkout").
- Only diverge from the provided rule-based category in
  "suggested_category" if the evidence clearly supports a different one;
  otherwise repeat it.
- "recommended_action" should be concrete: what to check, reproduce, or
  fix first — not generic advice like "investigate further"."""


class LLMClient(Protocol):
    """Same interface as its siblings in `ingestion.story_parser` / `planning.test_planner`."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True)
class RootCauseAnalysis:
    summary: str
    likely_cause: str
    contributing_factors: list[str]
    confidence: float
    suggested_category: FailureCategory
    recommended_action: str
    raw_llm_output: str


class RootCauseAnalysisError(Exception):
    """Raised when the LLM's output can't be parsed into a well-formed analysis."""


def _parse_llm_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RootCauseAnalysisError(f"LLM response was not valid JSON: {raw!r}") from exc


def _build_evidence_summary(evidence: FailureEvidence, classification: ClassificationResult) -> str:
    lines = [
        f"Rule-based classification: {classification.category.value} "
        f"(confidence={classification.confidence:.2f})",
        f"Classification signals: {classification.signals}",
        "",
        "Failed assertions:",
    ]
    lines += [
        f"  - {a.reason} (expected={a.expected!r}, actual={a.actual!r})"
        for a in evidence.failed_assertions
    ] or ["  (none)"]

    lines += ["", "Failed execution steps:"]
    lines += [
        f"  - step {e.step_id}: {e.error} (after {e.duration_ms:.0f}ms)"
        for e in evidence.failed_executions
    ] or ["  (none)"]

    lines += ["", "Console errors:"]
    lines += [f"  - [{e.source}] {e.text[:200]}" for e in evidence.console_errors] or ["  (none)"]

    lines += ["", "Network failures:"]
    lines += [
        f"  - {e.method} {e.url} -> status={e.status}" for e in evidence.network_failures
    ] or ["  (none)"]

    lines += ["", "Element resolutions used in this run:"]
    lines += [
        f"  - step {step_id}: strategy={r.strategy.value}, confidence={r.confidence:.2f}, "
        f"element=<{r.node.tag} role={r.node.role} name={r.node.accessible_name!r}>"
        for step_id, r in evidence.resolved_elements.items()
    ] or ["  (none)"]

    if evidence.historical_pass_rate is not None:
        lines.append(f"\nHistorical pass rate for this assertion/step: {evidence.historical_pass_rate:.0%}")

    return "\n".join(lines)


class RootCauseAnalyzer:
    """
    Example:
        analyzer = RootCauseAnalyzer(llm_client=my_llm_client)
        rca = await analyzer.analyze(evidence, classification)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def analyze(
        self, evidence: FailureEvidence, classification: ClassificationResult
    ) -> RootCauseAnalysis:
        user_prompt = _build_evidence_summary(evidence, classification)

        raw = await self._llm_client.complete(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)
        parsed = _parse_llm_json(raw)

        try:
            suggested_category = FailureCategory(
                parsed.get("suggested_category", classification.category.value)
            )
        except ValueError:
            logger.warning(
                "RCA returned an unrecognized suggested_category %r; falling back to "
                "the rule-based classification.",
                parsed.get("suggested_category"),
            )
            suggested_category = classification.category

        try:
            analysis = RootCauseAnalysis(
                summary=parsed["summary"],
                likely_cause=parsed["likely_cause"],
                contributing_factors=list(parsed.get("contributing_factors", [])),
                confidence=float(parsed.get("confidence", classification.confidence)),
                suggested_category=suggested_category,
                recommended_action=parsed.get("recommended_action", "Reproduce manually and inspect."),
                raw_llm_output=raw,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RootCauseAnalysisError(f"Malformed RCA JSON from LLM: {exc}") from exc

        logger.info(
            "Root cause analysis complete: %s (suggested_category=%s, confidence=%.2f).",
            analysis.summary,
            analysis.suggested_category.value,
            analysis.confidence,
        )
        return analysis