"""
Classifies a failed run into one of a small set of root categories:
a genuine application bug, a flaky test, selector drift (the app's
markup changed enough that perception's confidence was low), an
environment issue (third-party/infra failure unrelated to the app under
test), a plan-validation problem, or unknown.

Classification runs a deterministic, weighted rule engine first — cheap,
fast, and fully explainable, which matters because this classification
gates whether a bug actually gets filed. An optional LLM fallback (same
`Protocol`-based provider abstraction used throughout the codebase) is
only consulted when the rules produce a low-confidence or ambiguous
result, consistent with the project's "cheap heuristics first" principle
from `docs/adr/0002-selector-strategy.md`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.execution.action_executor import ExecutionResult
from qa_agent.execution.network_interceptor import RecordedExchange
from qa_agent.perception.element_resolver import ResolutionStrategy, ResolvedElement
from qa_agent.verification.assertion_engine import AssertionResult
from qa_agent.verification.console_error_monitor import RecordedConsoleEntry

logger = get_logger(__name__)


class FailureCategory(str, Enum):
    APP_BUG = "app_bug"
    FLAKY_TEST = "flaky_test"
    SELECTOR_DRIFT = "selector_drift"
    ENVIRONMENT_ISSUE = "environment_issue"
    PLAN_ERROR = "plan_error"
    UNKNOWN = "unknown"


@dataclass
class FailureEvidence:
    """
    Everything gathered about one failed run, handed to the classifier
    (and, downstream, to `RootCauseAnalyzer` and `DedupeEngine`) as a
    single bundle so none of those modules need to know how to reach
    into execution/verification internals themselves.
    """

    assertion_results: list[AssertionResult] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    console_errors: list[RecordedConsoleEntry] = field(default_factory=list)
    network_failures: list[RecordedExchange] = field(default_factory=list)
    resolved_elements: dict[str, ResolvedElement] = field(default_factory=dict)  # step_id -> resolution
    historical_pass_rate: float | None = field(
        default=None,
        metadata={
            "doc": "Fraction (0.0-1.0) of prior runs of this same assertion/step that "
            "passed, if the caller has that history available (typically from "
            "DedupeEngine's cluster stats). A mid-range value (neither ~0 nor ~1) "
            "is the strongest single signal of flakiness this classifier has access to."
        },
    )

    @property
    def failed_assertions(self) -> list[AssertionResult]:
        return [a for a in self.assertion_results if not a.passed]

    @property
    def failed_executions(self) -> list[ExecutionResult]:
        return [e for e in self.execution_results if not e.success]

    def has_any_failure(self) -> bool:
        return bool(self.failed_assertions or self.failed_executions)


@dataclass(frozen=True)
class ClassificationResult:
    category: FailureCategory
    confidence: float  # 0.0-1.0
    reasoning: str
    signals: list[str]
    scores: dict[FailureCategory, float]


# Weight each rule contributes to its target category. Kept as named
# constants (rather than inlined numbers) so tuning classification
# behavior is a one-line change, not a hunt through the rule bodies.
_WEIGHT_SELECTOR_DRIFT_RESOLUTION_ERROR = 3.0
_WEIGHT_SELECTOR_DRIFT_LOW_CONFIDENCE_VISION = 1.5
_WEIGHT_APP_BUG_SERVER_ERROR = 2.5
_WEIGHT_APP_BUG_CONSOLE_ERROR = 1.5
_WEIGHT_APP_BUG_BASELINE_ASSERTION_FAILURE = 1.0
_WEIGHT_ENVIRONMENT_THIRD_PARTY_FAILURE = 2.0
_WEIGHT_FLAKY_HISTORICAL_MID_PASS_RATE = 3.0
_WEIGHT_PLAN_ERROR_NO_TARGET_DESCRIPTION = 2.0

# A resolved element's confidence below this, even on a nominally
# "successful" resolution, is worth treating as selector-drift evidence —
# the agent got *an* element, but wasn't very sure it was the right one.
_LOW_CONFIDENCE_THRESHOLD = 0.75

# Below this score margin between the top two categories, treat the
# result as ambiguous and defer to the LLM fallback (if configured)
# rather than reporting a confident-sounding but arbitrary tie-break.
_AMBIGUITY_MARGIN = 0.15

_ELEMENT_RESOLUTION_ERROR_PATTERN = re.compile(
    r"could not (?:be )?resolv|elementresolutionerror|not attached to the dom",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """You are a senior QA engineer triaging a failed automated test run.
Given a summary of the failure evidence, classify it into exactly one of:
app_bug, flaky_test, selector_drift, environment_issue, plan_error, unknown.

Return ONLY a JSON object (no markdown, no prose):
{
  "category": "<one of the categories above>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one or two sentences explaining the classification>"
}

Guidance:
- app_bug: the application itself behaved incorrectly (wrong data, broken
  flow, backend error) despite the agent correctly interacting with it.
- flaky_test: evidence suggests non-deterministic timing/state rather than
  a real defect (e.g. this assertion/step has a mixed pass/fail history).
- selector_drift: the agent struggled to reliably locate the right
  element (low-confidence resolution, resolution errors) — likely a
  markup change, not an app defect.
- environment_issue: a third-party/infrastructure dependency failed, not
  the application under test itself.
- plan_error: the test plan itself is malformed or targets something
  that doesn't correspond to real UI.
- unknown: insufficient evidence to confidently pick another category."""


class LLMClient(Protocol):
    """Same interface as `ingestion.story_parser.LLMClient` and its siblings."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class FailureClassificationError(Exception):
    """Raised only for unrecoverable errors (e.g. malformed LLM fallback output) — a low-confidence UNKNOWN result is a normal, non-error outcome."""


def _parse_llm_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise FailureClassificationError(f"LLM fallback response was not valid JSON: {raw!r}") from exc


class FailureClassifier:
    """
    Example:
        classifier = FailureClassifier(llm_client=my_llm_client)
        result = await classifier.classify(evidence)
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    async def classify(self, evidence: FailureEvidence) -> ClassificationResult:
        if not evidence.has_any_failure():
            return ClassificationResult(
                category=FailureCategory.UNKNOWN,
                confidence=0.0,
                reasoning="No failed assertions or execution results in the provided evidence.",
                signals=[],
                scores={},
            )

        scores: dict[FailureCategory, float] = {}
        signals: list[str] = []

        self._score_selector_drift(evidence, scores, signals)
        self._score_app_bug(evidence, scores, signals)
        self._score_environment_issue(evidence, scores, signals)
        self._score_flaky_test(evidence, scores, signals)
        self._score_plan_error(evidence, scores, signals)

        if not scores:
            return ClassificationResult(
                category=FailureCategory.UNKNOWN,
                confidence=0.0,
                reasoning="Evidence contains failures but matched no known classification rule.",
                signals=signals,
                scores={},
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_category, top_score = ranked[0]
        total_score = sum(scores.values())
        confidence = round(top_score / total_score, 4) if total_score > 0 else 0.0

        is_ambiguous = len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < _AMBIGUITY_MARGIN

        if is_ambiguous and self._llm_client is not None:
            llm_result = await self._classify_via_llm(evidence, scores, signals)
            if llm_result is not None:
                return llm_result

        reasoning = f"Rule-based classification: {'; '.join(signals)}"
        logger.info(
            "Classified failure as %s (confidence=%.2f, ambiguous=%s).",
            top_category.value,
            confidence,
            is_ambiguous,
        )
        return ClassificationResult(
            category=top_category,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals,
            scores=scores,
        )

    @staticmethod
    def _bump(scores: dict[FailureCategory, float], category: FailureCategory, weight: float) -> None:
        scores[category] = scores.get(category, 0.0) + weight

    def _score_selector_drift(
        self, evidence: FailureEvidence, scores: dict[FailureCategory, float], signals: list[str]
    ) -> None:
        for exec_result in evidence.failed_executions:
            if exec_result.error and _ELEMENT_RESOLUTION_ERROR_PATTERN.search(exec_result.error):
                self._bump(scores, FailureCategory.SELECTOR_DRIFT, _WEIGHT_SELECTOR_DRIFT_RESOLUTION_ERROR)
                signals.append(f"execution step {exec_result.step_id} failed with a resolution error")

        for step_id, resolved in evidence.resolved_elements.items():
            if (
                resolved.strategy in (ResolutionStrategy.VISUAL_GROUNDING, ResolutionStrategy.HEURISTIC)
                and resolved.confidence < _LOW_CONFIDENCE_THRESHOLD
            ):
                self._bump(
                    scores, FailureCategory.SELECTOR_DRIFT, _WEIGHT_SELECTOR_DRIFT_LOW_CONFIDENCE_VISION
                )
                signals.append(
                    f"step {step_id} resolved via {resolved.strategy.value} at low "
                    f"confidence ({resolved.confidence:.2f})"
                )

    def _score_app_bug(
        self, evidence: FailureEvidence, scores: dict[FailureCategory, float], signals: list[str]
    ) -> None:
        server_errors = [e for e in evidence.network_failures if e.status and e.status >= 500]
        if server_errors:
            self._bump(scores, FailureCategory.APP_BUG, _WEIGHT_APP_BUG_SERVER_ERROR)
            signals.append(f"{len(server_errors)} backend 5xx response(s) observed")

        if evidence.console_errors:
            self._bump(scores, FailureCategory.APP_BUG, _WEIGHT_APP_BUG_CONSOLE_ERROR)
            signals.append(f"{len(evidence.console_errors)} console error(s) captured")

        # Baseline signal: an assertion failed cleanly with no execution
        # errors and no low-confidence resolution — the agent did exactly
        # what the plan asked and the app just didn't behave as expected.
        # This is deliberately the smallest weight so any other signal
        # (selector drift, environment, flaky history) can outweigh it.
        clean_assertion_failures = [
            a for a in evidence.failed_assertions if "could not be resolved" not in (a.reason or "")
        ]
        if clean_assertion_failures and not evidence.failed_executions:
            self._bump(scores, FailureCategory.APP_BUG, _WEIGHT_APP_BUG_BASELINE_ASSERTION_FAILURE)
            signals.append(
                f"{len(clean_assertion_failures)} assertion(s) failed with no execution or "
                "resolution errors"
            )

    def _score_environment_issue(
        self, evidence: FailureEvidence, scores: dict[FailureCategory, float], signals: list[str]
    ) -> None:
        # A response with no status (connection refused/timeout/DNS
        # failure) suggests infrastructure trouble rather than the
        # application under test returning a real error response.
        connection_failures = [e for e in evidence.network_failures if e.status is None]
        if connection_failures:
            self._bump(scores, FailureCategory.ENVIRONMENT_ISSUE, _WEIGHT_ENVIRONMENT_THIRD_PARTY_FAILURE)
            signals.append(f"{len(connection_failures)} request(s) failed to connect entirely")

    def _score_flaky_test(
        self, evidence: FailureEvidence, scores: dict[FailureCategory, float], signals: list[str]
    ) -> None:
        rate = evidence.historical_pass_rate
        if rate is not None and 0.0 < rate < 1.0:
            # Weight peaks at a 50/50 historical split (maximally
            # non-deterministic) and tapers toward the edges, where a
            # failure is more likely a first-time real regression (near
            # 1.0 pass rate) or a consistently broken feature (near 0.0)
            # than genuine flakiness.
            flakiness_factor = 1.0 - abs(rate - 0.5) * 2  # 1.0 at rate=0.5, 0.0 at rate=0 or 1
            weight = _WEIGHT_FLAKY_HISTORICAL_MID_PASS_RATE * flakiness_factor
            if weight > 0:
                self._bump(scores, FailureCategory.FLAKY_TEST, weight)
                signals.append(f"historical pass rate {rate:.0%} suggests non-deterministic behavior")

    def _score_plan_error(
        self, evidence: FailureEvidence, scores: dict[FailureCategory, float], signals: list[str]
    ) -> None:
        # A failed execution with an empty/near-empty error message paired
        # with zero resolved elements for that step suggests the plan
        # asked for something structurally invalid, rather than the app
        # or perception layer being at fault.
        for exec_result in evidence.failed_executions:
            if exec_result.step_id not in evidence.resolved_elements and not (
                exec_result.error and _ELEMENT_RESOLUTION_ERROR_PATTERN.search(exec_result.error)
            ):
                self._bump(scores, FailureCategory.PLAN_ERROR, _WEIGHT_PLAN_ERROR_NO_TARGET_DESCRIPTION)
                signals.append(
                    f"step {exec_result.step_id} failed without ever resolving a target element"
                )

    async def _classify_via_llm(
        self,
        evidence: FailureEvidence,
        scores: dict[FailureCategory, float],
        signals: list[str],
    ) -> ClassificationResult | None:
        assert self._llm_client is not None
        summary_lines = [
            f"Rule-based scores so far: {[(c.value, round(s, 2)) for c, s in scores.items()]}",
            f"Signals observed: {signals}",
            f"Failed assertions: {[a.reason for a in evidence.failed_assertions]}",
            f"Failed execution errors: {[e.error for e in evidence.failed_executions]}",
            f"Console error count: {len(evidence.console_errors)}",
            f"Network failure count: {len(evidence.network_failures)}",
            f"Historical pass rate: {evidence.historical_pass_rate}",
        ]
        try:
            raw = await self._llm_client.complete(
                system_prompt=_SYSTEM_PROMPT, user_prompt="\n".join(summary_lines)
            )
            parsed = _parse_llm_json(raw)
            category = FailureCategory(parsed["category"])
            return ClassificationResult(
                category=category,
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", "LLM fallback classification (ambiguous rule scores)."),
                signals=signals,
                scores=scores,
            )
        except (FailureClassificationError, KeyError, ValueError) as exc:
            logger.warning("LLM classification fallback failed, using rule-based result instead: %s", exc)
            return None