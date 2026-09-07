"""
Extraction of structured AcceptanceCriterion objects from raw text.

Handles two input shapes:
  1. Well-formed Gherkin ("Given ... When ... Then ...", optionally with a
     "Scenario:" header and "And"/"But" continuations).
  2. Loose, unstructured acceptance-criteria bullet points that don't use
     Gherkin keywords at all (common in Jira/Linear tickets written by
     PMs). These are heuristically classified into Given/When/Then buckets.

No LLM call is required for well-formed Gherkin — regex parsing is exact
and free. The heuristic fallback is intentionally conservative: anything
it can't confidently classify is kept as a THEN clause (an expected
outcome), which is the safest default for downstream test planning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from qa_agent.ingestion.schemas import AcceptanceCriterion, GherkinKeyword

_KEYWORD_PATTERN = re.compile(
    r"^\s*(given|when|then|and|but)\b[:\-]?\s*(.*)$", re.IGNORECASE
)
_SCENARIO_PATTERN = re.compile(r"^\s*scenario\s*[:\-]?\s*(.*)$", re.IGNORECASE)

# Heuristic cues for loose (non-Gherkin) acceptance criteria bullets.
_PRECONDITION_CUES = (
    "logged in",
    "given that",
    "assuming",
    "as a precondition",
    "prior to",
    "before",
)
_ACTION_CUES = (
    "when the user",
    "clicks",
    "submits",
    "enters",
    "navigates",
    "selects",
    "uploads",
    "user should be able to",
    "user can",
)


@dataclass
class ParsedGherkinBlock:
    scenario_name: str | None
    criteria: list[AcceptanceCriterion]


def parse_gherkin(text: str) -> ParsedGherkinBlock:
    """
    Parse a Gherkin-formatted block of text into ordered AcceptanceCriterion
    objects. Lines that don't start with a Gherkin keyword are ignored,
    except an optional leading "Scenario:" line.

    Falls back gracefully: if no Gherkin keywords are found at all, returns
    an empty criteria list so the caller can try `infer_criteria_from_text`
    instead.
    """
    scenario_name: str | None = None
    criteria: list[AcceptanceCriterion] = []
    order = 0
    last_primary_keyword: GherkinKeyword | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        scenario_match = _SCENARIO_PATTERN.match(line)
        if scenario_match:
            scenario_name = scenario_match.group(1).strip() or None
            continue

        match = _KEYWORD_PATTERN.match(line)
        if not match:
            continue

        raw_keyword, clause_text = match.groups()
        keyword = GherkinKeyword(raw_keyword.lower())
        clause_text = clause_text.strip().rstrip(".")
        if not clause_text:
            continue

        # "And"/"But" inherit the semantic role of the preceding primary
        # keyword (Given/When/Then) so downstream consumers that group by
        # keyword still treat them correctly.
        effective_keyword = keyword
        if keyword in (GherkinKeyword.AND, GherkinKeyword.BUT):
            effective_keyword = last_primary_keyword or GherkinKeyword.THEN
        else:
            last_primary_keyword = keyword

        criteria.append(
            AcceptanceCriterion(
                scenario_name=scenario_name,
                keyword=effective_keyword,
                text=clause_text,
                order=order,
            )
        )
        order += 1

    return ParsedGherkinBlock(scenario_name=scenario_name, criteria=criteria)


def infer_criteria_from_text(text: str, scenario_name: str | None = None) -> list[AcceptanceCriterion]:
    """
    Heuristically convert loose, non-Gherkin acceptance-criteria bullets
    (e.g. Jira tickets written as a plain checklist) into
    AcceptanceCriterion objects.

    This is intentionally simple pattern matching, not an LLM call — it
    exists to give the planner *something* structured for tickets that
    were never written in Given/When/Then form. Ambiguous lines default
    to THEN (treated as an expected outcome) since over-classifying a
    line as a precondition or action risks silently dropping an
    assertion the planner should verify.
    """
    lines = [
        re.sub(r"^[\-\*\d\.\)\s]+", "", raw.strip())
        for raw in text.splitlines()
        if raw.strip()
    ]

    criteria: list[AcceptanceCriterion] = []
    for order, line in enumerate(lines):
        lowered = line.lower()
        if any(cue in lowered for cue in _PRECONDITION_CUES):
            keyword = GherkinKeyword.GIVEN
        elif any(cue in lowered for cue in _ACTION_CUES):
            keyword = GherkinKeyword.WHEN
        else:
            keyword = GherkinKeyword.THEN

        criteria.append(
            AcceptanceCriterion(
                scenario_name=scenario_name,
                keyword=keyword,
                text=line.rstrip("."),
                order=order,
            )
        )
    return criteria


def extract_acceptance_criteria(text: str) -> list[AcceptanceCriterion]:
    """
    Primary entrypoint: try strict Gherkin parsing first, and only fall
    back to heuristic inference if the text contains no recognizable
    Gherkin keywords at all.
    """
    if not text or not text.strip():
        return []

    gherkin_result = parse_gherkin(text)
    if gherkin_result.criteria:
        return gherkin_result.criteria

    return infer_criteria_from_text(text)