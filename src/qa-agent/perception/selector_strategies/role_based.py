"""
Scores DOM element candidates against a target description using ARIA
role and accessible-name signal — the same signal a screen reader would
announce, and the first-choice resolution strategy per the project's
selector-strategy ADR (role/text before LLM/vision, for cost and
reliability).

This module is pure and dependency-free (no Playwright, no LLM) so it
can be unit-tested with hand-built `DomElementNode` fixtures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from qa_agent.perception.dom_snapshot import DomElementNode, DomSnapshot

# Maps common phrasing in a target description to the ARIA role it implies.
# Checked as whole-word matches against the description, so e.g. "link" in
# "linked account" does not spuriously imply role=link.
_ROLE_KEYWORDS: dict[str, str] = {
    "button": "button",
    "link": "link",
    "checkbox": "checkbox",
    "radio button": "radio",
    "radio": "radio",
    "dropdown": "combobox",
    "select": "combobox",
    "combobox": "combobox",
    "text field": "textbox",
    "text box": "textbox",
    "input field": "textbox",
    "input": "textbox",
    "textbox": "textbox",
    "slider": "slider",
    "tab": "tab",
    "menu item": "menuitem",
    "heading": "heading",
    "image": "img",
}

_ROLE_KEYWORD_PATTERNS = {
    phrase: re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
    for phrase in _ROLE_KEYWORDS
}

# Weights for the two signals that make up a role-based score. Role match
# is weighted higher than name similarity because it's a much stronger,
# lower-noise signal (a "button" role is either right or wrong; text
# similarity is inherently fuzzy).
_ROLE_MATCH_WEIGHT = 0.55
_NAME_SIMILARITY_WEIGHT = 0.45


@dataclass(frozen=True)
class ScoredCandidate:
    node: DomElementNode
    score: float  # 0.0-1.0
    reason: str


def _infer_expected_role(description: str) -> str | None:
    """
    Look for the longest matching role keyword phrase in the description
    (checked longest-first so "radio button" wins over the bare "button"
    substring match).
    """
    for phrase in sorted(_ROLE_KEYWORDS, key=len, reverse=True):
        if _ROLE_KEYWORD_PATTERNS[phrase].search(description):
            return _ROLE_KEYWORDS[phrase]
    return None


def extract_quoted_phrase(description: str) -> str | None:
    """Extract text a user wrapped in quotes, e.g. the 'Add to Cart' button."""
    match = re.search(r"['\"]([^'\"]+)['\"]", description)
    return match.group(1).strip() if match else None


def _name_similarity(description: str, node: DomElementNode) -> float:
    name = (node.accessible_name or "").strip().lower()
    if not name:
        return 0.0

    quoted = extract_quoted_phrase(description)
    if quoted:
        quoted_lower = quoted.lower()
        if quoted_lower == name:
            return 1.0
        if quoted_lower in name or name in quoted_lower:
            return 0.85
        return SequenceMatcher(None, quoted_lower, name).ratio()

    # No quoted phrase: fall back to comparing the whole description
    # against the accessible name — weaker signal, but still useful when
    # the description closely paraphrases visible text.
    return SequenceMatcher(None, description.lower(), name).ratio()


def score_candidates(
    snapshot: DomSnapshot,
    description: str,
    only_visible: bool = True,
    only_enabled: bool = True,
) -> list[ScoredCandidate]:
    """
    Score every eligible element in `snapshot` against `description` using
    role and accessible-name signal. Returns candidates sorted
    highest-score-first; callers typically only care about the top few.
    """
    expected_role = _infer_expected_role(description)

    elements = snapshot.elements
    if only_visible:
        elements = [e for e in elements if e.is_visible]
    if only_enabled:
        elements = [e for e in elements if e.is_enabled]

    scored: list[ScoredCandidate] = []
    for node in elements:
        if not node.accessible_name and not node.role:
            continue

        role_score = 0.0
        role_reason = "no role expectation"
        if expected_role is not None:
            if node.role == expected_role:
                role_score = 1.0
                role_reason = f"role matches expected {expected_role!r}"
            else:
                # Wrong role is a strong negative signal, not just an
                # absence of positive signal — an <a> tag should rarely
                # win when the description explicitly says "button".
                role_score = 0.0
                role_reason = f"role {node.role!r} != expected {expected_role!r}"

        name_score = _name_similarity(description, node)

        if expected_role is not None:
            combined = role_score * _ROLE_MATCH_WEIGHT + name_score * _NAME_SIMILARITY_WEIGHT
        else:
            # With no explicit role expectation, name similarity carries
            # the full weight of the score.
            combined = name_score

        scored.append(
            ScoredCandidate(
                node=node,
                score=round(combined, 4),
                reason=f"{role_reason}; name_similarity={name_score:.2f}",
            )
        )

    return sorted(scored, key=lambda c: c.score, reverse=True)