"""
Scores DOM element candidates by matching visible text content against a
target description — the second-line resolution strategy, used to
complement or break ties left by `role_based.py` (e.g. two buttons with
generic roles but different labels, or elements with no meaningful ARIA
role at all, like a plain `<div>` styled as a card).

Pure and dependency-free, same testability rationale as `role_based.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from qa_agent.perception.dom_snapshot import DomElementNode, DomSnapshot
from qa_agent.perception.selector_strategies.role_based import (
    ScoredCandidate,
    extract_quoted_phrase,
)

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "is", "are", "click", "select", "button", "field",
    "element", "that", "which",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _token_overlap_score(description_tokens: set[str], candidate_text: str) -> float:
    """Jaccard-style overlap between description keywords and an element's text."""
    candidate_tokens = _tokenize(candidate_text)
    if not description_tokens or not candidate_tokens:
        return 0.0
    intersection = description_tokens & candidate_tokens
    union = description_tokens | candidate_tokens
    return len(intersection) / len(union) if union else 0.0


def _best_text_similarity(description: str, node: DomElementNode) -> tuple[float, str]:
    """
    Compare the description (or a quoted phrase within it) against every
    text field the element carries, returning the strongest match and
    which field it came from.
    """
    quoted = extract_quoted_phrase(description)
    compare_against = quoted.lower() if quoted else description.lower()

    candidates: dict[str, str | None] = {
        "text": node.text,
        "accessible_name": node.accessible_name,
        "value": node.value,
        "placeholder": node.attributes.get("placeholder"),
    }

    best_score = 0.0
    best_field = "none"
    for field_name, field_value in candidates.items():
        if not field_value:
            continue
        field_lower = field_value.strip().lower()
        if not field_lower:
            continue

        if quoted:
            # Exact/substring match on a quoted phrase is a very strong
            # signal and short-circuits to a near-perfect score.
            if compare_against == field_lower:
                return 1.0, field_name
            if compare_against in field_lower or field_lower in compare_against:
                score = 0.9
            else:
                score = SequenceMatcher(None, compare_against, field_lower).ratio()
        else:
            score = SequenceMatcher(None, compare_against, field_lower).ratio()

        if score > best_score:
            best_score = score
            best_field = field_name

    return best_score, best_field


def score_candidates(
    snapshot: DomSnapshot,
    description: str,
    only_visible: bool = True,
) -> list[ScoredCandidate]:
    """
    Score every eligible element in `snapshot` against `description` using
    text-similarity signal (quoted-phrase matching, fuzzy string
    similarity, and keyword token overlap). Returns candidates sorted
    highest-score-first.
    """
    description_tokens = _tokenize(description)
    elements = [e for e in snapshot.elements if e.is_visible] if only_visible else snapshot.elements

    scored: list[ScoredCandidate] = []
    for node in elements:
        searchable = node.searchable_text()
        if not searchable:
            continue

        similarity, matched_field = _best_text_similarity(description, node)
        overlap = _token_overlap_score(description_tokens, searchable)

        # Similarity (quoted-phrase-aware) carries most of the weight;
        # token overlap is a secondary signal that helps rank elements
        # when there's no exact quoted phrase to anchor on.
        combined = round(0.75 * similarity + 0.25 * overlap, 4)
        if combined <= 0:
            continue

        scored.append(
            ScoredCandidate(
                node=node,
                score=combined,
                reason=f"text_similarity={similarity:.2f} (via {matched_field}), "
                f"token_overlap={overlap:.2f}",
            )
        )

    return sorted(scored, key=lambda c: c.score, reverse=True)