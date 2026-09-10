"""
Detects and classifies state changes between two DOM snapshots — used by
the execution loop to decide whether an action actually did something
(navigated, mutated the DOM) or was a no-op, and by verification to know
when it's safe to re-snapshot and evaluate assertions.

Comparison is structural, not pixel/text based: it looks at each
element's (tag, role, accessible_name) identity, which is stable across
re-renders of the same logical content and is the same identity signal
`element_resolver.py` and `selector_cache.py` use elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from qa_agent.perception.dom_snapshot import DomElementNode, DomSnapshot


class StateChangeType(str, Enum):
    NO_CHANGE = "no_change"
    DOM_MUTATED = "dom_mutated"  # same URL, structurally different DOM
    NAVIGATION = "navigation"  # URL changed


def _element_identity(node: DomElementNode) -> tuple[str, str | None, str | None]:
    """
    The identity tuple used to match "the same element" across two
    snapshots. Deliberately excludes `index` (not stable across
    snapshots) and `dom_path` (can shift with unrelated DOM insertions
    elsewhere on the page even when this element itself didn't change).
    """
    return (node.tag, node.role, node.accessible_name)


@dataclass(frozen=True)
class PageStateDiff:
    change_type: StateChangeType
    url_changed: bool
    title_changed: bool
    dom_similarity: float  # 0.0 (completely different) - 1.0 (identical structure)
    added_element_count: int
    removed_element_count: int
    before_url: str
    after_url: str

    @property
    def has_meaningful_change(self) -> bool:
        return self.change_type != StateChangeType.NO_CHANGE


def _jaccard_similarity(before: DomSnapshot, after: DomSnapshot) -> float:
    before_ids = {_element_identity(e) for e in before.elements}
    after_ids = {_element_identity(e) for e in after.elements}
    if not before_ids and not after_ids:
        return 1.0
    union = before_ids | after_ids
    if not union:
        return 1.0
    intersection = before_ids & after_ids
    return len(intersection) / len(union)


def diff_states(
    before: DomSnapshot,
    after: DomSnapshot,
    dom_mutation_threshold: float = 0.95,
) -> PageStateDiff:
    """
    Compare two snapshots and classify the transition between them.

    Args:
        before: Snapshot captured before an action.
        after: Snapshot captured after an action.
        dom_mutation_threshold: Similarity below which the DOM is
            considered meaningfully mutated (same URL). 0.95 tolerates
            small dynamic noise (a live clock, an ad slot) without
            flagging every step as a "change".
    """
    before_url = before.url.split("#")[0]
    after_url = after.url.split("#")[0]
    url_changed = before_url != after_url
    title_changed = before.title != after.title

    before_ids = {_element_identity(e) for e in before.elements}
    after_ids = {_element_identity(e) for e in after.elements}
    added = after_ids - before_ids
    removed = before_ids - after_ids

    similarity = _jaccard_similarity(before, after)

    if url_changed:
        change_type = StateChangeType.NAVIGATION
    elif similarity < dom_mutation_threshold:
        change_type = StateChangeType.DOM_MUTATED
    else:
        change_type = StateChangeType.NO_CHANGE

    return PageStateDiff(
        change_type=change_type,
        url_changed=url_changed,
        title_changed=title_changed,
        dom_similarity=round(similarity, 4),
        added_element_count=len(added),
        removed_element_count=len(removed),
        before_url=before.url,
        after_url=after.url,
    )


def has_meaningfully_changed(
    before: DomSnapshot, after: DomSnapshot, dom_mutation_threshold: float = 0.95
) -> bool:
    """Convenience predicate for callers that only need a yes/no answer."""
    return diff_states(before, after, dom_mutation_threshold).has_meaningful_change