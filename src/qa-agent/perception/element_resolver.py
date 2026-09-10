"""
Orchestrates element resolution: cache lookup -> role/text heuristics ->
vision-model fallback, in that order, per the project's layered
selector-resolution principle (docs/adr/0002-selector-strategy.md).

This is the single entrypoint `execution/action_executor.py` calls to
turn a plan step's `target_description` into something it can actually
act on (a `dom_path` usable as a Playwright selector).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings
from qa_agent.perception.dom_snapshot import DomElementNode, DomSnapshot
from qa_agent.perception.selector_strategies import role_based, text_based
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache
from qa_agent.perception.selector_strategies.visual_grounding import (
    VisionLLMClient,
    resolve_via_vision,
)

logger = get_logger(__name__)

# Below this combined heuristic score, we don't trust role/text matching
# enough to act on it and fall through to vision grounding instead.
_HEURISTIC_CONFIDENCE_THRESHOLD = 0.6

# If the top heuristic candidate doesn't clear the runner-up by at least
# this margin, the match is ambiguous even if its absolute score is high
# (e.g. two near-identical "Add to Wishlist" buttons on a listing page) —
# vision grounding, which can see spatial/visual context, is more likely
# to disambiguate correctly than either heuristic score alone.
_MIN_AMBIGUITY_MARGIN = 0.12

# Relative weighting when combining role-based and text-based scores for
# the same candidate into one ranking.
_ROLE_STRATEGY_WEIGHT = 0.5
_TEXT_STRATEGY_WEIGHT = 0.5


class ResolutionStrategy(str, Enum):
    CACHE_HIT = "cache_hit"
    HEURISTIC = "heuristic"
    VISUAL_GROUNDING = "visual_grounding"


class ScreenshotProvider(Protocol):
    """The minimal Playwright `Page` surface needed for the vision fallback."""

    async def screenshot(self) -> bytes: ...


@dataclass(frozen=True)
class ResolvedElement:
    node: DomElementNode
    strategy: ResolutionStrategy
    confidence: float
    selector: str  # execution-time-usable Playwright selector (the element's dom_path)


class ElementResolutionError(Exception):
    """
    Raised when no strategy — cache, heuristics, or vision — could
    resolve the description to an element with acceptable confidence.
    """

    def __init__(self, description: str, page_signature: str) -> None:
        self.description = description
        self.page_signature = page_signature
        super().__init__(
            f"Could not resolve element for description {description!r} "
            f"on page signature {page_signature!r}: all resolution strategies exhausted."
        )


def _combine_heuristic_scores(
    snapshot: DomSnapshot, description: str
) -> list[tuple[DomElementNode, float, str]]:
    """
    Run both heuristic strategies and merge their per-element scores into
    one ranked list. An element only scored by one strategy still gets
    ranked (missing signal from the other strategy counts as 0 for that
    strategy, not as disqualifying).
    """
    role_scores = {c.node.index: c for c in role_based.score_candidates(snapshot, description)}
    text_scores = {c.node.index: c for c in text_based.score_candidates(snapshot, description)}

    all_indices = set(role_scores) | set(text_scores)
    combined: list[tuple[DomElementNode, float, str]] = []

    for index in all_indices:
        role_candidate = role_scores.get(index)
        text_candidate = text_scores.get(index)
        role_score = role_candidate.score if role_candidate else 0.0
        text_score = text_candidate.score if text_candidate else 0.0
        node = (role_candidate or text_candidate).node  # type: ignore[union-attr]

        combined_score = round(
            role_score * _ROLE_STRATEGY_WEIGHT + text_score * _TEXT_STRATEGY_WEIGHT, 4
        )
        reason = (
            f"role_score={role_score:.2f}"
            + (f" ({role_candidate.reason})" if role_candidate else "")
            + f", text_score={text_score:.2f}"
            + (f" ({text_candidate.reason})" if text_candidate else "")
        )
        combined.append((node, combined_score, reason))

    return sorted(combined, key=lambda item: item[1], reverse=True)


class ElementResolver:
    """
    Resolves a natural-language `target_description` to a concrete DOM
    element on the current page.

    Example:
        resolver = ElementResolver(vision_client=my_vision_client)
        resolved = await resolver.resolve(page, snapshot, "the 'Add to Cart' button")
        await page.locator(resolved.selector).click()
    """

    def __init__(
        self,
        vision_client: VisionLLMClient | None = None,
        cache: SelectorCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._vision_client = vision_client
        self._cache = cache or SelectorCache()
        self._settings = settings or get_settings()

    async def resolve(
        self,
        page: ScreenshotProvider,
        snapshot: DomSnapshot,
        description: str,
    ) -> ResolvedElement:
        cached = await self._try_cache(snapshot, description)
        if cached is not None:
            return cached

        heuristic = self._try_heuristics(snapshot, description)
        if heuristic is not None:
            await self._store_in_cache(snapshot, description, heuristic)
            return heuristic

        if self._vision_client is not None:
            vision_result = await self._try_vision(page, snapshot, description)
            if vision_result is not None:
                await self._store_in_cache(snapshot, description, vision_result)
                return vision_result

        raise ElementResolutionError(description, snapshot.page_signature)

    async def _try_cache(
        self, snapshot: DomSnapshot, description: str
    ) -> ResolvedElement | None:
        cached_entry = await self._cache.get(snapshot.page_signature, description)
        if cached_entry is None:
            return None

        # Never trust a cache hit blindly: re-validate against the
        # *current* snapshot. An element matching the cached dom_path
        # AND role/name is required — a dom_path alone can silently
        # point at a different element after a markup change.
        match = next(
            (
                el
                for el in snapshot.visible_elements()
                if el.dom_path == cached_entry.dom_path
                and el.role == cached_entry.role
                and el.accessible_name == cached_entry.accessible_name
            ),
            None,
        )
        if match is None:
            logger.debug(
                "Cache entry for %r on page %s is stale; invalidating.",
                description,
                snapshot.page_signature,
            )
            await self._cache.invalidate(snapshot.page_signature, description)
            return None

        logger.debug("Cache hit for %r on page %s.", description, snapshot.page_signature)
        return ResolvedElement(
            node=match,
            strategy=ResolutionStrategy.CACHE_HIT,
            confidence=cached_entry.confidence,
            selector=match.dom_path,
        )

    def _try_heuristics(self, snapshot: DomSnapshot, description: str) -> ResolvedElement | None:
        ranked = _combine_heuristic_scores(snapshot, description)
        if not ranked:
            return None

        top_node, top_score, top_reason = ranked[0]
        if top_score < _HEURISTIC_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Top heuristic candidate for %r scored %.2f, below threshold %.2f.",
                description,
                top_score,
                _HEURISTIC_CONFIDENCE_THRESHOLD,
            )
            return None

        if len(ranked) > 1:
            runner_up_score = ranked[1][1]
            if (top_score - runner_up_score) < _MIN_AMBIGUITY_MARGIN:
                logger.debug(
                    "Top two heuristic candidates for %r are ambiguous (%.2f vs %.2f); "
                    "deferring to vision grounding if available.",
                    description,
                    top_score,
                    runner_up_score,
                )
                return None

        logger.debug(
            "Resolved %r via heuristics: index=%d, score=%.2f (%s)",
            description,
            top_node.index,
            top_score,
            top_reason,
        )
        return ResolvedElement(
            node=top_node,
            strategy=ResolutionStrategy.HEURISTIC,
            confidence=top_score,
            selector=top_node.dom_path,
        )

    async def _try_vision(
        self, page: ScreenshotProvider, snapshot: DomSnapshot, description: str
    ) -> ResolvedElement | None:
        assert self._vision_client is not None
        screenshot_bytes = await page.screenshot()
        result = await resolve_via_vision(
            screenshot_bytes=screenshot_bytes,
            elements=snapshot.visible_elements(),
            description=description,
            vision_client=self._vision_client,
        )
        if result is None:
            return None

        return ResolvedElement(
            node=result.node,
            strategy=ResolutionStrategy.VISUAL_GROUNDING,
            confidence=result.confidence,
            selector=result.node.dom_path,
        )

    async def _store_in_cache(
        self, snapshot: DomSnapshot, description: str, resolved: ResolvedElement
    ) -> None:
        # Cache hits themselves don't need re-storing — this only fires
        # for freshly-resolved (heuristic/vision) results.
        if resolved.strategy == ResolutionStrategy.CACHE_HIT:
            return
        await self._cache.set(
            page_signature=snapshot.page_signature,
            description=description,
            dom_path=resolved.node.dom_path,
            role=resolved.node.role,
            accessible_name=resolved.node.accessible_name,
            strategy=resolved.strategy.value,
            confidence=resolved.confidence,
        )