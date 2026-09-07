"""
Turns a raw UserStory into a structured TestIntent.

Two-tier strategy:
  1. A fast, free, deterministic regex parser handles the canonical
     "As a <persona>, I want <goal>, so that <rationale>" phrasing, which
     covers a large fraction of real-world tickets.
  2. Anything that doesn't match is handed to an LLM (via the
     `LLMClient` protocol below) to extract persona/goal/rationale from
     free-form narrative text.

The LLM dependency is expressed as a small local `Protocol` rather than a
concrete import from `qa_agent.llm.provider_router`, so this module has
zero hard dependency on which provider (Anthropic/OpenAI/local) is wired
up, and is trivially unit-testable with a stub.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.ingestion.acceptance_criteria import extract_acceptance_criteria
from qa_agent.ingestion.schemas import TestIntent, TestIntentKind, UserStory

logger = get_logger(__name__)

_CANONICAL_STORY_PATTERN = re.compile(
    r"""
    as\s+an?\s+(?P<persona>.+?)          # "As a/an <persona>"
    ,?\s*i\s+(?:want|need)\s+(?:to\s+)?  # "I want / I need to"
    (?P<goal>.+?)                        # "<goal>"
    (?:,?\s*so\s+that\s+(?P<rationale>[^\n]+?))?  # optional "so that <rationale>", same line only
    \.?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _first_paragraph(narrative: str) -> str:
    """
    The canonical "As a ... I want ... so that ..." phrasing is always the
    opening statement of a story; anything after the first blank line is
    typically a separate Gherkin/acceptance-criteria block and must not
    be considered part of the rationale clause.
    """
    return narrative.strip().split("\n\n", 1)[0].strip()

_SYSTEM_PROMPT = """You extract structured test intent from a software user story.
Return ONLY a JSON object (no markdown, no prose) with these exact keys:
{
  "persona": "<who is acting, short phrase>",
  "goal": "<what they are trying to accomplish, short phrase>",
  "rationale": "<why it matters, short phrase, or null>",
  "preconditions": ["<state that must hold before the test begins>", ...],
  "success_criteria": ["<observable, checkable condition that defines success>", ...]
}
Infer preconditions and success_criteria even if not explicitly stated,
based on standard behavior for this kind of feature. Keep every field
concise (under 20 words each)."""


class LLMClient(Protocol):
    """
    Minimal interface this module depends on. The real implementation
    lives in `qa_agent.llm.provider_router.LLMRouter`, which wraps
    Anthropic/OpenAI/local backends behind this same call shape.
    """

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw text completion for the given prompts."""
        ...


class StoryParsingError(Exception):
    """Raised when a story cannot be parsed into a TestIntent by any strategy."""


def _try_canonical_parse(narrative: str) -> dict[str, str | None] | None:
    match = _CANONICAL_STORY_PATTERN.search(_first_paragraph(narrative))
    if not match:
        return None
    groups = match.groupdict()
    return {
        "persona": groups["persona"].strip(),
        "goal": groups["goal"].strip().rstrip(","),
        "rationale": (groups["rationale"] or "").strip() or None,
    }


def _parse_llm_json(raw: str) -> dict:
    """
    Defensively parse an LLM completion as JSON, stripping common
    wrapper artifacts (markdown code fences) models sometimes add
    despite instructions not to.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise StoryParsingError(f"LLM response was not valid JSON: {raw!r}") from exc


class StoryParser:
    """
    Converts a UserStory into a TestIntent.

    Args:
        llm_client: Optional LLM backend used only when the canonical
            regex parse fails. If omitted, non-canonical stories raise
            `StoryParsingError` instead of silently guessing — callers
            that need best-effort parsing without an LLM should catch
            this and fall back to manual review.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    async def parse(self, story: UserStory) -> TestIntent:
        acceptance_criteria = story.acceptance_criteria or extract_acceptance_criteria(
            story.narrative
        )

        canonical = _try_canonical_parse(story.narrative)
        if canonical is not None:
            logger.debug("Parsed story %s via canonical regex pattern.", story.id)
            return TestIntent(
                story_id=story.id,
                kind=TestIntentKind.FUNCTIONAL,
                persona=canonical["persona"],
                goal=canonical["goal"],
                rationale=canonical["rationale"],
                success_criteria=[c.text for c in acceptance_criteria if c.keyword.value == "then"],
                preconditions=[c.text for c in acceptance_criteria if c.keyword.value == "given"],
                acceptance_criteria=acceptance_criteria,
                target_url=story.target_url,
                priority=story.priority,
            )

        if self._llm_client is None:
            raise StoryParsingError(
                f"Story {story.id!r} does not match the canonical "
                "'As a ... I want ... so that ...' pattern and no LLM "
                "client was provided to parse free-form narrative."
            )

        logger.debug("Story %s is non-canonical; falling back to LLM extraction.", story.id)
        raw = await self._llm_client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=(
                f"Title: {story.title}\n\n"
                f"Narrative:\n{story.narrative}\n\n"
                f"Acceptance criteria:\n"
                + "\n".join(f"- {c.as_gherkin_line()}" for c in acceptance_criteria)
            ),
        )
        extracted = _parse_llm_json(raw)

        return TestIntent(
            story_id=story.id,
            kind=TestIntentKind.FUNCTIONAL,
            persona=extracted.get("persona", "a user"),
            goal=extracted.get("goal", story.title),
            rationale=extracted.get("rationale"),
            preconditions=extracted.get("preconditions", []),
            success_criteria=extracted.get("success_criteria", []),
            acceptance_criteria=acceptance_criteria,
            target_url=story.target_url,
            priority=story.priority,
        )