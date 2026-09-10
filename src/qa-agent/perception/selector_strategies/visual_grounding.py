"""
Last-resort element resolution via a vision-capable LLM, used only when
`role_based.py` and `text_based.py` heuristics fail to produce a
sufficiently confident match — e.g. a custom-styled `<div>` "button" with
no ARIA role and generic/no text (an icon-only control, a drag handle, a
canvas-rendered widget).

Uses a "set-of-marks" prompting strategy: candidates are numbered and
overlaid on a screenshot as labeled boxes, and the vision model is asked
to return the number of the box matching the description, rather than
raw pixel coordinates — this is both more reliable across model
providers and trivially maps back to a `DomElementNode` via its index.

This is deliberately the most expensive and slowest strategy in the
pipeline (a screenshot + vision-model round trip), consistent with the
project's layered selector-resolution principle: reach for it only when
cheaper heuristics are inconclusive.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageDraw, ImageFont

from qa_agent.config.logging_config import get_logger
from qa_agent.perception.dom_snapshot import DomElementNode

logger = get_logger(__name__)

# Only offer the model a bounded number of candidates — beyond this, the
# overlay becomes visually unreadable and the prompt/image cost balloons
# for diminishing accuracy return.
_MAX_ANNOTATED_CANDIDATES = 30

_MARK_BOX_COLOR = (255, 0, 90)
_MARK_TEXT_COLOR = (255, 255, 255)
_MARK_BG_COLOR = (255, 0, 90)


class VisionLLMClient(Protocol):
    """
    Minimal interface this module depends on. A real implementation
    wraps a multimodal model call (e.g. Claude with an image content
    block) behind this shape; kept as a local Protocol for the same
    provider-independence reason as `LLMClient` in `story_parser.py` /
    `test_planner.py`.
    """

    async def locate_marked_element(
        self, image_bytes: bytes, description: str, mark_count: int
    ) -> int | None:
        """
        Given a screenshot annotated with numbered marks 0..mark_count-1,
        return the mark number that best matches `description`, or None
        if no mark is a plausible match.
        """
        ...


@dataclass(frozen=True)
class VisualGroundingResult:
    node: DomElementNode
    confidence: float
    raw_mark_index: int


class VisualGroundingError(Exception):
    """Raised when the screenshot cannot be annotated or the model call fails unrecoverably."""


def _select_candidates(
    elements: list[DomElementNode], max_candidates: int
) -> list[DomElementNode]:
    """
    Choose which visible elements to annotate and offer to the model.
    Prioritizes elements with a bounding box (unpositioned elements can't
    be marked) and, beyond the cap, prefers smaller elements first since
    oversized marks (e.g. a full-page `<div>`) are rarely the intended
    target of a specific description.
    """
    positioned = [e for e in elements if e.is_visible and e.bounding_box is not None]
    positioned.sort(key=lambda e: e.bounding_box.width * e.bounding_box.height)  # type: ignore[union-attr]
    return positioned[:max_candidates]


def _annotate_screenshot(screenshot_bytes: bytes, candidates: list[DomElementNode]) -> bytes:
    """
    Draw a numbered box around each candidate's bounding box. Mark index
    corresponds to the candidate's position in `candidates` (0-indexed),
    which is what the vision model is asked to return.
    """
    try:
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise VisualGroundingError(f"Could not decode screenshot: {exc}") from exc

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001 - default font should always load, but never fail hard on it
        font = None

    for mark_index, node in enumerate(candidates):
        box = node.bounding_box
        assert box is not None  # guaranteed by _select_candidates
        x0, y0 = box.x, box.y
        x1, y1 = box.x + box.width, box.y + box.height
        draw.rectangle([x0, y0, x1, y1], outline=_MARK_BOX_COLOR, width=2)

        label = str(mark_index)
        label_pos = (max(x0 - 2, 0), max(y0 - 14, 0))
        if font is not None:
            text_bbox = draw.textbbox(label_pos, label, font=font)
        else:
            text_bbox = (label_pos[0], label_pos[1], label_pos[0] + 12, label_pos[1] + 12)
        draw.rectangle(text_bbox, fill=_MARK_BG_COLOR)
        draw.text(label_pos, label, fill=_MARK_TEXT_COLOR, font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def resolve_via_vision(
    screenshot_bytes: bytes,
    elements: list[DomElementNode],
    description: str,
    vision_client: VisionLLMClient,
    max_candidates: int = _MAX_ANNOTATED_CANDIDATES,
) -> VisualGroundingResult | None:
    """
    Attempt to resolve `description` to one of `elements` using a
    screenshot + vision model. Returns None (not an exception) if the
    model reports no confident match — that is a legitimate outcome the
    caller (`element_resolver.py`) should handle as "resolution failed",
    not a system error.
    """
    candidates = _select_candidates(elements, max_candidates)
    if not candidates:
        logger.warning("No positioned, visible elements available for visual grounding.")
        return None

    annotated = _annotate_screenshot(screenshot_bytes, candidates)

    mark_index = await vision_client.locate_marked_element(
        image_bytes=annotated,
        description=description,
        mark_count=len(candidates),
    )

    if mark_index is None:
        logger.info("Vision model found no confident match for %r.", description)
        return None

    if not (0 <= mark_index < len(candidates)):
        raise VisualGroundingError(
            f"Vision model returned out-of-range mark index {mark_index} "
            f"for {len(candidates)} candidate(s)."
        )

    node = candidates[mark_index]
    logger.info(
        "Visual grounding resolved %r to element index=%d (mark=%d, tag=%s).",
        description,
        node.index,
        mark_index,
        node.tag,
    )
    # Vision-grounded matches are inherently less certain than an exact
    # role+name match, so confidence is capped below what heuristic
    # strategies can report even on a "successful" resolution.
    return VisualGroundingResult(node=node, confidence=0.7, raw_mark_index=mark_index)