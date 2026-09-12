"""Unit tests for qa_agent.perception.element_resolver."""

import io

import pytest
from PIL import Image

from qa_agent.perception.dom_snapshot import BoundingBox, DomElementNode, DomSnapshot
from qa_agent.perception.element_resolver import (
    ElementResolutionError,
    ElementResolver,
    ResolutionStrategy,
)
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache


def _el(index, name, dom_path=None) -> DomElementNode:
    return DomElementNode(
        index=index,
        tag="button",
        role="button",
        accessible_name=name,
        text=name,
        value=None,
        attributes={},
        dom_path=dom_path or f"button#{index}",
        bounding_box=BoundingBox(index * 10, 0, 10, 10),
        is_visible=True,
        is_enabled=True,
    )


def _snapshot() -> DomSnapshot:
    return DomSnapshot(
        url="https://shop.example.com",
        title="Shop",
        elements=[_el(0, "Add to Cart"), _el(1, "Add to Wishlist")],
    )


class FakeScreenshotPage:
    async def screenshot(self) -> bytes:
        img = Image.new("RGB", (200, 100), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


class StubVisionClient:
    def __init__(self, mark_index):
        self.mark_index = mark_index

    async def locate_marked_element(self, image_bytes, description, mark_count):
        return self.mark_index


@pytest.mark.asyncio
class TestElementResolver:
    async def test_resolves_unambiguous_match_via_heuristics(self):
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        resolved = await resolver.resolve(
            FakeScreenshotPage(), _snapshot(), "the 'Add to Cart' button"
        )
        assert resolved.strategy == ResolutionStrategy.HEURISTIC
        assert resolved.node.accessible_name == "Add to Cart"

    async def test_cache_hit_on_second_call(self):
        cache = SelectorCache()
        resolver = ElementResolver(vision_client=None, cache=cache)
        snapshot = _snapshot()

        first = await resolver.resolve(FakeScreenshotPage(), snapshot, "the 'Add to Cart' button")
        second = await resolver.resolve(FakeScreenshotPage(), snapshot, "the 'Add to Cart' button")

        assert first.strategy == ResolutionStrategy.HEURISTIC
        assert second.strategy == ResolutionStrategy.CACHE_HIT
        assert second.selector == first.selector

    async def test_stale_cache_entry_is_invalidated_and_recomputed(self):
        cache = SelectorCache()
        resolver = ElementResolver(vision_client=None, cache=cache)
        snapshot = _snapshot()
        await resolver.resolve(FakeScreenshotPage(), snapshot, "the 'Add to Cart' button")

        # Simulate the DOM changing: same description, different dom_path.
        changed_snapshot = DomSnapshot(
            url=snapshot.url,
            title=snapshot.title,
            elements=[_el(0, "Add to Cart", dom_path="button#new-add-to-cart")],
        )
        resolved = await resolver.resolve(
            FakeScreenshotPage(), changed_snapshot, "the 'Add to Cart' button"
        )
        # Cache entry pointed at the old dom_path, which no longer matches
        # exactly - resolver should fall through to heuristics again, not
        # blindly trust the stale path.
        assert resolved.selector == "button#new-add-to-cart"

    async def test_ambiguous_case_without_vision_client_raises(self):
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        with pytest.raises(ElementResolutionError):
            await resolver.resolve(FakeScreenshotPage(), _snapshot(), "the add button")

    async def test_ambiguous_case_resolved_via_vision_fallback(self):
        resolver = ElementResolver(vision_client=StubVisionClient(mark_index=0), cache=SelectorCache())
        resolved = await resolver.resolve(FakeScreenshotPage(), _snapshot(), "the add button")
        assert resolved.strategy == ResolutionStrategy.VISUAL_GROUNDING
