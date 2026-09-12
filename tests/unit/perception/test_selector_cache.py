"""Unit tests for qa_agent.perception.selector_strategies.selector_cache."""

import pytest

from qa_agent.perception.selector_strategies.selector_cache import InMemoryKeyValueStore, SelectorCache


@pytest.mark.asyncio
class TestSelectorCache:
    async def test_miss_returns_none(self):
        cache = SelectorCache()
        assert await cache.get("sig1", "the add to cart button") is None

    async def test_set_then_get_round_trips(self):
        cache = SelectorCache()
        await cache.set(
            page_signature="sig1",
            description="the 'Add to Cart' button",
            dom_path="#add-to-cart",
            role="button",
            accessible_name="Add to Cart",
            strategy="heuristic",
            confidence=0.95,
        )
        entry = await cache.get("sig1", "the 'Add to Cart' button")
        assert entry is not None
        assert entry.dom_path == "#add-to-cart"
        assert entry.confidence == 0.95

    async def test_normalizes_description_whitespace_and_case(self):
        cache = SelectorCache()
        await cache.set("sig1", "  The Add To Cart Button  ", "#x", "button", "x", "heuristic", 0.9)
        entry = await cache.get("sig1", "the add to cart button")
        assert entry is not None

    async def test_different_page_signature_does_not_collide(self):
        cache = SelectorCache()
        await cache.set("sig1", "the submit button", "#a", "button", "a", "heuristic", 0.9)
        entry = await cache.get("sig2", "the submit button")
        assert entry is None

    async def test_invalidate_removes_entry(self):
        cache = SelectorCache()
        await cache.set("sig1", "desc", "#a", "button", "a", "heuristic", 0.9)
        await cache.invalidate("sig1", "desc")
        assert await cache.get("sig1", "desc") is None

    async def test_corrupt_entry_treated_as_miss(self):
        store = InMemoryKeyValueStore()
        cache = SelectorCache(store=store)
        key = cache._make_key("sig1", "desc")
        await store.set(key, "not valid json")
        assert await cache.get("sig1", "desc") is None
