"""Unit tests for qa_agent.perception.dom_snapshot."""

import pytest

from qa_agent.perception.dom_snapshot import DomSnapshotExtractor, DomSnapshotError


class FakePage:
    def __init__(self, raw: dict):
        self._raw = raw
        self.url = raw["url"]

    async def title(self) -> str:
        return self._raw["title"]

    async def evaluate(self, script: str):
        return self._raw


def _raw_snapshot(elements=None) -> dict:
    return {
        "url": "https://shop.example.com/product/42",
        "title": "Product 42",
        "elements": elements
        or [
            {
                "index": 0,
                "tag": "button",
                "role": "button",
                "accessible_name": "Add to Cart",
                "text": "Add to Cart",
                "value": None,
                "attributes": {},
                "dom_path": "button#add-to-cart",
                "bounding_box": {"x": 10, "y": 20, "width": 120, "height": 40},
                "is_visible": True,
                "is_enabled": True,
            }
        ],
    }


@pytest.mark.asyncio
class TestDomSnapshotExtractor:
    async def test_captures_and_parses_elements(self):
        page = FakePage(_raw_snapshot())
        snapshot = await DomSnapshotExtractor().capture(page)

        assert snapshot.url == "https://shop.example.com/product/42"
        assert len(snapshot.elements) == 1
        assert snapshot.elements[0].accessible_name == "Add to Cart"
        assert snapshot.elements[0].bounding_box is not None

    async def test_page_signature_is_stable_for_same_structure(self):
        page1 = FakePage(_raw_snapshot())
        page2 = FakePage(_raw_snapshot())
        snap1 = await DomSnapshotExtractor().capture(page1)
        snap2 = await DomSnapshotExtractor().capture(page2)
        assert snap1.page_signature == snap2.page_signature

    async def test_page_signature_differs_for_different_structure(self):
        page1 = FakePage(_raw_snapshot())
        page2 = FakePage(
            _raw_snapshot(
                elements=[
                    {
                        "index": 0,
                        "tag": "a",
                        "role": "link",
                        "accessible_name": "Home",
                        "text": "Home",
                        "value": None,
                        "attributes": {},
                        "dom_path": "a#home",
                        "bounding_box": None,
                        "is_visible": True,
                        "is_enabled": True,
                    }
                ]
            )
        )
        snap1 = await DomSnapshotExtractor().capture(page1)
        snap2 = await DomSnapshotExtractor().capture(page2)
        assert snap1.page_signature != snap2.page_signature

    async def test_evaluate_failure_raises_dom_snapshot_error(self):
        class BrokenPage(FakePage):
            async def evaluate(self, script: str):
                raise RuntimeError("boom")

        with pytest.raises(DomSnapshotError):
            await DomSnapshotExtractor().capture(BrokenPage(_raw_snapshot()))

    async def test_visible_elements_filters_correctly(self):
        base = _raw_snapshot()["elements"][0]
        raw = _raw_snapshot(
            elements=[
                {**base, "index": 0, "is_visible": True},
                {**base, "index": 1, "is_visible": False, "dom_path": "#hidden"},
            ]
        )
        snapshot = await DomSnapshotExtractor().capture(FakePage(raw))
        assert len(snapshot.visible_elements()) == 1

    async def test_element_by_index(self):
        snapshot = await DomSnapshotExtractor().capture(FakePage(_raw_snapshot()))
        assert snapshot.element_by_index(0) is not None
        assert snapshot.element_by_index(99) is None
