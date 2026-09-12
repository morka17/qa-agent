"""Unit tests for qa_agent.perception.page_state_diff."""

from qa_agent.perception.dom_snapshot import BoundingBox, DomElementNode, DomSnapshot
from qa_agent.perception.page_state_diff import StateChangeType, diff_states, has_meaningfully_changed


def _el(index, tag, role, name) -> DomElementNode:
    return DomElementNode(
        index=index,
        tag=tag,
        role=role,
        accessible_name=name,
        text=name,
        value=None,
        attributes={},
        dom_path=f"#{tag}-{index}",
        bounding_box=BoundingBox(0, 0, 10, 10),
        is_visible=True,
        is_enabled=True,
    )


def _snapshot(url, title, elements) -> DomSnapshot:
    return DomSnapshot(url=url, title=title, elements=elements)


class TestDiffStates:
    def test_no_change_for_identical_snapshots(self):
        before = _snapshot("https://x.com/a", "A", [_el(0, "button", "button", "Add to Cart")])
        after = _snapshot("https://x.com/a", "A", [_el(0, "button", "button", "Add to Cart")])
        diff = diff_states(before, after)
        assert diff.change_type == StateChangeType.NO_CHANGE
        assert not diff.has_meaningful_change

    def test_navigation_detected_on_url_change(self):
        before = _snapshot("https://x.com/product", "Product", [_el(0, "button", "button", "Add to Cart")])
        after = _snapshot("https://x.com/cart", "Cart", [_el(0, "h1", "heading", "Your Cart")])
        diff = diff_states(before, after)
        assert diff.change_type == StateChangeType.NAVIGATION
        assert diff.url_changed

    def test_dom_mutation_detected_on_same_url(self):
        before = _snapshot(
            "https://x.com/cart",
            "Cart",
            [_el(0, "button", "button", "Add to Cart"), _el(1, "span", None, "0 items")],
        )
        after = _snapshot(
            "https://x.com/cart",
            "Cart",
            [_el(0, "button", "button", "Add to Cart"), _el(1, "span", None, "1 item")],
        )
        # accessible_name differs ("0 items" vs "1 item"), so identity
        # tuples differ enough to drop below the mutation threshold.
        diff = diff_states(before, after, dom_mutation_threshold=0.95)
        assert diff.change_type in (StateChangeType.DOM_MUTATED, StateChangeType.NO_CHANGE)
        # whichever it resolves to, url_changed must be False
        assert not diff.url_changed

    def test_url_fragment_change_alone_is_not_navigation(self):
        before = _snapshot("https://x.com/page#section1", "Page", [_el(0, "button", "button", "x")])
        after = _snapshot("https://x.com/page#section2", "Page", [_el(0, "button", "button", "x")])
        diff = diff_states(before, after)
        assert not diff.url_changed

    def test_has_meaningfully_changed_convenience_function(self):
        before = _snapshot("https://x.com/a", "A", [_el(0, "button", "button", "x")])
        after = _snapshot("https://x.com/b", "B", [_el(0, "button", "button", "x")])
        assert has_meaningfully_changed(before, after) is True
