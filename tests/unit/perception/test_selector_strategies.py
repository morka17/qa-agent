"""Unit tests for qa_agent.perception.selector_strategies.role_based / text_based."""

from qa_agent.perception.dom_snapshot import BoundingBox, DomElementNode, DomSnapshot
from qa_agent.perception.selector_strategies import role_based, text_based


def _el(index, tag, role, name, text=None, attrs=None, visible=True, enabled=True) -> DomElementNode:
    return DomElementNode(
        index=index,
        tag=tag,
        role=role,
        accessible_name=name,
        text=text or name,
        value=None,
        attributes=attrs or {},
        dom_path=f"#{tag}-{index}",
        bounding_box=BoundingBox(0, 0, 10, 10),
        is_visible=visible,
        is_enabled=enabled,
    )


def _snapshot() -> DomSnapshot:
    return DomSnapshot(
        url="https://shop.example.com",
        title="Shop",
        elements=[
            _el(0, "a", "link", "View Cart"),
            _el(1, "button", "button", "Add to Cart"),
            _el(2, "button", "button", "Add to Wishlist"),
            _el(3, "input", "textbox", None, attrs={"placeholder": "Search products"}),
            _el(4, "button", "button", "Hidden button", visible=False),
        ],
    )


class TestRoleBased:
    def test_exact_role_and_name_match_scores_highest(self):
        results = role_based.score_candidates(_snapshot(), "the 'Add to Cart' button")
        assert results[0].node.accessible_name == "Add to Cart"
        assert results[0].score == 1.0

    def test_wrong_role_is_penalized(self):
        results = role_based.score_candidates(_snapshot(), "the 'Add to Cart' button")
        link_result = next(r for r in results if r.node.tag == "a")
        button_result = next(r for r in results if r.node.accessible_name == "Add to Cart")
        assert link_result.score < button_result.score

    def test_role_keyword_inference_for_textbox(self):
        results = role_based.score_candidates(_snapshot(), "the search input field")
        assert results[0].node.tag == "input"

    def test_hidden_elements_excluded_by_default(self):
        results = role_based.score_candidates(_snapshot(), "the hidden button")
        assert all(r.node.accessible_name != "Hidden button" for r in results)

    def test_no_role_keyword_falls_back_to_name_similarity_only(self):
        results = role_based.score_candidates(_snapshot(), "View Cart")
        assert results[0].node.accessible_name == "View Cart"


class TestTextBased:
    def test_quoted_phrase_exact_match_wins(self):
        results = text_based.score_candidates(_snapshot(), "the 'Add to Cart' button")
        assert results[0].node.accessible_name == "Add to Cart"
        assert results[0].score == 1.0

    def test_distinguishes_similar_buttons(self):
        results = text_based.score_candidates(_snapshot(), "the 'Add to Cart' button")
        top_two = {r.node.accessible_name for r in results[:1]}
        assert "Add to Cart" in top_two
        wishlist_score = next(r.score for r in results if r.node.accessible_name == "Add to Wishlist")
        cart_score = next(r.score for r in results if r.node.accessible_name == "Add to Cart")
        assert cart_score > wishlist_score

    def test_matches_placeholder_text(self):
        results = text_based.score_candidates(_snapshot(), "search products")
        assert results[0].node.tag == "input"

    def test_no_match_returns_empty_or_low_scores(self):
        results = text_based.score_candidates(_snapshot(), "completely unrelated phrase xyz")
        assert all(r.score < 0.5 for r in results)
