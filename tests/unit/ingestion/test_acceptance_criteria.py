"""Unit tests for qa_agent.ingestion.acceptance_criteria."""

from qa_agent.ingestion.acceptance_criteria import (
    extract_acceptance_criteria,
    infer_criteria_from_text,
    parse_gherkin,
)
from qa_agent.ingestion.schemas import GherkinKeyword


class TestParseGherkin:
    def test_extracts_given_when_then_in_order(self):
        text = """
        Given I am on the product page
        When I click "Add to Cart"
        Then the item should appear in my cart
        """
        result = parse_gherkin(text)
        assert [c.keyword for c in result.criteria] == [
            GherkinKeyword.GIVEN,
            GherkinKeyword.WHEN,
            GherkinKeyword.THEN,
        ]
        assert result.criteria[0].text == "I am on the product page"
        assert [c.order for c in result.criteria] == [0, 1, 2]

    def test_and_inherits_preceding_primary_keyword(self):
        text = """
        Given I am logged in
        And my cart is empty
        Then I should see an empty cart message
        """
        result = parse_gherkin(text)
        # "And my cart is empty" should inherit GIVEN, not default to THEN
        assert result.criteria[1].keyword == GherkinKeyword.GIVEN

    def test_captures_scenario_name(self):
        text = """
        Scenario: Successful checkout
        Given I have items in my cart
        Then I can complete checkout
        """
        result = parse_gherkin(text)
        assert result.scenario_name == "Successful checkout"
        assert all(c.scenario_name == "Successful checkout" for c in result.criteria)

    def test_non_gherkin_text_returns_empty(self):
        result = parse_gherkin("The checkout flow needs to support Apple Pay.")
        assert result.criteria == []


class TestInferCriteriaFromText:
    def test_classifies_precondition_action_and_outcome(self):
        text = """
        - The user is logged in
        - When the user clicks submit
        - The order should be confirmed
        """
        criteria = infer_criteria_from_text(text)
        keywords = [c.keyword for c in criteria]
        assert GherkinKeyword.GIVEN in keywords
        assert GherkinKeyword.WHEN in keywords
        assert GherkinKeyword.THEN in keywords

    def test_ambiguous_line_defaults_to_then(self):
        criteria = infer_criteria_from_text("Something entirely unrelated to any cue")
        assert criteria[0].keyword == GherkinKeyword.THEN

    def test_strips_bullet_markers(self):
        criteria = infer_criteria_from_text("- Item one\n* Item two\n1. Item three")
        texts = [c.text for c in criteria]
        assert texts == ["Item one", "Item two", "Item three"]


class TestExtractAcceptanceCriteria:
    def test_prefers_gherkin_when_present(self):
        text = "Given a thing\nThen another thing"
        criteria = extract_acceptance_criteria(text)
        assert len(criteria) == 2

    def test_falls_back_to_heuristic_when_no_gherkin(self):
        text = "- user is logged in\n- order total should be correct"
        criteria = extract_acceptance_criteria(text)
        assert len(criteria) == 2

    def test_empty_text_returns_empty_list(self):
        assert extract_acceptance_criteria("") == []
        assert extract_acceptance_criteria("   ") == []
