"""Unit tests for qa_agent.ingestion.story_parser."""

import pytest

from qa_agent.ingestion.schemas import StorySource, UserStory
from qa_agent.ingestion.story_parser import StoryParser, StoryParsingError


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


@pytest.mark.asyncio
class TestCanonicalParsing:
    async def test_parses_canonical_story_without_llm(self):
        story = UserStory(
            source=StorySource.MANUAL,
            title="Add to cart",
            narrative="As a logged-in customer, I want to add an item to my cart, "
            "so that I can purchase it later.",
        )
        parser = StoryParser()  # no LLM client - canonical parse must not need one
        intent = await parser.parse(story)

        assert intent.persona == "logged-in customer"
        assert intent.goal == "add an item to my cart"
        assert intent.rationale == "I can purchase it later"

    async def test_rationale_does_not_swallow_trailing_gherkin_block(self):
        """Regression test: the rationale clause must not greedily consume
        acceptance criteria that follow on subsequent lines."""
        narrative = (
            "As a customer, I want to add an item to my cart, so that I can buy it later.\n\n"
            "Given I am on the product page\n"
            "When I click 'Add to Cart'\n"
            "Then the item appears in my cart\n"
        )
        story = UserStory(source=StorySource.MANUAL, title="t", narrative=narrative)
        intent = await StoryParser().parse(story)

        assert intent.rationale == "I can buy it later"
        assert "Given" not in intent.rationale
        assert intent.preconditions == ["I am on the product page"]
        assert intent.success_criteria == ["the item appears in my cart"]

    async def test_extracts_preconditions_and_success_criteria_from_gherkin(self):
        narrative = (
            "As a shopper, I want to apply a discount code, so that I pay less.\n\n"
            "Given my cart has items\n"
            "When I enter a valid code\n"
            "Then the total should be reduced\n"
            "And a confirmation message should appear\n"
        )
        story = UserStory(source=StorySource.MANUAL, title="t", narrative=narrative)
        intent = await StoryParser().parse(story)

        assert intent.preconditions == ["my cart has items"]
        assert intent.success_criteria == [
            "the total should be reduced",
            "a confirmation message should appear",
        ]


@pytest.mark.asyncio
class TestNonCanonicalParsing:
    async def test_raises_without_llm_client(self):
        story = UserStory(
            source=StorySource.MANUAL,
            title="Weird story",
            narrative="The checkout flow needs to support Apple Pay.",
        )
        with pytest.raises(StoryParsingError):
            await StoryParser().parse(story)

    async def test_falls_back_to_llm_for_non_canonical_narrative(self):
        story = UserStory(
            source=StorySource.MANUAL,
            title="Weird story",
            narrative="The checkout flow needs to support Apple Pay.",
        )
        llm_response = (
            '{"persona": "a mobile shopper", "goal": "pay with Apple Pay", '
            '"rationale": "faster checkout", "preconditions": ["Apple Pay is enabled"], '
            '"success_criteria": ["payment completes via Apple Pay"]}'
        )
        parser = StoryParser(llm_client=StubLLM(llm_response))
        intent = await parser.parse(story)

        assert intent.persona == "a mobile shopper"
        assert intent.goal == "pay with Apple Pay"
        assert intent.preconditions == ["Apple Pay is enabled"]

    async def test_malformed_llm_json_raises(self):
        story = UserStory(source=StorySource.MANUAL, title="t", narrative="Not canonical at all.")
        parser = StoryParser(llm_client=StubLLM("not json"))
        with pytest.raises(StoryParsingError):
            await parser.parse(story)
