"""
Integration test that exercises `DomSnapshotExtractor` and
`ElementResolver` against a *real* Playwright browser rendering real
HTML - unlike the unit tests in tests/unit/perception/, which use
hand-built fake DOM dicts, this proves the actual JS snapshot script
(`dom_snapshot._SNAPSHOT_JS`) executes correctly in a real browser and
that resolution works against its real output.

Requires Playwright browsers to be installed (`playwright install
chromium`). Skips gracefully - rather than failing - when they aren't,
so this test suite still runs in environments (e.g. a fast unit-test-only
CI stage) that haven't provisioned a browser.
"""

import pytest

from qa_agent.perception.dom_snapshot import DomSnapshotExtractor
from qa_agent.perception.element_resolver import ElementResolver
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache

PRODUCT_PAGE_HTML = """
<!doctype html>
<html>
<body>
  <header><a href="/">Home</a> <a href="/cart">View Cart</a></header>
  <main>
    <h1>Wireless Headphones</h1>
    <p>Noise-cancelling, 30 hour battery life.</p>
    <button id="add-to-cart">Add to Cart</button>
    <button id="add-to-wishlist">Add to Wishlist</button>
    <input type="text" placeholder="Enter a promo code" />
  </main>
</body>
</html>
"""


@pytest.fixture
async def real_page():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        pytest.skip("playwright is not installed")

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
    except Exception as exc:  # noqa: BLE001 - browser binaries not installed, or launch failed for any reason
        pytest.skip(f"Playwright browser unavailable, skipping live-DOM test: {exc}")
        return

    context = await browser.new_context()
    page = await context.new_page()
    await page.set_content(PRODUCT_PAGE_HTML)

    yield page

    await context.close()
    await browser.close()
    await playwright.stop()


@pytest.mark.asyncio
class TestSelectorResolutionLiveDom:
    async def test_snapshot_extracts_real_elements(self, real_page):
        snapshot = await DomSnapshotExtractor().capture(real_page)

        names = {e.accessible_name for e in snapshot.elements if e.accessible_name}
        assert "Add to Cart" in names
        assert "Add to Wishlist" in names
        assert "View Cart" in names

    async def test_resolves_unambiguous_button_on_real_dom(self, real_page):
        snapshot = await DomSnapshotExtractor().capture(real_page)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())

        resolved = await resolver.resolve(real_page, snapshot, "the 'Add to Cart' button")

        # Confirm the resolved selector genuinely targets the real element
        # in the live page, not just a plausible-looking string.
        locator = real_page.locator(resolved.selector)
        assert await locator.count() == 1
        assert (await locator.text_content()).strip() == "Add to Cart"

    async def test_resolves_input_by_placeholder_on_real_dom(self, real_page):
        snapshot = await DomSnapshotExtractor().capture(real_page)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())

        resolved = await resolver.resolve(real_page, snapshot, "the promo code input field")

        locator = real_page.locator(resolved.selector)
        assert await locator.get_attribute("placeholder") == "Enter a promo code"

    async def test_click_via_resolved_selector_actually_works(self, real_page):
        # Add an onclick handler dynamically to prove the resolved
        # selector really is clickable and targets the right element.
        await real_page.evaluate(
            "document.getElementById('add-to-cart').addEventListener("
            "'click', () => { document.title = 'clicked'; })"
        )
        snapshot = await DomSnapshotExtractor().capture(real_page)
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        resolved = await resolver.resolve(real_page, snapshot, "the 'Add to Cart' button")

        await real_page.locator(resolved.selector).click()
        assert await real_page.title() == "clicked"
