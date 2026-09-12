"""Unit tests for qa_agent.execution.network_interceptor."""

import pytest

from qa_agent.execution.network_interceptor import NetworkInterceptor, url_matches_any


class FakeRequest:
    def __init__(self, url, method="GET", body=None):
        self.url = url
        self.method = method
        self._body = body
        self.headers: dict[str, str] = {}

    def post_data(self):
        return self._body


class FakeRoute:
    def __init__(self, request):
        self.request = request
        self.fulfilled = None
        self.continued = False

    async def fulfill(self, **kwargs):
        self.fulfilled = kwargs

    async def continue_(self, **kwargs):
        self.continued = True

    async def abort(self, error_code="failed"):
        pass


class FakePage:
    def __init__(self):
        self.routed_handler = None

    async def route(self, pattern, handler):
        self.routed_handler = handler

    async def unroute(self, pattern):
        self.routed_handler = None

    def on(self, event, handler):
        pass


@pytest.mark.asyncio
class TestNetworkInterceptor:
    async def test_mocked_request_is_fulfilled_not_passed_through(self):
        interceptor = NetworkInterceptor()
        interceptor.mock("**/api/payment", status=200, body={"status": "approved"})
        page = FakePage()
        await interceptor.attach(page)

        route = FakeRoute(FakeRequest("https://shop.example.com/api/payment", method="POST", body='{"amount":10}'))
        await interceptor._handle_route(route)

        assert route.fulfilled is not None
        assert route.fulfilled["status"] == 200
        assert not route.continued

    async def test_unmocked_request_passes_through(self):
        interceptor = NetworkInterceptor()
        interceptor.mock("**/api/payment", status=200, body={})
        page = FakePage()
        await interceptor.attach(page)

        route = FakeRoute(FakeRequest("https://shop.example.com/api/other"))
        await interceptor._handle_route(route)

        assert route.continued
        assert route.fulfilled is None

    async def test_exchanges_are_recorded(self):
        interceptor = NetworkInterceptor()
        interceptor.mock("**/api/payment", status=200, body={})
        page = FakePage()
        await interceptor.attach(page)

        await interceptor._handle_route(FakeRoute(FakeRequest("https://x.com/api/payment", method="POST")))
        assert len(interceptor.exchanges) == 1
        assert interceptor.exchanges[0].was_mocked

    async def test_failed_exchanges_filters_by_status_threshold(self):
        interceptor = NetworkInterceptor()
        interceptor.mock("**/api/ok", status=200, body={})
        interceptor.mock("**/api/broken", status=500, body={"error": "oops"})
        page = FakePage()
        await interceptor.attach(page)

        await interceptor._handle_route(FakeRoute(FakeRequest("https://x.com/api/ok")))
        await interceptor._handle_route(FakeRoute(FakeRequest("https://x.com/api/broken")))

        failed = interceptor.failed_exchanges()
        assert len(failed) == 1
        assert failed[0].status == 500

    async def test_detach_clears_routed_handler(self):
        interceptor = NetworkInterceptor()
        page = FakePage()
        await interceptor.attach(page)
        assert page.routed_handler is not None
        await interceptor.detach()
        assert page.routed_handler is None

    async def test_later_registered_rule_takes_precedence(self):
        interceptor = NetworkInterceptor()
        interceptor.mock("**/api/*", status=200, body={"generic": True})
        interceptor.mock("**/api/special", status=201, body={"specific": True})
        page = FakePage()
        await interceptor.attach(page)

        route = FakeRoute(FakeRequest("https://x.com/api/special"))
        await interceptor._handle_route(route)
        assert route.fulfilled["status"] == 201


class TestUrlMatchesAny:
    def test_glob_pattern_matches(self):
        assert url_matches_any("https://x.com/api/checkout", ["**/api/checkout"])

    def test_no_match_returns_false(self):
        assert not url_matches_any("https://x.com/api/other", ["**/api/checkout"])

    def test_does_not_crash_on_double_star_glob_as_invalid_regex(self):
        # "**/foo" is invalid regex syntax ("nothing to repeat") but must
        # not raise - it should just fail the regex branch and rely on
        # the glob check.
        result = url_matches_any("https://x.com/foo", ["**/foo"])
        assert result is True
