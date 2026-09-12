"""Unit tests for qa_agent.execution.recorder."""

import pytest

from qa_agent.execution.recorder import Recorder, RecorderError


class FakeTracing:
    def __init__(self):
        self.started = False
        self.stop_path = None

    async def start(self, **kwargs):
        self.started = True

    async def stop(self, path=None):
        self.stop_path = path
        with open(path, "wb") as f:
            f.write(b"fake-trace-zip")


class FakeContext:
    def __init__(self):
        self.tracing = FakeTracing()


class FakeVideo:
    async def path(self) -> str:
        return "/tmp/fake-video.webm"


class FakePage:
    def __init__(self):
        self.video = FakeVideo()

    async def screenshot(self, **kwargs) -> bytes:
        return b"fake-png-bytes"


class BrokenScreenshotPage(FakePage):
    async def screenshot(self, **kwargs):
        raise RuntimeError("screenshot backend unavailable")


class NoVideoPage:
    """A page whose `video` attribute is genuinely None (recording wasn't enabled for this context)."""

    video = None

    async def screenshot(self, **kwargs) -> bytes:
        return b"fake-png-bytes"


@pytest.mark.asyncio
class TestRecorder:
    async def test_full_lifecycle(self, tmp_path):
        recorder = Recorder(run_id="test-run-1", artifact_root=tmp_path)
        context = FakeContext()
        page = FakePage()

        await recorder.start_tracing(context)
        assert context.tracing.started

        path1 = await recorder.capture_screenshot(page, label="before click")
        path2 = await recorder.capture_screenshot(page, label="after click")
        assert path1.exists()
        assert path1.read_bytes() == b"fake-png-bytes"
        assert path1 != path2

        artifacts = await recorder.finalize(context, page)
        assert artifacts.trace_path is not None
        assert artifacts.trace_path.exists()
        assert artifacts.video_path is not None
        assert len(artifacts.screenshots) == 2

    async def test_stop_tracing_without_start_raises(self, tmp_path):
        recorder = Recorder(run_id="test-run-2", artifact_root=tmp_path)
        with pytest.raises(RecorderError):
            await recorder.stop_tracing(FakeContext())

    async def test_screenshot_failure_raises_recorder_error(self, tmp_path):
        recorder = Recorder(run_id="test-run-3", artifact_root=tmp_path)
        with pytest.raises(RecorderError):
            await recorder.capture_screenshot(BrokenScreenshotPage(), label="x")

    async def test_finalize_without_video_does_not_raise(self, tmp_path):
        recorder = Recorder(run_id="test-run-4", artifact_root=tmp_path)
        context = FakeContext()
        await recorder.start_tracing(context)
        page = NoVideoPage()

        artifacts = await recorder.finalize(context, page)
        assert artifacts.video_path is None

    async def test_manifest_serialization(self, tmp_path):
        recorder = Recorder(run_id="test-run-5", artifact_root=tmp_path)
        context = FakeContext()
        page = FakePage()
        await recorder.start_tracing(context)
        await recorder.capture_screenshot(page, label="x")
        artifacts = await recorder.finalize(context, page)

        manifest = artifacts.to_manifest()
        assert manifest["run_id"] == "test-run-5"
        assert manifest["trace"] is not None
        assert len(manifest["screenshots"]) == 1
