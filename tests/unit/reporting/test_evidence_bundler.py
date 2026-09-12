"""Unit tests for qa_agent.reporting.evidence_bundler."""

from pathlib import Path

import pytest

from qa_agent.execution.network_interceptor import RecordedExchange
from qa_agent.execution.recorder import RunArtifacts
from qa_agent.reporting.evidence_bundler import EvidenceBundler
from qa_agent.reporting.report_schema import EvidenceType
from qa_agent.verification.console_error_monitor import ConsoleMessageLevel, RecordedConsoleEntry


class FailingUploader:
    async def upload(self, local_path: Path, run_id: str) -> str:
        raise ConnectionError("upload service unreachable")


class WorkingUploader:
    async def upload(self, local_path: Path, run_id: str) -> str:
        return f"https://cdn.example.com/{run_id}/{local_path.name}"


@pytest.mark.asyncio
class TestEvidenceBundler:
    async def test_bundles_trace_video_and_screenshots(self, tmp_path):
        trace = tmp_path / "trace.zip"
        trace.write_bytes(b"trace")
        video = tmp_path / "video.webm"
        video.write_bytes(b"video")
        screenshot = tmp_path / "shot.png"
        screenshot.write_bytes(b"png")

        artifacts = RunArtifacts(
            run_id="run-1", run_dir=tmp_path, trace_path=trace, video_path=video, screenshots=[screenshot]
        )
        bundler = EvidenceBundler()
        evidence = await bundler.bundle(artifacts)

        types = {e.type for e in evidence}
        assert types == {EvidenceType.TRACE, EvidenceType.VIDEO, EvidenceType.SCREENSHOT}
        assert all(not e.is_uploaded for e in evidence)

    async def test_uploader_success_marks_is_uploaded(self, tmp_path):
        trace = tmp_path / "trace.zip"
        trace.write_bytes(b"trace")
        artifacts = RunArtifacts(run_id="run-1", run_dir=tmp_path, trace_path=trace)

        bundler = EvidenceBundler(uploader=WorkingUploader())
        evidence = await bundler.bundle(artifacts)

        assert evidence[0].is_uploaded
        assert evidence[0].path_or_url.startswith("https://cdn.example.com/")

    async def test_uploader_failure_falls_back_to_local_path(self, tmp_path):
        trace = tmp_path / "trace.zip"
        trace.write_bytes(b"trace")
        artifacts = RunArtifacts(run_id="run-1", run_dir=tmp_path, trace_path=trace)

        bundler = EvidenceBundler(uploader=FailingUploader())
        evidence = await bundler.bundle(artifacts)

        assert not evidence[0].is_uploaded
        assert evidence[0].path_or_url == str(trace)

    async def test_console_errors_bundled_as_single_attachment(self, tmp_path):
        artifacts = RunArtifacts(run_id="run-1", run_dir=tmp_path)
        entries = [
            RecordedConsoleEntry(level=ConsoleMessageLevel.ERROR, text=f"error {i}", source="console", url=None)
            for i in range(3)
        ]
        bundler = EvidenceBundler()
        evidence = await bundler.bundle(artifacts, console_errors=entries)

        assert len(evidence) == 1
        assert evidence[0].type == EvidenceType.CONSOLE_LOG
        assert "3 console error" in evidence[0].description

    async def test_network_failures_bundled_as_single_attachment(self, tmp_path):
        artifacts = RunArtifacts(run_id="run-1", run_dir=tmp_path)
        failures = [
            RecordedExchange(
                url="https://x.com/api",
                method="POST",
                request_body=None,
                status=500,
                response_body=None,
                was_mocked=False,
            )
        ]
        bundler = EvidenceBundler()
        evidence = await bundler.bundle(artifacts, network_failures=failures)

        assert len(evidence) == 1
        assert evidence[0].type == EvidenceType.NETWORK_LOG

    async def test_empty_artifacts_produce_no_evidence(self, tmp_path):
        artifacts = RunArtifacts(run_id="run-1", run_dir=tmp_path)
        bundler = EvidenceBundler()
        evidence = await bundler.bundle(artifacts)
        assert evidence == []
