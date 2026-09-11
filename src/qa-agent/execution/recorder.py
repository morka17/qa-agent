"""
Captures the evidence a filed bug report is built on: a Playwright trace
(full timeline, DOM snapshots, network, console), video, and point-in-time
screenshots. This is what makes Sentinel's bug reports "evidence-first" —
`reporting/evidence_bundler.py` packages whatever this module wrote to
disk/artifact storage into the final filed report.

Tracing/video are context-level concerns in Playwright (started before
any page exists), while screenshots are page-level — the `Recorder`
wraps both behind one lifecycle so `orchestration/agent_loop.py` doesn't
need to know the difference.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings

logger = get_logger(__name__)


class Tracing(Protocol):
    async def start(self, **kwargs: object) -> None: ...
    async def stop(self, path: str | None = None) -> None: ...


class TraceableContext(Protocol):
    @property
    def tracing(self) -> Tracing: ...


class Video(Protocol):
    async def path(self) -> str: ...


class RecordablePage(Protocol):
    async def screenshot(self, **kwargs: object) -> bytes: ...
    @property
    def video(self) -> Video | None: ...


@dataclass
class RunArtifacts:
    """
    Paths (relative to the run's artifact directory) of everything a
    `Recorder` produced for one run. `evidence_bundler.py` reads this to
    know what to attach to a filed bug report.
    """

    run_id: str
    run_dir: Path
    trace_path: Path | None = None
    video_path: Path | None = None
    screenshots: list[Path] = field(default_factory=list)

    def to_manifest(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "trace": str(self.trace_path) if self.trace_path else None,
            "video": str(self.video_path) if self.video_path else None,
            "screenshots": [str(p) for p in self.screenshots],
        }


class RecorderError(Exception):
    """Raised for capture failures that shouldn't silently produce an incomplete evidence set."""


class Recorder:
    """
    Example:
        recorder = Recorder(run_id="2026-09-01-a1b2c3")
        await recorder.start_tracing(context)
        ...
        await recorder.capture_screenshot(page, label="after_add_to_cart")
        ...
        artifacts = await recorder.finalize(context, page)
    """

    def __init__(
        self,
        run_id: str | None = None,
        settings: Settings | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.run_id = run_id or str(uuid.uuid4())
        root = Path(artifact_root or self._settings.artifact_local_path)
        self._run_dir = root / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._screenshots_dir = self._run_dir / "screenshots"
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._artifacts = RunArtifacts(run_id=self.run_id, run_dir=self._run_dir)
        self._tracing_started = False

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    async def start_tracing(
        self,
        context: TraceableContext,
        screenshots: bool = True,
        snapshots: bool = True,
        sources: bool = False,
    ) -> None:
        """
        Start Playwright's built-in trace recorder. Call this immediately
        after context creation — a trace only captures what happens
        after `start()`, so starting late silently produces an
        incomplete trace rather than an error, which is why this should
        be one of the very first calls in a run's lifecycle.
        """
        await context.tracing.start(
            screenshots=screenshots, snapshots=snapshots, sources=sources
        )
        self._tracing_started = True
        logger.debug("Started tracing for run %s.", self.run_id)

    async def stop_tracing(self, context: TraceableContext) -> Path:
        if not self._tracing_started:
            raise RecorderError(
                f"stop_tracing() called for run {self.run_id} but start_tracing() was never called."
            )
        trace_path = self._run_dir / "trace.zip"
        await context.tracing.stop(path=str(trace_path))
        self._artifacts.trace_path = trace_path
        self._tracing_started = False
        logger.debug("Stopped tracing for run %s -> %s.", self.run_id, trace_path)
        return trace_path

    async def capture_screenshot(
        self, page: RecordablePage, label: str, full_page: bool = True
    ) -> Path:
        """
        Capture a point-in-time screenshot, e.g. immediately after a
        failed assertion — the single most useful artifact for a human
        reviewing a filed bug. `label` becomes part of the filename, so
        callers should pass something descriptive ("after_add_to_cart",
        "assertion_failure").
        """
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        index = len(self._artifacts.screenshots)
        path = self._screenshots_dir / f"{index:03d}_{safe_label}.png"

        try:
            image_bytes = await page.screenshot(full_page=full_page)
        except Exception as exc:  # noqa: BLE001
            raise RecorderError(f"Failed to capture screenshot {label!r}: {exc}") from exc

        path.write_bytes(image_bytes)
        self._artifacts.screenshots.append(path)
        logger.debug("Captured screenshot %r -> %s.", label, path)
        return path

    async def finalize(
        self, context: TraceableContext | None = None, page: RecordablePage | None = None
    ) -> RunArtifacts:
        """
        Stop any still-running trace and locate the video file (Playwright
        writes video to disk only once the page/context closes), then
        return the complete `RunArtifacts` manifest.
        """
        if context is not None and self._tracing_started:
            await self.stop_tracing(context)

        if page is not None and page.video is not None:
            try:
                raw_video_path = await page.video.path()
                self._artifacts.video_path = Path(raw_video_path)
            except Exception as exc:  # noqa: BLE001 - video capture is best-effort, never blocks a run
                logger.warning("Could not resolve video path for run %s: %s", self.run_id, exc)

        logger.info(
            "Finalized artifacts for run %s: trace=%s, video=%s, screenshots=%d.",
            self.run_id,
            bool(self._artifacts.trace_path),
            bool(self._artifacts.video_path),
            len(self._artifacts.screenshots),
        )
        return self._artifacts