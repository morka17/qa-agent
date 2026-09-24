"""
Turns everything gathered during a run - trace, video, screenshots
(`execution/recorder.py`), console errors (`verification/console_error_monitor.py`),
and network failures (`execution/network_interceptor.py`) - into the
`EvidenceAttachment` list a `BugReport` carries.

Local artifact paths are optionally pushed to durable storage via an
injected `ArtifactUploader`, so a report filed against an external issue
tracker links to something reachable from outside the agent's own
filesystem rather than a path only the worker process can see. Without
an uploader, attachments simply carry their local path - fine for local
dev/CI where the tracker and the agent share a filesystem or artifact
volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.execution.network_interceptor import RecordedExchange
from qa_agent.execution.recorder import RunArtifacts
from qa_agent.reporting.report_schema import EvidenceAttachment, EvidenceType
from qa_agent.verification.console_error_monitor import RecordedConsoleEntry

logger = get_logger(__name__)

# Cap how many individual console/network log lines get their own
# attachment entry — beyond this, they're rolled into one combined log
# attachment instead, so a chatty page doesn't produce a 200-item
# evidence list that buries the screenshots/trace/video that matter most.
_MAX_INDIVIDUAL_LOG_ATTACHMENTS = 10


class ArtifactUploader(Protocol):
    """
    Pushes a local file to durable storage and returns a public/artifact-store
    URL. Real implementations back onto S3/GCS (per `artifact_store_provider`
    in settings); kept as a Protocol here so `EvidenceBundler` has zero
    dependency on which backend is configured.
    """

    async def upload(self, local_path: Path, run_id: str) -> str: ...


class EvidenceBundler:
    """
    Example:
        bundler = EvidenceBundler(uploader=my_s3_uploader)
        evidence = await bundler.bundle(
            artifacts=run_artifacts,
            console_errors=monitor.errors(),
            network_failures=interceptor.failed_exchanges(),
        )
    """

    def __init__(self, uploader: ArtifactUploader | None = None) -> None:
        self._uploader = uploader

    async def bundle(
        self,
        artifacts: RunArtifacts,
        console_errors: list[RecordedConsoleEntry] | None = None,
        network_failures: list[RecordedExchange] | None = None,
    ) -> list[EvidenceAttachment]:
        attachments: list[EvidenceAttachment] = []

        if artifacts.trace_path is not None:
            attachments.append(await self._attach_file(
                artifacts.trace_path, EvidenceType.TRACE, "Full Playwright trace (timeline, DOM, network, console).", artifacts.run_id
            ))

        if artifacts.video_path is not None:
            attachments.append(await self._attach_file(
                artifacts.video_path, EvidenceType.VIDEO, "Screen recording of the full run.", artifacts.run_id
            ))

        for screenshot_path in artifacts.screenshots:
            attachments.append(await self._attach_file(
                screenshot_path,
                EvidenceType.SCREENSHOT,
                f"Screenshot: {screenshot_path.stem}",
                artifacts.run_id,
            ))

        if console_errors:
            attachments.append(self._bundle_console_errors(console_errors))

        if network_failures:
            attachments.append(self._bundle_network_failures(network_failures))

        logger.info(
            "Bundled %d evidence attachment(s) for run %s.", len(attachments), artifacts.run_id
        )
        return attachments

    async def _attach_file(
        self, path: Path, evidence_type: EvidenceType, description: str, run_id: str
    ) -> EvidenceAttachment:
        if self._uploader is not None:
            try:
                url = await self._uploader.upload(path, run_id)
                return EvidenceAttachment(
                    type=evidence_type, description=description, path_or_url=url, is_uploaded=True
                )
            except Exception as exc:  # noqa: BLE001 - a failed upload shouldn't drop the evidence, just its durability
                logger.warning(
                    "Failed to upload %s (%s) for run %s, falling back to local path: %s",
                    evidence_type.value,
                    path,
                    run_id,
                    exc,
                )
        return EvidenceAttachment(
            type=evidence_type, description=description, path_or_url=str(path), is_uploaded=False
        )

    @staticmethod
    def _bundle_console_errors(entries: list[RecordedConsoleEntry]) -> EvidenceAttachment:
        shown = entries[:_MAX_INDIVIDUAL_LOG_ATTACHMENTS]
        overflow = len(entries) - len(shown)
        lines = [f"[{e.source}] {e.text}" for e in shown]
        if overflow > 0:
            lines.append(f"... and {overflow} more")
        return EvidenceAttachment(
            type=EvidenceType.CONSOLE_LOG,
            description=f"{len(entries)} console error(s) captured during the run.",
            path_or_url="data:text/plain," + "\n".join(lines),
            is_uploaded=False,
        )

    @staticmethod
    def _bundle_network_failures(exchanges: list[RecordedExchange]) -> EvidenceAttachment:
        shown = exchanges[:_MAX_INDIVIDUAL_LOG_ATTACHMENTS]
        overflow = len(exchanges) - len(shown)
        lines = [f"{e.method} {e.url} -> status={e.status}" for e in shown]
        if overflow > 0:
            lines.append(f"... and {overflow} more")
        return EvidenceAttachment(
            type=EvidenceType.NETWORK_LOG,
            description=f"{len(exchanges)} failed network request(s) captured during the run.",
            path_or_url="data:text/plain," + "\n".join(lines),
            is_uploaded=False,
        )