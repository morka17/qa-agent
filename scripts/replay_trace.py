#!/usr/bin/env python3
"""
Debug tool: opens a run's Playwright trace in the interactive trace
viewer (`playwright show-trace`), optionally after fetching it from a
run's artifact record in the database rather than requiring a local
file path directly.

Usage:
    # Open a trace file directly
    python scripts/replay_trace.py --file runs/2026-09-01-a1b2c3/trace.zip

    # Look up a run by ID and open its trace (requires the trace to be
    # reachable at the path/URL stored in its ArtifactORM row - a local
    # path works out of the box; a remote URL is downloaded first)
    python scripts/replay_trace.py --run-id 2026-09-01-a1b2c3

    # Just print a summary of what's in the trace without launching the
    # interactive viewer (useful in a CI log or over SSH)
    python scripts/replay_trace.py --file trace.zip --summary-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from sqlalchemy import select

from qa_agent.config.settings import get_settings
from qa_agent.storage.models import ArtifactORM, get_engine, get_sessionmaker


class ReplayError(Exception):
    pass


async def _find_trace_path_for_run(run_id: str) -> str:
    settings = get_settings()
    engine = get_engine(settings)
    sessionmaker = get_sessionmaker(engine)

    async with sessionmaker() as session:
        result = await session.execute(
            select(ArtifactORM).where(ArtifactORM.run_id == run_id, ArtifactORM.type == "trace")
        )
        artifact = result.scalar_one_or_none()

    await engine.dispose()

    if artifact is None:
        raise ReplayError(f"No trace artifact found for run {run_id!r}.")
    return artifact.path_or_url


def _resolve_local_path(path_or_url: str) -> Path:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="sentinel-trace-"))
        local_path = tmp_dir / "trace.zip"
        print(f"Downloading trace from {path_or_url} ...")
        urllib.request.urlretrieve(path_or_url, local_path)  # noqa: S310 - operator-supplied trusted URL
        return local_path

    local_path = Path(path_or_url)
    if not local_path.exists():
        raise ReplayError(f"Trace file not found at {local_path}.")
    return local_path


def _summarize_trace(trace_path: Path) -> None:
    """
    Playwright traces are zip files containing one or more JSONL
    `trace.trace` files. Each recorded API call appears as a paired
    `before`/`after` record — `before` carries `method` (e.g. "click",
    "goto") and `startTime`; the matching `after` record carries
    `endTime` and, on failure, an `error` field — rather than a single
    `"type": "action"` record. This parses that pairing to report a real
    action count, error count, and duration without needing the
    interactive viewer or a browser at all.
    """
    with zipfile.ZipFile(trace_path) as zf:
        trace_entries = [n for n in zf.namelist() if n.endswith("trace.trace")]
        if not trace_entries:
            print("No trace.trace entry found in this archive - is it a valid Playwright trace?")
            return

        action_count = 0
        error_count = 0
        start_times: list[float] = []
        end_times: list[float] = []

        for entry_name in trace_entries:
            with zf.open(entry_name) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    record_type = record.get("type")
                    if record_type == "before" and record.get("method"):
                        action_count += 1
                        if "startTime" in record:
                            start_times.append(record["startTime"])
                    elif record_type == "after":
                        if record.get("error"):
                            error_count += 1
                        if "endTime" in record:
                            end_times.append(record["endTime"])

        print(f"Trace file: {trace_path}")
        print(f"Actions recorded: {action_count}")
        print(f"Actions with errors: {error_count}")
        if start_times and end_times:
            duration_s = (max(end_times) - min(start_times)) / 1000
            print(f"Duration: {duration_s:.2f}s")


def _launch_viewer(trace_path: Path) -> None:
    if shutil.which("playwright") is None:
        raise ReplayError(
            "The 'playwright' CLI is not on PATH. Install it with "
            "'pip install playwright' or run 'npx playwright show-trace' instead."
        )
    subprocess.run(["playwright", "show-trace", str(trace_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="Path to a trace.zip file.")
    source.add_argument("--run-id", help="Run ID to look up the trace for via the database.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a text summary instead of launching the interactive trace viewer.",
    )
    args = parser.parse_args()

    try:
        path_or_url = args.file or asyncio.run(_find_trace_path_for_run(args.run_id))
        trace_path = _resolve_local_path(path_or_url)

        _summarize_trace(trace_path)
        if not args.summary_only:
            _launch_viewer(trace_path)
    except ReplayError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
