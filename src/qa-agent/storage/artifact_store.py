"""
Durable storage for run artifacts (videos, traces, screenshots) —
implements `reporting.evidence_bundler.ArtifactUploader`, so any of these
backends can be passed straight into `EvidenceBundler(uploader=...)`.

Backend selection follows `settings.artifact_store_provider`
("s3" / "gcs" / "local"); `build_artifact_store()` is the single factory
function production code (`api/app.py`'s startup) should call rather than
importing a specific backend directly, so switching providers is a
settings change, not a code change.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from qa_agent.config.logging_config import get_logger
from qa_agent.config.settings import Settings, get_settings

logger = get_logger(__name__)


class ArtifactStoreError(Exception):
    pass


class LocalArtifactStore:
    """
    Copies artifacts into a locally-served directory and returns a
    configured base-URL-relative path. Suitable for local dev/CI where
    the agent worker and whatever serves artifacts (e.g. a simple static
    file server, or a shared volume mounted into the issue tracker's
    environment) share a filesystem.
    """

    def __init__(self, root: str | Path, public_base_url: str | None = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None

    async def upload(self, local_path: Path, run_id: str) -> str:
        dest_dir = self._root / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / local_path.name

        if local_path.resolve() != dest_path.resolve():
            shutil.copy2(local_path, dest_path)

        if self._public_base_url:
            url = f"{self._public_base_url}/{run_id}/{local_path.name}"
        else:
            url = str(dest_path)

        logger.debug("Copied artifact %s -> %s.", local_path, url)
        return url


class S3ArtifactStore:
    """
    Uploads to S3 (or an S3-compatible store, e.g. MinIO/R2 via a custom
    endpoint). Lazily imports `boto3`, which is synchronous — wrapped
    with `asyncio.to_thread` so it doesn't block the event loop other
    run stages are running on.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "sentinel-qa",
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ArtifactStoreError(
                "S3ArtifactStore requires the 'boto3' package to be installed."
            ) from exc
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url)

    async def upload(self, local_path: Path, run_id: str) -> str:
        import asyncio

        key = f"{self._prefix}/{run_id}/{local_path.name}"
        await asyncio.to_thread(self._client.upload_file, str(local_path), self._bucket, key)
        url = f"https://{self._bucket}.s3.amazonaws.com/{key}"
        logger.info("Uploaded artifact %s -> s3://%s/%s.", local_path, self._bucket, key)
        return url


class GCSArtifactStore:
    """Uploads to Google Cloud Storage. Lazily imports `google-cloud-storage`, also sync-wrapped via `asyncio.to_thread`."""

    def __init__(self, bucket: str, prefix: str = "sentinel-qa") -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ArtifactStoreError(
                "GCSArtifactStore requires the 'google-cloud-storage' package to be installed."
            ) from exc
        self._prefix = prefix.strip("/")
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self._bucket_name = bucket

    async def upload(self, local_path: Path, run_id: str) -> str:
        import asyncio

        blob_name = f"{self._prefix}/{run_id}/{local_path.name}"
        blob = self._bucket.blob(blob_name)
        await asyncio.to_thread(blob.upload_from_filename, str(local_path))
        url = f"https://storage.googleapis.com/{self._bucket_name}/{blob_name}"
        logger.info("Uploaded artifact %s -> gs://%s/%s.", local_path, self._bucket_name, blob_name)
        return url


def build_artifact_store(settings: Settings | None = None):
    """Factory selecting the configured backend. Returns an object satisfying `reporting.evidence_bundler.ArtifactUploader`."""
    settings = settings or get_settings()

    if settings.artifact_store_provider == "s3":
        return S3ArtifactStore(bucket=settings.artifact_bucket)
    if settings.artifact_store_provider == "gcs":
        return GCSArtifactStore(bucket=settings.artifact_bucket)
    return LocalArtifactStore(root=settings.artifact_local_path)