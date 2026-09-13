"""
Groups failures that are really "the same bug" across runs so
`reporting/bug_report_writer.py` files one ticket and updates its
occurrence count, rather than spamming the tracker with a new issue every
time a flaky-but-real bug reproduces.

Two failures are clustered together when their normalized fingerprints
are similar enough — normalization strips run-specific noise (numbers,
UUIDs, timestamps) from error text before comparing, since "expected 42
got 41" and "expected 58 got 57" are almost certainly the same underlying
defect, not two different ones. Similarity uses the same lightweight,
dependency-free fuzzy-matching approach as
`perception/selector_strategies/text_based.py`, kept deliberately simple
and explainable rather than embedding-based, since cluster decisions
directly affect whether a human sees one bug report or ten.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Protocol

from qa_agent.config.logging_config import get_logger
from qa_agent.triage.failure_classifier import FailureCategory

logger = get_logger(__name__)


class KeyValueStore(Protocol):
    """Same minimal async interface as `perception.selector_strategies.selector_cache.KeyValueStore`, kept as a separate local Protocol so `triage/` doesn't take a dependency on `perception/` purely for storage plumbing."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...


class InMemoryKeyValueStore:
    """Process-local store. Fine for a single worker/dev; production should back DedupeEngine with the same DB used for run/bug records so clusters are visible across workers."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


# Strips run-specific noise from error/assertion text before fingerprinting
# and similarity comparison: numbers, UUIDs, and quoted dynamic values —
# the parts of an error message that differ between occurrences of the
# same underlying bug.
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
_QUOTED_VALUE_PATTERN = re.compile(r"['\"][^'\"]{1,80}['\"]")

# Two failures are considered the same cluster above this similarity score.
_DEFAULT_SIMILARITY_THRESHOLD = 0.82

# How many of a target's existing clusters to compare a new failure
# against. Bounded so dedupe stays cheap even once a target app has
# accumulated a long failure history.
_MAX_CLUSTERS_TO_COMPARE = 200


def normalize_error_text(text: str) -> str:
    """Strip run-specific noise so two occurrences of the same bug fingerprint identically."""
    normalized = _UUID_PATTERN.sub("<uuid>", text)
    normalized = _QUOTED_VALUE_PATTERN.sub("<value>", normalized)
    normalized = _NUMBER_PATTERN.sub("<n>", normalized)
    return " ".join(normalized.lower().split())


@dataclass(frozen=True)
class FailureFingerprint:
    """
    The identity of one failure occurrence, used both to key which
    target-app "bucket" of clusters to search and to compute similarity
    against existing clusters in that bucket.
    """

    target_key: str  # e.g. domain + path, scopes clustering to the same app/page
    category: FailureCategory
    step_description: str
    assertion_description: str | None
    normalized_error: str

    def comparison_text(self) -> str:
        return " | ".join(
            filter(
                None,
                [
                    self.category.value,
                    self.step_description,
                    self.assertion_description or "",
                    self.normalized_error,
                ],
            )
        )


@dataclass
class FailureCluster:
    id: str
    fingerprint: FailureFingerprint
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    run_ids: list[str]
    filed_bug_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fingerprint": {
                "target_key": self.fingerprint.target_key,
                "category": self.fingerprint.category.value,
                "step_description": self.fingerprint.step_description,
                "assertion_description": self.fingerprint.assertion_description,
                "normalized_error": self.fingerprint.normalized_error,
            },
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count,
            "run_ids": self.run_ids,
            "filed_bug_ref": self.filed_bug_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FailureCluster":
        fp = data["fingerprint"]
        return cls(
            id=data["id"],
            fingerprint=FailureFingerprint(
                target_key=fp["target_key"],
                category=FailureCategory(fp["category"]),
                step_description=fp["step_description"],
                assertion_description=fp.get("assertion_description"),
                normalized_error=fp["normalized_error"],
            ),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            occurrence_count=data["occurrence_count"],
            run_ids=list(data["run_ids"]),
            filed_bug_ref=data.get("filed_bug_ref"),
        )


def build_fingerprint(
    target_key: str,
    category: FailureCategory,
    step_description: str,
    assertion_description: str | None,
    error_text: str,
) -> FailureFingerprint:
    return FailureFingerprint(
        target_key=target_key,
        category=category,
        step_description=step_description,
        assertion_description=assertion_description,
        normalized_error=normalize_error_text(error_text),
    )


class DedupeEngine:
    """
    Example:
        engine = DedupeEngine()
        fingerprint = build_fingerprint(
            target_key="shop.example.com/checkout",
            category=FailureCategory.APP_BUG,
            step_description="Submit the order",
            assertion_description="Order confirmation is shown",
            error_text="expected 'Order #4821 confirmed' but got 'Internal Server Error'",
        )
        cluster, is_new = await engine.find_or_create_cluster(fingerprint, run_id="run-123")
    """

    def __init__(
        self,
        store: KeyValueStore | None = None,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._store = store or InMemoryKeyValueStore()
        self._similarity_threshold = similarity_threshold

    @staticmethod
    def _bucket_key(target_key: str) -> str:
        digest = hashlib.sha256(target_key.encode("utf-8")).hexdigest()[:24]
        return f"dedupe_bucket:{digest}"

    async def _load_clusters(self, target_key: str) -> list[FailureCluster]:
        raw = await self._store.get(self._bucket_key(target_key))
        if raw is None:
            return []
        try:
            return [FailureCluster.from_dict(d) for d in json.loads(raw)]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Corrupt dedupe bucket for %r, resetting: %s", target_key, exc)
            return []

    async def _save_clusters(self, target_key: str, clusters: list[FailureCluster]) -> None:
        payload = json.dumps([c.to_dict() for c in clusters])
        await self._store.set(self._bucket_key(target_key), payload)

    def _best_match(
        self, fingerprint: FailureFingerprint, clusters: list[FailureCluster]
    ) -> tuple[FailureCluster, float] | None:
        candidates = [c for c in clusters if c.fingerprint.category == fingerprint.category]
        if not candidates:
            return None

        target_text = fingerprint.comparison_text()
        scored = [
            (cluster, SequenceMatcher(None, target_text, cluster.fingerprint.comparison_text()).ratio())
            for cluster in candidates[:_MAX_CLUSTERS_TO_COMPARE]
        ]
        best_cluster, best_score = max(scored, key=lambda item: item[1])
        return (best_cluster, best_score) if best_score >= self._similarity_threshold else None

    async def find_or_create_cluster(
        self, fingerprint: FailureFingerprint, run_id: str
    ) -> tuple[FailureCluster, bool]:
        """
        Returns `(cluster, is_new)`. `is_new=True` means this failure
        started a fresh cluster — the caller (`orchestration/agent_loop.py`
        or `reporting/`) should file a new bug; `is_new=False` means an
        existing bug's occurrence count was updated instead of filing a
        duplicate.
        """
        clusters = await self._load_clusters(fingerprint.target_key)
        match = self._best_match(fingerprint, clusters)
        now = datetime.now(timezone.utc)

        if match is not None:
            cluster, score = match
            cluster.occurrence_count += 1
            cluster.last_seen = now
            if run_id not in cluster.run_ids:
                cluster.run_ids.append(run_id)
            await self._save_clusters(fingerprint.target_key, clusters)
            logger.info(
                "Matched failure to existing cluster %s (similarity=%.2f, occurrence #%d).",
                cluster.id,
                score,
                cluster.occurrence_count,
            )
            return cluster, False

        new_cluster = FailureCluster(
            id=str(uuid.uuid4()),
            fingerprint=fingerprint,
            first_seen=now,
            last_seen=now,
            occurrence_count=1,
            run_ids=[run_id],
        )
        clusters.append(new_cluster)
        await self._save_clusters(fingerprint.target_key, clusters)
        logger.info(
            "Created new failure cluster %s for target %r.", new_cluster.id, fingerprint.target_key
        )
        return new_cluster, True

    async def mark_filed(self, target_key: str, cluster_id: str, bug_ref: str) -> None:
        """Record that a cluster has been filed against an issue tracker, so future matches can reference the existing ticket instead of re-filing."""
        clusters = await self._load_clusters(target_key)
        for cluster in clusters:
            if cluster.id == cluster_id:
                cluster.filed_bug_ref = bug_ref
                await self._save_clusters(target_key, clusters)
                return
        logger.warning("mark_filed: no cluster %r found for target %r.", cluster_id, target_key)