"""
Memoizes resolved element locators so repeat runs against a stable page
don't re-pay the cost of role/text scoring (cheap) or, more importantly,
a vision-model fallback call (slow and not free) for a target description
that's already been resolved before.

A cache hit is never trusted blindly: `element_resolver.py` always
re-validates a cached entry against the *current* snapshot (does an
element with this dom_path/role/name still exist and is it visible?)
before using it, since the whole point of semantic resolution is
surviving markup changes that would silently break a raw selector.
Entries that fail re-validation are treated as stale and evicted.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Protocol


class KeyValueStore(Protocol):
    """
    The minimal async key-value interface this cache depends on. The
    default implementation is in-process (`InMemoryKeyValueStore`); a
    production deployment would back this with Redis so the cache
    survives process restarts and is shared across browser workers.
    """

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


class InMemoryKeyValueStore:
    """Process-local cache backend. Fine for local dev/tests; not shared across workers."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


@dataclass(frozen=True)
class CachedSelector:
    """
    A previously successful resolution, stored so it can be re-validated
    and reused. Deliberately does NOT store a raw index (indices aren't
    stable across snapshots) — it stores the identifying signal
    (dom_path/role/accessible_name) needed to re-find the same logical
    element in a fresh snapshot.
    """

    dom_path: str
    role: str | None
    accessible_name: str | None
    strategy: str
    confidence: float
    cached_at: float  # unix timestamp

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "CachedSelector":
        return cls(**json.loads(raw))


_DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days; app UIs change slower than that on average


class SelectorCache:
    """
    High-level cache facade used by `element_resolver.py`.

    Cache keys are scoped by `page_signature` (see `DomSnapshot.page_signature`)
    so a description like "the submit button" cached on one page never
    collides with the same phrase on a structurally different page.
    """

    def __init__(self, store: KeyValueStore | None = None, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._store = store or InMemoryKeyValueStore()
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _normalize_description(description: str) -> str:
        return " ".join(description.strip().lower().split())

    def _make_key(self, page_signature: str, description: str) -> str:
        normalized = self._normalize_description(description)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return f"selector:{page_signature}:{digest}"

    async def get(self, page_signature: str, description: str) -> CachedSelector | None:
        key = self._make_key(page_signature, description)
        raw = await self._store.get(key)
        if raw is None:
            return None
        try:
            return CachedSelector.from_json(raw)
        except (json.JSONDecodeError, TypeError):
            # Corrupt/incompatible entry (e.g. after a schema change) —
            # treat as a miss rather than propagating an error.
            await self._store.delete(key)
            return None

    async def set(
        self,
        page_signature: str,
        description: str,
        dom_path: str,
        role: str | None,
        accessible_name: str | None,
        strategy: str,
        confidence: float,
    ) -> None:
        key = self._make_key(page_signature, description)
        entry = CachedSelector(
            dom_path=dom_path,
            role=role,
            accessible_name=accessible_name,
            strategy=strategy,
            confidence=confidence,
            cached_at=time.time(),
        )
        await self._store.set(key, entry.to_json(), ttl_seconds=self._ttl_seconds)

    async def invalidate(self, page_signature: str, description: str) -> None:
        key = self._make_key(page_signature, description)
        await self._store.delete(key)