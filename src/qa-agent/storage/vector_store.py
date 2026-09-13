"""
Embedding-backed memory for selector/element resolution — a semantic
complement to `perception/selector_strategies/selector_cache.py`'s exact
`(page_signature, description)` lookup. Where the text cache only hits on
a near-identical description string, a vector store can match "the
button to finalize the purchase" against a previously cached "the 'Place
Order' button" by semantic similarity rather than surface text overlap.

`element_resolver.py` does not currently consume this — it's an
optional, additive resolution signal a deployment can wire in ahead of
the vision-grounding fallback (cheaper and faster than a vision call) as
the system matures past exact-text caching. Kept as its own module,
independent of `selector_cache.py`, so adopting it is opt-in.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol

from qa_agent.config.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Turns text into a vector. A real implementation wraps an embedding model API call; kept as a Protocol for the same provider-independence reason as every other LLM-adjacent interface in this codebase."""

    async def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    metadata: dict[str, str]


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float  # cosine similarity, 1.0 = identical direction
    metadata: dict[str, str]


class VectorStoreError(Exception):
    pass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise VectorStoreError(f"Vector dimension mismatch: {len(a)} vs {len(b)}.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore(Protocol):
    async def upsert(self, record: VectorRecord) -> None: ...
    async def query(self, vector: list[float], top_k: int = 5) -> list[VectorMatch]: ...
    async def delete(self, id: str) -> None: ...


class InMemoryVectorStore:
    """
    Brute-force cosine-similarity search over an in-process list. Fine
    for local dev/tests and for a single app's selector history (a few
    thousand records at most); a deployment with many target apps and a
    long resolution history should use `PgVectorStore` instead.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, record: VectorRecord) -> None:
        self._records[record.id] = record

    async def query(self, vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        scored = [
            VectorMatch(id=r.id, score=cosine_similarity(vector, r.vector), metadata=r.metadata)
            for r in self._records.values()
        ]
        return sorted(scored, key=lambda m: m.score, reverse=True)[:top_k]

    async def delete(self, id: str) -> None:
        self._records.pop(id, None)

    def __len__(self) -> int:
        return len(self._records)


class PgVectorStore:
    """
    Postgres + `pgvector` extension backed implementation, for production
    deployments (matches `infra/terraform/modules/vector_store/` and
    `docs/adr/0002-selector-strategy.md`'s pgvector default). Uses a
    SQLAlchemy async engine directly with raw SQL (rather than the ORM
    layer in `models.py`) since pgvector's `<=>` distance operator and
    vector column type aren't part of the core run/step/bug schema.

    Requires the `pgvector` Python package and a Postgres instance with
    `CREATE EXTENSION vector;` already applied (see
    `storage/migrations/versions/` for the migration that does this).
    """

    def __init__(self, engine: object, table_name: str = "selector_embeddings", dimensions: int = 1536) -> None:
        try:
            import pgvector  # noqa: F401
        except ImportError as exc:
            raise VectorStoreError(
                "PgVectorStore requires the 'pgvector' package to be installed "
                "(pip install pgvector)."
            ) from exc
        self._engine = engine
        self._table_name = table_name
        self._dimensions = dimensions

    async def upsert(self, record: VectorRecord) -> None:
        from sqlalchemy import text

        if len(record.vector) != self._dimensions:
            raise VectorStoreError(
                f"Vector has {len(record.vector)} dimensions, expected {self._dimensions}."
            )
        async with self._engine.begin() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                text(
                    f"INSERT INTO {self._table_name} (id, embedding, metadata) "
                    f"VALUES (:id, :embedding, :metadata) "
                    f"ON CONFLICT (id) DO UPDATE SET embedding = :embedding, metadata = :metadata"
                ),
                {"id": record.id, "embedding": str(record.vector), "metadata": json.dumps(record.metadata)},
            )

    async def query(self, vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        from sqlalchemy import text

        async with self._engine.connect() as conn:  # type: ignore[attr-defined]
            result = await conn.execute(
                text(
                    f"SELECT id, metadata, 1 - (embedding <=> :vector) AS score "
                    f"FROM {self._table_name} ORDER BY embedding <=> :vector LIMIT :top_k"
                ),
                {"vector": str(vector), "top_k": top_k},
            )
            rows = result.fetchall()
        return [
            VectorMatch(id=row.id, score=float(row.score), metadata=json.loads(row.metadata))
            for row in rows
        ]

    async def delete(self, id: str) -> None:
        from sqlalchemy import text

        async with self._engine.begin() as conn:  # type: ignore[attr-defined]
            await conn.execute(text(f"DELETE FROM {self._table_name} WHERE id = :id"), {"id": id})