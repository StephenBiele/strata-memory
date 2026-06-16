"""MemoryVectorStore interface (spec §12 adapter model).

The core API stays stable even if zvec/TurboVec are replaced. Required methods:
upsert/search/remove/persist/stats; ``contains`` supports deletion verification. Vectors are
versioned artifacts tagged with an embedding generation so migrations and shadow indexing work
(spec §8). Canonical text is never stored here — only ids, vectors, content_hash, and the
minimal metadata needed for filters.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class VectorRecord:
    id: int                       # uint64-compatible canonical id
    vector: list[float]
    content_hash: str
    embedding_model_id: str
    embedding_generation: int
    metadata: dict = field(default_factory=dict)  # status/tier/sensitivity for filters


@dataclass
class VectorHit:
    id: int
    score: float
    content_hash: str
    embedding_generation: int


class MemoryVectorStore(abc.ABC):
    name: str

    @abc.abstractmethod
    def upsert(self, records: Iterable[VectorRecord]) -> dict[int, bool]: ...

    @abc.abstractmethod
    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        generation: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[VectorHit]: ...

    @abc.abstractmethod
    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        """Soft-delete ids from active query results; return per-ID success."""

    @abc.abstractmethod
    def contains(self, record_id: int) -> bool:
        """True if the id is present and not soft-deleted (counts for verification)."""

    @abc.abstractmethod
    def persist(self) -> None: ...

    @abc.abstractmethod
    def stats(self) -> dict: ...
