"""In-memory fake vector store — the GATE target for the invariant suites (spec step 6).

Models the semantics the real adapters must honor:
* **Soft-delete**: removed ids stay in storage but are excluded from search/contains (mirrors
  zvec's roaring-bitmap DeleteStore).
* **Generation tagging**: multiple embedding generations can coexist for shadow indexing /
  migration; search is scoped to a generation.
* **Similarity-only ranking** by cosine; no diversity on the raw vectors (diversity is applied
  post-hydration, per spec §9 for the archive).
"""

from __future__ import annotations

from typing import Iterable, Optional

from strata.vector.base import MemoryVectorStore, VectorHit, VectorRecord
from strata.vector.embedder import cosine


class InMemoryVectorStore(MemoryVectorStore):
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        # (id, generation) -> VectorRecord
        self._by_key: dict[tuple[int, int], VectorRecord] = {}
        self._deleted: set[int] = set()  # soft-deleted ids (across all generations)

    def upsert(self, records: Iterable[VectorRecord]) -> dict[int, bool]:
        out: dict[int, bool] = {}
        for rec in records:
            self._by_key[(rec.id, rec.embedding_generation)] = rec
            # Re-adding an id clears any prior soft-delete (upsert semantics).
            self._deleted.discard(rec.id)
            out[rec.id] = True
        return out

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        generation: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[VectorHit]:
        hits = []
        for (rid, gen), rec in list(self._by_key.items()):  # snapshot: safe under concurrent writes
            if rid in self._deleted:
                continue
            if generation is not None and gen != generation:
                continue
            if filters and not self._matches(rec, filters):
                continue
            if len(rec.vector) != len(vector):
                continue  # cross-generation vectors are not comparable
            hits.append(VectorHit(rid, cosine(vector, rec.vector), rec.content_hash, gen))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _matches(rec: VectorRecord, filters: dict) -> bool:
        return all(rec.metadata.get(k) == v for k, v in filters.items())

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        out = {}
        for i in ids:
            self._deleted.add(i)
            out[i] = True
        return out

    def contains(self, record_id: int) -> bool:
        if record_id in self._deleted:
            return False
        return any(rid == record_id for (rid, _gen) in list(self._by_key))

    def purge_deleted(self) -> int:
        """Physically drop soft-deleted rows (compaction analogue). Returns rows purged."""
        keys = [(rid, gen) for (rid, gen) in self._by_key if rid in self._deleted]
        for k in keys:
            del self._by_key[k]
        n = len(keys)
        self._deleted -= {rid for rid, _ in keys}
        return n

    def persist(self) -> None:
        # In-memory: nothing to flush.
        return None

    def stats(self) -> dict:
        return {
            "name": self.name,
            "rows": len(self._by_key),
            "soft_deleted": len(self._deleted),
            "generations": sorted({gen for _id, gen in self._by_key}),
        }
