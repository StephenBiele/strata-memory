"""TurboVec archive adapter (spec §9, §11, §12). Requires ``turbovec==0.8.0``.

Drives ``IdMapIndex`` directly with canonical uint64 record ids — no LangChain/Haystack/
LlamaIndex wrapper, and no JSON/docstore sidecar. The SQLite canonical store remains the sole
source of truth for text, metadata, deletion state, and provenance; this index stores only
vectors keyed by id. Verified semantics (probed against 0.8.0, see docs/PINS.md):

* **Stable uint64 ids** via ``add_with_ids``; ``contains``/``remove`` operate by id.
* **upsert = remove-if-present then add**: a duplicate ``add_with_ids`` of an existing id
  raises ``id already present``. ``remove`` then re-add of the same id works cleanly.
* **Similarity-only ranking**: ``search`` returns (scores, ids) with higher = better. Vectors
  are 2–4 bit quantized, so max-marginal-relevance / diversity reranking cannot run on the raw
  index vectors — diversity is applied AFTER hydration from canonical (see resolver.diversity).
* **Per-generation indices**: quantized vectors are not comparable across embedding
  generations, so each generation is its own ``IdMapIndex`` / ``.tvim`` file (spec §8, §11).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from strata.vector.base import MemoryVectorStore, VectorHit, VectorRecord


def _require():
    try:
        import numpy as np
        import turbovec
        return np, turbovec
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "turbovec + numpy are required for the archive adapter: "
            "pip install 'strata-memory[adapters]'"
        ) from exc


class TurboVecArchiveAdapter(MemoryVectorStore):
    def __init__(self, path: str, *, dim: int, name: str = "archive") -> None:
        self._np, self._tv = _require()
        self.name = name
        self.path = path
        self.dim = dim
        os.makedirs(path, exist_ok=True)
        # generation -> IdMapIndex. Each generation is an independent quantized index.
        self._indices: dict[int, object] = {}
        # generation -> set of live ids (id-only bookkeeping, not a text sidecar): lets us
        # size search k safely (IdMapIndex exposes no count) and report stats.
        self._ids: dict[int, set[int]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for fn in os.listdir(self.path):
            if fn.startswith(f"{self.name}.gen") and fn.endswith(".tvim"):
                gen = int(fn[len(f"{self.name}.gen"):-len(".tvim")])
                self._indices[gen] = self._tv.IdMapIndex.load(os.path.join(self.path, fn))
                self._ids.setdefault(gen, set())

    def _index_for(self, generation: int):
        idx = self._indices.get(generation)
        if idx is None:
            idx = self._tv.IdMapIndex(self.dim)
            self._indices[generation] = idx
        return idx

    # -- mutation --------------------------------------------------------------
    def upsert(self, records: Iterable[VectorRecord]) -> dict[int, bool]:
        np = self._np
        by_gen: dict[int, list[VectorRecord]] = {}
        for r in records:
            if len(r.vector) != self.dim:
                raise ValueError(
                    f"vector dim {len(r.vector)} != index dim {self.dim} "
                    "(a different embedding dimension is a separate generation index)"
                )
            by_gen.setdefault(int(r.embedding_generation), []).append(r)
        out: dict[int, bool] = {}
        for gen, recs in by_gen.items():
            idx = self._index_for(gen)
            live = self._ids.setdefault(gen, set())
            for r in recs:
                if idx.contains(r.id):
                    idx.remove(r.id)  # upsert: duplicate add would raise
            vectors = np.asarray([r.vector for r in recs], dtype="float32")
            ids = np.asarray([r.id for r in recs], dtype="uint64")
            idx.add_with_ids(vectors, ids)
            idx.prepare()
            for r in recs:
                live.add(r.id)
                out[r.id] = True
        return out

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        ids = list(ids)
        for gen, idx in self._indices.items():
            removed = False
            for i in ids:
                if idx.contains(i):
                    idx.remove(i)
                    self._ids.get(gen, set()).discard(i)
                    removed = True
            if removed:
                idx.prepare()
        return {i: not self.contains(i) for i in ids}

    # -- query -----------------------------------------------------------------
    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        generation: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[VectorHit]:
        np = self._np
        if len(vector) != self.dim:
            return []
        gens = [generation] if generation is not None else list(self._indices)
        hits: list[VectorHit] = []
        q = np.asarray([vector], dtype="float32")
        for gen in gens:
            idx = self._indices.get(gen)
            if idx is None:
                continue
            live = len(self._ids.get(gen, ()))
            if live == 0:
                continue
            k = min(top_k, live)
            scores, ids = idx.search(q, k)
            for score, rid in zip(scores[0].tolist(), ids[0].tolist()):
                # Similarity-only (higher = better). content_hash not stored in the archive;
                # hydration integrity is enforced against canonical, not the index.
                hits.append(VectorHit(int(rid), float(score), "", gen))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def contains(self, record_id: int) -> bool:
        return any(idx.contains(record_id) for idx in self._indices.values())

    # -- persistence -----------------------------------------------------------
    def persist(self) -> None:
        for gen, idx in self._indices.items():
            idx.write(os.path.join(self.path, f"{self.name}.gen{gen}.tvim"))

    def purge_generation(self, generation: int) -> None:
        """Drop an obsolete .tvim generation (spec §11 hard-delete for TurboVec)."""
        self._indices.pop(generation, None)
        self._ids.pop(generation, None)
        fn = os.path.join(self.path, f"{self.name}.gen{generation}.tvim")
        if os.path.exists(fn):
            os.remove(fn)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "generations": sorted(self._indices),
            "rows": sum(len(s) for s in self._ids.values()),
        }
