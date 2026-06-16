"""zvec hot adapter (spec §12, §11). Requires ``zvec==0.5.0`` (extras: adapters).

Encodes the backend's verified semantics (see docs/PINS.md, probed against 0.5.0):

* **Soft-delete**: ``delete(ids)`` removes ids from query results immediately and durably
  (roaring-bitmap DeleteStore, WAL-first). ``contains`` is checked via ``fetch``.
* **upsert, never insert**: ``delete_by_filter`` leaves the PK→doc_id map stale, so an
  ``insert`` of a previously-deleted id silently fails to become retrievable. The adapter
  always uses ``upsert()`` so re-adding a canonical id after any deletion works.
* **Physical purge is threshold-gated** (~30% deleted ratio): ``compact()`` calls
  ``optimize()`` but cannot guarantee disk reclamation below the threshold. The canonical
  tombstone — not zvec — is the authoritative deletion guarantee.
* **Similarity orientation**: zvec COSINE score is a distance (0.0 = identical); the adapter
  returns ``similarity = 1 - distance`` so higher = better.

Single-writer: a zvec collection is single-process-exclusive for writes, which is why the
Write Coordinator owns all mutation (only the one writer calls upsert/remove).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from strata.vector.base import MemoryVectorStore, VectorHit, VectorRecord

_VEC_FIELD = "vec"


def _require_zvec():
    try:
        import zvec  # noqa: F401
        return zvec
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "zvec is required for the hot adapter: pip install 'strata-memory[adapters]'"
        ) from exc


def _lit(value) -> str:
    """Render a filter literal: ints bare, strings single-quoted (zvec filter grammar)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


class ZvecHotAdapter(MemoryVectorStore):
    def __init__(self, path: str, *, dim: int, name: str = "hot", metric: str = "COSINE") -> None:
        zvec = _require_zvec()
        self.name = name
        self.path = path
        self.dim = dim
        self._zvec = zvec
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.isdir(path) and os.listdir(path):
            self._col = zvec.open(path)
        else:
            schema = zvec.CollectionSchema(
                name=name,
                fields=[
                    zvec.FieldSchema("content_hash", zvec.DataType.STRING),
                    zvec.FieldSchema("generation", zvec.DataType.INT32),
                    zvec.FieldSchema("status", zvec.DataType.STRING),
                    zvec.FieldSchema("tier", zvec.DataType.STRING),
                    zvec.FieldSchema("sensitivity", zvec.DataType.STRING),
                ],
                vectors=[
                    zvec.VectorSchema(
                        _VEC_FIELD, zvec.DataType.VECTOR_FP32, dimension=dim,
                        index_param=zvec.FlatIndexParam(metric_type=getattr(zvec.MetricType, metric)),
                    )
                ],
            )
            self._col = zvec.create_and_open(path, schema)

    # -- mutation --------------------------------------------------------------
    def upsert(self, records: Iterable[VectorRecord]) -> dict[int, bool]:
        zvec = self._zvec
        docs, ids = [], []
        for r in records:
            if len(r.vector) != self.dim:
                raise ValueError(
                    f"vector dim {len(r.vector)} != collection dim {self.dim} "
                    "(a different embedding dimension needs a separate shadow collection)"
                )
            md = r.metadata or {}
            docs.append(zvec.Doc(
                id=str(r.id),
                vectors={_VEC_FIELD: list(r.vector)},
                fields={
                    "content_hash": r.content_hash,
                    "generation": int(r.embedding_generation),
                    "status": str(md.get("status", "")),
                    "tier": str(md.get("tier", "")),
                    "sensitivity": str(md.get("sensitivity", "")),
                },
            ))
            ids.append(r.id)
        if docs:
            self._col.upsert(docs)   # always upsert (never insert) — flag #4
            self._col.flush()
        return {i: True for i in ids}

    def remove(self, ids: Iterable[int]) -> dict[int, bool]:
        ids = list(ids)
        if ids:
            self._col.delete([str(i) for i in ids])
            self._col.flush()
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
        if len(vector) != self.dim:
            return []  # cross-generation/dim vectors are not comparable
        zvec = self._zvec
        clauses = []
        if generation is not None:
            clauses.append(f"generation = {int(generation)}")
        for k, v in (filters or {}).items():
            clauses.append(f"{k} = {_lit(v)}")
        filter_expr = " AND ".join(clauses) if clauses else None
        res = self._col.query(
            zvec.Query(_VEC_FIELD, vector=list(vector)),
            topk=top_k,
            filter=filter_expr,
            output_fields=["content_hash", "generation"],
        )
        hits = []
        for doc in res:
            gen = doc.field("generation") if doc.has_field("generation") else (generation or 0)
            chash = doc.field("content_hash") if doc.has_field("content_hash") else ""
            hits.append(VectorHit(int(doc.id), 1.0 - float(doc.score), chash, int(gen)))
        return hits

    def contains(self, record_id: int) -> bool:
        result = self._col.fetch([str(record_id)])
        return str(record_id) in result

    # -- maintenance -----------------------------------------------------------
    def compact(self) -> None:
        """Run zvec compaction (optimize). Physical purge only above ~30% deleted ratio;
        not a deletion guarantee on its own (see docs/PINS.md flag #2)."""
        self._col.optimize()
        self._col.flush()

    def persist(self) -> None:
        self._col.flush()

    def stats(self) -> dict:
        s = self._col.stats
        return {"name": self.name, "rows": s.doc_count, "index_completeness": s.index_completeness}
