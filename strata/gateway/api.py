"""Strata gateway — the stable, framework-neutral API (spec §12).

In-process facade over the MemoryEngine exposing the durable API surface: write_event,
write_memory, recall, update_memory, supersede_memory, delete_memory, deletion_status,
explain_memory, run_reflection. ``gateway/http.py`` wraps this as JSON/HTTP.
"""

from __future__ import annotations

from typing import Optional

from strata.canonical.records import (
    MemoryRecord,
    RecordType,
    Sensitivity,
    Status,
    Tier,
)
from strata.canonical.store import CanonicalStore
from strata.coordinator.coordinator import WriteCoordinator
from strata.engine import MemoryEngine
from strata.lexical.fts import LexicalStore
from strata.reflection.engine import ReflectionEngine
from strata.resolver.resolver import Resolver
from strata.vector.embedder import DeterministicHashEmbedder
from strata.vector.fake import InMemoryVectorStore


def _summary(rec: MemoryRecord) -> dict:
    return {
        "id": rec.id,
        "content": rec.content,
        "tier": rec.tier.value,
        "record_type": rec.record_type.value,
        "status": rec.status.value,
        "sensitivity": rec.sensitivity.value,
        "content_hash": rec.content_hash,
    }


class Strata:
    def __init__(self, engine: MemoryEngine, reflection: ReflectionEngine) -> None:
        self.engine = engine
        self.reflection = reflection
        self.store = engine.store

    # -- construction ----------------------------------------------------------
    @classmethod
    def open(cls, *, db_path: str = ":memory:", vector_factory=None,
             embedder=None) -> "Strata":
        store = CanonicalStore(db_path)
        lexical = LexicalStore(store)
        coordinator = WriteCoordinator(store)
        resolver = Resolver(store)
        # A host may inject a real embedding model; the default stays offline and
        # reproducible (spec §8). A custom embedder gets its own generation so the
        # embedding_generations table stays honest about which model wrote vectors.
        if embedder is None:
            embedder = DeterministicHashEmbedder()
            generation = 1
        else:
            generation = 2
        vstore = vector_factory(embedder.dim) if vector_factory else InMemoryVectorStore()
        engine = MemoryEngine(store, lexical, {"hot": vstore}, coordinator, resolver,
                              embedder, active_generation=generation)
        return cls(engine, ReflectionEngine(engine))

    def close(self) -> None:
        self.store.close()

    # -- writes ----------------------------------------------------------------
    def write_event(self, content: str, **fields) -> dict:
        rec = MemoryRecord.create(
            content, record_type=RecordType.SYSTEM, tier=Tier.L0, **fields
        )
        return _summary(self.engine.write_memory(rec))

    def write_memory(
        self,
        content: str,
        *,
        tier: str = "L1",
        record_type: str = "fact",
        sensitivity: str = "normal",
        confidence: Optional[float] = None,
        **fields,
    ) -> dict:
        rec = MemoryRecord.create(
            content,
            tier=Tier(tier),
            record_type=RecordType(record_type),
            sensitivity=Sensitivity(sensitivity),
            confidence=confidence,
            **fields,
        )
        return _summary(self.engine.write_memory(rec))

    def update_memory(self, record_id: int, *, content: Optional[str] = None, **patch) -> dict:
        old = self.store.read_one(record_id)
        if old is None:
            raise KeyError(record_id)
        if content is not None:
            # FTS5-first ordering: drop old postings before canonical content changes.
            self.engine.lexical.remove([record_id])
            updated = old.with_content(content)
            self.engine.write_memory(updated)
            return _summary(self.store.read_one(record_id))
        # metadata-only patch
        if "status" in patch:
            self.store.set_status(record_id, Status(patch["status"]))
        return _summary(self.store.read_one(record_id))

    def supersede_memory(self, old_id: int, content: str, **fields) -> dict:
        new = MemoryRecord.create(content, **fields)
        return _summary(self.engine.correct(old_id, new))

    # -- recall ----------------------------------------------------------------
    def recall(self, query: str, *, top_k: int = 10, diversity: bool = False, budget_ms: Optional[int] = None) -> dict:
        # budget_ms is accepted for API compatibility; the local path returns fast. Graceful
        # degradation (omit archive when over budget) is a post-MVP refinement.
        bundle = self.engine.recall(query, top_k=top_k, diversity=diversity)
        return bundle.to_dict()

    # -- deletion --------------------------------------------------------------
    def delete_memory(self, record_id: int, *, mode: str = "logical") -> dict:
        return {"job_id": self.engine.delete_memory(record_id, mode=mode)}

    def deletion_status(self, job_id: str) -> dict:
        return self.engine.deletion_status(job_id)

    # -- inspection ------------------------------------------------------------
    def explain_memory(self, record_id: int) -> dict:
        rec = self.store.read_one(record_id)
        if rec is None:
            raise KeyError(record_id)
        spans = self.store._conn().execute(
            "SELECT span_id, event_id, char_start, char_end, raw_ref FROM source_spans WHERE record_id = ?",
            (record_id,),
        ).fetchall()
        return {
            "record": _summary(rec),
            "confidence": rec.confidence,
            "confidence_reason": rec.confidence_reason,
            "supersedes": self.store.superseders_of(record_id),
            "contradicts": self.store.contradictors_of(record_id),
            "related": sorted(self.store.dependency_graph(record_id)),
            "source_spans": [dict(s) for s in spans],
            "tombstoned": self.store.is_tombstoned(record_id),
        }

    # -- reflection ------------------------------------------------------------
    def run_reflection(self, job: str = "consolidate", window: Optional[str] = None) -> dict:
        if job == "consolidate":
            ids = self.reflection.consolidate()
        elif job == "contradiction_audit":
            ids = self.reflection.contradiction_audit()
        else:
            raise ValueError(f"unknown reflection job {job!r}")
        return {"job": job, "proposal_ids": ids, "count": len(ids)}
